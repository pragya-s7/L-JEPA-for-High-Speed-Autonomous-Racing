"""
Follow-the-Gap controller ported from the ROS2 node in
lab-4-follow-the-gap-team13/gap_follow/scripts/reactive_node.py

This version is decoupled from ROS2 and works directly with numpy arrays,
making it suitable for use with the f1tenth_gym.
"""
import numpy as np


class FollowTheGap:
    """
    Race-tuned Follow-The-Gap (mapless) controller.

    Call: steer, speed = controller.plan(scan_ranges, angle_min, angle_increment, current_speed)
    """

    def __init__(self):
        # LiDAR + gap params
        self.fov_deg = 85.0
        self.range_max = 6.0
        self.range_min_valid = 0.05

        # Bubble around closest obstacle (meters)
        self.bubble_radius = 0.32
        # Reject corridors that are too tight (meters)
        self.min_corridor = 0.35
        # Threshold small ranges in the forward window (meters)
        self.hard_zero_thresh = 0.25

        # Vehicle/control params
        self.wheelbase = 0.33
        self.steer_limit = 0.4189  # ~24 deg

        # Pure pursuit lookahead: Lh = clamp(Lh_min + Lh_k*|v|, Lh_min, Lh_max)
        self.Lh_min = 0.9
        self.Lh_max = 2.0
        self.Lh_k = 0.25

        # Steering smoothing/rate limit
        self.max_steer_rate = 2.5   # rad/s
        self.steer_alpha = 0.40     # EMA weight for new command

        # Speed planning
        self.v_min = 1.2
        self.v_max = 4.5
        self.a_lat_max = 6.0        # m/s^2
        self.speed_alpha = 0.25

        # Front-distance cap
        self.front_deg = 10.0
        self.front_margin = 0.35
        self.front_k = 1.3

        # Scoring weights
        self.w_progress = 1.00
        self.w_corridor = 1.20
        self.w_balance = 1.40
        self.w_centering = 0.15
        self.w_smooth = 0.70

        # State
        self.prev_delta = 0.0
        self.prev_speed_mag = self.v_min
        self.dt = 0.01  # will be updated each call if desired

    def preprocess(self, ranges):
        r = np.asarray(ranges, dtype=np.float64)
        r[np.isnan(r)] = 0.0
        r[np.isinf(r)] = 0.0
        r[r < 0.0] = 0.0
        r = np.clip(r, 0.0, self.range_max)
        if len(r) >= 5:
            k = np.array([1, 2, 3, 2, 1], dtype=np.float64)
            k /= k.sum()
            rs = r.copy()
            for i in range(2, len(r) - 2):
                rs[i] = np.dot(k, r[i - 2:i + 3])
            r = rs
        return r

    def find_max_gap(self, r):
        best_s, best_e = 0, -1
        curr_s = None
        for i, v in enumerate(r):
            if v > 0.0:
                if curr_s is None:
                    curr_s = i
            else:
                if curr_s is not None:
                    if (i - 1) - curr_s > best_e - best_s:
                        best_s, best_e = curr_s, i - 1
                    curr_s = None
        if curr_s is not None and (len(r) - 1) - curr_s > best_e - best_s:
            best_s, best_e = curr_s, len(r) - 1
        return best_s, best_e

    def apply_bubble(self, r, angles, closest_i, closest_dist):
        if closest_dist <= 1e-6 or len(angles) < 2:
            return r
        ang_inc = float(abs(angles[1] - angles[0]))
        bubble_ang = np.arcsin(np.clip(self.bubble_radius / closest_dist, 0.0, 1.0))
        w = int(bubble_ang / (ang_inc + 1e-9))
        i0 = max(0, closest_i - w)
        i1 = min(len(r) - 1, closest_i + w)
        r[i0:i1 + 1] = 0.0
        return r

    def corridor_clearance(self, r, i, half_window):
        a = max(0, i - half_window)
        b = min(len(r) - 1, i + half_window)
        seg = r[a:b + 1]
        seg = seg[seg > 0.0]
        return float(np.min(seg)) if seg.size > 0 else 0.0

    def balance_score(self, r, i, half_window):
        left = r[max(0, i - half_window):i]
        right = r[i + 1:min(len(r), i + 1 + half_window)]
        left = left[left > 0.0]
        right = right[right > 0.0]
        if left.size == 0 or right.size == 0:
            return 0.0
        dl, dr = float(np.min(left)), float(np.min(right))
        return 1.0 - abs(dl - dr) / (max(dl, dr) + 1e-6)

    def pick_best_point(self, r, angles, s, e):
        if e <= s:
            return None
        candidates = np.arange(s, e + 1, 2)
        if candidates.size == 0:
            return None
        ang_inc = float(abs(angles[1] - angles[0])) if len(angles) > 1 else 0.01
        half_window = max(1, int(np.deg2rad(5.0) / (ang_inc + 1e-9)))

        r_cand = r[candidates]
        a_cand = angles[candidates]

        prog = r_cand / (np.max(r_cand) + 1e-6)
        corr = np.array([self.corridor_clearance(r, int(i), half_window) for i in candidates])
        corr_n = np.clip(corr, 0.0, self.range_max) / self.range_max
        tight_penalty = np.where(corr < self.min_corridor, -1.0, 0.0)
        bal = np.array([self.balance_score(r, int(i), half_window) for i in candidates])
        center = 1.0 - np.abs(a_cand) / (np.max(np.abs(a_cand)) + 1e-6)
        diff = np.abs(a_cand - self.prev_delta)
        smooth = 1.0 - diff / (np.max(diff) + 1e-6)

        score = (
            self.w_progress * prog +
            self.w_corridor * corr_n +
            self.w_balance * bal +
            self.w_centering * center +
            self.w_smooth * smooth +
            tight_penalty
        )
        return int(candidates[int(np.argmax(score))])

    def plan(self, scan_ranges, angle_min, angle_increment, current_speed=None, dt=0.01):
        """
        Compute (steer, speed) from a raw LiDAR scan.

        Args:
            scan_ranges: array of range readings (num_beams,)
            angle_min: minimum scan angle in radians
            angle_increment: angular step between beams in radians
            current_speed: current forward speed (m/s), optional
            dt: time since last call (s)

        Returns:
            steer (float): steering angle in radians
            speed (float): forward speed in m/s
        """
        self.dt = dt
        if current_speed is not None:
            self.prev_speed_mag = abs(current_speed)

        r = self.preprocess(scan_ranges)
        N = len(r)

        # FOV slice
        fov = np.deg2rad(self.fov_deg)
        start = int((-fov - angle_min) / angle_increment)
        end = int((fov - angle_min) / angle_increment)
        start = max(0, start)
        end = min(N - 1, end)
        if end <= start:
            return 0.0, self.v_min

        idx = np.arange(start, end + 1)
        angles = angle_min + idx * angle_increment

        sub = r[start:end + 1].copy()
        sub[sub < self.range_min_valid] = 0.0
        if np.all(sub <= 0.0):
            return 0.0, self.v_min

        valid = sub > 0.0
        closest_local = int(np.argmin(np.where(valid, sub, np.inf)))
        closest_dist = float(sub[closest_local])

        sub = self.apply_bubble(sub, angles, closest_local, closest_dist)
        sub[sub < self.hard_zero_thresh] = 0.0

        s, e = self.find_max_gap(sub)
        if e <= s:
            return 0.0, self.v_min

        best_i = self.pick_best_point(sub, angles, s, e)
        if best_i is None:
            return 0.0, self.v_min

        target_angle = float(angles[best_i])

        # Pure-pursuit steering
        Lh = float(np.clip(self.Lh_min + self.Lh_k * self.prev_speed_mag, self.Lh_min, self.Lh_max))
        delta_target = float(np.clip(
            np.arctan2(2.0 * self.wheelbase * np.sin(target_angle), Lh),
            -self.steer_limit, self.steer_limit
        ))

        # Rate limit + EMA smooth
        max_step = self.max_steer_rate * self.dt
        delta_rl = float(self.prev_delta + np.clip(delta_target - self.prev_delta, -max_step, max_step))
        delta_cmd = float(np.clip(
            self.steer_alpha * delta_rl + (1.0 - self.steer_alpha) * self.prev_delta,
            -self.steer_limit, self.steer_limit
        ))

        # Speed from curvature
        kappa = np.tan(delta_cmd) / self.wheelbase
        v_curve = float(np.clip(np.sqrt(self.a_lat_max / (abs(kappa) + 1e-3)), self.v_min, self.v_max))

        # Front-distance cap
        front_mask = np.abs(angles) < np.deg2rad(self.front_deg)
        front_vals = sub[front_mask]
        front_vals = front_vals[front_vals > 0.0]
        d_front = float(np.min(front_vals)) if front_vals.size > 0 else self.range_max
        v_dist = float(np.clip(
            self.front_k * np.sqrt(max(0.0, d_front - self.front_margin)),
            self.v_min, self.v_max
        ))

        v_mag = min(v_curve, v_dist)
        v_cmd = float(self.speed_alpha * v_mag + (1.0 - self.speed_alpha) * self.prev_speed_mag)

        self.prev_delta = delta_cmd
        self.prev_speed_mag = v_cmd

        return delta_cmd, v_cmd

    def reset(self):
        self.prev_delta = 0.0
        self.prev_speed_mag = self.v_min
