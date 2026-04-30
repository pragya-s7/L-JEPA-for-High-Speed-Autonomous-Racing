"""
Wall-Follow controller ported from the ROS2 node in
wall_follow_node.py.

This version is decoupled from ROS2 and works directly with numpy arrays,
making it suitable for use with the f1tenth_gym. Interface matches
follow_the_gap.py:

    steer, speed = controller.plan(scan_ranges, angle_min, angle_increment,
                                   current_speed, dt)
"""
import numpy as np


class WallFollow:
    """
    PID-based left-wall follower (counter-clockwise Levine loop).

    Call: steer, speed = controller.plan(scan_ranges, angle_min, angle_increment,
                                         current_speed, dt)
    """

    def __init__(self):
        # PID gains (from original ROS2 node)
        self.kp = 0.6
        self.kd = 0.2
        self.ki = 0.01

        # Wall-follow geometry
        self.theta = np.pi / 4      # angle between the two beams used for wall estimation
        self.look_ahead = 1.0       # lookahead distance (m)
        self.desired_dist = 2.0    # target distance to the left wall (m)

        # Vehicle/control params (kept consistent with follow_the_gap.py)
        self.steer_limit = 0.4189   # ~24 deg

        # Speed schedule thresholds (deg). Matches original scan_callback logic.
        self.speed_small_deg = 10.0
        self.speed_med_deg = 20.0
        self.v_fast = 1.5
        self.v_med = 1.0
        self.v_slow = 0.5

        # Also expose v_min / v_max so callers/loggers that expect them work.
        self.v_min = self.v_slow
        self.v_max = self.v_fast

        # State
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_delta = 0.0
        self.prev_speed_mag = self.v_min
        self.dt = 0.01  # updated each call

    # ------------------------------------------------------------------
    # Scan helpers
    # ------------------------------------------------------------------
    def get_range(self, range_data, angle, angle_min, angle_increment):
        """
        Return the range measurement at a given angle (radians), handling
        NaN/inf/non-positive values by returning +inf.
        """
        n = len(range_data)
        idx = int(round((angle - angle_min) / angle_increment))
        if idx < 0 or idx >= n:
            return np.inf
        r = float(range_data[idx])
        if not np.isfinite(r) or r <= 0.0:
            return np.inf
        return r

    def get_error(self, range_data, angle_min, angle_increment, dist):
        """
        Compute the signed lateral error to the left wall using the two-beam
        method from the original node:

            a : beam at (pi/2 - theta)   (angled forward-left)
            b : beam at  pi/2            (straight left)

        alpha is the wall-orientation angle; Dt is current perpendicular
        distance; Dt_1 is the projected distance one lookahead ahead.
        """
        horizontal_angle = np.pi / 2
        another_angle = horizontal_angle - self.theta

        horizontal_dist = self.get_range(range_data, horizontal_angle,
                                         angle_min, angle_increment)
        another_dist = self.get_range(range_data, another_angle,
                                      angle_min, angle_increment)

        # Guard against invalid readings so the controller fails gracefully.
        if not np.isfinite(horizontal_dist) or not np.isfinite(another_dist):
            return 0.0

        denom = another_dist * np.sin(self.theta)
        if abs(denom) < 1e-9:
            return 0.0

        alpha = np.arctan((another_dist * np.cos(self.theta) - horizontal_dist)
                          / denom)
        Dt = horizontal_dist * np.cos(alpha)
        Dt_1 = Dt + self.look_ahead * np.sin(alpha)
        return float(Dt_1 - dist)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def plan(self, scan_ranges, angle_min, angle_increment,
             current_speed=None, dt=0.01):
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
        self.dt = float(dt) if dt is not None else 0.01
        if current_speed is not None:
            self.prev_speed_mag = abs(float(current_speed))

        ranges = np.asarray(scan_ranges, dtype=np.float64)

        # 1. Error to the left wall
        error = self.get_error(ranges, angle_min, angle_increment,
                               self.desired_dist)

        # 2. PID -> steering angle
        if self.dt <= 0.0:
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / self.dt

        self.integral += error * self.dt

        angle = (self.kp * error
                 + self.ki * self.integral
                 + self.kd * derivative)
        angle = float(np.clip(angle, -self.steer_limit, self.steer_limit))

        # 3. Speed schedule based on commanded steering magnitude
        est_angle_deg = abs(np.rad2deg(angle))
        if est_angle_deg <= self.speed_small_deg:
            velocity = self.v_fast
        elif est_angle_deg <= self.speed_med_deg:
            velocity = self.v_med
        else:
            velocity = self.v_slow

        # 4. Bookkeeping
        self.prev_error = error
        self.prev_delta = angle
        self.prev_speed_mag = velocity

        return angle, float(velocity)

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_delta = 0.0
        self.prev_speed_mag = self.v_min