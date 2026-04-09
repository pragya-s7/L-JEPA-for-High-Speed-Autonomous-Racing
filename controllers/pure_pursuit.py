"""
Pure Pursuit controller ported from:
  f1tenth_gym/examples/waypoint_follow.py

Works directly with numpy; no ROS2 dependency.
Requires a waypoints CSV file.
"""
import numpy as np
from numba import njit


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        tmp = point - projections[i]
        dists[i] = np.sqrt(np.sum(tmp * tmp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


@njit(fastmath=False, cache=True)
def first_point_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    start_i = int(t)
    start_t = t % 1.0
    first_p = None
    first_i = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :] + 1e-6
        V = np.ascontiguousarray(end - start)
        a = np.dot(V, V)
        b = 2.0 * np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            continue
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_p = start + t1 * V
                first_i = i
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_p = start + t2 * V
                first_i = i
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_p = start + t1 * V
            first_i = i
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_p = start + t2 * V
            first_i = i
            break
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + 1e-6
            V = end - start
            a = np.dot(V, V)
            b = 2.0 * np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
            discriminant = b * b - 4 * a * c
            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_p = start + t1 * V
                first_i = i
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_p = start + t2 * V
                first_i = i
                break
    return first_p, first_i


class PurePursuitPlanner:
    """
    Pure Pursuit waypoint-following controller.

    Args:
        waypoint_path (str): path to CSV waypoints file
        delimiter (str): CSV delimiter
        x_col (int): column index for x
        y_col (int): column index for y
        v_col (int): column index for speed
        wheelbase (float): vehicle wheelbase in meters
        lookahead (float): fixed lookahead distance in meters (overrides adaptive if set)
        vgain (float): speed multiplier applied on top of waypoint speed
    """

    def __init__(
        self,
        waypoint_path,
        delimiter=';',
        skiprows=0,
        x_col=1,
        y_col=2,
        v_col=5,
        wheelbase=0.33,
        lookahead=1.5,
        vgain=1.0,
    ):
        self.waypoints = np.loadtxt(waypoint_path, delimiter=delimiter, skiprows=skiprows)
        self.wpts_xy = np.ascontiguousarray(
            np.vstack((self.waypoints[:, x_col], self.waypoints[:, y_col])).T
        )
        self.v_col = v_col
        self.wheelbase = wheelbase
        self.lookahead = lookahead
        self.vgain = vgain
        self.max_reacquire = 20.0

    def plan(self, pose_x, pose_y, pose_theta):
        """
        Compute (speed, steer) given current pose.

        Returns:
            steer (float): steering angle in radians
            speed (float): forward speed in m/s
        """
        position = np.array([pose_x, pose_y])
        nearest_pt, nearest_dist, t, i = nearest_point_on_trajectory(position, self.wpts_xy)

        if nearest_dist < self.lookahead:
            lookahead_pt, _ = first_point_intersecting_circle(
                position, self.lookahead, self.wpts_xy, i + t, wrap=True
            )
            if lookahead_pt is None:
                return 0.0, 1.0
            speed = self.vgain * float(self.waypoints[i, self.v_col])
        elif nearest_dist < self.max_reacquire:
            lookahead_pt = self.wpts_xy[i]
            speed = self.vgain * float(self.waypoints[i, self.v_col])
        else:
            return 0.0, 1.0

        # Transform lookahead point into vehicle frame
        waypoint_y = np.dot(
            np.array([np.sin(-pose_theta), np.cos(-pose_theta)]),
            lookahead_pt - position
        )
        if abs(waypoint_y) < 1e-6:
            return speed, 0.0

        radius = 1.0 / (2.0 * waypoint_y / self.lookahead ** 2)
        steer = float(np.arctan(self.wheelbase / radius))
        return speed, steer
