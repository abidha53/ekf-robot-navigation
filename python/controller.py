import math


def _normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class PIDController:

    def __init__(self, kp: float, ki: float, kd: float,
                 output_limit: float, integral_limit: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit

        self._integral = 0.0
        self._previous_error = 0.0
        self._has_previous_error = False

    def reset(self) -> None:
        self._integral = 0.0
        self._previous_error = 0.0
        self._has_previous_error = False

    def compute(self, error: float, dt: float) -> float:
        self._integral = max(-self.integral_limit,
                              min(self.integral_limit, self._integral + error * dt))

        if self._has_previous_error:
            derivative = (error - self._previous_error) / dt
        else:
            derivative = 0.0
        self._previous_error = error
        self._has_previous_error = True

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(-self.output_limit, min(self.output_limit, output))


HEADING_KP = 2.5
HEADING_KI = 0.0
HEADING_KD = 0.3
MAX_ANGULAR_VELOCITY = 2.0

SPEED_TAPER_POWER = 4

LINEAR_KP = 0.8
LINEAR_KI = 0.0
LINEAR_KD = 0.0
MAX_LINEAR_VELOCITY = 1.5

WAYPOINT_TOLERANCE = 0.3


class WaypointController:

    def __init__(self):
        self._heading_pid = PIDController(
            HEADING_KP, HEADING_KI, HEADING_KD,
            output_limit=MAX_ANGULAR_VELOCITY, integral_limit=1.0)
        self._linear_pid = PIDController(
            LINEAR_KP, LINEAR_KI, LINEAR_KD,
            output_limit=MAX_LINEAR_VELOCITY, integral_limit=1.0)

    def reset(self) -> None:
        self._heading_pid.reset()
        self._linear_pid.reset()

    def compute_control(self, estimated_pose: tuple, target_xy: tuple, dt: float) -> tuple:
        x, y, theta = estimated_pose
        target_x, target_y = target_xy

        dx = target_x - x
        dy = target_y - y
        distance = math.hypot(dx, dy)

        desired_heading = math.atan2(dy, dx)
        heading_error = _normalize_angle(desired_heading - theta)

        omega = self._heading_pid.compute(heading_error, dt)
        v = self._linear_pid.compute(distance, dt)

        v *= max(0.0, math.cos(heading_error)) ** SPEED_TAPER_POWER

        return v, omega

    @staticmethod
    def has_reached(estimated_pose: tuple, target_xy: tuple) -> bool:
        distance = math.hypot(target_xy[0] - estimated_pose[0],
                               target_xy[1] - estimated_pose[1])
        return distance < WAYPOINT_TOLERANCE
