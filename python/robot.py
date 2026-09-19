import math


class DifferentialDriveRobot:

    def __init__(self, x: float, y: float, theta: float):
        self.x = x
        self.y = y
        self.theta = theta

    @property
    def pose(self):
        return (self.x, self.y, self.theta)

    def step(self, v: float, omega: float, dt: float) -> None:
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += omega * dt
        self.theta = _wrap_angle(self.theta)


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
