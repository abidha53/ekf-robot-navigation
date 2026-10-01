import math


class DifferentialDriveRobot:

    def __init__(self, x: float, y: float, theta: float):
        self.x = x
        self.y = y
        self.theta = theta

    @property
    def pose(self):
        return (self.x, self.y, self.theta)

    def predict_pose(self, v: float, omega: float, dt: float) -> tuple:
        x = self.x + v * math.cos(self.theta) * dt
        y = self.y + v * math.sin(self.theta) * dt
        theta = _wrap_angle(self.theta + omega * dt)
        return (x, y, theta)

    def step(self, v: float, omega: float, dt: float) -> None:
        self.x, self.y, self.theta = self.predict_pose(v, omega, dt)


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
