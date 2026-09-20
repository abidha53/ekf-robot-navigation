from dataclasses import dataclass


WORLD_WIDTH = 20.0
WORLD_HEIGHT = 20.0

START_POSE = (2.0, 2.0, 0.0)
GOAL_POSITION = (18.0, 18.0)


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    width: float
    height: float

    def contains(self, px: float, py: float, margin: float = 0.0) -> bool:
        return (self.x - margin <= px <= self.x + self.width + margin and
                self.y - margin <= py <= self.y + self.height + margin)


OBSTACLES = [
    Obstacle(x=0.0, y=5.0, width=15.0, height=1.5),
    Obstacle(x=5.0, y=10.0, width=15.0, height=1.5),
    Obstacle(x=0.0, y=14.5, width=15.0, height=1.5),
]


@dataclass(frozen=True)
class Landmark:
    id: int
    x: float
    y: float
    color: tuple
    shape: str


LANDMARKS = [
    Landmark(id=0, x=4.0, y=17.0, color=(220, 30, 30), shape="circle"),
    Landmark(id=1, x=17.0, y=4.0, color=(30, 60, 220), shape="square"),
    Landmark(id=2, x=10.0, y=18.0, color=(40, 180, 60), shape="triangle"),
    Landmark(id=3, x=19.0, y=9.0, color=(230, 200, 20), shape="circle"),
    Landmark(id=4, x=2.0, y=10.0, color=(200, 30, 200), shape="square"),
]


def is_point_free(px: float, py: float, margin: float = 0.0) -> bool:
    if not (0.0 <= px <= WORLD_WIDTH and 0.0 <= py <= WORLD_HEIGHT):
        return False
    return not any(obs.contains(px, py, margin) for obs in OBSTACLES)
