from dataclasses import dataclass


WORLD_WIDTH = 20.0
WORLD_HEIGHT = 20.0


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    width: float
    height: float

    def contains(self, px: float, py: float, margin: float = 0.0) -> bool:
        return (self.x - margin <= px <= self.x + self.width + margin and
                self.y - margin <= py <= self.y + self.height + margin)


@dataclass(frozen=True)
class Landmark:
    id: int
    x: float
    y: float
    color: tuple
    shape: str


@dataclass(frozen=True)
class Scenario:
    name: str
    difficulty: int
    description: str
    start_pose: tuple
    goal_position: tuple
    obstacles: list
    landmarks: list


_PALETTE = [
    ((220, 30, 30), "circle"),
    ((30, 60, 220), "square"),
    ((40, 180, 60), "triangle"),
    ((230, 200, 20), "circle"),
    ((200, 30, 200), "square"),
]


def _landmarks_at(positions: list) -> list:
    return [Landmark(id=i, x=x, y=y, color=color, shape=shape)
            for i, ((x, y), (color, shape)) in enumerate(zip(positions, _PALETTE))]


_START_POSE = (2.0, 2.0, 0.0)
_GOAL_POSITION = (18.0, 18.0)

SCENARIOS = {
    "easy": Scenario(
        name="easy", difficulty=1,
        description="4 scattered boxes with one detour off the direct path.",
        start_pose=_START_POSE, goal_position=_GOAL_POSITION,
        obstacles=[
            Obstacle(x=5.0, y=5.0, width=3.0, height=8.0),
            Obstacle(x=10.0, y=2.0, width=2.0, height=6.0),
            Obstacle(x=13.0, y=10.0, width=5.0, height=2.0),
            Obstacle(x=6.0, y=14.0, width=4.0, height=3.0),
        ],
        landmarks=_landmarks_at([(4.0, 17.0), (17.0, 4.0), (10.0, 18.0), (19.0, 9.0), (2.0, 10.0)]),
    ),

    "medium": Scenario(
        name="medium", difficulty=2,
        description="3 near full width walls with alternating gaps, crossing the world twice. 5 landmarks.",
        start_pose=_START_POSE, goal_position=_GOAL_POSITION,
        obstacles=[
            Obstacle(x=0.0, y=5.0, width=15.0, height=1.5),
            Obstacle(x=5.0, y=10.0, width=15.0, height=1.5),
            Obstacle(x=0.0, y=14.5, width=15.0, height=1.5),
        ],
        landmarks=_landmarks_at([(4.0, 17.0), (17.0, 4.0), (10.0, 18.0), (19.0, 9.0), (2.0, 10.0)]),
    ),

    "hard": Scenario(
        name="hard", difficulty=3,
        description="4 walls with 4 unit gaps, crossing the world 3 times. 4 landmarks.",
        start_pose=_START_POSE, goal_position=_GOAL_POSITION,
        obstacles=[
            Obstacle(x=0.0, y=4.0, width=16.0, height=1.2),
            Obstacle(x=4.0, y=8.0, width=16.0, height=1.2),
            Obstacle(x=0.0, y=12.0, width=16.0, height=1.2),
            Obstacle(x=4.0, y=16.0, width=16.0, height=1.2),
        ],
        landmarks=_landmarks_at([(10.0, 2.0), (10.0, 10.5), (10.0, 14.5), (10.0, 18.5)]),
    ),

    "very_hard": Scenario(
        name="very_hard", difficulty=4,
        description="5 walls with 4.5 unit gaps, crossing the world 4 times. 3 landmarks: near the start, the middle and the goal.",
        start_pose=_START_POSE, goal_position=_GOAL_POSITION,
        obstacles=[
            Obstacle(x=0.0, y=3.5, width=15.5, height=1.1),
            Obstacle(x=4.5, y=7.0, width=15.5, height=1.1),
            Obstacle(x=0.0, y=10.5, width=15.5, height=1.1),
            Obstacle(x=4.5, y=14.0, width=15.5, height=1.1),
            Obstacle(x=0.0, y=17.5, width=15.5, height=1.1),
        ],
        landmarks=_landmarks_at([(10.0, 1.0), (10.0, 10.75), (10.0, 19.0)]),
    ),

    "extreme": Scenario(
        name="extreme", difficulty=5,
        description="Same maze as 'very_hard' with only 2 landmarks: near the start and near the goal.",
        start_pose=_START_POSE, goal_position=_GOAL_POSITION,
        obstacles=[
            Obstacle(x=0.0, y=3.5, width=15.5, height=1.1),
            Obstacle(x=4.5, y=7.0, width=15.5, height=1.1),
            Obstacle(x=0.0, y=10.5, width=15.5, height=1.1),
            Obstacle(x=4.5, y=14.0, width=15.5, height=1.1),
            Obstacle(x=0.0, y=17.5, width=15.5, height=1.1),
        ],
        landmarks=_landmarks_at([(10.0, 1.0), (10.0, 19.0)]),
    ),
}

DEFAULT_SCENARIO = "medium"


def load_scenario(name: str) -> None:
    global START_POSE, GOAL_POSITION, OBSTACLES, LANDMARKS, ACTIVE_SCENARIO
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Choices: {sorted(SCENARIOS)}")

    scenario = SCENARIOS[name]
    START_POSE = scenario.start_pose
    GOAL_POSITION = scenario.goal_position
    OBSTACLES = scenario.obstacles
    LANDMARKS = scenario.landmarks
    ACTIVE_SCENARIO = name


load_scenario(DEFAULT_SCENARIO)


def is_point_free(px: float, py: float, margin: float = 0.0) -> bool:
    if not (0.0 <= px <= WORLD_WIDTH and 0.0 <= py <= WORLD_HEIGHT):
        return False
    return not any(obs.contains(px, py, margin) for obs in OBSTACLES)
