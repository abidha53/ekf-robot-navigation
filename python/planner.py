import heapq
import math

import environment as env


GRID_RESOLUTION = 0.5

ROBOT_RADIUS = 0.4

_NEIGHBORS = [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
]


def _world_to_grid(x: float, y: float) -> tuple:
    return (int(round(x / GRID_RESOLUTION)), int(round(y / GRID_RESOLUTION)))


def _grid_to_world(cell: tuple) -> tuple:
    return (cell[0] * GRID_RESOLUTION, cell[1] * GRID_RESOLUTION)


def _is_cell_free(cell: tuple) -> bool:
    wx, wy = _grid_to_world(cell)
    return env.is_point_free(wx, wy, margin=ROBOT_RADIUS)


def _heuristic(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def plan_path(start_xy: tuple, goal_xy: tuple) -> list:
    start_cell = _world_to_grid(*start_xy)
    goal_cell = _world_to_grid(*goal_xy)

    g_score = {start_cell: 0.0}
    came_from = {}

    open_heap = [(_heuristic(start_cell, goal_cell), start_cell)]
    visited = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        if current == goal_cell:
            return simplify_path(_reconstruct_path(came_from, current))

        for dx, dy, step_cost in _NEIGHBORS:
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor in visited or not _is_cell_free(neighbor):
                continue

            tentative_g = g_score[current] + step_cost
            if tentative_g < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f_score = tentative_g + _heuristic(neighbor, goal_cell)
                heapq.heappush(open_heap, (f_score, neighbor))

    return []


def _reconstruct_path(came_from: dict, current: tuple) -> list:
    path_cells = [current]
    while current in came_from:
        current = came_from[current]
        path_cells.append(current)
    path_cells.reverse()
    return [_grid_to_world(cell) for cell in path_cells]


def _has_line_of_sight(a: tuple, b: tuple) -> bool:
    dist = math.hypot(b[0] - a[0], b[1] - a[1])
    num_samples = max(2, int(dist / (GRID_RESOLUTION / 2)))
    for i in range(num_samples + 1):
        t = i / num_samples
        x = a[0] + t * (b[0] - a[0])
        y = a[1] + t * (b[1] - a[1])
        if not env.is_point_free(x, y, margin=ROBOT_RADIUS):
            return False
    return True


def simplify_path(waypoints: list) -> list:
    if len(waypoints) <= 2:
        return waypoints

    simplified = [waypoints[0]]
    anchor_idx = 0
    while anchor_idx < len(waypoints) - 1:
        furthest_visible = anchor_idx + 1
        for candidate_idx in range(len(waypoints) - 1, anchor_idx, -1):
            if _has_line_of_sight(waypoints[anchor_idx], waypoints[candidate_idx]):
                furthest_visible = candidate_idx
                break
        simplified.append(waypoints[furthest_visible])
        anchor_idx = furthest_visible

    return simplified
