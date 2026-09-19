import math

import cv2
import numpy as np

import environment as env


IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
CAMERA_FOV_DEG = 90.0
CAMERA_MAX_RANGE = 10.0
BACKGROUND_COLOR = (245, 245, 245)

_HALF_FOV_RAD = math.radians(CAMERA_FOV_DEG / 2.0)
FOCAL_LENGTH_PX = (IMAGE_WIDTH / 2.0) / math.tan(_HALF_FOV_RAD)

LANDMARK_REAL_RADIUS = 0.4

BLUR_KERNEL_SIZE = 5
BRIGHTNESS_JITTER_FRACTION = 0.15
OCCLUSION_PROBABILITY = 0.10
OCCLUSION_COVER_FRACTION = 0.5

MIN_CONTOUR_AREA_PX = 15
_LANDMARK_SIGNATURES = {lm.id: (lm.color, lm.shape) for lm in env.LANDMARKS}


def _relative_range_bearing(true_pose: tuple, landmark_x: float, landmark_y: float) -> tuple:
    x, y, theta = true_pose
    dx = landmark_x - x
    dy = landmark_y - y
    rng = math.hypot(dx, dy)
    bearing = (math.atan2(dy, dx) - theta + math.pi) % (2.0 * math.pi) - math.pi
    return rng, bearing


def _equivalent_pixel_radius(rng: float) -> float:
    return FOCAL_LENGTH_PX * LANDMARK_REAL_RADIUS / rng


def _draw_shape(image: np.ndarray, shape: str, center: tuple, equiv_radius_px: float, color: tuple) -> tuple:
    cx, cy = center
    target_area = math.pi * equiv_radius_px ** 2

    if shape == "circle":
        r = equiv_radius_px
        cv2.circle(image, (int(cx), int(cy)), int(r), color, thickness=-1)
        bbox = (cx - r, cy - r, cx + r, cy + r)

    elif shape == "square":
        half_side = equiv_radius_px * math.sqrt(math.pi) / 2.0
        top_left = (int(cx - half_side), int(cy - half_side))
        bottom_right = (int(cx + half_side), int(cy + half_side))
        cv2.rectangle(image, top_left, bottom_right, color, thickness=-1)
        bbox = (cx - half_side, cy - half_side, cx + half_side, cy + half_side)

    elif shape == "triangle":
        side = equiv_radius_px * math.sqrt(4.0 * math.pi / math.sqrt(3.0))
        h = side * math.sqrt(3.0) / 2.0
        p1 = (cx, cy - 2.0 * h / 3.0)
        p2 = (cx - side / 2.0, cy + h / 3.0)
        p3 = (cx + side / 2.0, cy + h / 3.0)
        pts = np.array([p1, p2, p3], dtype=np.int32)
        cv2.fillPoly(image, [pts], color)
        bbox = (cx - side / 2.0, cy - 2.0 * h / 3.0, cx + side / 2.0, cy + h / 3.0)

    else:
        raise ValueError(f"unknown landmark shape: {shape}")

    del target_area
    return bbox


def render_camera_view(true_pose: tuple, rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    image = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), BACKGROUND_COLOR, dtype=np.uint8)

    for lm in env.LANDMARKS:
        distance, bearing = _relative_range_bearing(true_pose, lm.x, lm.y)

        if distance <= 0.0 or distance > CAMERA_MAX_RANGE:
            continue
        if abs(bearing) > _HALF_FOV_RAD:
            continue

        cx = IMAGE_WIDTH / 2.0 + FOCAL_LENGTH_PX * math.tan(bearing)
        cy = IMAGE_HEIGHT / 2.0

        equiv_radius_px = _equivalent_pixel_radius(distance)
        bbox = _draw_shape(image, lm.shape, (cx, cy), equiv_radius_px, lm.color)

        if rng.random() < OCCLUSION_PROBABILITY:
            _apply_partial_occlusion(image, bbox, rng)

    _apply_blur_and_brightness(image, rng)
    return image


def _apply_partial_occlusion(image: np.ndarray, bbox: tuple, rng: np.random.Generator) -> None:
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    cover_width = width * OCCLUSION_COVER_FRACTION

    if rng.random() < 0.5:
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_min + cover_width), int(y_max)),
                       BACKGROUND_COLOR, thickness=-1)
    else:
        cv2.rectangle(image, (int(x_max - cover_width), int(y_min)), (int(x_max), int(y_max)),
                       BACKGROUND_COLOR, thickness=-1)


def _apply_blur_and_brightness(image: np.ndarray, rng: np.random.Generator) -> None:
    blurred = cv2.GaussianBlur(image, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)

    brightness_scale = 1.0 + rng.uniform(-BRIGHTNESS_JITTER_FRACTION, BRIGHTNESS_JITTER_FRACTION)
    brightened = np.clip(blurred.astype(np.float32) * brightness_scale, 0, 255).astype(np.uint8)

    image[:] = brightened


def _classify_shape(contour) -> str:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
    vertices = len(approx)

    area = cv2.contourArea(contour)
    circularity = 4.0 * math.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0

    if vertices == 3:
        return "triangle"
    if vertices == 4 and circularity < 0.85:
        return "square"
    return "circle"


def _hsv_threshold_range(rgb_color: tuple) -> tuple:
    color_patch = np.uint8([[list(rgb_color)]])
    hsv_color = cv2.cvtColor(color_patch, cv2.COLOR_RGB2HSV)[0, 0]
    hue, sat, val = int(hsv_color[0]), int(hsv_color[1]), int(hsv_color[2])

    hue_tolerance = 10
    lower = np.array([max(0, hue - hue_tolerance), max(40, sat - 90), max(40, val - 90)])
    upper = np.array([min(179, hue + hue_tolerance), 255, 255])
    return lower, upper


def detect_landmarks(image: np.ndarray) -> list:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    detections = []

    for landmark_id, (color, expected_shape) in _LANDMARK_SIGNATURES.items():
        lower, upper = _hsv_threshold_range(color)
        mask = cv2.inRange(hsv_image, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < MIN_CONTOUR_AREA_PX:
            continue

        detected_shape = _classify_shape(largest)
        if detected_shape != expected_shape:
            continue

        moments = cv2.moments(largest)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]

        equiv_radius_px = math.sqrt(area / math.pi)
        estimated_range = FOCAL_LENGTH_PX * LANDMARK_REAL_RADIUS / equiv_radius_px

        estimated_bearing = math.atan2(cx - IMAGE_WIDTH / 2.0, FOCAL_LENGTH_PX)

        detections.append((landmark_id, estimated_range, estimated_bearing))

    return detections
