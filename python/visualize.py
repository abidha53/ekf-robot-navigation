import math
import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import environment as env
import perception
from simulate import SimulationLog, run_simulation

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def _draw_static_world(ax) -> None:
    for obs in env.OBSTACLES:
        ax.add_patch(plt.Rectangle((obs.x, obs.y), obs.width, obs.height,
                                    color="gray", alpha=0.4, zorder=1))
    for lm in env.LANDMARKS:
        color = tuple(c / 255.0 for c in lm.color)
        ax.plot(lm.x, lm.y, marker="o", markersize=10, color=color, zorder=2)
        ax.annotate(f"L{lm.id}", (lm.x, lm.y), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)

    ax.set_xlim(0, env.WORLD_WIDTH)
    ax.set_ylim(0, env.WORLD_HEIGHT)
    ax.set_aspect("equal")


def plot_static_overview(log: SimulationLog, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_static_world(ax)

    path_xs = [p[0] for p in log.planned_path]
    path_ys = [p[1] for p in log.planned_path]
    ax.plot(path_xs, path_ys, "o--", color="orange", alpha=0.7,
            label="planned path (A*)", zorder=2)

    true_xs = [p[0] for p in log.true_poses]
    true_ys = [p[1] for p in log.true_poses]
    ekf_xs = [p[0] for p in log.ekf_estimates]
    ekf_ys = [p[1] for p in log.ekf_estimates]
    dr_xs = [p[0] for p in log.dead_reckoning_poses]
    dr_ys = [p[1] for p in log.dead_reckoning_poses]

    ax.plot(true_xs, true_ys, color="black", linewidth=2, label="true trajectory", zorder=3)
    ax.plot(ekf_xs, ekf_ys, color="limegreen", linewidth=2, label="EKF estimate", zorder=3)
    ax.plot(dr_xs, dr_ys, color="red", linestyle="--", label="dead-reckoning only", zorder=3)

    ax.plot(*env.START_POSE[:2], marker="^", markersize=12, color="green", label="start")
    ax.plot(*env.GOAL_POSITION, marker="*", markersize=15, color="blue", label="goal")

    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Planned path vs true / EKF / dead-reckoning trajectories")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def animate_trajectories(log: SimulationLog, out_path: str, stride: int = 2) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_static_world(ax)
    ax.plot(*env.START_POSE[:2], marker="^", markersize=12, color="green")
    ax.plot(*env.GOAL_POSITION, marker="*", markersize=15, color="blue")

    true_line, = ax.plot([], [], color="black", linewidth=2, label="true trajectory")
    ekf_line, = ax.plot([], [], color="limegreen", linewidth=2, label="EKF estimate")
    dr_line, = ax.plot([], [], color="red", linestyle="--", label="dead-reckoning only")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Robot localization: true vs EKF vs dead-reckoning")

    frame_indices = list(range(0, len(log.true_poses), stride))

    def update(frame_num):
        i = frame_indices[frame_num]
        true_line.set_data([p[0] for p in log.true_poses[:i + 1]],
                            [p[1] for p in log.true_poses[:i + 1]])
        ekf_line.set_data([p[0] for p in log.ekf_estimates[:i + 1]],
                           [p[1] for p in log.ekf_estimates[:i + 1]])
        dr_line.set_data([p[0] for p in log.dead_reckoning_poses[:i + 1]],
                          [p[1] for p in log.dead_reckoning_poses[:i + 1]])
        return true_line, ekf_line, dr_line

    anim = animation.FuncAnimation(fig, update, frames=len(frame_indices),
                                    interval=40, blit=True)
    anim.save(out_path, writer="pillow", fps=25)
    plt.close(fig)


def compute_position_rmse(log: SimulationLog) -> dict:
    ekf_errors = []
    dr_errors = []
    for true_pose, ekf_pose, dr_pose in zip(log.true_poses, log.ekf_estimates, log.dead_reckoning_poses):
        ekf_errors.append(math.hypot(true_pose[0] - ekf_pose[0], true_pose[1] - ekf_pose[1]))
        dr_errors.append(math.hypot(true_pose[0] - dr_pose[0], true_pose[1] - dr_pose[1]))

    return {
        "ekf_errors": ekf_errors,
        "dr_errors": dr_errors,
        "ekf_rmse": math.sqrt(np.mean(np.square(ekf_errors))),
        "dr_rmse": math.sqrt(np.mean(np.square(dr_errors))),
    }


def plot_position_error(rmse_result: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rmse_result["ekf_errors"], color="limegreen", label="EKF-fused error")
    ax.plot(rmse_result["dr_errors"], color="red", linestyle="--", label="dead-reckoning-only error")
    ax.set_xlabel("timestep")
    ax.set_ylabel("position error (world units)")
    ax.set_title(f"Position error over time  (EKF RMSE={rmse_result['ekf_rmse']:.3f}, "
                 f"dead-reckoning RMSE={rmse_result['dr_rmse']:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def evaluate_perception_accuracy(log: SimulationLog) -> dict:
    landmarks_by_id = {lm.id: lm for lm in env.LANDMARKS}

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    range_errors = []
    bearing_errors = []

    for true_pose, detections in zip(log.true_poses, log.detections_per_step):
        expected_ids = set()
        ground_truth = {}
        for lm in env.LANDMARKS:
            d, b = perception._relative_range_bearing(true_pose, lm.x, lm.y)
            if 0 < d <= perception.CAMERA_MAX_RANGE and abs(b) <= perception._HALF_FOV_RAD:
                expected_ids.add(lm.id)
                ground_truth[lm.id] = (d, b)

        detected_ids = set()
        for landmark_id, est_range, est_bearing in detections:
            detected_ids.add(landmark_id)
            if landmark_id in ground_truth:
                true_range, true_bearing = ground_truth[landmark_id]
                range_errors.append(abs(est_range - true_range))
                bearing_errors.append(abs(est_bearing - true_bearing))

        true_positives += len(expected_ids & detected_ids)
        false_positives += len(detected_ids - expected_ids)
        false_negatives += len(expected_ids - detected_ids)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else float("nan")
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else float("nan")

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "mean_range_error": float(np.mean(range_errors)) if range_errors else float("nan"),
        "mean_bearing_error_deg": math.degrees(float(np.mean(bearing_errors))) if bearing_errors else float("nan"),
    }


def write_metrics_summary(rmse_result: dict, perception_result: dict, log: SimulationLog, out_path: str) -> None:
    lines = [
        "=== Simulation result ===",
        f"Goal reached: {log.goal_reached}",
        f"Steps taken: {log.steps_taken}",
        "",
        "=== Metric 1: Localization accuracy (EKF-fused vs dead-reckoning-only) ===",
        f"EKF-fused RMSE:      {rmse_result['ekf_rmse']:.4f} world units",
        f"Dead-reckoning RMSE: {rmse_result['dr_rmse']:.4f} world units",
        f"Improvement factor:  {rmse_result['dr_rmse'] / rmse_result['ekf_rmse']:.2f}x",
        "",
        "=== Metric 2/3: Perception (classical CV) accuracy ===",
        f"True positives:  {perception_result['true_positives']}",
        f"False positives: {perception_result['false_positives']}",
        f"False negatives: {perception_result['false_negatives']}",
        f"Precision (of reported detections, how many were real): {perception_result['precision']:.1%}",
        f"Recall (of visible landmarks, how many were caught):     {perception_result['recall']:.1%}",
        f"Mean range error on correct detections:   {perception_result['mean_range_error']:.4f} world units",
        f"Mean bearing error on correct detections: {perception_result['mean_bearing_error_deg']:.3f} degrees",
        "",
        "This shows how perception noise propagates into localization: the",
        "EKF's accuracy above is bounded by these range/bearing errors, not",
        "by the EKF math itself (which is exact given its inputs).",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log = run_simulation()

    plot_static_overview(log, os.path.join(OUTPUT_DIR, "overview.png"))
    animate_trajectories(log, os.path.join(OUTPUT_DIR, "animation.gif"))

    rmse_result = compute_position_rmse(log)
    plot_position_error(rmse_result, os.path.join(OUTPUT_DIR, "position_error.png"))

    perception_result = evaluate_perception_accuracy(log)
    write_metrics_summary(rmse_result, perception_result, log,
                           os.path.join(OUTPUT_DIR, "metrics_summary.txt"))

    print(f"\nAll outputs saved to {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
