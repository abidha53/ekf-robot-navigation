import math
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Ellipse, Wedge
import numpy as np

import environment as env
import perception
import planner
from simulate import SimulationLog, run_simulation, scenario_from_argv, seed_from_argv

ROBOT_DRAW_RADIUS = planner.ROBOT_RADIUS
HEADING_ARROW_LENGTH = 0.7

COVARIANCE_ELLIPSE_N_SIGMA = 2

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


def plot_static_overview(log: SimulationLog, out_path: str) -> plt.Figure:
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
    return fig


def _covariance_ellipse_params(covariance_2x2: np.ndarray) -> tuple:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_2x2)
    major_vec = eigenvectors[:, -1]
    angle_deg = math.degrees(math.atan2(major_vec[1], major_vec[0]))
    width = 2 * COVARIANCE_ELLIPSE_N_SIGMA * math.sqrt(max(eigenvalues[-1], 0.0))
    height = 2 * COVARIANCE_ELLIPSE_N_SIGMA * math.sqrt(max(eigenvalues[0], 0.0))
    return width, height, angle_deg


def _build_animation(ax, log: SimulationLog, stride: int = 2):
    ax.plot(*env.START_POSE[:2], marker="^", markersize=12, color="green")
    ax.plot(*env.GOAL_POSITION, marker="*", markersize=15, color="blue")

    true_line, = ax.plot([], [], color="black", linewidth=2, label="true trajectory")
    ekf_line, = ax.plot([], [], color="limegreen", linewidth=2, label="EKF estimate")
    dr_line, = ax.plot([], [], color="red", linestyle="--", label="dead-reckoning only")
    ax.legend(loc="upper left", fontsize=8)

    frame_indices = list(range(0, len(log.true_poses), stride))
    landmarks_by_id = {lm.id: lm for lm in env.LANDMARKS}
    dynamic_artists = []

    def update(frame_num):
        i = frame_indices[frame_num]

        true_line.set_data([p[0] for p in log.true_poses[:i + 1]],
                            [p[1] for p in log.true_poses[:i + 1]])
        ekf_line.set_data([p[0] for p in log.ekf_estimates[:i + 1]],
                           [p[1] for p in log.ekf_estimates[:i + 1]])
        dr_line.set_data([p[0] for p in log.dead_reckoning_poses[:i + 1]],
                          [p[1] for p in log.dead_reckoning_poses[:i + 1]])

        for artist in dynamic_artists:
            artist.remove()
        dynamic_artists.clear()

        true_x, true_y, true_theta = log.true_poses[i]
        heading_deg = math.degrees(true_theta)

        fov_cone = Wedge((true_x, true_y), perception.CAMERA_MAX_RANGE,
                          heading_deg - perception.CAMERA_FOV_DEG / 2,
                          heading_deg + perception.CAMERA_FOV_DEG / 2,
                          color="deepskyblue", alpha=0.12, zorder=1)
        ax.add_patch(fov_cone)
        dynamic_artists.append(fov_cone)

        robot_body = Circle((true_x, true_y), ROBOT_DRAW_RADIUS,
                             facecolor="navy", edgecolor="black", zorder=4)
        ax.add_patch(robot_body)
        dynamic_artists.append(robot_body)

        heading_line, = ax.plot(
            [true_x, true_x + HEADING_ARROW_LENGTH * math.cos(true_theta)],
            [true_y, true_y + HEADING_ARROW_LENGTH * math.sin(true_theta)],
            color="white", linewidth=2, zorder=5)
        dynamic_artists.append(heading_line)

        ekf_x, ekf_y, _ = log.ekf_estimates[i]
        width, height, angle_deg = _covariance_ellipse_params(log.ekf_covariances[i][:2, :2])
        uncertainty_ellipse = Ellipse((ekf_x, ekf_y), width, height, angle=angle_deg,
                                       facecolor="limegreen", edgecolor="darkgreen",
                                       alpha=0.25, zorder=2)
        ax.add_patch(uncertainty_ellipse)
        dynamic_artists.append(uncertainty_ellipse)

        for landmark_id, _range, _bearing in log.detections_per_step[i]:
            lm = landmarks_by_id[landmark_id]
            highlight = Circle((lm.x, lm.y), 0.6, fill=False,
                                edgecolor="gold", linewidth=2.5, zorder=3)
            ax.add_patch(highlight)
            dynamic_artists.append(highlight)

    return update, len(frame_indices)


def animate_trajectories(log: SimulationLog, out_path: str, stride: int = 2) -> tuple:
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_static_world(ax)
    ax.set_title("Robot localization: true vs EKF vs dead-reckoning")

    update, n_frames = _build_animation(ax, log, stride)

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=40, blit=False)
    anim.save(out_path, writer="pillow", fps=25)
    return fig, anim


def show_live_simulation(log: SimulationLog, stride: int = 2) -> None:
    from matplotlib.widgets import Button

    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_static_world(ax)
    fig.subplots_adjust(bottom=0.14)

    update, n_frames = _build_animation(ax, log, stride)
    update(0)

    title = ax.set_title("Press SPACE or click \"Start Simulation\" to begin")

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                    interval=40, blit=False, repeat=False)
    anim.event_source.stop()

    state = {"started": False}

    def start(_event=None):
        if state["started"]:
            return
        state["started"] = True
        title.set_text("Robot localization: true vs EKF vs dead-reckoning")
        start_button.ax.set_visible(False)
        fig.canvas.draw_idle()
        anim.event_source.start()

    button_ax = fig.add_axes([0.4, 0.02, 0.22, 0.06])
    start_button = Button(button_ax, "Start Simulation")
    start_button.on_clicked(start)

    def on_key(event):
        if event.key == " ":
            start()
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()


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
    live = "--live" in sys.argv
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env.load_scenario(scenario_from_argv())
    scenario = env.SCENARIOS[env.ACTIVE_SCENARIO]
    print(f"Scenario: {scenario.name} (difficulty {scenario.difficulty}/5), {scenario.description}")
    suffix = f"_{scenario.name}"

    seed = seed_from_argv()
    log = run_simulation(odometry_seed=seed, perception_seed=seed + 1)

    overview_fig = plot_static_overview(log, os.path.join(OUTPUT_DIR, f"overview{suffix}.png"))
    anim_fig, anim = animate_trajectories(log, os.path.join(OUTPUT_DIR, f"animation{suffix}.gif"))

    rmse_result = compute_position_rmse(log)
    plot_position_error(rmse_result, os.path.join(OUTPUT_DIR, f"position_error{suffix}.png"))

    perception_result = evaluate_perception_accuracy(log)
    write_metrics_summary(rmse_result, perception_result, log,
                           os.path.join(OUTPUT_DIR, f"metrics_summary{suffix}.txt"))

    print(f"\nAll outputs saved to {os.path.abspath(OUTPUT_DIR)}")

    plt.close(overview_fig)
    plt.close(anim_fig)

    if live:
        print('Opening interactive window, press SPACE or click "Start Simulation" to begin...')
        show_live_simulation(log)


if __name__ == "__main__":
    main()
