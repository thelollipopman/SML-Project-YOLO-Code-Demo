from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence
from ultralytics import YOLO


# --------------------------
# Model and keypoint indices
# --------------------------
MODEL = YOLO("yolo26n-pose.pt")

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10

CONF_THRESH = 0.4
FOREARM_MARGIN = 15
HISTORY_LEN = 25
DEFAULT_COOLDOWN_FRAMES = 20
DEFAULT_GIF_SCALE = 0.6
WINDOW_NAME = "Meme pose detector"

debug = None


# --------------------------
# Utility helpers
# --------------------------
def load_gif_frames(path: Path, scale: float = 0.5):
    gif = Image.open(path)
    frames = []
    durations = []

    for frame in ImageSequence.Iterator(gif):
        frame = frame.convert("RGBA")
        frame_np = np.array(frame)

        if scale != 1.0:
            height, width = frame_np.shape[:2]
            frame_np = cv2.resize(
                frame_np,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )

        frames.append(frame_np)
        duration_ms = frame.info.get("duration", 40)
        durations.append(max(duration_ms / 1000.0, 0.02))

    return frames, durations


def overlay_rgba(background, overlay, x, y):
    """Overlay an RGBA image on top of a BGR OpenCV frame."""
    bg_height, bg_width = background.shape[:2]
    ov_height, ov_width = overlay.shape[:2]

    if x >= bg_width or y >= bg_height:
        return background
    if x + ov_width <= 0 or y + ov_height <= 0:
        return background

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + ov_width, bg_width)
    y2 = min(y + ov_height, bg_height)

    ov_x1 = x1 - x
    ov_y1 = y1 - y
    ov_x2 = ov_x1 + (x2 - x1)
    ov_y2 = ov_y1 + (y2 - y1)

    overlay_crop = overlay[ov_y1:ov_y2, ov_x1:ov_x2]
    overlay_rgb = overlay_crop[:, :, :3]
    alpha = overlay_crop[:, :, 3:] / 255.0
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

    bg_crop = background[y1:y2, x1:x2].astype(float)
    fg_crop = overlay_bgr.astype(float)
    blended = alpha * fg_crop + (1 - alpha) * bg_crop
    background[y1:y2, x1:x2] = blended.astype(np.uint8)
    return background


def get_now_seconds():
    return cv2.getTickCount() / cv2.getTickFrequency()


def point_dist(a, b):
    return np.linalg.norm(a[:2] - b[:2])


def angle_deg(a, b, c):
    """Return angle ABC in degrees."""
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return None

    cos_theta = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def keypoints_confident(kp, indices, conf_thresh=CONF_THRESH):
    return all(kp[i, 2] >= conf_thresh for i in indices)


def get_largest_person_index(result):
    if result.boxes is None or len(result.boxes) == 0:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()
    areas = []
    for x1, y1, x2, y2 in boxes:
        areas.append((x2 - x1) * (y2 - y1))

    return int(np.argmax(areas))


# --------------------------
# Pose detectors
# --------------------------
def get_motion_state(kp):
    needed = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_ELBOW,
        RIGHT_ELBOW,
        LEFT_WRIST,
        RIGHT_WRIST,
    ]

    if not keypoints_confident(kp, needed):
        return None

    left_elbow = kp[LEFT_ELBOW]
    right_elbow = kp[RIGHT_ELBOW]
    left_wrist = kp[LEFT_WRIST]
    right_wrist = kp[RIGHT_WRIST]

    left_up = left_wrist[1] < left_elbow[1] - FOREARM_MARGIN
    left_down = left_wrist[1] > left_elbow[1] + FOREARM_MARGIN
    right_up = right_wrist[1] < right_elbow[1] - FOREARM_MARGIN
    right_down = right_wrist[1] > right_elbow[1] + FOREARM_MARGIN

    if left_up and right_down:
        return "L_up_R_down"
    if left_down and right_up:
        return "L_down_R_up"
    return "mid"


def detect_six_seven(history):
    compact = []
    valid_states = ["L_up_R_down", "L_down_R_up"]

    for state in history:
        if state in valid_states:
            if not compact or compact[-1] != state:
                compact.append(state)

    if len(compact) < 3:
        return False

    last3 = compact[-3:]
    return last3 in [
        ["L_up_R_down", "L_down_R_up", "L_up_R_down"],
        ["L_down_R_up", "L_up_R_down", "L_down_R_up"],
    ]


def detect_dab_pose(kp):
    needed = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_ELBOW,
        RIGHT_ELBOW,
        LEFT_WRIST,
        RIGHT_WRIST,
    ]

    if not keypoints_confident(kp, needed):
        return None

    ls = kp[LEFT_SHOULDER]
    rs = kp[RIGHT_SHOULDER]
    le = kp[LEFT_ELBOW]
    re = kp[RIGHT_ELBOW]
    lw = kp[LEFT_WRIST]
    rw = kp[RIGHT_WRIST]

    shoulder_width = point_dist(ls, rs)
    if shoulder_width < 1:
        return None

    left_angle = angle_deg(ls, le, lw)
    right_angle = angle_deg(rs, re, rw)
    if left_angle is None or right_angle is None:
        return None

    left_arm_up = lw[1] < le[1] < ls[1]
    left_arm_straight = left_angle > 145
    right_arm_bent = right_angle < 110
    right_wrist_near_left_side = point_dist(rw, ls) < 1.2 * shoulder_width

    if left_arm_up and left_arm_straight and right_arm_bent and right_wrist_near_left_side:
        return "left_dab"

    right_arm_up = rw[1] < re[1] < rs[1]
    right_arm_straight = right_angle > 145
    left_arm_bent = left_angle < 110
    left_wrist_near_right_side = point_dist(lw, rs) < 1.2 * shoulder_width

    if right_arm_up and right_arm_straight and left_arm_bent and left_wrist_near_right_side:
        return "right_dab"

    return None


def detect_t_pose(kp):
    global debug
    needed = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_ELBOW,
        RIGHT_ELBOW,
        LEFT_WRIST,
        RIGHT_WRIST,
    ]

    if not keypoints_confident(kp, needed):
        return None

    ls = kp[LEFT_SHOULDER]
    rs = kp[RIGHT_SHOULDER]
    le = kp[LEFT_ELBOW]
    re = kp[RIGHT_ELBOW]
    lw = kp[LEFT_WRIST]
    rw = kp[RIGHT_WRIST]

    shoulder_width = point_dist(ls, rs)
    if shoulder_width < 1:
        return None

    left_angle = angle_deg(ls, le, lw)
    right_angle = angle_deg(rs, re, rw)
    if left_angle is None or right_angle is None:
        return None

    shoulder_y = 0.5 * (ls[1] + rs[1])

    arms_straight = left_angle > 155 and right_angle > 155
    wrists_outward = (
        lw[0] < ls[0] - 0.35 * shoulder_width
        and rw[0] > rs[0] + 0.35 * shoulder_width
    )
    elbows_outward = le[0] < ls[0] and re[0] > rs[0]
    wrists_near_shoulder_height = (
        abs(lw[1] - shoulder_y) < 0.55 * shoulder_width
        and abs(rw[1] - shoulder_y) < 0.55 * shoulder_width
    )
    elbows_near_shoulder_height = (
        abs(le[1] - shoulder_y) < 0.50 * shoulder_width
        and abs(re[1] - shoulder_y) < 0.50 * shoulder_width
    )
    wrists_level = abs(lw[1] - rw[1]) < 0.20 * shoulder_width
    wide_span = abs(rw[0] - lw[0]) > 2.3 * shoulder_width

    debug = f"{lw[0]: .2f} {ls[0]: .2f}"

    if (
        arms_straight
        # and wrists_outward
        # and elbows_outward
        and wrists_near_shoulder_height
        and elbows_near_shoulder_height
        and wrists_level
        and wide_span
    ):
        return "tpose"

    return None


def detect_triggered_pose(kp, history):
    dab_state = detect_dab_pose(kp)
    tpose_state = detect_t_pose(kp)

    if detect_six_seven(history):
        return "67", {"dab_state": dab_state, "tpose_state": tpose_state}

    if dab_state is not None:
        return "dab", {"dab_state": dab_state, "tpose_state": tpose_state}

    if tpose_state is not None:
        return "tpose", {"dab_state": dab_state, "tpose_state": tpose_state}

    return None, {"dab_state": dab_state, "tpose_state": tpose_state}


# --------------------------
# Effect / animation control
# --------------------------
def init_app_state():
    return {
        "cooldown": 0,
        "detection_timer": 0,
        "active_detection_label": "",
        "gif_playing": False,
        "gif_index": 0,
        "gif_loops_left": 0,
        "last_gif_time": 0.0,
        "active_gif_name": None,
        "active_gif_frames": [],
        "active_gif_durations": [],
    }


def start_pose_effect(app_state, pose_name, pose_config, gifs):
    app_state["detection_timer"] = pose_config.get("timer_frames", 20)
    app_state["cooldown"] = pose_config.get("cooldown_frames", DEFAULT_COOLDOWN_FRAMES)
    app_state["active_detection_label"] = pose_config["label"]

    gif_key = pose_config.get("gif")
    gif_data = gifs.get(gif_key) if gif_key else None

    if gif_data is None:
        app_state["gif_playing"] = False
        app_state["gif_index"] = 0
        app_state["gif_loops_left"] = 0
        app_state["last_gif_time"] = 0.0
        app_state["active_gif_name"] = None
        app_state["active_gif_frames"] = []
        app_state["active_gif_durations"] = []
        return

    frames, durations = gif_data
    app_state["gif_playing"] = True
    app_state["gif_index"] = 0
    app_state["gif_loops_left"] = pose_config.get("loops", 1)
    app_state["last_gif_time"] = get_now_seconds()
    app_state["active_gif_name"] = pose_name
    app_state["active_gif_frames"] = frames
    app_state["active_gif_durations"] = durations


def update_timers(app_state):
    if app_state["cooldown"] > 0:
        app_state["cooldown"] -= 1
    if app_state["detection_timer"] > 0:
        app_state["detection_timer"] -= 1

def draw_status_text(frame, current_state, dab_state, tpose_state, app_state):
    global debug
    cv2.putText(
        frame,
        f"state: {current_state}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # cv2.putText(
    #     frame,
    #     f"dab: {dab_state}",
    #     (20, 85),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.8,
    #     (255, 255, 0),
    #     2,
    # )

    # cv2.putText(
    #     frame,
    #     f"tpose: {tpose_state}",
    #     (20, 115),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.8,
    #     (255, 200, 0),
    #     2,
    # )
    # cv2.putText(
    #     frame,
    #     f"debug: {debug}",
    #     (20, 140),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.8,
    #     (255, 200, 0),
    #     2,
    # )

    if app_state["detection_timer"] > 0:
        cv2.putText(
            frame,
            app_state["active_detection_label"],
            (20, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )


def update_and_draw_gif(frame, app_state, x_margin=20, y_margin=20):
    if not app_state["gif_playing"] or not app_state["active_gif_frames"]:
        return frame

    current_gif = app_state["active_gif_frames"][app_state["gif_index"]]
    gif_height, gif_width = current_gif.shape[:2]
    frame_height, frame_width = frame.shape[:2]

    x = frame_width - gif_width - x_margin
    y = y_margin
    frame = overlay_rgba(frame, current_gif, x, y)

    now = get_now_seconds()
    current_duration = app_state["active_gif_durations"][app_state["gif_index"]]

    if now - app_state["last_gif_time"] >= current_duration:
        app_state["gif_index"] += 1
        app_state["last_gif_time"] = now

        if app_state["gif_index"] >= len(app_state["active_gif_frames"]):
            app_state["gif_index"] = 0
            app_state["gif_loops_left"] -= 1
            if app_state["gif_loops_left"] <= 0:
                app_state["gif_playing"] = False
                app_state["active_gif_name"] = None
                app_state["active_gif_frames"] = []
                app_state["active_gif_durations"] = []

    return frame


# --------------------------
# Pose registry and assets
# --------------------------
def build_pose_registry():
    return {
        "67": {
            "label": "6 7 DETECTED",
            "gif": "67",
            "loops": 2,
            "cooldown_frames": 20,
            "timer_frames": 20,
        },
        "dab": {
            "label": "DAB DETECTED",
            "gif": "dab",
            "loops": 1,
            "cooldown_frames": 20,
            "timer_frames": 20,
        },
        "tpose": {
            "label": "T POSE DETECTED",
            "gif": "tpose",
            "loops": 1,
            "cooldown_frames": 20,
            "timer_frames": 20,
        },
    }


def build_gif_registry(base_dir: Path):
    gif_dir = base_dir / "gifs"
    gifs = {}

    for name in ["67", "dab", "tpose"]:
        gif_path = gif_dir / f"{name}.gif"
        if gif_path.exists():
            gifs[name] = load_gif_frames(gif_path, scale=DEFAULT_GIF_SCALE)

    return gifs


# --------------------------
# Main application loop
# --------------------------
def run():
    base_dir = Path(__file__).resolve().parent
    poses = build_pose_registry()
    gifs = build_gif_registry(base_dir)
    app_state = init_app_state()

    cap = cv2.VideoCapture(0)
    history = deque(maxlen=HISTORY_LEN)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = frame.copy()
            current_state = None
            dab_state = None
            tpose_state = None

            results = MODEL(frame, verbose=False)

            if results:
                result = results[0]
                person_idx = get_largest_person_index(result)

                if result.keypoints is not None and person_idx is not None:
                    kpts = result.keypoints.data

                    if kpts is not None and len(kpts) > person_idx:
                        kp = kpts[person_idx].cpu().numpy()
                        current_state = get_motion_state(kp)
                        history.append(current_state)

                        pose_name, debug_info = detect_triggered_pose(kp, history)
                        dab_state = debug_info["dab_state"]
                        tpose_state = debug_info["tpose_state"]

                        if app_state["cooldown"] == 0 and pose_name is not None:
                            start_pose_effect(app_state, pose_name, poses[pose_name], gifs)

                        annotated = result.plot()
                    else:
                        history.append(None)
                else:
                    history.append(None)
            else:
                history.append(None)

            update_timers(app_state)
            draw_status_text(annotated, current_state, dab_state, tpose_state, app_state)
            annotated = update_and_draw_gif(annotated, app_state)

            cv2.imshow(WINDOW_NAME, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
