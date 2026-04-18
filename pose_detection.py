from ultralytics import YOLO
import cv2
import math
from PIL import Image, ImageSequence
from collections import deque
import numpy as np
from pathlib import Path

# Call the pose model 
model = YOLO("yolo26n-pose.pt")


# Joint connection points
left_shoulder = 5
right_shoulder = 6
left_elbow = 7
right_elbow = 8
left_wrist = 9
right_wrist = 10

# Thresholds
conf_thresh = 0.4
motion_thresh = 5
band_tolerance = 25
forearm_margin = 15

# Temporal / Time-related
history_len = 25
cooldown_frames = 20

cap = cv2.VideoCapture(0)

history = deque(maxlen=history_len)
prev_kp = None

cooldown = 0
detection_timer = 0

def load_gif_frames(path, scale=0.5):
    gif = Image.open(path)
    frames = []
    durations = []

    for frame in ImageSequence.Iterator(gif):
        frame = frame.convert("RGBA")
        frame_np = np.array(frame)

        if scale != 1.0:
            h, w = frame_np.shape[:2]
            frame_np = cv2.resize(
                frame_np,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA
            )

        frames.append(frame_np)

        # duration is in milliseconds in PIL
        duration_ms = frame.info.get("duration", 40)
        durations.append(max(duration_ms / 1000.0, 0.02))

    return frames, durations


def overlay_rgba(background, overlay, x, y):
    """
    background: BGR image (OpenCV frame)
    overlay:    RGBA image
    x, y:       top-left corner where overlay is placed
    """
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]

    # Clip to screen
    if x >= bw or y >= bh:
        return background
    if x + ow <= 0 or y + oh <= 0:
        return background

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + ow, bw)
    y2 = min(y + oh, bh)

    overlay_x1 = x1 - x
    overlay_y1 = y1 - y
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    overlay_crop = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

    # Split RGBA
    overlay_rgb = overlay_crop[:, :, :3]
    alpha = overlay_crop[:, :, 3:] / 255.0

    # PIL gives RGB, OpenCV uses BGR
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

    bg_crop = background[y1:y2, x1:x2].astype(float)
    fg_crop = overlay_bgr.astype(float)

    blended = alpha * fg_crop + (1 - alpha) * bg_crop
    background[y1:y2, x1:x2] = blended.astype(np.uint8)

    return background

def keypoints_confident(kp, indices, conf_thresh=conf_thresh):
    for i in indices:
        if kp[i, 2] < conf_thresh:
            return False
    return True


def get_largest_person_index(result):
    if result.boxes is None or len(result.boxes) == 0:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()
    areas = []

    for box in boxes:
        x1, y1, x2, y2 = box
        areas.append((x2 - x1) * (y2 - y1))

    return int(np.argmax(areas))


def get_motion_state(kp, prev_kp=None):
    needed = [
        left_shoulder, right_shoulder,
        left_elbow, right_elbow,
        left_wrist, right_wrist
    ]

    if not keypoints_confident(kp, needed):
        return None

    le = kp[left_elbow]
    re = kp[right_elbow]
    lw = kp[left_wrist]
    rw = kp[right_wrist]

    left_up = lw[1] < le[1] - forearm_margin
    left_down = lw[1] > le[1] + forearm_margin

    right_up = rw[1] < re[1] - forearm_margin
    right_down = rw[1] > re[1] + forearm_margin

    if left_up and right_down:
        return "L_up_R_down"
    elif left_down and right_up:
        return "L_down_R_up"
    else:
        return "mid"


def detect_six_seven(history):
    compact = []
    valid_states = ["L_up_R_down", "L_down_R_up"]

    for state in history:
        if state in valid_states:
            if len(compact) == 0 or compact[-1] != state:
                compact.append(state)

    if len(compact) < 3:
        return False

    last3 = compact[-3:]

    return (
        last3 == ["L_up_R_down", "L_down_R_up", "L_up_R_down"] or
        last3 == ["L_down_R_up", "L_up_R_down", "L_down_R_up"]
    )




def point_dist(a, b):
    return np.linalg.norm(a[:2] - b[:2])


def angle_deg(a, b, c):
    """
    Angle ABC in degrees
    """
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return None

    cos_theta = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def detect_dab_pose(kp):
    needed = [
        left_shoulder, right_shoulder,
        left_elbow, right_elbow,
        left_wrist, right_wrist
    ]

    if not keypoints_confident(kp, needed):
        return None

    ls = kp[left_shoulder]
    rs = kp[right_shoulder]
    le = kp[left_elbow]
    re = kp[right_elbow]
    lw = kp[left_wrist]
    rw = kp[right_wrist]

    shoulder_width = point_dist(ls, rs)
    if shoulder_width < 1:
        return None

    # Elbow angles
    left_angle = angle_deg(ls, le, lw)
    right_angle = angle_deg(rs, re, rw)

    # ----- Left dab -----
    # left arm straight-ish and raised
    left_arm_up = lw[1] < le[1] < ls[1]
    left_arm_straight = left_angle is not None and left_angle > 145

    # right arm bent across face/chest area
    right_arm_bent = right_angle is not None and right_angle < 110
    right_wrist_near_left_side = point_dist(rw, ls) < 1.2 * shoulder_width

    if left_arm_up and left_arm_straight and right_arm_bent and right_wrist_near_left_side:
        return "left_dab"

    # ----- Right dab -----
    # right arm straight-ish and raised
    right_arm_up = rw[1] < re[1] < rs[1]
    right_arm_straight = right_angle is not None and right_angle > 145

    # left arm bent across face/chest area
    left_arm_bent = left_angle is not None and left_angle < 110
    left_wrist_near_right_side = point_dist(lw, rs) < 1.2 * shoulder_width

    if right_arm_up and right_arm_straight and left_arm_bent and left_wrist_near_right_side:
        return "right_dab"

    return None

# Load the gifs

directory = Path(__file__).resolve().parent

gif_67_frames, gif_67_durations = load_gif_frames(directory / "gifs" / "67.gif", scale=0.6)
gif_dab_frames, gif_dab_durations = load_gif_frames(directory / "gifs" / "dab.gif", scale=0.6)

active_gif_frames = []
active_gif_durations = []
active_gif_name = None

gif_index = 0
gif_playing = False
gif_loops_left = 0
last_gif_time = 0

active_detection_label = ""


while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated = frame.copy()
    current_state = None
    dab_state = None

    # Run YOLO pose
    results = model(frame, verbose=False)

    if len(results) > 0:
        result = results[0]

        person_idx = get_largest_person_index(result)

        if result.keypoints is not None and person_idx is not None:
            kpts = result.keypoints.data

            if kpts is not None and len(kpts) > person_idx:
                kp = kpts[person_idx].cpu().numpy()
                dab_state = detect_dab_pose(kp)

                # Get motion state
                current_state = get_motion_state(kp, prev_kp)

                # Update history
                history.append(current_state)

                # Detect gesture
                detected = False
                if cooldown == 0 and detect_six_seven(history):
                    detected = True
                    detection_timer = 20
                    cooldown = cooldown_frames

                    active_gif_frames = gif_67_frames
                    active_gif_durations = gif_67_durations
                    active_gif_name = "67"

                    gif_playing = True
                    gif_index = 0
                    gif_loops_left = 2
                    last_gif_time = cv2.getTickCount() / cv2.getTickFrequency()

                    active_detection_label = "6 7 DETECTED"

                elif cooldown == 0 and dab_state is not None:
                    detection_timer = 20
                    cooldown = cooldown_frames

                    active_gif_frames = gif_dab_frames
                    active_gif_durations = gif_dab_durations
                    active_gif_name = "dab"

                    gif_playing = True
                    gif_index = 0
                    gif_loops_left = 1
                    last_gif_time = cv2.getTickCount() / cv2.getTickFrequency()

                    active_detection_label = "DAB DETECTED"

                prev_kp = kp

                # Draw YOLO output
                annotated = result.plot()
            else:
                history.append(None)
                prev_kp = None
        else:
            history.append(None)
            prev_kp = None
    else:
        history.append(None)
        prev_kp = None

    # Update timers
    if cooldown > 0:
        cooldown -= 1

    if detection_timer > 0:
        detection_timer -= 1

    # Overlay text
    cv2.putText(
        annotated,
        f"state: {current_state}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    if detection_timer > 0:
        cv2.putText(
            annotated,
            active_detection_label,
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )


    # Overlay gif
    if gif_playing and len(active_gif_frames) > 0:
        current_gif = active_gif_frames[gif_index]

        gh, gw = current_gif.shape[:2]
        h, w = annotated.shape[:2]
        x = w - gw - 20
        y = 20

        annotated = overlay_rgba(annotated, current_gif, x, y)

        now = cv2.getTickCount() / cv2.getTickFrequency()
        if now - last_gif_time >= active_gif_durations[gif_index]:
            gif_index += 1
            last_gif_time = now

            if gif_index >= len(active_gif_frames):
                gif_index = 0
                gif_loops_left -= 1
                if gif_loops_left <= 0:
                    gif_playing = False
                    active_gif_name = None

    # Show frame
    cv2.imshow("6 7 detector", annotated)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break