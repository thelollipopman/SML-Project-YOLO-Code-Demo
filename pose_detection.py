from ultralytics import YOLO
import cv2
import math
from PIL import Image
from IPython.display import display, clear_output
from collections import deque
import numpy as np

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated = frame.copy()
    current_state = None

    # Run YOLO pose
    results = model(frame, verbose=False)

    if len(results) > 0:
        result = results[0]

        person_idx = get_largest_person_index(result)

        if result.keypoints is not None and person_idx is not None:
            kpts = result.keypoints.data

            if kpts is not None and len(kpts) > person_idx:
                kp = kpts[person_idx].cpu().numpy()

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
            "6 7 DETECTED",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

    # Show frame
    cv2.imshow("6 7 detector", annotated)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break