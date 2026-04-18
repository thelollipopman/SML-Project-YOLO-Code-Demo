from ultralytics import YOLO
import cv2
import math
from PIL import Image
from IPython.display import display, clear_output
from collections import deque
import numpy as np


# Call the object detection model
model = YOLO("yolo26n.pt")

# Thresholds
conf_thresh = 0.4

# Temporal / Time-related
history_len = 25
cooldown_frames = 20

cap = cv2.VideoCapture(0)

history = deque(maxlen=history_len)

cooldown = 0
detection_timer = 0


def get_largest_object_index(result):
    if result.boxes is None or len(result.boxes) == 0:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()
    areas = []

    for box in boxes:
        x1, y1, x2, y2 = box
        areas.append((x2 - x1) * (y2 - y1))

    return int(np.argmax(areas))


def get_detected_object_name(result, obj_idx, conf_thresh=conf_thresh):
    if result.boxes is None or obj_idx is None:
        return None

    boxes = result.boxes
    conf = float(boxes.conf[obj_idx].cpu().numpy())

    if conf < conf_thresh:
        return None

    cls_id = int(boxes.cls[obj_idx].cpu().numpy())
    cls_name = model.names[cls_id]

    return cls_name


while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated = frame.copy()
    current_object = None

    # Run YOLO object detection
    results = model(frame, verbose=False)

    if len(results) > 0:
        result = results[0]

        obj_idx = get_largest_object_index(result)

        if result.boxes is not None and obj_idx is not None:
            current_object = get_detected_object_name(result, obj_idx)

            # Update history
            history.append(current_object)

            if cooldown == 0 and current_object is not None:
                detection_timer = 20
                cooldown = cooldown_frames

            # Draw YOLO output
            annotated = result.plot()
        else:
            history.append(None)
    else:
        history.append(None)

    # Update timers
    if cooldown > 0:
        cooldown -= 1

    if detection_timer > 0:
        detection_timer -= 1

    # Overlay text
    cv2.putText(
        annotated,
        f"object: {current_object}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    # Show frame
    cv2.imshow("Object Detector", annotated)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()