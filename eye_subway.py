import cv2
import mediapipe as mp
import keyboard
import time
import winsound
from collections import deque

# ---- Setup ----
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# ---- Blink thresholds ----
EYE_CLOSED_THRESH = 0.018
COOLDOWN = 0.6

last_left_blink = 0
last_right_blink = 0

# ---- Sensitivity settings ----
LOOK_THRESHOLD = 0.015  # <-- smaller = more sensitive, larger = less sensitive
SMOOTHING = 7  # number of frames for averaging

# ---- Smoothing buffer ----
pupil_buffer = deque(maxlen=SMOOTHING)
last_action = None
last_move_time = time.time()

print("👁️ Eye control ready — Left blink=DOWN, Right blink=UP, Look Left/Right = ←/→")
winsound.Beep(1500, 300)


# ---- Helper ----
def get_ratio(landmarks):
    # Measure horizontal position of the pupil relative to the eye corners
    left_corner = landmarks[33]  # left eye outer corner
    right_corner = landmarks[133]  # left eye inner corner
    pupil = landmarks[468]  # left eye pupil

    eye_width = abs(right_corner.x - left_corner.x)
    ratio = (pupil.x - left_corner.x) / (eye_width + 1e-6)
    return ratio


# ---- Main Loop ----
while True:
    ret, frame = cam.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb)
    landmark_points = output.multi_face_landmarks

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # Compute relative eye movement (ratio stays stable even if head moves)
        gaze_ratio = get_ratio(landmarks)
        pupil_buffer.append(gaze_ratio)
        avg_ratio = sum(pupil_buffer) / len(pupil_buffer)

        # Blink detection
        left_eye_top, left_eye_bottom = landmarks[159].y, landmarks[145].y
        right_eye_top, right_eye_bottom = landmarks[386].y, landmarks[374].y
        left_dist = abs(left_eye_top - left_eye_bottom)
        right_dist = abs(right_eye_top - right_eye_bottom)

        left_closed = left_dist < EYE_CLOSED_THRESH
        right_closed = right_dist < EYE_CLOSED_THRESH
        now = time.time()

        # ---- Blink actions ----
        if left_closed and not right_closed and now - last_left_blink > COOLDOWN:
            keyboard.press_and_release("down")
            last_left_blink = now
            cv2.putText(frame, "⬇️ DOWN (Left Blink)", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if right_closed and not left_closed and now - last_right_blink > COOLDOWN:
            keyboard.press_and_release("up")
            last_right_blink = now
            cv2.putText(frame, "⬆️ UP (Right Blink)", (40, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # ---- Gaze direction ----
        if now - last_move_time > 0.4:
            if avg_ratio < 0.42 - LOOK_THRESHOLD:  # look left
                keyboard.press_and_release("left")
                cv2.putText(frame, "← LEFT", (40, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                last_action = "left"
                last_move_time = now
            elif avg_ratio > 0.58 + LOOK_THRESHOLD:  # look right
                keyboard.press_and_release("right")
                cv2.putText(frame, "→ RIGHT", (40, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                last_action = "right"
                last_move_time = now

        # ---- Debug Info ----
        cv2.putText(frame, f"Gaze Ratio: {avg_ratio:.3f}", (40, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Optional: draw eye landmarks (for visual feedback)
        h, w, _ = frame.shape
        for idx in [33, 133, 468]:
            x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    cv2.imshow("Eye Control", frame)
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

cam.release()
face_mesh.close()
cv2.destroyAllWindows()