import cv2
import mediapipe as mp
import pyautogui
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Camera not found")
    detector.close()
    raise SystemExit

screen_w, screen_h = pyautogui.size()
prev_x, prev_y = 0, 0
smooth_factor = 0.5
previous_hand_x = None
swipe_threshold = 0.15
last_gesture = None
last_action_time = 0
cooldown = 0.8


def count_fingers(hand_landmarks, handedness):
    count = 0
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks[tip].y < hand_landmarks[pip].y:
            count += 1

    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]
    hand_type = handedness[0].category_name

    if hand_type == "Right":
        if thumb_tip.x > thumb_ip.x:
            count += 1
    else:
        if thumb_tip.x < thumb_ip.x:
            count += 1

    return count


def perform_action(fingers, hand_landmarks):
    global prev_x, prev_y

    if fingers == 0:
        pyautogui.press("esc")
        return "ESC"

    if fingers == 1:
        index_tip = hand_landmarks[8]
        target_x = int(index_tip.x * screen_w)
        target_y = int(index_tip.y * screen_h)
        smooth_x = int(prev_x + (target_x - prev_x) * smooth_factor)
        smooth_y = int(prev_y + (target_y - prev_y) * smooth_factor)
        pyautogui.moveTo(smooth_x, smooth_y, duration=0.02)
        prev_x, prev_y = smooth_x, smooth_y
        return "MOUSE MOVE"

    if fingers == 2:
        pyautogui.click()
        return "LEFT CLICK"

    if fingers == 3:
        palm_y = hand_landmarks[9].y
        if palm_y < 0.45:
            pyautogui.scroll(5)
            return "SCROLL UP"
        pyautogui.scroll(-5)
        return "SCROLL DOWN"

    if fingers == 4:
        current_x = hand_landmarks[9].x
        if previous_hand_x is not None:
            movement = current_x - previous_hand_x
            if movement > swipe_threshold:
                pyautogui.press("nexttrack")
                return "NEXT TRACK"
            if movement < -swipe_threshold:
                pyautogui.press("prevtrack")
                return "PREVIOUS TRACK"
        pyautogui.press("volumedown")
        return "VOLUME DOWN"

    if fingers == 5:
        pyautogui.press("playpause")
        return "PLAY / PAUSE"

    return "NONE"


print("Offline Hand Gesture Laptop Control")
print("Press Q to exit.")
timestamp_ms = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("Camera frame not received")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms += 33
    result = detector.detect_for_video(mp_image, timestamp_ms)

    gesture_text = "NO HAND"
    command_text = "NONE"
    fingers = 0

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]

        # Draw the 21 hand landmarks and simple connections.
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]
        for start_i, end_i in connections:
            start = hand_landmarks[start_i]
            end = hand_landmarks[end_i]
            x1, y1 = int(start.x * frame.shape[1]), int(start.y * frame.shape[0])
            x2, y2 = int(end.x * frame.shape[1]), int(end.y * frame.shape[0])
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        for landmark in hand_landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        handedness = result.handedness[0]
        fingers = count_fingers(hand_landmarks, handedness)

        gesture_names = {
            0: "FIST",
            1: "ONE",
            2: "TWO",
            3: "THREE",
            4: "FOUR",
            5: "OPEN HAND"
        }
        default_commands = {
            0: "ESC",
            1: "MOUSE MOVE",
            2: "LEFT CLICK",
            3: "SCROLL",
            4: "MEDIA / VOLUME",
            5: "PLAY / PAUSE"
        }
        gesture_text = gesture_names.get(fingers, "UNKNOWN")
        default_command = default_commands.get(fingers, "NONE")
        current_x = hand_landmarks[9].x
        current_time = time.time()

        if fingers == 1:
            command_text = perform_action(fingers, hand_landmarks)
        elif last_gesture != fingers and current_time - last_action_time > cooldown:
            command_text = perform_action(fingers, hand_landmarks)
            last_gesture = fingers
            last_action_time = current_time
        else:
            command_text = default_command

        previous_hand_x = current_x
    else:
        previous_hand_x = None
        last_gesture = None

    cv2.rectangle(frame, (10, 10), (450, 120), (30, 30, 30), -1)
    cv2.putText(frame, f"Fingers: {fingers}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Gesture: {gesture_text}", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Command: {command_text}", (25, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("Offline Hand Gesture Laptop Control", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
detector.close()
cv2.destroyAllWindows()
