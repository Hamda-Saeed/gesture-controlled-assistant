import cv2
import mediapipe as mp
import pyautogui
import time


cap = cv2.VideoCapture(0)

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# To avoid repeating actions too fast
prev_action_time = 0
action_delay = 0.25  # seconds between actions

# Finger tip landmarks 
tip_ids = [4, 8, 12, 16, 20]

def count_fingers(hand_landmarks):
    fingers = []

    # Thumb
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Mirror image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    fingers_up = 0
    current_time = time.time()

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

        fingers_up = count_fingers(handLms)

        # Perform actions with delay
        if current_time - prev_action_time > action_delay:
            if fingers_up == 1:
                pyautogui.scroll(-100)  # Scroll down
                action_text = "Scrolling Down..."
            elif fingers_up == 2:
                pyautogui.scroll(100)   # Scroll up
                action_text = "Scrolling Up..."
            elif fingers_up == 4:
                pyautogui.hotkey('ctrl', '+')  # Zoom in
                action_text = "Zooming In..."
            elif fingers_up == 5:
                pyautogui.hotkey('ctrl', '-')  # Zoom out
                action_text = "Zooming Out..."
            else:
                action_text = "No Action"

            prev_action_time = current_time
        else:
            action_text = "Waiting..."

        # Display finger count and action
        cv2.putText(img, f"Fingers: {fingers_up}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"{action_text}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 100), 2)

    else:
        cv2.putText(img, "No Hand Detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the webcam feed
    cv2.imshow("Gesture Assistant", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
