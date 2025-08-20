import cv2
import mediapipe as mp
import pyautogui
import pygetwindow as gw
import math
import time

# ----------------- CONFIG -----------------
FIST_DETECTION_OFFSET = 0.05

# Pitch thresholds (right hand Y)
PITCH_UP_THRESHOLD = 0.6    # right hand low -> 'up' arrow (climb)
PITCH_DOWN_THRESHOLD = 0.4  # right hand high -> 'down' arrow (dive)

TILT_THRESHOLD = 0.25       # radians, for roll detection

# Distance-based throttle thresholds
DISTANCE_SPEED_UP = 0.18
DISTANCE_SLOW_DOWN = 0.6

SPEED_COMMAND_COOLDOWN = 0.35   # seconds between throttle commands
GOOGLE_EARTH_FOCUS_INTERVAL = 2.0  # seconds, try to re-activate GE window periodically

# Preview window size / position
PREVIEW_WIDTH = 320
PREVIEW_HEIGHT = 240
PREVIEW_X = 20
PREVIEW_Y = 20

# ------------------------------------------

pyautogui.FAILSAFE = False

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

keys_pressed = set()
last_speed_command_time = 0.0
last_focus_time = 0.0

print("Starting Google Earth Gesture Controller.")
print("Make sure Google Earth Flight Simulator is running and visible.")
print("Controls: Right-hand up/down -> Pitch (arrow up/down). Tilt between hands -> Roll (left/right). Hands distance -> Throttle (PageUp/PageDown).")
print("Press Ctrl+C in terminal to stop gracefully.")

# Try activate Google Earth once at start
def activate_google_earth_once():
    try:
        wins = gw.getWindowsWithTitle("Google Earth")
        if not wins:
            wins = gw.getWindowsWithTitle("Earth")  # fallback scan
        if wins:
            w = wins[0]
            try:
                w.activate()
                print("[INFO] Activated Google Earth window.")
            except Exception:
                # Sometimes .activate() fails if window minimized; try restore then activate
                try:
                    w.restore()
                    time.sleep(0.1)
                    w.activate()
                    print("[INFO] Restored + activated Google Earth window.")
                except Exception:
                    print("[WARN] Could not programmatically activate Google Earth window. Click it manually.")
        else:
            print("[WARN] Google Earth window not found. Click it manually.")
    except Exception as e:
        print(f"[WARN] Exception while trying to activate Google Earth: {e}")

activate_google_earth_once()

# Setup preview window (always on top, small)
cv2.namedWindow("Gesture Preview", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gesture Preview", PREVIEW_WIDTH, PREVIEW_HEIGHT)
cv2.moveWindow("Gesture Preview", PREVIEW_X, PREVIEW_Y)
cv2.setWindowProperty("Gesture Preview", cv2.WND_PROP_TOPMOST, 1)

def is_fist(hand_landmarks):
    try:
        idx_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        idx_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        mid_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        mid_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
        ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
        pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y

        if (idx_tip > idx_pip + FIST_DETECTION_OFFSET and
            mid_tip > mid_pip + FIST_DETECTION_OFFSET and
            ring_tip > ring_pip + FIST_DETECTION_OFFSET and
            pinky_tip > pinky_pip + FIST_DETECTION_OFFSET):
            return True
    except Exception:
        return False
    return False

def send_throttle_up():
    """Send a stronger throttle-up that Google Earth will notice."""
    # two quick presses with tiny hold each
    for _ in range(2):
        pyautogui.keyDown('pageup')
        time.sleep(0.04)
        pyautogui.keyUp('pageup')
        time.sleep(0.03)

def send_throttle_down():
    """Send a stronger throttle-down that Google Earth will notice."""
    for _ in range(2):
        pyautogui.keyDown('pagedown')
        time.sleep(0.04)
        pyautogui.keyUp('pagedown')
        time.sleep(0.03)

try:
    while True:
        # Periodically ensure Google Earth has focus so key events go to it
        now = time.time()
        if now - last_focus_time > GOOGLE_EARTH_FOCUS_INTERVAL:
            activate_google_earth_once()
            last_focus_time = now

        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        pitch_state = "Neutral"
        roll_state = "Neutral"
        throttle_state = "Neutral"

        current_keys = set()

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
            # take first two detected hands
            hand1, hand2 = results.multi_hand_landmarks[0], results.multi_hand_landmarks[1]

            # determine fists
            hand1_fist = is_fist(hand1)
            hand2_fist = is_fist(hand2)

            # draw landmarks for feedback
            mp.solutions.drawing_utils.draw_landmarks(frame, hand1, mp_hands.HAND_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(frame, hand2, mp_hands.HAND_CONNECTIONS)

            if not (hand1_fist and hand2_fist):
                # identify left/right by x coordinate of wrist
                w1 = hand1.landmark[mp_hands.HandLandmark.WRIST]
                w2 = hand2.landmark[mp_hands.HandLandmark.WRIST]

                if w1.x < w2.x:
                    left_wrist, right_wrist = w1, w2
                else:
                    left_wrist, right_wrist = w2, w1

                # PITCH: right hand y
                if right_wrist.y < PITCH_DOWN_THRESHOLD:
                    # right hand high on camera -> nose down in sim -> press 'down' arrow key
                    current_keys.add('down')
                    pitch_state = "Nose Down (DOWN key)"
                elif right_wrist.y > PITCH_UP_THRESHOLD:
                    current_keys.add('up')
                    pitch_state = "Nose Up (UP key)"
                else:
                    pitch_state = "Neutral"

                # ROLL: tilt between wrists
                angle = math.atan2(right_wrist.y - left_wrist.y, right_wrist.x - left_wrist.x)
                if angle > TILT_THRESHOLD:
                    current_keys.add('right')
                    roll_state = "Roll Right (RIGHT key)"
                elif angle < -TILT_THRESHOLD:
                    current_keys.add('left')
                    roll_state = "Roll Left (LEFT key)"
                else:
                    roll_state = "Neutral"

                # THROTTLE: distance between wrists
                distance = math.hypot(right_wrist.x - left_wrist.x, right_wrist.y - left_wrist.y)
                if now - last_speed_command_time > SPEED_COMMAND_COOLDOWN:
                    if distance < DISTANCE_SPEED_UP:
                        send_throttle_up()
                        last_speed_command_time = now
                        throttle_state = "Throttle UP (PageUp)"
                        print("[EVENT] Throttle UP triggered.")
                    elif distance > DISTANCE_SLOW_DOWN:
                        send_throttle_down()
                        last_speed_command_time = now
                        throttle_state = "Throttle DOWN (PageDown)"
                        print("[EVENT] Throttle DOWN triggered.")
                    else:
                        throttle_state = "Neutral"
                else:
                    # still in cooldown, show last throttle state unchanged
                    throttle_state = "Waiting (cooldown)"

                # Debug telemetry in terminal
                print(f"[TELEMETRY] pitch: {pitch_state} | roll: {roll_state} | throttle: {throttle_state} | dist:{distance:.3f} angle:{angle:.3f}")

            else:
                # Both fists = neutral; no keys
                pitch_state = roll_state = throttle_state = "Neutral"
                # no throttle event; release keys by leaving current_keys empty

        else:
            # less than two hands detected: treat as neutral
            pitch_state = roll_state = throttle_state = "No/Single hand detected"

        # Manage keyDown/keyUp for continuous controls (arrow keys)
        # Press keys that are in current_keys but not yet pressed
        for key in current_keys:
            if key not in keys_pressed:
                pyautogui.keyDown(key)
                keys_pressed.add(key)
                print(f"[KEY] DOWN: {key}")

        # Release keys that are pressed but not currently requested
        to_release = list(keys_pressed - current_keys)
        for key in to_release:
            pyautogui.keyUp(key)
            keys_pressed.remove(key)
            print(f"[KEY] UP: {key}")

        # Overlay telemetry text onto preview frame for on-screen feedback
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (PREVIEW_WIDTH, 70), (0, 0, 0), -1)  # semi background
        alpha = 0.6
        frame[0:70, 0:PREVIEW_WIDTH] = cv2.addWeighted(overlay[0:70, 0:PREVIEW_WIDTH], alpha, frame[0:70, 0:PREVIEW_WIDTH], 1 - alpha, 0)

        # Draw text lines
        cv2.putText(frame, f"Pitch: {pitch_state}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Roll: {roll_state}", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Throttle: {throttle_state}", (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

        # Resize to preview size and show
        preview = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
        cv2.imshow("Gesture Preview", preview)

        # Use a short wait; preview is topmost and non-focus stealing; quitting via terminal (Ctrl+C) recommended
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # if the preview has focus you can press 'q' on it to quit too
            break

except KeyboardInterrupt:
    print("\n[INFO] KeyboardInterrupt received. Exiting.")

finally:
    print("[INFO] Releasing keys and cleaning up.")
    # Ensure all pressed keys are released
    for k in list(keys_pressed):
        try:
            pyautogui.keyUp(k)
        except Exception:
            pass
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("[INFO] Shutdown complete.")
