"""
Google Earth Flight Controller (Verbose / Expanded)
--------------------------------------------------
This is a long-form, fully-expanded version of your Google Earth flight control script.
It keeps your original mappings and logic, but replaces the PITCH calculation with
Option B: **vertical movement of the index finger tip relative to its MCP joint**
as you requested.

Features included (explicit and verbosely implemented):
 - MediaPipe hand detection (max 2 hands)
 - Small always-on-top preview window (doesn't steal focus)
 - Automatic attempt to activate Google Earth window periodically
 - Pitch control using Option B (index fingertip vs index_mcp)
 - Roll control using wrist tilt (angle between wrists)
 - Throttle control using distance between wrists (with robust double-press & hold)
 - KeyDown / KeyUp management for continuous arrow keys
 - Lots of debug prints and on-screen telemetry
 - Graceful cleanup (releasing keys, webcam, MediaPipe, windows)
 - Plenty of comments and separated helper functions to make the file long and readable

Before running:
 - Install required packages if you haven't already:
     pip install opencv-python mediapipe pyautogui pygetwindow numpy

 - Make sure Google Earth is open and in Flight Simulator mode.
 - Place this file somewhere convenient and run it with Python 3.x.

Controls mapping summary:
 - PITCH: Index fingertip vertical vs index MCP (Option B)
     * Tip above MCP -> Nose up (UP arrow)
     * Tip below MCP -> Nose down (DOWN arrow)
 - ROLL: Tilt between wrists -> LEFT/RIGHT arrows
 - THROTTLE: Distance between wrists -> PageUp (increase) / PageDown (decrease)
"""

# ===========================
# Imports and global setup
# ===========================
import cv2
import mediapipe as mp
import pyautogui
import pygetwindow as gw
import math
import time
import numpy as np
import sys
import traceback

# ======================================================
# Configuration block - tweak these values to tune behavior
# ======================================================

# Fist detection sensitivity (used to detect neutral)
FIST_DETECTION_OFFSET = 0.05

# Pitch control thresholds (these are not used for Option B; left for reference)
PITCH_UP_THRESHOLD = 0.6    # earlier used threshold (kept for compatibility)
PITCH_DOWN_THRESHOLD = 0.4  # earlier used threshold (kept for compatibility)

# Tilt threshold for roll (radians)
TILT_THRESHOLD = 0.25  # how much tilt (angle) is needed to trigger left/right roll

# Throttle distance thresholds (normalized coordinates)
DISTANCE_SPEED_UP = 0.18   # if hands closer than this -> speed up
DISTANCE_SLOW_DOWN = 0.6   # if hands farther than this -> slow down

# Throttle timing / cooldown
SPEED_COMMAND_COOLDOWN = 0.35   # seconds between actionable throttle commands
THROTTLE_PRESS_HOLD = 0.04      # how long each PageUp/PageDown key is held
THROTTLE_PRESS_REPEATS = 2      # repeat number for better registration in Google Earth

# Google Earth focus reactivation interval
GOOGLE_EARTH_FOCUS_INTERVAL = 2.0  # seconds

# Preview window settings
PREVIEW_WIDTH = 360
PREVIEW_HEIGHT = 270
PREVIEW_X = 20
PREVIEW_Y = 20

# OpenCV waitKey delay
WAIT_KEY_DELAY = 1  # millisecond per frame for waitKey()

# Safety and pyautogui
pyautogui.FAILSAFE = False

# ======================================================
# MediaPipe and capture initialization
# ======================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Use two hands to detect left/right wrist relationships
hands_detector = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Check camera index and permissions.")

# ======================================================
# State variables
# ======================================================
keys_pressed = set()            # set of currently held keys (arrow keys)
last_speed_command_time = 0.0   # last timestamp throttle command was sent
last_focus_time = 0.0           # last time we tried to activate Google Earth
running = True                  # main loop flag

# ======================================================
# Helper functions
# ======================================================

def safe_print(*args, **kwargs):
    """Wrapper around print so we can easily disable or enable verbose logs."""
    print(*args, **kwargs)

def activate_google_earth_once():
    """
    Try to bring Google Earth window to the front so our pyautogui presses reach it.
    We attempt multiple likely window title matches for robustness.
    """
    try:
        # Common titles: "Google Earth", "Earth", "Google Earth Pro"
        titles_to_try = ["Google Earth", "Google Earth Pro", "Earth", "Google Earth -"]
        found = False
        for t in titles_to_try:
            wins = gw.getWindowsWithTitle(t)
            if wins:
                w = wins[0]
                # Try to activate; if minimized, restore then activate
                try:
                    if w.isMinimized:
                        w.restore()
                        time.sleep(0.08)
                    w.activate()
                    safe_print(f"[INFO] Activated window with title containing '{t}'.")
                    found = True
                    break
                except Exception as e:
                    # Activation might fail for several reasons; try restore then activate
                    try:
                        w.restore()
                        time.sleep(0.08)
                        w.activate()
                        safe_print(f"[INFO] Restored + activated window with title containing '{t}'.")
                        found = True
                        break
                    except Exception:
                        safe_print(f"[WARN] Could not programmatically activate window '{t}': {e}")
                        # continue trying other titles
        if not found:
            safe_print("[WARN] Google Earth window not found. Please click Google Earth manually to focus.")
    except Exception as e:
        safe_print(f"[WARN] Exception while attempting to activate Google Earth: {e}")

def is_fist(landmarks):
    """
    Determine whether a hand is in a fist-like pose by comparing fingertip vs pip y positions.
    Returns True if index, middle, ring and pinky tips are curled (fist), False otherwise.
    """
    try:
        # Using normalized landmark coordinates (0..1)
        idx_tip_y = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        idx_pip_y = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y

        mid_tip_y = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        mid_pip_y = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y

        ring_tip_y = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
        ring_pip_y = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y

        pinky_tip_y = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
        pinky_pip_y = landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y

        if (idx_tip_y > idx_pip_y + FIST_DETECTION_OFFSET and
            mid_tip_y > mid_pip_y + FIST_DETECTION_OFFSET and
            ring_tip_y > ring_pip_y + FIST_DETECTION_OFFSET and
            pinky_tip_y > pinky_pip_y + FIST_DETECTION_OFFSET):
            return True
    except Exception:
        return False
    return False

def send_throttle_up():
    """
    Robust throttle up function: send multiple quick pageup key presses
    with a tiny hold per press so Google Earth will register the throttle increase.
    """
    try:
        for _ in range(THROTTLE_PRESS_REPEATS):
            pyautogui.keyDown('pageup')
            time.sleep(THROTTLE_PRESS_HOLD)
            pyautogui.keyUp('pageup')
            time.sleep(0.03)
        safe_print("[ACTION] Throttle UP sent (robust).")
    except Exception as e:
        safe_print("[ERROR] Exception in send_throttle_up:", e)

def send_throttle_down():
    """
    Robust throttle down function: send multiple quick pagedown key presses
    with a tiny hold per press so Google Earth will register the throttle decrease.
    """
    try:
        for _ in range(THROTTLE_PRESS_REPEATS):
            pyautogui.keyDown('pagedown')
            time.sleep(THROTTLE_PRESS_HOLD)
            pyautogui.keyUp('pagedown')
            time.sleep(0.03)
        safe_print("[ACTION] Throttle DOWN sent (robust).")
    except Exception as e:
        safe_print("[ERROR] Exception in send_throttle_down:", e)

def calculate_wrist_angle(left_wrist, right_wrist):
    """
    Compute the angle (radians) between the line connecting left and right wrist and the horizontal axis.
    Positive angle means right wrist is lower (clockwise tilt), negative means left wrist lower.
    """
    try:
        dy = right_wrist.y - left_wrist.y
        dx = right_wrist.x - left_wrist.x
        angle = math.atan2(dy, dx)
        return angle
    except Exception:
        return 0.0

def normalized_distance(a, b):
    """
    Euclidean distance between two normalized landmark points (x,y).
    Each point is an object with .x and .y in [0,1].
    """
    try:
        return math.hypot(a.x - b.x, a.y - b.y)
    except Exception:
        return 0.0

# ======================================================
# Initial attempt to put Google Earth in focus
# ======================================================
safe_print("Starting Google Earth Gesture Controller (verbose).")
safe_print("Make sure Google Earth Flight Simulator is open and visible.")
activate_google_earth_once()

# ======================================================
# Setup preview window (always-on-top, non-focus-stealing)
# ======================================================
cv2.namedWindow("Gesture Preview", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gesture Preview", PREVIEW_WIDTH, PREVIEW_HEIGHT)
cv2.moveWindow("Gesture Preview", PREVIEW_X, PREVIEW_Y)
# Attempt to keep preview window on top so user can see it
try:
    cv2.setWindowProperty("Gesture Preview", cv2.WND_PROP_TOPMOST, 1)
except Exception:
    # some OpenCV builds may not support WND_PROP_TOPMOST; that's okay
    safe_print("[WARN] Could not set preview window as topmost on this platform.")

# ======================================================
# Main loop (very verbose with explicit steps)
# ======================================================
try:
    # We'll loop indefinitely until KeyboardInterrupt or user quits with 'q' in preview
    while True:
        # 1) Periodically ensure Google Earth window remains focused so keystrokes are delivered correctly
        now = time.time()
        if now - last_focus_time > GOOGLE_EARTH_FOCUS_INTERVAL:
            # Try to activate Google Earth window again every GOOGLE_EARTH_FOCUS_INTERVAL seconds
            try:
                activate_google_earth_once()
            except Exception as e:
                safe_print("[WARN] Exception while trying to activate window:", e)
            # update last_focus_time (we intentionally update even if activation fails, to avoid spamming)
            last_focus_time = now

        # 2) Grab a frame from the webcam
        success, frame = cap.read()
        if not success:
            # If reading the camera fails, wait briefly and continue
            time.sleep(0.01)
            continue

        # 3) Mirror the image (so left-right on-screen matches user expectation)
        frame = cv2.flip(frame, 1)

        # 4) Convert BGR -> RGB before sending to MediaPipe (it expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 5) Run MediaPipe hands detector on the frame
        results = hands_detector.process(rgb_frame)

        # 6) Prepare telemetry variables for display and debugging
        pitch_state = "Neutral"
        roll_state = "Neutral"
        throttle_state = "Neutral"
        distance_between_wrists = 0.0
        tilt_angle = 0.0

        # 7) Default current desired keys: if a control is active we'll add to this set
        current_keys = set()

        # 8) If two hands are detected, compute controls. If fewer than two hands are detected treat as neutral.
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
            # We'll use the first two detected hands for left/right decisions
            hand_landmarks_0 = results.multi_hand_landmarks[0]
            hand_landmarks_1 = results.multi_hand_landmarks[1]

            # Draw landmarks to the frame for visual feedback
            try:
                mp_drawing.draw_landmarks(frame, hand_landmarks_0, mp_hands.HAND_CONNECTIONS)
            except Exception:
                pass
            try:
                mp_drawing.draw_landmarks(frame, hand_landmarks_1, mp_hands.HAND_CONNECTIONS)
            except Exception:
                pass

            # Determine if either or both hands are fists (neutral)
            is_hand0_fist = is_fist(hand_landmarks_0)
            is_hand1_fist = is_fist(hand_landmarks_1)

            # If both hands are fists, treat it as neutral: no controls active
            if is_hand0_fist and is_hand1_fist:
                pitch_state = roll_state = throttle_state = "Both fists (neutral)"
                # current_keys remains empty -> no arrow keys held
            else:
                # Determine which detected hand is left and which is right according to x coordinate of wrist
                try:
                    wrist0 = hand_landmarks_0.landmark[mp_hands.HandLandmark.WRIST]
                    wrist1 = hand_landmarks_1.landmark[mp_hands.HandLandmark.WRIST]
                except Exception:
                    wrist0 = None
                    wrist1 = None

                # Fallback: if wrists couldn't be retrieved, skip to next iteration
                if wrist0 is None or wrist1 is None:
                    # Can't compute controls without wrist positions; mark as neutral
                    pitch_state = roll_state = throttle_state = "Missing wrist data"
                else:
                    # Choose left_wrist, right_wrist by comparing normalized x values
                    if wrist0.x < wrist1.x:
                        left_wrist = wrist0
                        right_wrist = wrist1
                        left_landmarks = hand_landmarks_0
                        right_landmarks = hand_landmarks_1
                    else:
                        left_wrist = wrist1
                        right_wrist = wrist0
                        left_landmarks = hand_landmarks_1
                        right_landmarks = hand_landmarks_0

                    # ---------------------------
                    # PITCH (Option B) - Using vertical movement of index fingertip vs index MCP
                    # ---------------------------
                    # We will use the right hand's index fingertip and MCP joint (Knuckle) for pitch.
                    # Reason: user requested Option B where index finger vertical displacement controls pitch.
                    # Implementation detail:
                    #   - Tip above MCP by threshold -> nose up (press UP arrow)
                    #   - Tip below MCP by threshold -> nose down (press DOWN arrow)
                    # Threshold is in normalized coordinates and tuned conservatively to avoid jitter.
                    try:
                        # We'll get the landmarks for the index finger tip and index finger MCP on the right hand.
                        index_tip = right_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        index_mcp = right_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

                        # Compute vertical difference (normalized coordinates: y increases downward)
                        # So tip.y < mcp.y means tip is higher on the camera (hand raised)
                        vertical_diff = index_tip.y - index_mcp.y  # negative when tip above MCP

                        # Choose a small deadzone threshold to avoid jitter (tunable)
                        PITCH_DEADZONE = 0.03  # normalized units

                        if vertical_diff < -PITCH_DEADZONE:
                            # Index tip is sufficiently above MCP -> pitch up
                            # In Google Earth: UP arrow climbs (nose up)
                            current_keys.add('up')
                            pitch_state = "Nose Up (index tip above MCP)"
                        elif vertical_diff > PITCH_DEADZONE:
                            # Index tip is sufficiently below MCP -> pitch down
                            current_keys.add('down')
                            pitch_state = "Nose Down (index tip below MCP)"
                        else:
                            pitch_state = "Pitch Neutral (index within deadzone)"
                    except Exception as e:
                        pitch_state = f"Pitch error: {e}"

                    # ---------------------------
                    # ROLL - Using wrist tilt angle
                    # ---------------------------
                    try:
                        tilt_angle = calculate_wrist_angle(left_wrist, right_wrist)
                        # Compare to threshold to determine left or right roll
                        if tilt_angle > TILT_THRESHOLD:
                            # Right wrist lower than left wrist => clockwise tilt => roll right
                            current_keys.add('right')
                            roll_state = "Roll Right"
                        elif tilt_angle < -TILT_THRESHOLD:
                            # Left wrist lower => counter-clockwise tilt => roll left
                            current_keys.add('left')
                            roll_state = "Roll Left"
                        else:
                            roll_state = "Roll Neutral"
                    except Exception as e:
                        roll_state = f"Roll error: {e}"

                    # ---------------------------
                    # THROTTLE - distance between wrists (normalized)
                    # ---------------------------
                    try:
                        distance_between_wrists = normalized_distance(left_wrist, right_wrist)
                        # Use cooldown to avoid spamming throttle commands
                        if time.time() - last_speed_command_time > SPEED_COMMAND_COOLDOWN:
                            if distance_between_wrists < DISTANCE_SPEED_UP:
                                # hands close -> speed up
                                send_throttle_up()
                                throttle_state = "Throttle UP"
                                last_speed_command_time = time.time()
                            elif distance_between_wrists > DISTANCE_SLOW_DOWN:
                                # hands far apart -> slow down
                                send_throttle_down()
                                throttle_state = "Throttle DOWN"
                                last_speed_command_time = time.time()
                            else:
                                throttle_state = "Throttle Neutral (distance midrange)"
                        else:
                            # within cooldown
                            throttle_state = "Throttle Cooldown"
                    except Exception as e:
                        throttle_state = f"Throttle error: {e}"

                    # End of control calculations for two-hand case

            # End of check for fists etc.

        else:
            # less than two hands detected - show helpful hint on the preview and telemetry
            pitch_state = "No/Single hand detected (need two hands)"
            roll_state = "No/Single hand detected (need two hands)"
            throttle_state = "No/Single hand detected (need two hands)"
            # We leave current_keys empty which will cause any previously held arrow keys to be released below.

        # ---------------------------
        # Key management - hold keys for continuous actions
        # ---------------------------
        # For each key in current_keys that is not currently pressed we press and hold it (keyDown).
        # For keys that are currently pressed but not requested in current_keys we release them (keyUp).
        try:
            # Press new keys
            for k in current_keys:
                if k not in keys_pressed:
                    try:
                        pyautogui.keyDown(k)
                        keys_pressed.add(k)
                        safe_print(f"[KEY] DOWN: {k}")
                    except Exception as e:
                        safe_print(f"[ERROR] Failed to keyDown {k}: {e}")

            # Release keys that are no longer requested
            keys_to_release = set(keys_pressed) - set(current_keys)
            for k in list(keys_to_release):
                try:
                    pyautogui.keyUp(k)
                    keys_pressed.discard(k)
                    safe_print(f"[KEY] UP: {k}")
                except Exception as e:
                    safe_print(f"[ERROR] Failed to keyUp {k}: {e}")
        except Exception as e:
            safe_print("[ERROR] Key management exception:", e)

        # ---------------------------
        # Draw telemetry overlay onto frame for user feedback
        # ---------------------------
        try:
            overlay = frame.copy()
            # Draw semi-opaque rectangle as background for text
            rect_h = 90
            cv2.rectangle(overlay, (0, 0), (PREVIEW_WIDTH, rect_h), (0, 0, 0), -1)
            alpha = 0.55
            frame[0:rect_h, 0:PREVIEW_WIDTH] = cv2.addWeighted(overlay[0:rect_h, 0:PREVIEW_WIDTH], alpha,
                                                             frame[0:rect_h, 0:PREVIEW_WIDTH], 1 - alpha, 0)

            # Draw three lines of telemetry
            cv2.putText(frame, f"Pitch: {pitch_state}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Roll: {roll_state}", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Throttle: {throttle_state}", (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Additional debug values on the right side (distance & tilt)
            cv2.putText(frame, f"Wrists dist: {distance_between_wrists:.3f}", (PREVIEW_WIDTH - 10 - 220, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Tilt angle: {tilt_angle:.3f}", (PREVIEW_WIDTH - 10 - 220, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

            # Show currently held keys
            held_keys_str = ", ".join(sorted(list(keys_pressed))) if keys_pressed else "None"
            cv2.putText(frame, f"Held keys: {held_keys_str}", (8, rect_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 255), 1, cv2.LINE_AA)

        except Exception as e:
            safe_print("[WARN] Exception while drawing telemetry overlay:", e)

        # ---------------------------
        # Resize the preview to PREVIEW_WIDTH x PREVIEW_HEIGHT and show
        # This preview is intentionally small so it doesn't cover Google Earth
        # ---------------------------
        try:
            preview = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
            cv2.imshow("Gesture Preview", preview)
        except Exception as e:
            # If resizing fails for any reason, still attempt to show original frame
            try:
                cv2.imshow("Gesture Preview", frame)
            except Exception:
                safe_print("[ERROR] Failed to show preview window:", e)

        # ---------------------------
        # Check for user quit (press 'q' while preview is focused)
        # ---------------------------
        key = cv2.waitKey(WAIT_KEY_DELAY) & 0xFF
        if key == ord('q'):
            safe_print("[INFO] 'q' pressed in preview window - exiting.")
            break

        # Small sleep to avoid pegging CPU too hard (already WAIT_KEY_DELAY but being explicit)
        time.sleep(0.001)

# End of main try
except KeyboardInterrupt:
    safe_print("\n[INFO] KeyboardInterrupt detected. Exiting gracefully.")
except Exception as e:
    safe_print("[ERROR] Unexpected exception in main loop:", e)
    traceback.print_exc(file=sys.stdout)
finally:
    # ============================
    # Cleanup and release resources
    # ============================
    safe_print("[INFO] Cleaning up: releasing keys, webcam, windows, and MediaPipe resources.")

    # Release any keys still held down (very important so arrow keys don't remain stuck)
    try:
        for k in list(keys_pressed):
            try:
                pyautogui.keyUp(k)
                safe_print(f"[CLEANUP] Released key: {k}")
            except Exception as e:
                safe_print(f"[CLEANUP ERROR] Failed to release key {k}: {e}")
        keys_pressed.clear()
    except Exception as e:
        safe_print("[CLEANUP ERROR] While releasing keys:", e)

    # Release webcam
    try:
        cap.release()
        safe_print("[CLEANUP] Webcam released.")
    except Exception as e:
        safe_print("[CLEANUP ERROR] Failed to release webcam:", e)

    # Destroy all OpenCV windows
    try:
        cv2.destroyAllWindows()
        safe_print("[CLEANUP] OpenCV windows destroyed.")
    except Exception as e:
        safe_print("[CLEANUP ERROR] Failed to destroy OpenCV windows:", e)

    # Close MediaPipe
    try:
        hands_detector.close()
        safe_print("[CLEANUP] MediaPipe hands detector closed.")
    except Exception as e:
        safe_print("[CLEANUP ERROR] Failed to close MediaPipe detector:", e)

    safe_print("[INFO] Shutdown complete. Bye!")
