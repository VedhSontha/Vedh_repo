"""
TrackMania Gesture Controller - Right-hand pinch throttle + wrist-tilt steering
Verbose, ready-to-run (~250+ lines)

Controls:
 - Steering: both-hands wrist tilt -> hold LEFT/RIGHT while tilt persists
 - Accel/Decel: RIGHT-hand pinch (thumb <-> index) -> proportional throttle:
     * small pinch -> hold UP (full throttle)
     * medium pinch -> intermittent UP presses (partial throttle)
     * large pinch -> release UP (coast)
 - Brake: RIGHT-hand closed fist -> hold DOWN (brake)
 - Camera priority: camera gestures override physical keyboard (pynput) fallback
 - Preview: on-screen MediaPipe landmarks and telemetry drawn
"""

# ---------------------------
# Imports and configuration
# ---------------------------
import cv2
import mediapipe as mp
import pydirectinput
import pygetwindow as gw
import math
import time
import numpy as np
import threading
import traceback
import sys

# keyboard listener for fallback
try:
    from pynput import keyboard as pynput_keyboard
except Exception:
    pynput_keyboard = None

# Safety
pydirectinput.FAILSAFE = False

# ---------------------------
# TUNABLE PARAMETERS
# ---------------------------
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Steering (wrist-tilt) thresholds (radians)
STEER_TILT_THRESHOLD = 0.18   # smaller -> more sensitive steering

# Pinch distance thresholds (normalized, tuned roughly)
PINCH_FULL_ACCEL = 0.03      # if index-thumb distance < this -> hold UP (full throttle)
PINCH_PARTIAL_ACCEL = 0.08   # if between PINCH_FULL_ACCEL and this -> partial throttle
PINCH_MAX = 0.25             # anything above this is considered "coast"

# Partial throttle parameters
# We'll simulate partial throttle by sending short UP presses with a frequency that
# depends on how strong the pinch is (closer tip -> higher frequency).
PARTIAL_PRESS_MIN_INTERVAL = 0.08   # fastest repeat for strong pinch (seconds)
PARTIAL_PRESS_MAX_INTERVAL = 0.5    # slowest repeat for weak pinch (seconds)

# Brake: right-hand fist detection threshold (fingertips near pip)
FIST_DETECTION_OFFSET = 0.05

# Cooldowns (seconds)
THROTTLE_COOLDOWN = 0.05  # minimal time between automated thrust presses (safety)
FOCUS_INTERVAL = 2.0      # seconds to attempt re-focusing TrackMania window

# Preview window config
PREVIEW_W = 480
PREVIEW_H = 360
PREVIEW_X = 20
PREVIEW_Y = 20

# Keys mapping (change if you remapped)
KEY_LEFT = 'left'
KEY_RIGHT = 'right'
KEY_ACCEL = 'up'
KEY_BRAKE = 'down'

# OpenCV waitKey delay (ms)
WAIT_KEY_DELAY = 1

# ---------------------------
# Globals and state
# ---------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Make sure your camera is accessible.")

# Keys currently held by camera logic vs keyboard fallback
keys_held_camera = set()
keys_held_keyboard = set()

# Physical keyboard tracking (pynput)
physical_keys_active = set()
physical_keys_lock = threading.Lock()
keyboard_listener = None

# Timing for partial throttle pressing
last_partial_press_time = 0.0

# Last focus attempt time
last_focus_time = 0.0

# Run flag
running = True

# Helper printing
def safe_print(*args, **kwargs):
    print(*args, **kwargs)

# ---------------------------
# Helper functions
# ---------------------------
def activate_trackmania():
    """
    Try to bring TrackMania to foreground using likely window title substrings.
    This helps simulated keys reach the game.
    """
    try:
        candidates = ["TrackMania", "TrackMania Nations", "Trackmania", "TMN"]
        found = False
        for t in candidates:
            wins = gw.getWindowsWithTitle(t)
            if wins:
                w = wins[0]
                try:
                    if w.isMinimized:
                        w.restore()
                        time.sleep(0.06)
                    w.activate()
                    safe_print(f"[INFO] Activated window matching '{t}'.")
                    found = True
                    break
                except Exception as e:
                    try:
                        w.restore()
                        time.sleep(0.06)
                        w.activate()
                        safe_print(f"[INFO] Restored & activated '{t}'.")
                        found = True
                        break
                    except Exception:
                        safe_print(f"[WARN] Could not activate '{t}': {e}")
                        continue
        if not found:
            safe_print("[WARN] TrackMania window not found; click it manually.")
    except Exception as e:
        safe_print("[WARN] Exception in activate_trackmania():", e)

def normalized_distance_pts(a, b):
    """Euclidean distance between two normalized landmarks (.x/.y)."""
    try:
        return math.hypot(a.x - b.x, a.y - b.y)
    except Exception:
        return 1.0

def calc_wrist_angle(left_wrist, right_wrist):
    """Angle (radians) of line between left & right wrist relative to horizontal axis."""
    try:
        dy = right_wrist.y - left_wrist.y
        dx = right_wrist.x - left_wrist.x
        return math.atan2(dy, dx)
    except Exception:
        return 0.0

def is_fist(landmarks):
    """
    Return True if hand is roughly in a fist:
    fingertip y > corresponding pip y + offset (i.e., curled downward on image).
    """
    try:
        idx_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        idx_pip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        mid_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        mid_pip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
        ring_pip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
        pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
        pinky_pip = landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y

        if (idx_tip > idx_pip + FIST_DETECTION_OFFSET and
            mid_tip > mid_pip + FIST_DETECTION_OFFSET and
            ring_tip > ring_pip + FIST_DETECTION_OFFSET and
            pinky_tip > pinky_pip + FIST_DETECTION_OFFSET):
            return True
    except Exception:
        return False
    return False

def right_hand_pinch_strength(landmarks):
    """
    Compute normalized pinch distance (index_tip <-> thumb_tip) on a single hand landmarks object.
    Returns a value in [0, ~0.5] where smaller is stronger pinch.
    """
    try:
        thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        # normalized Euclidean distance
        d = normalized_distance_pts(thumb_tip, index_tip)
        return d
    except Exception:
        return 1.0

def robust_hold_key(keyname):
    """
    A short robust press: keyDown -> small hold -> keyUp
    Used for single-shot or brief presses for partial throttle pulses.
    """
    try:
        pydirectinput.keyDown(keyname)
        time.sleep(0.03)
        pydirectinput.keyUp(keyname)
    except Exception as e:
        safe_print(f"[ERROR] robust_hold_key({keyname}) failed: {e}")

# ---------------------------
# pynput keyboard listener (fallback)
# ---------------------------
def on_press(key):
    """Add pressed physical keys to set for fallback."""
    try:
        name = None
        if hasattr(key, 'char') and key.char is not None:
            name = key.char.lower()
        else:
            name = str(key).replace('Key.', '').lower()
        with physical_keys_lock:
            physical_keys_active.add(name)
    except Exception:
        pass

def on_release(key):
    """Remove released keys from set."""
    try:
        name = None
        if hasattr(key, 'char') and key.char is not None:
            name = key.char.lower()
        else:
            name = str(key).replace('Key.', '').lower()
        with physical_keys_lock:
            if name in physical_keys_active:
                physical_keys_active.remove(name)
    except Exception:
        pass

if pynput_keyboard is not None:
    try:
        keyboard_listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
        keyboard_listener.daemon = True
        keyboard_listener.start()
        safe_print("[INFO] Physical keyboard listener started (pynput).")
    except Exception as e:
        safe_print("[WARN] Could not start pynput listener:", e)
        keyboard_listener = None
else:
    safe_print("[WARN] pynput not available; physical-key fallback disabled.")

# ---------------------------
# Preview window setup
# ---------------------------
cv2.namedWindow("TM Gesture Preview", cv2.WINDOW_NORMAL)
cv2.resizeWindow("TM Gesture Preview", PREVIEW_W, PREVIEW_H)
cv2.moveWindow("TM Gesture Preview", PREVIEW_X, PREVIEW_Y)
try:
    cv2.setWindowProperty("TM Gesture Preview", cv2.WND_PROP_TOPMOST, 1)
except Exception:
    safe_print("[WARN] setWindowProperty(WND_PROP_TOPMOST) not supported on this system.")

# Attempt initial activation
safe_print("[INFO] Starting TrackMania gesture controller. Attempting to activate TrackMania window...")
activate_trackmania()

# ---------------------------
# Main loop
# ---------------------------
try:
    # last time we did a partial throttle pulse (for rate limiting)
    last_partial_time = 0.0

    while True:
        now = time.time()
        # periodically try to activate TrackMania
        if now - last_focus_time > FOCUS_INTERVAL:
            activate_trackmania()
            last_focus_time = now

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)  # mirror
        frame_h, frame_w = frame.shape[:2]

        # media pipe processing
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)

        # default telemetry
        steer_state = "Neutral"
        throttle_state = "Coast"
        brake_state = "No"
        pinch_val = 1.0
        tilt_angle = 0.0

        # desired keys from camera this frame
        camera_desired_keys = set()

        # detect two hands (we need both for reliable steering)
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
            # draw both hands
            try:
                mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            except Exception:
                pass
            try:
                mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[1], mp_hands.HAND_CONNECTIONS)
            except Exception:
                pass

            # pick first two hands
            h0 = results.multi_hand_landmarks[0]
            h1 = results.multi_hand_landmarks[1]

            # compute wrist landmarks for tilt/steer
            try:
                w0 = h0.landmark[mp_hands.HandLandmark.WRIST]
                w1 = h1.landmark[mp_hands.HandLandmark.WRIST]
            except Exception:
                w0 = None
                w1 = None

            # determine left/right visually by x coordinate
            if w0 is not None and w1 is not None:
                if w0.x < w1.x:
                    left_hand = h0
                    right_hand = h1
                    left_wrist = w0
                    right_wrist = w1
                else:
                    left_hand = h1
                    right_hand = h0
                    left_wrist = w1
                    right_wrist = w0

                # Steering via tilt angle
                tilt_angle = calc_wrist_angle(left_wrist, right_wrist)
                if tilt_angle > STEER_TILT_THRESHOLD:
                    camera_desired_keys.add(KEY_RIGHT)
                    steer_state = "Right"
                elif tilt_angle < -STEER_TILT_THRESHOLD:
                    camera_desired_keys.add(KEY_LEFT)
                    steer_state = "Left"
                else:
                    steer_state = "Neutral"

                # Right-hand pinch detection for throttle
                pinch_val = right_hand_pinch_strength(right_hand)  # normalized distance
                # pinch_val smaller -> stronger pinch -> we accelerate more

                # Decide throttle behavior:
                # - pinch < PINCH_FULL_ACCEL -> full hold UP
                # - PINCH_FULL_ACCEL <= pinch < PINCH_PARTIAL_ACCEL -> partial throttle (intermittent presses)
                # - pinch >= PINCH_PARTIAL_ACCEL -> coast (no UP)
                if pinch_val < PINCH_FULL_ACCEL:
                    # full throttle
                    camera_desired_keys.add(KEY_ACCEL)
                    throttle_state = "Full"
                elif pinch_val < PINCH_PARTIAL_ACCEL:
                    # partial throttle: do intermittent short presses at frequency depending on pinch_val
                    # compute normalized strength (0..1) where 1 is just at PINCH_FULL_ACCEL and 0 at PINCH_PARTIAL_ACCEL
                    # stronger pinch -> higher frequency -> smaller interval
                    norm = (PINCH_PARTIAL_ACCEL - pinch_val) / (PINCH_PARTIAL_ACCEL - PINCH_FULL_ACCEL)
                    norm = max(0.0, min(1.0, norm))
                    # compute interval between presses
                    interval = PARTIAL_PRESS_MIN_INTERVAL + (1.0 - norm) * (PARTIAL_PRESS_MAX_INTERVAL - PARTIAL_PRESS_MIN_INTERVAL)
                    # if enough time passed since last partial press, emit a short UP press
                    if time.time() - last_partial_time > interval:
                        robust_hold_key(KEY_ACCEL)
                        last_partial_time = time.time()
                    throttle_state = f"Partial ({interval:.2f}s interval)"
                else:
                    # coast
                    throttle_state = "Coast"

                # Right-hand fist -> brake (hold DOWN)
                if is_fist(right_hand):
                    camera_desired_keys.add(KEY_BRAKE)
                    brake_state = "Brake (Right fist)"
                else:
                    brake_state = "No"

            else:
                # missing wrist data
                steer_state = "Missing wrists"
                throttle_state = "Missing wrists"
                brake_state = "No"

            # Additional: If both hands are fists -> we could map to emergency brake or other,
            # but per your last instruction brake is right-hand fist only.
            # So we don't override.

        else:
            # less than two hands -> no camera steering; check single-hand situations
            # check if a single right hand is visible and acting for throttle/brake
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
                single = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, single, mp_hands.HAND_CONNECTIONS)
                # We need to determine whether this is the right hand. MediaPipe doesn't give handedness reliably without extra output;
                # assume single is right for throttle control in that case
                pinch_val = right_hand_pinch_strength(single)
                if pinch_val < PINCH_FULL_ACCEL:
                    camera_desired_keys.add(KEY_ACCEL)
                    throttle_state = "Full(single)"
                elif pinch_val < PINCH_PARTIAL_ACCEL:
                    norm = (PINCH_PARTIAL_ACCEL - pinch_val) / (PINCH_PARTIAL_ACCEL - PINCH_FULL_ACCEL)
                    norm = max(0.0, min(1.0, norm))
                    interval = PARTIAL_PRESS_MIN_INTERVAL + (1.0 - norm) * (PARTIAL_PRESS_MAX_INTERVAL - PARTIAL_PRESS_MIN_INTERVAL)
                    if time.time() - last_partial_time > interval:
                        robust_hold_key(KEY_ACCEL)
                        last_partial_time = time.time()
                    throttle_state = f"Partial(single, {interval:.2f}s)"
                else:
                    throttle_state = "Coast(single)"

                if is_fist(single):
                    camera_desired_keys.add(KEY_BRAKE)
                    brake_state = "Brake(single)"
                else:
                    brake_state = "No"
                steer_state = "Single-hand mode (no steer)"
            else:
                # no hands detected
                steer_state = "No hands"
                throttle_state = "No hands"
                brake_state = "No hands"

        # ---------------------------
        # Apply camera-priority keys
        # ---------------------------
        try:
            # Hold keys requested by camera (keyDown)
            for k in camera_desired_keys:
                if k not in keys_held_camera:
                    try:
                        pydirectinput.keyDown(k)
                        keys_held_camera.add(k)
                        safe_print(f"[CAMERA] keyDown {k}")
                    except Exception as e:
                        safe_print(f"[ERROR] keyDown {k} failed: {e}")

            # Release camera-held keys that are no longer desired
            to_release = list(keys_held_camera - camera_desired_keys)
            for k in to_release:
                try:
                    pydirectinput.keyUp(k)
                    keys_held_camera.discard(k)
                    safe_print(f"[CAMERA] keyUp {k}")
                except Exception as e:
                    safe_print(f"[ERROR] keyUp {k} failed: {e}")

            # If camera controls nothing relevant, fall back to physical keyboard pressed keys
            if not camera_desired_keys:
                snapshot = set()
                if pynput_keyboard is not None:
                    with physical_keys_lock:
                        snapshot = set(physical_keys_active)
                # normalize names, take only relevant ones
                normalized = set([str(x).lower() for x in snapshot])
                phys_should_hold = set()
                if 'left' in normalized or KEY_LEFT in normalized:
                    phys_should_hold.add(KEY_LEFT)
                if 'right' in normalized or KEY_RIGHT in normalized:
                    phys_should_hold.add(KEY_RIGHT)
                if 'up' in normalized or KEY_ACCEL in normalized:
                    phys_should_hold.add(KEY_ACCEL)
                if 'down' in normalized or KEY_BRAKE in normalized:
                    phys_should_hold.add(KEY_BRAKE)

                for k in phys_should_hold:
                    if k not in keys_held_keyboard:
                        try:
                            pydirectinput.keyDown(k)
                            keys_held_keyboard.add(k)
                            safe_print(f"[PHYS->GAME] keyDown {k}")
                        except Exception as e:
                            safe_print(f"[ERROR] phys keyDown {k}: {e}")

                phys_to_release = list(keys_held_keyboard - phys_should_hold)
                for k in phys_to_release:
                    try:
                        pydirectinput.keyUp(k)
                        keys_held_keyboard.discard(k)
                        safe_print(f"[PHYS->GAME] keyUp {k}")
                    except Exception as e:
                        safe_print(f"[ERROR] phys keyUp {k}: {e}")

        except Exception as e:
            safe_print("[ERROR] while applying camera/keyboard keys:", e)

        # ---------------------------
        # Telemetry overlay drawing
        # ---------------------------
        try:
            overlay = frame.copy()
            header_h = 120
            cv2.rectangle(overlay, (0, 0), (PREVIEW_W, header_h), (0, 0, 0), -1)
            alpha = 0.55
            frame[0:header_h, 0:PREVIEW_W] = cv2.addWeighted(overlay[0:header_h, 0:PREVIEW_W], alpha,
                                                             frame[0:header_h, 0:PREVIEW_W], 1 - alpha, 0)

            cv2.putText(frame, f"Steer: {steer_state}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            cv2.putText(frame, f"Throttle: {throttle_state}", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            cv2.putText(frame, f"Brake (R fist): {brake_state}", (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            cv2.putText(frame, f"Pinch dist (right): {pinch_val:.3f}", (8, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
            cv2.putText(frame, f"Tilt rad: {tilt_angle:.3f}", (8, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

            camkeys = ", ".join(sorted(keys_held_camera)) if keys_held_camera else "None"
            physkeys = ", ".join(sorted(keys_held_keyboard)) if keys_held_keyboard else "None"
            cv2.putText(frame, f"Camera-held: {camkeys}", (8, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,255), 1)
            cv2.putText(frame, f"Phys-held: {physkeys}", (8, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,255,180), 1)
        except Exception as e:
            safe_print("[WARN] Telemetry overlay error:", e)

        # show preview (resized)
        try:
            preview = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
            cv2.imshow("TM Gesture Preview", preview)
        except Exception:
            cv2.imshow("TM Gesture Preview", frame)

        # quit with 'q' on preview
        if (cv2.waitKey(WAIT_KEY_DELAY) & 0xFF) == ord('q'):
            safe_print("[INFO] 'q' pressed in preview - exiting.")
            break

        # small sleep
        time.sleep(0.001)

# ---------------------------
# exception handling and cleanup
# ---------------------------
except KeyboardInterrupt:
    safe_print("\n[INFO] KeyboardInterrupt received, exiting.")
except Exception as e:
    safe_print("[ERROR] Unhandled exception:", e)
    traceback.print_exc(file=sys.stdout)
finally:
    safe_print("[CLEANUP] releasing keys, camera, listeners...")
    # Release camera-held keys
    try:
        for k in list(keys_held_camera):
            try:
                pydirectinput.keyUp(k)
                safe_print(f"[CLEANUP] Released camera key: {k}")
            except Exception as e:
                safe_print(f"[CLEANUP ERROR] keyUp {k}: {e}")
        keys_held_camera.clear()
    except Exception as e:
        safe_print("[CLEANUP ERROR] releasing camera keys:", e)

    # Release keyboard-held keys
    try:
        for k in list(keys_held_keyboard):
            try:
                pydirectinput.keyUp(k)
                safe_print(f"[CLEANUP] Released phys key: {k}")
            except Exception as e:
                safe_print(f"[CLEANUP ERROR] phys keyUp {k}: {e}")
        keys_held_keyboard.clear()
    except Exception as e:
        safe_print("[CLEANUP ERROR] releasing phys keys:", e)

    # stop keyboard listener
    try:
        if keyboard_listener is not None:
            keyboard_listener.stop()
            safe_print("[CLEANUP] keyboard listener stopped.")
    except Exception as e:
        safe_print("[CLEANUP ERROR] stopping keyboard listener:", e)

    # release camera
    try:
        cap.release()
        safe_print("[CLEANUP] camera released.")
    except Exception as e:
        safe_print("[CLEANUP ERROR] releasing camera:", e)

    # destroy windows
    try:
        cv2.destroyAllWindows()
        safe_print("[CLEANUP] OpenCV windows destroyed.")
    except Exception as e:
        safe_print("[CLEANUP ERROR] destroying windows:", e)

    # close mediapipe
    try:
        hands_detector.close()
        safe_print("[CLEANUP] MediaPipe hands closed.")
    except Exception as e:
        safe_print("[CLEANUP ERROR] closing MediaPipe:", e)

    safe_print("[INFO] Shutdown complete. Have fun in TrackMania!")
