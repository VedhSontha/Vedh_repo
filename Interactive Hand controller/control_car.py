"""
TrackMania Gesture Controller - Both-wrist steering + per-hand pinch throttle
Verbose, long-form (250+ lines)

Controls:
 - Steering: both wrists tilt -> hold LEFT/RIGHT while tilt persists
 - Forward throttle: RIGHT-hand pinch (thumb <-> index)
     - pinch small -> full throttle (hold UP)
     - pinch medium -> partial throttle (intermittent UP presses; frequency depends on pinch strength)
     - pinch large -> coast (no UP)
 - Reverse throttle: LEFT-hand pinch (thumb <-> index) same rules -> sends DOWN
 - Emergency brake: both fists closed -> hold SPACE (configurable)
 - Camera gestures take priority; physical keyboard fallback supported (pynput)
 - Uses pydirectinput for game-friendly key events
 - Shows MediaPipe landmarks and telemetry overlay
"""

# ---------------------------
# Imports
# ---------------------------
import cv2
import mediapipe as mp
import pydirectinput
import pygetwindow as gw
import time
import math
import numpy as np
import threading
import traceback
import sys

# pynput for physical keyboard fallback (optional)
try:
    from pynput import keyboard as pynput_keyboard
except Exception:
    pynput_keyboard = None

# ---------------------------
# Safety & basic settings
# ---------------------------
pydirectinput.FAILSAFE = False

# ---------------------------
# TUNABLE PARAMETERS
# ---------------------------
# MediaPipe
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Steering sensitivity (radians)
STEER_TILT_THRESHOLD = 0.28  # smaller = more sensitive

# Pinch distance thresholds (normalized)
PINCH_FULL_THRESHOLD = 0.03    # distance <= this -> full throttle
PINCH_PARTIAL_THRESHOLD = 0.08 # PINCH_FULL < distance <= PARTIAL -> partial throttle
PINCH_COAST_THRESHOLD = 0.25   # anything above considered no throttle

# Partial throttle parameters (simulate partial throttle by intermittent presses)
PARTIAL_PRESS_MIN_INTERVAL = 0.08  # fastest repeat (strong pinch)
PARTIAL_PRESS_MAX_INTERVAL = 0.45  # slowest repeat (weak partial pinch)

# Fist detection offset (how much fingertip is below pip to count as curled)
FIST_DETECTION_OFFSET = 0.05

# Cooldowns
THROTTLE_COOLDOWN = 0.04  # minimal time between automated pulses
FOCUS_INTERVAL = 2.0      # seconds to attempt re-focusing TrackMania window

# Preview window
PREVIEW_W = 480
PREVIEW_H = 360
PREVIEW_X = 20
PREVIEW_Y = 20

# Keys to use (change if you remapped in TrackMania)
KEY_LEFT = 'left'
KEY_RIGHT = 'right'
KEY_ACCEL = 'up'
KEY_REVERSE = 'down'
KEY_BRAKE = 'space'  # emergency brake when both fists closed

# OpenCV waitKey
WAIT_KEY_DELAY = 1  # ms

# ---------------------------
# Globals & initialization
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
    raise RuntimeError("Cannot open webcam (index 0). Check camera permissions.")

# Held keys tracking
keys_held_camera = set()   # keys currently held by camera logic
keys_held_physical = set() # keys currently held by physical keyboard fallback

# Physical keyboard tracking
physical_keys_active = set()
physical_keys_lock = threading.Lock()
keyboard_listener = None

# Timing helpers
last_partial_press_time = 0.0
last_focus_time = 0.0

# Running flag
running = True

# Printing helper
def safe_print(*args, **kwargs):
    print(*args, **kwargs)

# ---------------------------
# Helper functions
# ---------------------------
def activate_trackmania_window():
    """
    Try to bring TrackMania to front so synthetic key events reach it.
    Attempts several likely window title substrings.
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
            safe_print("[WARN] TrackMania window not found. Click it manually.")
    except Exception as e:
        safe_print("[WARN] Exception in activate_trackmania_window():", e)

def normalized_distance(a, b):
    """
    Euclidean distance between two normalized landmarks (objects with .x and .y)
    """
    try:
        return math.hypot(a.x - b.x, a.y - b.y)
    except Exception:
        return 1.0

def calc_wrist_angle(left_wrist, right_wrist):
    """
    Angle between wrist line and horizontal axis (radians).
    Positive -> right wrist lower -> steer right.
    Negative -> left wrist lower -> steer left.
    """
    try:
        dy = right_wrist.y - left_wrist.y
        dx = right_wrist.x - left_wrist.x
        return math.atan2(dy, dx)
    except Exception:
        return 0.0

def is_fist_landmarks(hand_landmarks):
    """
    Return True if hand is roughly a fist (fingertips below pip + offset).
    Checks index/middle/ring/pinky finger tips vs pips.
    """
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

def pinch_distance(hand_landmarks):
    """
    Return normalized distance between thumb tip and index tip for a given hand landmarks object.
    Smaller = stronger pinch.
    """
    try:
        thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        idx = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        return normalized_distance(thumb, idx)
    except Exception:
        return 1.0

def robust_tap(keyname):
    """Single robust short press: keyDown -> tiny hold -> keyUp"""
    try:
        pydirectinput.keyDown(keyname)
        time.sleep(0.03)
        pydirectinput.keyUp(keyname)
    except Exception as e:
        safe_print(f"[ERROR] robust_tap({keyname}) failed: {e}")

def robust_hold_pulse(keyname, repeats=1, hold=0.03):
    """
    If you need multiple small pulses (used by partial throttle),
    call this. It performs quick down/up repeats.
    """
    try:
        for _ in range(repeats):
            pydirectinput.keyDown(keyname)
            time.sleep(hold)
            pydirectinput.keyUp(keyname)
            time.sleep(0.02)
    except Exception as e:
        safe_print(f"[ERROR] robust_hold_pulse({keyname}) failed: {e}")

# ---------------------------
# pynput keyboard listener (physical fallback)
# ---------------------------
def on_press(key):
    """Record physical pressed keys (pynput)."""
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
    """Remove released physical keys."""
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

# start pynput listener if available
if pynput_keyboard is not None:
    try:
        keyboard_listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
        keyboard_listener.daemon = True
        keyboard_listener.start()
        safe_print("[INFO] pynput keyboard listener started.")
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
    safe_print("[WARN] Cannot set preview always-on-top on this platform.")

# Try to focus TrackMania once
safe_print("[INFO] Starting; attempting to activate TrackMania window...")
activate_trackmania_window()

# ---------------------------
# Main loop (verbose)
# ---------------------------
try:
    # timestamp of last partial throttle pulse
    last_partial_time = 0.0

    while True:
        now = time.time()

        # periodically ensure TrackMania has focus
        if now - last_focus_time > FOCUS_INTERVAL:
            activate_trackmania_window()
            last_focus_time = now

        # read camera
        success, frame = cap.read()
        if not success:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)  # mirror
        frame_h, frame_w = frame.shape[:2]

        # process with MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)

        # telemetry defaults
        steer_state = "Neutral"
        forward_state = "Coast"
        reverse_state = "Coast"
        brake_state = "No"
        pinch_right = 1.0
        pinch_left = 1.0
        tilt_angle = 0.0

        # camera desired keys for this frame
        camera_desired_keys = set()

        # detect hands
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

            # get the first two hands
            h0 = results.multi_hand_landmarks[0]
            h1 = results.multi_hand_landmarks[1]

            # get wrist landmarks
            try:
                w0 = h0.landmark[mp_hands.HandLandmark.WRIST]
                w1 = h1.landmark[mp_hands.HandLandmark.WRIST]
            except Exception:
                w0 = None
                w1 = None

            # determine which hand is left/right visually (x coordinate)
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

                # steering via tilt angle
                tilt_angle = calc_wrist_angle(left_wrist, right_wrist)
                if tilt_angle > STEER_TILT_THRESHOLD:
                    camera_desired_keys.add(KEY_RIGHT)
                    steer_state = "Right"
                elif tilt_angle < -STEER_TILT_THRESHOLD:
                    camera_desired_keys.add(KEY_LEFT)
                    steer_state = "Left"
                else:
                    steer_state = "Neutral"

                # pinch detection per hand
                pinch_right = pinch_distance(right_hand)
                pinch_left = pinch_distance(left_hand)

                # RIGHT-hand pinch -> FORWARD throttle
                # smaller pinch_right -> stronger pinch -> more forward throttle
                if pinch_right <= PINCH_FULL_THRESHOLD:
                    camera_desired_keys.add(KEY_ACCEL)
                    forward_state = "Full (R pinch)"
                elif pinch_right <= PINCH_PARTIAL_THRESHOLD:
                    # partial forward: intermittent UP presses depending on pinch strength
                    # normalize strength 0..1 (1 -> near PINCH_FULL_THRESHOLD)
                    norm = (PINCH_PARTIAL_THRESHOLD - pinch_right) / (PINCH_PARTIAL_THRESHOLD - PINCH_FULL_THRESHOLD)
                    norm = max(0.0, min(1.0, norm))
                    interval = PARTIAL_PRESS_MIN_INTERVAL + (1.0 - norm) * (PARTIAL_PRESS_MAX_INTERVAL - PARTIAL_PRESS_MIN_INTERVAL)
                    # send a short pulse if interval elapsed
                    if time.time() - last_partial_time > interval:
                        robust_tap(KEY_ACCEL)
                        last_partial_time = time.time()
                        forward_state = f"Partial (interval {interval:.2f}s)"
                    else:
                        forward_state = f"Partial waiting ({interval:.2f}s)"
                else:
                    forward_state = "Coast (R no pinch)"

                # LEFT-hand pinch -> REVERSE throttle
                if pinch_left <= PINCH_FULL_THRESHOLD:
                    camera_desired_keys.add(KEY_REVERSE)
                    reverse_state = "Full (L pinch)"
                elif pinch_left <= PINCH_PARTIAL_THRESHOLD:
                    # partial reverse
                    norm = (PINCH_PARTIAL_THRESHOLD - pinch_left) / (PINCH_PARTIAL_THRESHOLD - PINCH_FULL_THRESHOLD)
                    norm = max(0.0, min(1.0, norm))
                    interval = PARTIAL_PRESS_MIN_INTERVAL + (1.0 - norm) * (PARTIAL_PRESS_MAX_INTERVAL - PARTIAL_PRESS_MIN_INTERVAL)
                    if time.time() - last_partial_time > interval:
                        robust_tap(KEY_REVERSE)
                        last_partial_time = time.time()
                        reverse_state = f"Partial (interval {interval:.2f}s)"
                    else:
                        reverse_state = f"Partial waiting ({interval:.2f}s)"
                else:
                    reverse_state = "Coast (L no pinch)"

                # BRAKE: both fists closed -> emergency brake (hold KEY_BRAKE)
                is_left_fist = is_fist_landmarks(left_hand)
                is_right_fist = is_fist_landmarks(right_hand)
                if is_left_fist and is_right_fist:
                    camera_desired_keys.add(KEY_BRAKE)
                    brake_state = "EMERGENCY (both fists)"
                else:
                    # If only right-hand fist and you want to map to brake, you could add:
                    # if is_right_fist: camera_desired_keys.add(KEY_BRAKE); brake_state = "Right fist brake"
                    brake_state = "No"

            else:
                # missing wrist data
                steer_state = "Missing wrists"
                forward_state = "Missing wrists"
                reverse_state = "Missing wrists"
                brake_state = "No"

        elif results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
            # one hand visible: treat it as right hand for throttle/brake (convenience)
            single = results.multi_hand_landmarks[0]
            try:
                mp_drawing.draw_landmarks(frame, single, mp_hands.HAND_CONNECTIONS)
            except Exception:
                pass

            # use single-hand pinch for forward (assume right)
            pinch_right = pinch_distance(single)
            if pinch_right <= PINCH_FULL_THRESHOLD:
                camera_desired_keys.add(KEY_ACCEL)
                forward_state = "Full (single-hand)"
            elif pinch_right <= PINCH_PARTIAL_THRESHOLD:
                norm = (PINCH_PARTIAL_THRESHOLD - pinch_right) / (PINCH_PARTIAL_THRESHOLD - PINCH_FULL_THRESHOLD)
                norm = max(0.0, min(1.0, norm))
                interval = PARTIAL_PRESS_MIN_INTERVAL + (1.0 - norm) * (PARTIAL_PRESS_MAX_INTERVAL - PARTIAL_PRESS_MIN_INTERVAL)
                if time.time() - last_partial_time > interval:
                    robust_tap(KEY_ACCEL)
                    last_partial_time = time.time()
                    forward_state = f"Partial(single, {interval:.2f}s)"
                else:
                    forward_state = f"Partial waiting(single, {interval:.2f}s)"
            else:
                forward_state = "Coast(single)"

            # single-hand fist -> brake
            if is_fist_landmarks(single):
                camera_desired_keys.add(KEY_BRAKE)
                brake_state = "Brake(single)"
            else:
                brake_state = "No"
            # steering neutral in single-hand mode
            steer_state = "Single-hand (no steer)"

        else:
            # no hands
            steer_state = "No hands"
            forward_state = "No hands"
            reverse_state = "No hands"
            brake_state = "No hands"

        # ---------------------------
        # Apply camera-priority keys
        # ---------------------------
        try:
            # press & hold camera desired keys
            for k in camera_desired_keys:
                if k not in keys_held_camera:
                    try:
                        pydirectinput.keyDown(k)
                        keys_held_camera.add(k)
                        safe_print(f"[CAMERA] keyDown {k}")
                    except Exception as e:
                        safe_print(f"[ERROR] keyDown {k} failed: {e}")

            # release camera-held keys no longer desired
            to_release = list(keys_held_camera - camera_desired_keys)
            for k in to_release:
                try:
                    pydirectinput.keyUp(k)
                    keys_held_camera.discard(k)
                    safe_print(f"[CAMERA] keyUp {k}")
                except Exception as e:
                    safe_print(f"[ERROR] keyUp {k} failed: {e}")

            # if camera not controlling, apply physical keyboard fallback
            if not camera_desired_keys:
                snapshot = set()
                if pynput_keyboard is not None:
                    with physical_keys_lock:
                        snapshot = set(physical_keys_active)
                lower_snapshot = set([str(x).lower() for x in snapshot])
                phys_should_hold = set()
                if 'left' in lower_snapshot or KEY_LEFT in lower_snapshot:
                    phys_should_hold.add(KEY_LEFT)
                if 'right' in lower_snapshot or KEY_RIGHT in lower_snapshot:
                    phys_should_hold.add(KEY_RIGHT)
                if 'up' in lower_snapshot or KEY_ACCEL in lower_snapshot:
                    phys_should_hold.add(KEY_ACCEL)
                if 'down' in lower_snapshot or KEY_REVERSE in lower_snapshot:
                    phys_should_hold.add(KEY_REVERSE)
                if 'space' in lower_snapshot or KEY_BRAKE in lower_snapshot:
                    phys_should_hold.add(KEY_BRAKE)

                # hold phys keys
                for k in phys_should_hold:
                    if k not in keys_held_physical:
                        try:
                            pydirectinput.keyDown(k)
                            keys_held_physical.add(k)
                            safe_print(f"[PHYS] keyDown {k}")
                        except Exception as e:
                            safe_print(f"[ERROR] physical keyDown {k}: {e}")

                # release physical keys no longer requested
                to_release_phys = list(keys_held_physical - phys_should_hold)
                for k in to_release_phys:
                    try:
                        pydirectinput.keyUp(k)
                        keys_held_physical.discard(k)
                        safe_print(f"[PHYS] keyUp {k}")
                    except Exception as e:
                        safe_print(f"[ERROR] physical keyUp {k}: {e}")

        except Exception as e:
            safe_print("[ERROR] applying keys failed:", e)

        # ---------------------------
        # Telemetry overlay
        # ---------------------------
        try:
            overlay = frame.copy()
            info_h = 140
            cv2.rectangle(overlay, (0, 0), (PREVIEW_W, info_h), (0, 0, 0), -1)
            alpha = 0.55
            frame[0:info_h, 0:PREVIEW_W] = cv2.addWeighted(overlay[0:info_h, 0:PREVIEW_W], alpha,
                                                          frame[0:info_h, 0:PREVIEW_W], 1 - alpha, 0)

            cv2.putText(frame, f"Steer: {steer_state}", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            cv2.putText(frame, f"Forward: {forward_state}", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            cv2.putText(frame, f"Reverse: {reverse_state}", (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            cv2.putText(frame, f"Brake: {brake_state}", (8, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            cv2.putText(frame, f"Pinch R: {pinch_right:.3f}", (8, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
            cv2.putText(frame, f"Pinch L: {pinch_left:.3f}", (8, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
            cv2.putText(frame, f"Tilt rad: {tilt_angle:.3f}", (8, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

            camkeys = ", ".join(sorted(keys_held_camera)) if keys_held_camera else "None"
            physkeys = ", ".join(sorted(keys_held_physical)) if keys_held_physical else "None"
            cv2.putText(frame, f"Camera-held: {camkeys}", (260, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,255), 1)
            cv2.putText(frame, f"Phys-held: {physkeys}", (260, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,255,180), 1)
        except Exception as e:
            safe_print("[WARN] telemetry overlay error:", e)

        # show preview
        try:
            preview = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
            cv2.imshow("TM Gesture Preview", preview)
        except Exception:
            cv2.imshow("TM Gesture Preview", frame)

        # quit if 'q' in preview
        if (cv2.waitKey(WAIT_KEY_DELAY) & 0xFF) == ord('q'):
            safe_print("[INFO] 'q' pressed in preview - exiting.")
            break

        # tiny sleep
        time.sleep(0.001)

# ---------------------------
# cleanup + exception handling
# ---------------------------
except KeyboardInterrupt:
    safe_print("\n[INFO] KeyboardInterrupt - exiting.")
except Exception as e:
    safe_print("[ERROR] Unhandled exception:", e)
    traceback.print_exc(file=sys.stdout)
finally:
    safe_print("[CLEANUP] releasing keys, camera, MediaPipe, listeners...")

    # release camera-held keys
    try:
        for k in list(keys_held_camera):
            try:
                pydirectinput.keyUp(k)
                safe_print(f"[CLEANUP] Released camera-held key: {k}")
            except Exception as e:
                safe_print(f"[CLEANUP ERROR] releasing camera key {k}: {e}")
        keys_held_camera.clear()
    except Exception as e:
        safe_print("[CLEANUP ERROR] releasing camera keys:", e)

    # release physical-held keys
    try:
        for k in list(keys_held_physical):
            try:
                pydirectinput.keyUp(k)
                safe_print(f"[CLEANUP] Released phys-held key: {k}")
            except Exception as e:
                safe_print(f"[CLEANUP ERROR] releasing phys key {k}: {e}")
        keys_held_physical.clear()
    except Exception as e:
        safe_print("[CLEANUP ERROR] releasing phys keys:", e)

    # stop keyboard listener
    try:
        if keyboard_listener is not None:
            keyboard_listener.stop()
            safe_print("[CLEANUP] Stopped keyboard listener.")
    except Exception as e:
        safe_print("[CLEANUP ERROR] stopping keyboard listener:", e)

    # release camera
    try:
        cap.release()
        safe_print("[CLEANUP] Camera released.")
    except Exception as e:
        safe_print("[CLEANUP ERROR] releasing camera:", e)

    # destroy windows
    try:
        cv2.destroyAllWindows()
        safe_print("[CLEANUP] OpenCV windows destroyed.")
    except Exception as e:
        safe_print("[CLEANUP ERROR] destroying windows:", e)

    # close MediaPipe
    try:
        hands_detector.close()
        safe_print("[CLEANUP] MediaPipe hands detector closed.")
    except Exception as e:
        safe_print("[CLEANUP ERROR] closing MediaPipe:", e)

    safe_print("[INFO] Shutdown complete. Have fun in TrackMania!")
