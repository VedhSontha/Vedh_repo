import cv2
import numpy as np
import imutils
import tensorflow as tf
from tensorflow.keras.models import load_model
import dlib

YOLO_CFG = '/home/shashank/yoloface/yoloface/cfg/yolov3-face.cfg'
YOLO_WEIGHTS = '/home/shashank/yoloface/yoloface/model-weights/yolov3-wider_16000.weights'

FACE_NET_MODEL_PATH = 'facenet_model.h5'
facenet_model = load_model(FACE_NET_MODEL_PATH)

def load_yolo_model(cfg, weights):
    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

net = load_yolo_model(YOLO_CFG, YOLO_WEIGHTS)
layer_names = net.getUnconnectedOutLayersNames()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
tracker = None
tracking = False
selected_embedding = None
selected_box = None
clicked = False

known_faces = []

def adjust_gamma(image, gamma=1.7):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_denoising(frame):
    return cv2.fastNlMeansDenoisingColored(frame, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

def detect_faces_yolo(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)
    boxes = []
    confidences = []
    for out in outs:
        for det in out:
            scores = det[5:]
            if scores.size == 0:
                continue
            confidence = scores[0]
            if confidence > 0.4:
                center_x = int(det[0] * w)
                center_y = int(det[1] * h)
                bw = int(det[2] * w)
                bh = int(det[3] * h)
                x = int(center_x - bw / 2)
                y = int(center_y - bh / 2)
                boxes.append([x, y, bw, bh])
                confidences.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    return [boxes[i] for i in indices.flatten()] if len(indices) > 0 else []

def extract_face_embedding(frame, box):
    x, y, w, h = box
    face_img = frame[y:y+h, x:x+w]
    if face_img.size == 0:
        return None
    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img_resized = cv2.resize(face_img_rgb, (160, 160)) / 255.0
    face_input = np.expand_dims(face_img_resized, axis=0)
    embedding = facenet_model.predict(face_input)
    embd=np.mean(embedding, axis=0)
    return embd, face_img

def cosine_similarity(a, b):
    if a is None or b is None:
        return 0
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def on_mouse_click(event, x, y, flags, param):
    global clicked, selected_box, tracker, tracking, selected_embedding
    if event == cv2.EVENT_LBUTTONDOWN:
        for box in detected_boxes:
            bx, by, bw, bh = box
            if bx <= x <= bx + bw and by <= y <= by + bh:
                selected_box = (bx, by, bw, bh)
                selected_embedding, _ = extract_face_embedding(frame, selected_box)
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, selected_box)
                tracking = True
                clicked = True
                print("Face selected and tracking started.")
                known_faces.append((selected_embedding, selected_box))
                break

cv2.namedWindow("Face Re-ID Tracker")
cv2.setMouseCallback("Face Re-ID Tracker", on_mouse_click)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = imutils.resize(frame, width=720)
    
    frame = adjust_gamma(frame, gamma=1.7)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < 100:
        frame = apply_denoising(frame)

    display = frame.copy()

    if tracking:
        success, box = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in box]
            selected_box = (x, y, w, h)
            cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(display, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        else:
            print("Tracking lost. Re-detecting...")
            tracking = False

    if not tracking and clicked:
        detected_boxes = detect_faces_yolo(frame)
        best_match = None
        best_score = 0.70
        last_x, last_y, last_w, last_h = selected_box if selected_box else (0, 0, 0, 0)
        for box in detected_boxes:
            emb, face_crop = extract_face_embedding(frame, box)
            if emb is None:
                continue
            brightness = np.mean(cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY))
            if brightness < 80:
                threshold = 0.9
            elif brightness < 130:
                threshold = 0.9
            else:
                threshold = 0.9
            score = cosine_similarity(emb, selected_embedding)
            position_diff = abs(box[0] - last_x) + abs(box[1] - last_y)
            combined_score = score * 0.8 + (1 - position_diff / 1000) * 0.2
            print(f"Cosine: {score:.2f}, Brightness: {brightness:.1f}, Combined: {combined_score:.2f}, Threshold: {threshold}")
            if combined_score > threshold and combined_score > best_score:
                best_score = combined_score
                best_match = box
        if best_match:
            print(f"Best match with score {best_score:.2f}")
            x, y, w, h = best_match
            selected_box = (x, y, w, h)
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, selected_box)
            tracking = True
            selected_embedding, _ = extract_face_embedding(frame, selected_box)
            known_faces.append((selected_embedding, selected_box))

    if not tracking:
        detected_boxes = detect_faces_yolo(frame)
        for (x, y, w, h) in detected_boxes:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Re-ID Tracker", display)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
