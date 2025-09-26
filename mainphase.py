import cv2, os
from detector import YOLOFaceDetector
from embedder import InsightFaceEmbedder
from utils import save_crop

VIDEO_SOURCE = "foot1.mp4"   
YOLO_MODEL = "yolov8n.pt"
MIN_DETECTION_CONF = 0.3
INSIGHTFACE_MODEL = "buffalo_l"   
CROP_SIZE = (112, 112)
LOGS_DIR = "logs"


cap = cv2.VideoCapture(VIDEO_SOURCE)
detector = YOLOFaceDetector(YOLO_MODEL, MIN_DETECTION_CONF)
embedder = InsightFaceEmbedder(INSIGHTFACE_MODEL, CROP_SIZE)

frame_idx = 0
os.makedirs(LOGS_DIR, exist_ok=True)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    boxes = detector.detect(frame)
    for i, bbox in enumerate(boxes):
        crop = detector.crop_face(frame, bbox, size=CROP_SIZE)
        if crop is None:
            continue
        emb = embedder.get_embedding(crop)
        if emb is None:
            continue
        img_path = save_crop(crop, LOGS_DIR, prefix=f"f{frame_idx}_{i}")
        print(f"[Frame {frame_idx}] Saved {img_path}, emb_dim={len(emb)}")

cap.release()
