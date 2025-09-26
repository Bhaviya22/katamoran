from ultralytics import YOLO
import cv2

class YOLOFaceDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.3):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box[:4]
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return boxes

    def crop_face(self, frame, bbox, size=(112, 112)):
        x1, y1, x2, y2 = bbox
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None
        return cv2.resize(face, size)
