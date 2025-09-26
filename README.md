# katamoran


 
Build an AI-driven unique visitor counter that processes a video stream to detect, track, and recognize faces in real-time. The system should automatically register new faces upon first detection, recognize them in subsequent frames, and track them continuously until they exit the frame. Every entry and exit must be logged with a timestamped image and stored both locally and in a database. The ultimate goal is to accurately count the number of unique visitors in the video stream.




# config.json:

{
  "model": {
    "yolo_model_path": "yolov8n.pt",
    "insightface_model": "arcface_r100_v1"
  },
  "paths": {
    "input_video": "input/video.mp4",
    "output_dir": "outputs/",
    "crops_dir": "outputs/crops/"
  },
  "settings": {
    "confidence_threshold": 0.3,
    "iou_threshold": 0.45,
    "save_crops": true,
    "draw_boxes": true
  },
  "hardware": {
    "use_gpu": true,
    "onnx_provider": "CUDAExecutionProvider"
  }
}

























# This project is a part of a hackathon run by https://katomaran.com 
