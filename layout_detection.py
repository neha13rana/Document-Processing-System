import cv2
from yolov5 import YOLO  # Replace with the YOLO model used in your repo

def detect_layout(image_path, model_path):
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    results = model.predict(image)
    return results