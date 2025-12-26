import cv2
import torch
import numpy as np
import time
import sys

sys.path.append('./yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device

# Define the working GStreamer pipeline
gst_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1, format=NV12 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "appsink"
)

# Open the video capture
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Failed to open camera.")
    exit()

# Load YOLOv5 model
weights = 'best2.pt'  # Path to your trained YOLOv5 weights
device = select_device('0')  # Change to 'cpu' if you are not using GPU
model = attempt_load(weights, device=device)
model.eval()
stride = int(model.stride.max())
names = model.module.names if hasattr(model, 'module') else model.names

# Define allowed labels
allowed_labels = ['person', 'car', 'bike']

# Preprocess image function
def preprocess_image(img, img_size=640):
    img0 = img.copy()
    img = cv2.resize(img, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB and rearrange
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img, img0

# FPS calculation
fps_list = []
smooth_window = 10
last_time = time.time()

# Define fixed color list (Blue, Green, Red)
fixed_colors = [
    (0, 0, 255),   # Blue (BGR format)
    (0, 255, 0),   # Green
    (255, 0, 0)    # Red
]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Compute FPS
    current_time = time.time()
    fps = 1 / (current_time - last_time)
    last_time = current_time

    fps_list.append(fps)
    if len(fps_list) > smooth_window:
        fps_list.pop(0)

    smooth_fps = sum(fps_list) / len(fps_list)

    # Preprocess frame
    img, img0 = preprocess_image(frame, img_size=640)

    # Object detection
    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # Draw detection boxes
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                label_name = names[int(cls)]
                # Only display allowed labels: person, car, bike
                if label_name in allowed_labels:
                    label = f'{label_name} {conf:.2f}'
                    # Select color based on class index (cycling through Red, Green, Blue)
                    color = fixed_colors[int(cls) % len(fixed_colors)]  # Use modulo to cycle colors
                    cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
                    cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display FPS
    cv2.putText(img0, f'FPS: {smooth_fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame with bounding boxes
    cv2.imshow('YOLOv5 Real-Time Detection', img0)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()