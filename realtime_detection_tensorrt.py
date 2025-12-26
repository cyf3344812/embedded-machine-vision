import cv2
import numpy as np
np.bool = np.bool_  # ? ?? PyCUDA ?? np.bool ?????(monkey patch)
import time
import sys
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch

# Add local YOLOv5 directory to system path (Jetson path)
yolov5_path = os.path.abspath('/home/jetson/Miniproject/yolov5')
sys.path.append(yolov5_path)

# Import YOLOv5 utilities for post-processing
from utils.general import non_max_suppression, scale_boxes

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Function to load TensorRT engine
def load_engine(engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(engine_data)

# Function to allocate buffers for TensorRT inference
def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream

# Function to perform TensorRT inference
def infer(engine, context, inputs, outputs, bindings, stream, input_shape):
    inputs[0]['host'] = np.ascontiguousarray(input_shape, dtype=np.float32)
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
    context.execute_async_v2(bindings, stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()
    return outputs[0]['host']

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

# Load TensorRT engine
engine_path = '/home/jetson/Miniproject/best2.trt'
if not os.path.exists(engine_path):
    print(f"Error: TensorRT engine file {engine_path} does not exist")
    exit()
engine = load_engine(engine_path)
context = engine.create_execution_context()
inputs, outputs, bindings, stream = allocate_buffers(engine)

# Define allowed labels and class names
names = ['person', 'car', 'bike']  # Adjust based on your model's actual class names
allowed_labels = ['person', 'car', 'bike']

# Preprocess image function (modified for TensorRT)
def preprocess_image(img, img_size=640):
    img0 = img.copy()
    img = cv2.resize(img, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB and rearrange to CHW
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0
    return img, img0

# FPS calculation
fps_list = []
smooth_window = 10
last_time = time.time()

# Define fixed color list (Blue, Green, Red)
fixed_colors = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0)
]

# Set parameters
conf_thres = 0.25
iou_thres = 0.45
img_size = 640

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
    img, img0 = preprocess_image(frame, img_size=img_size)

    # Perform TensorRT inference
    pred = infer(engine, context, inputs, outputs, bindings, stream, img[np.newaxis, :])
    pred = torch.from_numpy(pred.reshape(1, -1, 6))
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    # Draw detection boxes
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes((img_size, img_size), det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                label_name = names[int(cls)]
                if label_name in allowed_labels:
                    label = f'{label_name} {conf:.2f}'
                    color = fixed_colors[int(cls) % len(fixed_colors)]
                    cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
                    cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display FPS
    cv2.putText(img0, f'FPS: {smooth_fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame with bounding boxes
    cv2.imshow('YOLOv5 Real-Time Detection', img0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
time.sleep(1)
cv2.destroyAllWindows()
