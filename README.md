# 🎯 YOLOv8 Object Detection (OpenCV DNN - C++)

This project is a real-time **object detection system** using YOLOv8 (ONNX format) with OpenCV DNN in C++.

It captures live video from a webcam and detects objects using a pre-trained YOLOv8 model.

---

## 🚀 Technologies Used

- C++
- OpenCV (DNN module)
- YOLOv8 (ONNX model)
- COCO dataset labels
- Webcam (real-time video input)

---

## 📦 Model Files

Make sure you have:

- `yolov8n.onnx` → YOLOv8 model file
- `coco.names` → class labels file

Example classes:
- person
- car
- bicycle
- dog
- etc.

---

## ⚙️ How It Works

1. Webcam captures live video
2. Frame is resized and converted to blob
3. YOLOv8 ONNX model processes the frame
4. Bounding boxes + confidence scores are extracted
5. Non-Maximum Suppression (NMS) filters results
6. Final detections are drawn on frame

---

## ▶️ How to Run

### 1. Install OpenCV (with DNN module)

Make sure OpenCV is installed in Visual Studio.

---

### 2. Add required files

Place these in your project folder.
