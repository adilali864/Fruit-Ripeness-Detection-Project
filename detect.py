from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys
import os

app = Flask(__name__)

# Model paths - update these with your actual paths
MODEL_DIR = r"D:\AI Hackathon\ALML MultiFruit Ripeness detection Project\model_outputs"
model_path = os.path.join(MODEL_DIR, "final_model.keras")
yolo_weights = os.path.join(MODEL_DIR, "yolov3.weights")
yolo_cfg = os.path.join(MODEL_DIR, "yolov3.cfg")
coco_names = os.path.join(MODEL_DIR, "coco.names")

# Load ripeness detection model
print("Loading ripeness detection model...")
model = load_model(model_path)
labels = ["Unripe", "Ripe", "Overripe"]

# Initialize YOLO for object detection
print("Loading YOLO model...")
try:
    if not all(os.path.exists(f) for f in [yolo_weights, yolo_cfg, coco_names]):
        raise FileNotFoundError("YOLO files not found. Please download required files.")
        
    yolo_net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
    with open(coco_names, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = yolo_net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
except Exception as e:
    print(f"Error loading YOLO model: {str(e)}")
    print("\nPlease download the required YOLO files:")
    print("1. yolov3.weights (237MB) - from: https://pjreddie.com/media/files/yolov3.weights")
    print("2. yolov3.cfg - from: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg")
    print("3. coco.names - from: https://github.com/pjreddie/darknet/blob/master/data/coco.names")
    print("\nPlace these files in your project directory:", MODEL_DIR)
    sys.exit(1)

# Initialize video capture
print("Initializing video capture...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture device")
    sys.exit(1)

def detect_fruits(frame):
    """Detect only bananas and apples in the frame using YOLO."""
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(output_layers)

    fruit_boxes = []
    fruit_confidences = []
    fruit_class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5 and classes[class_id] in ['banana', 'apple']:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                fruit_boxes.append([x, y, w, h])
                fruit_confidences.append(float(confidence))
                fruit_class_ids.append(class_id)

    return fruit_boxes, fruit_confidences, fruit_class_ids

def analyze_ripeness(frame, box):
    """Analyze ripeness of detected fruit."""
    try:
        x, y, w, h = box
        # Ensure coordinates are within frame boundaries
        x, y = max(0, x), max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        fruit_roi = frame[y:y+h, x:x+w]
        if fruit_roi.size == 0:
            return "Unknown", 0
        
        resized_roi = cv2.resize(fruit_roi, (128, 128))
        normalized_roi = resized_roi / 255.0
        input_roi = np.expand_dims(normalized_roi, axis=0)
        
        predictions = model.predict(input_roi, verbose=0)  # Disable prediction verbosity
        predicted_label = labels[np.argmax(predictions)]
        confidence = np.max(predictions)
        
        return predicted_label, confidence
    except Exception as e:
        print(f"Error in analyze_ripeness: {str(e)}")
        return "Error", 0

def generate_frames():
    """Generator function to yield video frames with predictions."""
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            fruit_boxes, fruit_confidences, fruit_class_ids = detect_fruits(frame)
            indices = cv2.dnn.NMSBoxes(fruit_boxes, fruit_confidences, 0.5, 0.4)

            if len(indices) > 0:
                for i in indices.flatten():
                    box = fruit_boxes[i]
                    x, y, w, h = box
                    fruit_type = classes[fruit_class_ids[i]]
                    
                    ripeness, confidence = analyze_ripeness(frame, box)
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"{fruit_type} - {ripeness} ({confidence:.2f})"
                    cv2.putText(frame, label, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   
        except Exception as e:
            print(f"Error in generate_frames: {str(e)}")
            continue

@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route to provide the video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting Flask application...")
    sys.stdout.reconfigure(encoding='utf-8')
    app.run(host='0.0.0.0', port=5000, debug=True)