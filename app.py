from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('fruit_ripeness_model.h5')
class_labels = np.load('class_labels.npy', allow_pickle=True)

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        resized_frame = cv2.resize(frame, (128, 128))
        input_frame = np.expand_dims(resized_frame / 255.0, axis=0)
        prediction = model.predict(input_frame)
        ripeness_stage = class_labels[np.argmax(prediction)]

        # Overlay prediction
        cv2.putText(frame, f'Ripeness: {ripeness_stage}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
