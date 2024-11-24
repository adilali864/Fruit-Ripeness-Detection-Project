import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model and class labels
model = load_model('fruit_ripeness_model.h5')
class_labels = np.load('class_labels.npy', allow_pickle=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    resized_frame = cv2.resize(frame, (128, 128))
    input_frame = np.expand_dims(resized_frame / 255.0, axis=0)

    # Prediction
    prediction = model.predict(input_frame)
    ripeness_stage = class_labels[np.argmax(prediction)]

    # Display result
    cv2.putText(frame, f'Ripeness: {ripeness_stage}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Fruit Ripeness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
