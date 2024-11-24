import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load model
model_path = r"C:\Users\Adil Khan\Desktop\AI Hackathon\Ai 3\fruit_ripeness_model.h5"
model = load_model(model_path)
labels = ["Unripe", "Ripe", "Overripe"]  # Adjust this to your actual class names

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (128, 128))  # Match input size
    normalized_frame = resized_frame / 255.0      # Normalize pixel values
    input_frame = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(input_frame)
    predicted_label = labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Display predictions on the frame
    cv2.putText(frame, f"{predicted_label} ({confidence:.2f})", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Fruit Ripeness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
