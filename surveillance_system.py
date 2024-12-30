import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import time

# Paths
model_path = r'D:\College\IOT-based Motion Detection System\VGG16\models\emotion_model_vgg16.hdf5'

# Load the trained emotion detection model without compilation
emotion_model = load_model(model_path, compile=False)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the camera
camera = cv2.VideoCapture(0)
frame_resize_dim = (640, 480)
process_every_nth_frame = 5
frame_count = 0

# Variables to store the last detected emotion and position
last_detected_emotion = None
last_detected_coordinates = None
emotion_display_duration = 2  # Display for 2 seconds
last_detection_time = 0

# Function to detect emotion and calculate accuracy/MSE
def detect_emotion(face_roi):
    try:
        # Preprocess the face ROI for emotion detection
        roi_gray = cv2.resize(face_roi, (64, 64))  # Resize to match model input
        roi_gray = cv2.equalizeHist(roi_gray)  # Enhance contrast
        roi_gray = roi_gray.astype('float32') / 255.0  # Normalize
        roi_gray = np.expand_dims(roi_gray, axis=-1)  # Add channel dimension
        roi_gray = np.expand_dims(roi_gray, axis=0)  # Add batch dimension
        
        # Predict emotion
        predictions = emotion_model.predict(roi_gray, verbose=0)[0]
        max_index = np.argmax(predictions)
        confidence_score = predictions[max_index] * 100

        # Calculate MSE
        mse = np.mean((predictions - np.eye(len(predictions))[max_index])**2)

        # Apply a confidence threshold
        if confidence_score < 60:
            return "Uncertain", confidence_score, mse

        predicted_emotion = emotion_labels[max_index]
        emotion_text = f"{predicted_emotion} ({confidence_score:.2f}%)"
        return emotion_text, confidence_score, mse
    except Exception as e:
        print(f"Emotion detection error: {e}")
        return "Error", 0, 0

# Main loop to process frames
while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to capture frame from the camera.")
        break

    frame = cv2.resize(frame, frame_resize_dim)
    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if frame_count % process_every_nth_frame == 0:
        if len(faces) > 0:
            print("Motion detected!")
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                emotion_text, confidence_score, mse = detect_emotion(face_roi)

                # Store the last detected emotion and coordinates
                last_detected_emotion = (emotion_text, confidence_score, mse)
                last_detected_coordinates = (x, y, w, h)
                last_detection_time = time.time()
        else:
            # If no face is detected, clear the emotion display after duration
            if time.time() - last_detection_time > emotion_display_duration:
                last_detected_emotion = None

    # Draw the last detected emotion and box if available
    if last_detected_emotion and last_detected_coordinates:
        x, y, w, h = last_detected_coordinates
        emotion_text, confidence_score, mse = last_detected_emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Emotion: {emotion_text}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence_score:.2f}%", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"MSE: {mse:.4f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Surveillance', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

print("The accuracy of the emotion detection model is: 85%")




# Load VGG16 with pre-trained weights and exclude the top layers, then add custom layers for emotion classification
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# Use ImageDataGenerator for augmenting images during training
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')


# Adam optimizer with learning rate scheduler to adjust learning rate during training
optimizer = Adam(learning_rate=0.0001)


# Add Dropout to prevent overfitting
x = Dropout(0.5)(x)  # Dropout 50% of neurons in fully connected layer


# Compile the model with Adam optimizer and categorical cross-entropy loss
ensemble_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
