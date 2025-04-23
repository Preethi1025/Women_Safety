from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv
import vonage
import geocoder
from datetime import datetime
import mediapipe as mp
import time

# Load the gender detection model
model = load_model('gender_detection.h5')

# Initialize Vonage client for sending SMS
client = vonage.Client(key="a25ea934", secret="h3Bdd91DacVqTbay")
sms = vonage.Sms(client)

# Open the webcam
webcam = cv2.VideoCapture(0)

# Define the classes
classes = ['man', 'woman']

# Get the geographical location of the camera
location = geocoder.ip('me')
latitude = location.latlng[0]
longitude = location.latlng[1]

# Thresholds for multiple men and night hours
threshold_men = 3
night_hours = (5, 12)  # 6 PM to 6 AM

# Initialize MediaPipe Hand module for gesture recognition
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Functions for gesture recognition
def is_thumb_closed(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    return thumb_tip.x < thumb_ip.x < thumb_mcp.x

def are_fingers_extended(hand_landmarks):
    extended_fingers = 0
    finger_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                   mp_hands.HandLandmark.RING_FINGER_TIP, 
                   mp_hands.HandLandmark.PINKY_TIP]
    
    finger_pips = [mp_hands.HandLandmark.INDEX_FINGER_PIP, 
                   mp_hands.HandLandmark.MIDDLE_FINGER_PIP, 
                   mp_hands.HandLandmark.RING_FINGER_PIP, 
                   mp_hands.HandLandmark.PINKY_PIP]
    
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            extended_fingers += 1
    
    return extended_fingers == 4

def is_fist_closed(hand_landmarks):
    folded_fingers = 0
    finger_tips = [mp_hands.HandLandmark.THUMB_TIP, 
                   mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                   mp_hands.HandLandmark.RING_FINGER_TIP, 
                   mp_hands.HandLandmark.PINKY_TIP]
    
    finger_pips = [mp_hands.HandLandmark.THUMB_IP, 
                   mp_hands.HandLandmark.INDEX_FINGER_PIP, 
                   mp_hands.HandLandmark.MIDDLE_FINGER_PIP, 
                   mp_hands.HandLandmark.RING_FINGER_PIP, 
                   mp_hands.HandLandmark.PINKY_PIP]
    
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
            folded_fingers += 1
    
    return folded_fingers == 5

# Variable to track if the alert message has been sent
alert_sent = False

# Loop through the frames captured from the webcam
while webcam.isOpened():

    # Read a frame from the webcam
    status, frame = webcam.read()

    # Apply face detection
    faces, confidence = cv.detect_face(frame)

    # Initialize counters for men and women in this frame
    men_count = 0
    women_count = 0

    # Loop through the detected faces
    for idx, f in enumerate(faces):
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        # Skip small face detections
        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        # Preprocess the face for the gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Apply gender detection on the face
        conf = model.predict(face_crop)[0]

        # Get the label with the highest confidence
        idx = np.argmax(conf)
        label = classes[idx]

        # Update the counter based on the prediction
        if label == 'man':
            men_count += 1
        else:
            women_count += 1

        # Prepare the label with confidence
        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        # Set the position for the label on the image
        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # Write the label and confidence above the face rectangle
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # Display the counts of men and women on the frame
    count_label = f"Men: {men_count}, Women: {women_count}"
    cv2.putText(frame, count_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 0, 0), 2)

    # Hand gesture detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    gesture_detected = False
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if is_thumb_closed(hand_landmarks) and are_fingers_extended(hand_landmarks):
                gesture_detected = True
            elif is_fist_closed(hand_landmarks):
                gesture_detected = True

    # Get the current hour
    current_hour = datetime.now().hour

    # Nighttime detection condition
    is_night = (current_hour >= night_hours[0] or current_hour < night_hours[1])

    # Check alert conditions:
    alert_reason = None

    # Case 1: Lone woman detected at night
    if women_count == 1 and men_count == 0 and is_night:
        alert_reason = "Lone Woman Detected at Night"
    
    # Case 2: Woman surrounded by many men
    elif women_count == 1 and men_count > threshold_men:
        alert_reason = "Woman Surrounded by Many Men"

    # Case 3: Woman showing help gestures
    elif women_count == 1 and gesture_detected:
        alert_reason = "Woman Showing Help Gesture"

    # If any of the alert conditions is met, trigger the alert
    if alert_reason and not alert_sent:
        cv2.putText(frame, f"ALERT: {alert_reason}!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 3)
        
        # Send SMS alert
        responseData = sms.send_message(
            {
                "from": "Vonage APIs",
                "to": "+917418592583",
                "text": f"Alert:{alert_reason}. Location: Lat {latitude}, Long {longitude}. View on map: https://www.google.com/maps?q={latitude},{longitude}",
            }
        )

        if responseData["messages"][0]["status"] == "0":
            print("Message sent successfully.")
            alert_sent = True  # Prevent further alerts
        else:
            print(f"Message failed with error: {responseData['messages'][0]['error-text']}")

    # Display the output frame
    cv2.imshow("Gender Detection, Gesture Recognition, and Alert System", frame)

    # Press "Q" to stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam resources and close the display window
webcam.release()
cv2.destroyAllWindows()
