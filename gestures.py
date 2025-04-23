# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 21:49:16 2024

@author: preet
"""

import cv2
import mediapipe as mp

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to determine if thumb is closed (for the first "help" gesture)
def is_thumb_closed(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

    # Check if thumb tip is near the palm (lower x value than thumb MCP and IP)
    return thumb_tip.x < thumb_ip.x < thumb_mcp.x

# Function to check if all fingers (except thumb) are extended (for the first "help" gesture)
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
    
    # Check if each finger tip is above its PIP joint (indicating the finger is extended)
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            extended_fingers += 1
    
    return extended_fingers == 4  # All four fingers extended

# Function to check if all fingers are folded (for the second "fist" gesture)
def is_fist_closed(hand_landmarks):
    # We check if all fingers (tips) are below their respective PIP joints
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
    
    return folded_fingers == 5  # All fingers folded

def detect_with_webcam():
    cap = cv2.VideoCapture(0)  # Open webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform hand detection
        results = hands.process(rgb_frame)
        
        gesture_detected = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks and connections
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Check for first gesture (help gesture)
                if is_thumb_closed(hand_landmarks) and are_fingers_extended(hand_landmarks):
                    label = "Help Gesture Detected"
                    gesture_detected = True
                # Check for second gesture (fist gesture)
                elif is_fist_closed(hand_landmarks):
                    label = "Fist Gesture Detected"
                    gesture_detected = True
                else:
                    label = "No Gesture Detected"
        
        if not gesture_detected:
            label = "No Gesture Detected"
        
        # Display label
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display the result
        cv2.imshow('Hand Gesture Recognition', frame)
        
        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the hand detection
detect_with_webcam()
