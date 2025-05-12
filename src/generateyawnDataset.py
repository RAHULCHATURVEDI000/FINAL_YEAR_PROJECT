import cv2
import dlib
import numpy as np
import os
import time
import sys

# Retrieve starting counts from command-line arguments
if len(sys.argv) != 3:
    print("Usage: python script.py <yawn_start> <no_yawn_start>")
    sys.exit(1)

try:
    yawn_count = int(sys.argv[1])
    no_yawn_count = int(sys.argv[2])
except ValueError:
    print("Both arguments must be integers.")
    sys.exit(1)

# Set absolute paths for saving images
yawn_path = '/home/dhruv/Driver_Drowsiness_detection(CNN multithreading)/customYawndata/yawn'
no_yawn_path = '/home/dhruv/Driver_Drowsiness_detection(CNN multithreading)/customYawndata/no_yawn'
os.makedirs(yawn_path, exist_ok=True)
os.makedirs(no_yawn_path, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
else:
    print("Webcam opened successfully.")

# Initialize dlib's face detector (HOG-based) and create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor_path = "/home/dhruv/CollegeProject/Trial3/customYawndata/shape_predictor_68_face_landmarks.dat"
print("Loading shape predictor from:", predictor_path)
try:
    predictor = dlib.shape_predictor(predictor_path)
    print("Shape predictor loaded successfully.")
except RuntimeError as e:
    print("Error loading shape predictor:", e)
    exit()

last_capture_time = 0
capture_interval = 0.1  # Capture an image every 100ms

def extract_mouth(image, landmarks):
    # The mouth region corresponds to points 48-67 in the 68-point model
    mouth_points = []
    for i in range(48, 68):
        mouth_points.append((landmarks.part(i).x, landmarks.part(i).y))
    # Create a bounding rectangle around the mouth landmarks
    mouth_points = np.array(mouth_points)
    x, y, w, h = cv2.boundingRect(mouth_points)
    mouth_roi = image[y:y+h, x:x+w]
    return mouth_roi

def capture_image(label, gray):
    global yawn_count, no_yawn_count, last_capture_time, capture_interval
    
    current_time = time.time()
    if current_time - last_capture_time < capture_interval:
        return
    last_capture_time = current_time
    
    # Detect faces using dlib
    faces = detector(gray)
    print("Faces detected:", len(faces))
    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray, face)
        # Extract mouth region
        mouth_roi = extract_mouth(gray, landmarks)
        
        if mouth_roi.size == 0:
            continue
        
        if label == 'y':
            image_path = f'{yawn_path}/{yawn_count}.jpg'
            cv2.imwrite(image_path, mouth_roi)
            print(f"Saved yawn image: {image_path}")
            yawn_count += 1
        elif label == 'n':
            image_path = f'{no_yawn_path}/{no_yawn_count}.jpg'
            cv2.imwrite(image_path, mouth_roi)
            print(f"Saved no_yawn image: {image_path}")
            no_yawn_count += 1

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('y'):
        capture_image('y', gray)
    elif key == ord('n'):
        capture_image('n', gray)
    elif key == ord('q'):
        break

    # Optionally, display the frame with a rectangle around detected faces
    faces = detector(gray)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('Frame', frame)

cap.release()
cv2.destroyAllWindows()
