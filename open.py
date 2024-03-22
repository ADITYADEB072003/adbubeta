import cv2 as cv
import os
model_path = os.path.abspath("haarcascade_frontalface_default.xml")

# Load the model
face_cap = cv.CascadeClassifier(model_path)
# Load the cascade classifier

# Check if the cascade classifier was loaded successfully
if face_cap.empty():
    print("Error: Cascade classifier not loaded!")
    exit()

# Start video capture
video_capture = cv.VideoCapture(0)

# Check if the video capture device is opened successfully
if not video_capture.isOpened():
    print("Error: Video capture device cannot be opened!")
    exit()

while True:
    # Read frame from the video capture device
    ret, video_data = video_capture.read()
    
    # Convert frame to grayscale
    col = cv.cvtColor(video_data, cv.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE
    )
    
    # Draw rectangles around the detected faces
    for (x, y, width, height) in faces:
        cv.rectangle(video_data, (x, y), (x + width, y + height), (0, 255, 0), 2)
    
    # Display the frame with rectangles around faces
    cv.imshow("Face Recognition", video_data)
    
    # Check for key press 'a' to exit the loop
    if cv.waitKey(10) == ord("a"):
        break

# Release the video capture device and close all windows
video_capture.release()
cv.destroyAllWindows()
