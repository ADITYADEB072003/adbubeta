import cv2

# Path to the pre-trained model file
model_path = 'haarcascade_frontalface_default.xml'

# Load the model
cascade = cv2.CascadeClassifier(model_path)

# Initialize camera
cam = cv2.VideoCapture(0)

# Counter for saving images
count = 0

while True:
    # Read the frame from the camera
    _, img = cam.read()
    
    # Convert color image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = cascade.detectMultiScale(gray_img)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Crop the face region
        face_region = img[y:y+h, x:x+w]
        
        # Save the cropped face region
        cv2.imwrite(f'recognized_face_{count}.jpg', face_region)
        count += 1

    # Display the image with rectangles around faces
    cv2.imshow("Face Detection", img)

    key = cv2.waitKey(1)

    if key == 81 or key == 113:  # Exit when 'q' or 'Q' is pressed
        break

# Release the camera
cam.release()
cv2.destroyAllWindows()
