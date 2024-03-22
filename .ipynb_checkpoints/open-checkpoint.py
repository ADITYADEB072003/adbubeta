import cv2 as cv
face_cap=cv.CascadeClassifier("haarcascade_frontalface_default.xml")
video_capture = cv.VideoCapture(0)

while True:
    ret, video_data = video_capture.read()
    cv.imshow("Face Recognition", video_data)
    if cv.waitKey(10) == ord("a"):
        break

video_capture.release()
cv.destroyAllWindows()
