import cv2
import sys
import config

training_positive="../positive_samples/Person_l/"
#filename="Person_l"
cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)
t=0;
video_capture = cv2.VideoCapture(0)

while (t<10):
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the resulting frame

    if len(faces)==1:
    	t=t+1
	crop_height = int((config.FACE_HEIGHT / float(config.FACE_WIDTH)) * w)
	midy = y + h/2
	y1 = max(0, midy-crop_height/2)
	y2 = min(frame.shape[0]-1, midy+crop_height/2)
	face=frame[y1:y2, x:x+w]
	cv2.imshow('Video', face)
	cv2.imwrite(training_positive+str(x)+".jpg", face)
	

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
