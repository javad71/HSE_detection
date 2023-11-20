# import the opencv library and YOLO
import cv2
from ultralytics import YOLO

# define a video capture object with RTSP protocol
video = cv2.VideoCapture("rtsp://username, password@127.0.0.1:554/Streaming/Channels/401")

while True:
    # Capture the video frame
    # by frame
    ret, frame = video.read()

    # load yolo model
    yolo_model = YOLO("./models/best.pt")

    # get result of this frame 
    results = yolo_model(frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
video.release()
# Destroy all the windows
cv2.destroyAllWindows()
