# import the opencv library
import os

import cv2
import numpy as np
import time
from ultralytics import YOLO

# define a video capture object
vid = cv2.VideoCapture(0)

# Load a model
model = YOLO('./models/best.pt')  # load an official model

while True:
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Predict with the model
    results = model.predict(frame, conf=0.5, verbose=False)[0]

    font_scale = 1
    thickness = 1
    labels = open("class.names").read().strip().split("\n")
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
    print('start detection...')

    # Loop over the detections
    for data in results.boxes.data.tolist():
        # Get the bounding box coordinates, confidence, and class id
        xmin, ymin, xmax, ymax, confidence, class_id = data

        # Converting the coordinates and the class id to integers
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        class_id = int(class_id)

        # Draw a bounding box rectangle and label on the image
        color = [int(c) for c in colors[class_id]]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=color, thickness=thickness)
        text = f"{labels[class_id]}: {confidence:.2f}"
        # Calculate text width & height to draw the transparent boxes as background of the text
        (text_width, text_height) = \
            cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
        text_offset_x = xmin
        text_offset_y = ymin - 5
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
        overlay = frame.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
        # Add opacity (transparency to the box)
        img = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        # Now put the text (label: confidence %)
        cv2.putText(img, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

        # Display the resulting frame
        # cv2.imshow('detect...', img)
        time_str = time.strftime("%Y%m%d-%H%M%S")
        # Filename
        filename = './outputs/' + time_str + '-' + labels[class_id] + '.jpg'

        # save detected image
        cv2.imwrite(filename, img)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
