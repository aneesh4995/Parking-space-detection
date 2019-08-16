import os
import sys
import numpy as np
import cv2
ROOT_DIR = os.path.abspath("/home/aneesh/Desktop/project/Mask_RCNN")

sys.path.append(ROOT_DIR)
from mrcnn import config
from mrcnn import utils
from mrcnn.model import MaskRCNN
from pathlib import Path

class MaskRCNNConfig(config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6
def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)

    return np.array(car_boxes)


MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
VIDEO_SOURCE = "highway.avi"

model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())
model.load_weights(COCO_MODEL_PATH, by_name=True)
parked_car_boxes = None

video_capture = cv2.VideoCapture(VIDEO_SOURCE)
while(True):
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    #print cap.isOpened(), ret
    if frame is not None:
        # Display the resulting frame
        cv2.imshow('frame',frame)
    else:
        print ("Frame is None")
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up everything when finished
rgb_image = frame[:, :, ::-1]
results = model.detect([rgb_image], verbose=0)
r = results[0]
car_boxes = get_car_boxes(r['rois'], r['class_ids'])

print("Cars found in frame of video:")

    # Draw each box on the frame
for box in car_boxes:
    print("Car: ", box)

    y1, x1, y2, x2 = box

        # Draw the box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Show the frame of video on the screen
cv2.imshow('Video', frame)
video_capture.release()
cv2.destroyAllWindows()