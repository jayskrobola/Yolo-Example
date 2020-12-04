import numpy as np
import argparse
import time
from cv2 import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.5, help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained with
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize list of colors for each label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

#derive the paths to the YOLO weights and model config
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load image and grab spatial dimensions
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

# determine only the output layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct blob from input image then perform a forward pass of YOLO
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

print("[INFO] YOLO took {:.6f} seconds".format(end-start))

#initialize list of bounding boxes, confidences, and class IDs
boxes = []
confidences = []
classIDs = []

# loop over each of the layer outputs and each of its detections
for output in layerOutputs:
    for detection in output:
        # extract classID and confidence
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        # filter out week predictions
        if confidence > args["confidence"]:
            # scale the bounding box coordinates back relative to the size of the image
            # keeping in mind that YOLO actually returns the center (x, y)-coordinates
            # of the bounding box followed by the boxes' width and height
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")

            # use the center (x, y)-coordinates to derive the top left corner
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            #update our lists
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# non-maxima suppression
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

# DRAW
# ensure at least one detection exists
if len(idxs) > 0:
    for i in idxs.flatten():
        # extract coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        #draw bounding box and label
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x,y), (x+w, y+h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# show output
cv2.imshow("Image", image)
cv2.waitKey(0)
