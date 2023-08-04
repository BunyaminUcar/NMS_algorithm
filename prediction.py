import numpy as np
import cv2
from nms import non_max_suppression
import torch

# Load the `.pt` weight file.
model = torch.load("model.pt")

# Read the image.
image = cv2.imread("image.jpg")

# Convert the image to a NumPy array.
image_np = np.array(image)

# Perform the prediction.
prediction = model(image_np)

# Get the bounding boxes.
boxes = prediction["boxes"]

# Get the scores.
scores = prediction["scores"]

# Perform non-maximum suppression.
suppressed_boxes = non_max_suppression(boxes, scores, iou_threshold=0.5)

# Draw the bounding boxes on the image.
for box in suppressed_boxes:
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

# Save the image.
cv2.imwrite("output.jpg", image)
