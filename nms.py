import numpy as np


def non_max_suppression(boxes, scores, iou_threshold):
  """Performs non-maximum suppression on a set of bounding boxes.

  Args:
    boxes: A NumPy array of bounding boxes.
    scores: A NumPy array of scores for the bounding boxes.
    iou_threshold: The intersection-over-union threshold.

  Returns:
    A list of bounding boxes that have been suppressed.
  """

  # Sort the bounding boxes by score.
  sorted_indices = np.argsort(scores)[::-1]

  # Keep track of the suppressed bounding boxes.
  suppressed_boxes = []

  # Iterate over the sorted bounding boxes.
  for i in sorted_indices:
    # Check if the current bounding box is suppressed.
    if iou_threshold > 0:
      for j in suppressed_boxes:
        if iou(boxes[i], boxes[j]) > iou_threshold:
          break
    else:
      suppressed_boxes.append(i)

  # Return the suppressed bounding boxes.
  return boxes[suppressed_boxes]


def iou(box1, box2):
  """Calculates the intersection-over-union of two bounding boxes.

  Args:
    box1: A NumPy array of the coordinates of the first bounding box.
    box2: A NumPy array of the coordinates of the second bounding box.

  Returns:
    The intersection-over-union of the two bounding boxes.
  """

  # Calculate the intersection of the two bounding boxes.
  intersection = np.minimum(box1[2], box2[2]) - np.maximum(box1[0], box2[0])
  intersection = intersection * \
      np.minimum(box1[3], box2[3]) - np.maximum(box1[1], box2[1])

  # Calculate the union of the two bounding boxes.
  union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
      (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection

  # Return the intersection-over-union.
  return intersection / union
