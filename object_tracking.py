# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
#from pyimagesearch.centroidtracker import CentroidTracker
#from imutils.video import VideoStream
import numpy as np
from collections import OrderedDict

# import cv2

## construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--prototxt", required=True,
#	help="path to Caffe 'deploy' prototxt file")
#ap.add_argument("-m", "--model", required=True,
#	help="path to Caffe pre-trained model")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
	# help="minimum probability to filter weak detections")
#args = vars(ap.parse_args())

# initialize our centroid tracker and frame dimensions
#ct = CentroidTracker()
# (H, W) = (None, None)

# load our serialized model from disk
# print("[INFO] loading model...")
# net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the camera sensor to warmup
# print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
# time.sleep(2.0)

# read the next frame from the video stream and resize it
# frame = vs.read()
# frame = imutils.resize(frame, width=400)

# if the frame dimensions are None, grab them
# if W is None or H is None:
	# (H, W) = frame.shape[:2]

# construct a blob from the frame, pass it through the network,
# obtain our output predictions, and initialize the list of
# bounding box rectangles
# blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
# 	(104.0, 177.0, 123.0))
# net.setInput(blob)


def track_object(ct, objects, items, cata, size):
	detections = items
	rects = []
	H = size[0]
	W = size[1]

	# loop over the detections
	for i in range(0, len(detections)):
		# filter out weak detections by ensuring the predicted
		# probability is greater than a minimum threshold
		# if detections[0, 0, i, 2] > confidence:
			# compute the (x, y)-coordinates of the bounding box for
			# the object, then update the bounding box rectangles list
		box = detections[i] * np.array([W, H, W, H])
		rects.append(box.astype("int"))

			# draw a bounding box surrounding the object so we can
			# visualize it
		# (startX, startY, endX, endY) = box.astype("int")
		# cv2.rectangle(None, (startX, startY), (endX, endY),
			# (0, 255, 0), 2)

	# update our centroid tracker using the computed set of bounding
	# box rectangles
	# objects = ct.update(rects)

	count = 0
	for key, value in ct.update(rects).items():
		# update the coordinates of an object if it already exists
		if count >= len(detections):
			break
		if key in objects:
			objects[key][0] = detections[count]
		else:
			objects[key] = [detections[count], cata[count][0], cata[count][1], 0, None, None]
		count += 1

	# return objects
