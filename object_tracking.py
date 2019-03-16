# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
#from pyimagesearch.centroidtracker import CentroidTracker
#from imutils.video import VideoStream
import numpy as np
import cv2 as cv
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

def track_object(ct, objects, items, cata, size,frame, faceRec,x,strangerList,nameCount,faceList):
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
	test = ct.update(rects).items()

	for key, value in test:
		# update the coordinates of an object if it already exists
		
		if ct.disappeared[key] > 0:
			try:
				del objects[key]
			except:
				pass
			continue

		if key in objects.keys():
			objects[key][0] = detections[count]
			objects[key][1] = cata[count][0]
			objects[key][2] = cata[count][1]
			
		        		 
			if objects[key][2] == 'person' and objects[key][3] == 0:
				print("Perform face detection: ", objects[key][2])
				img = x.cropImage(frame,objects[key][0])
				try:
					print("try rec")
					label, confidence = faceRec.predict(img)
					objects[key][3] = 1
					if confidence >= 0.6:
						objects[key][4] = label
						faceList[label] = faceRec.detect_face(img)
						print("Known person: ",label)
					else:
						print("Unknown person")
					print("recognize done")
				except Exception as e:
					print(e)
					print("failed")
					pass
			elif objects[key][2] != "person" and objects[key][0][1] > 0.5 and objects[key][5] is None:
				print("some one put thing down!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
				personKey = nearstPerson(key,objects)

				if personKey is not None and objects[personKey][4] is None and personKey in strangerList and len(strangerList[personKey])>0:
					print("Find Nearst Person")
					nameCount += 1
					try:
						faceRec.updateModel(strangerList[personKey],nameCount)
						with open("count") as f:
							f.write(nameCount)
						objects[key][5] = nameCount
						print("train success~~~~~~`")
					except Exception as e:
						print(e)
						print("train failxxxxxxxx`")
						pass
				elif personKey is not None:
					objects[key][5] = objects[personKey][4]
				objects[key][6] = 0

			elif objects[key][2] != "person" and objects[key][0][1] < 0.5 and objects[key][5] is not None:
				print("some one take thing away00000000000000000000000000000")
				personKey = nearstPerson(key, objects)
				if personKey is not None:
					if objects[personKey][4] == objects[key][5]:
						objects[key][6] = 1
						#push to databse that object taken by master
					else:
						objects[key][6] = 2
						if objects[key][5] is None:
							nameCount += 1

							try:
								faceRec.updateModel(strangerList[personKey], nameCount)
								with open("count") as f:
									f.write(nameCount)
								objects[key][5] = nameCount
								print("train success~~~~~~`")
							except Exception as e:
								print(e)
								print("train failxxxxxxxx`")
								pass
							print("Taken by stranger")
						else:
							print("Taken by master")
					        # push to databse that object taken by master

			
			if objects[key][2] == "person" and objects[key][4] is None:
				print("collect stranger")
				if key not in strangerList:
					strangerList[key] = []
				img = x.cropImage(frame, objects[key][0])
				strangerList[key].append(img)
				if len(strangerList) > 10:
					strangerList[key].pop(0)

		else:
			try:
				objects[key] = [detections[count], cata[count][0], cata[count][1], 0, None, None, 0]
			except Exception as e:
				print(e)

		count += 1
	

	# return objects
def nearstPerson(objectKey,objects):
	personKey = None
	objectCoord = objects[objectKey][0]
	currdistance = 1000
	for key in objects:
		if key != objectKey and objects[key][2] == 'person':
			personCoord = objects[key][0]
			distance = (objectCoord[0]-personCoord[0])**2 + (objectCoord[1]-personCoord[1])**2
			if distance < currdistance:
				currdistance = distance
				personKey = key
	return personKey
