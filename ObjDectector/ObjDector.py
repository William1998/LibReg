import cv2 as cv
import numpy as np

class ObjDector():
    def __init__(self, modelPath,cfgPath,pixel):
        self.confThreshold = 0.2  # Confidence threshold
        self.nmsThreshold = 0.3  # Non-maximum suppression threshold
        self.inpWidth = pixel  # Width of network's input image
        self.inpHeight = pixel  # Height of network's input image

        # Load names of classes
        self.classesFile = "./pre-trained-model/coco.names"
        self.classes = None
        with open(self.classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        # Give the configuration and weight files for the model and load the network using them.
        modelConfiguration = cfgPath
        modelWeights = modelPath

        self.net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        print("Initialized")


    def getOutputsNames(self,net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def getCoord(self,frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        tempItems = []
        tempConfidences = []
        tempBoxes = []
        tempCata=[]
        for out in outs:
            for detection in out:

                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)

                    coord = detection[:4]
                    label = None
                    if self.classes:
                        assert (classId < len(self.classes))
                        label = '%s:%s' % (self.classes[classId], label)
                    catainfo = [classId,label]

                    tempItems.append(coord)
                    tempCata.append(catainfo)
                    tempConfidences.append(float(confidence))
                    tempBoxes.append([left, top, width, height])

        indices = cv.dnn.NMSBoxes(tempBoxes, tempConfidences, self.confThreshold, self.nmsThreshold)
        items = []
        confidences = []
        boxes = []
        cata = []
        for i in indices:

            i=i[0]
            items.append(tempItems[i])
            confidences.append(tempConfidences[i])
            boxes.append(tempBoxes[i])
            cata.append(tempCata[i])

        return items,cata,confidences,boxes

    def drawBox(self,items,cata,confidences,boxes,frame):
        for i in range(len(items)):

            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            classId =cata[i][0]
            conf = confidences[i]
            right = left + width
            bottom = top + height

            # Draw a bounding box.
            cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

            label = '%.2f' % conf

            # Get the label for the class name and its confidence
            label = '%s:%s' % (cata[i][1], label)

            # Display the label at the top of the bounding box
            labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])),
                         (left + round(1.5 * labelSize[0]), top + baseLine),
                         (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

        # Remove the bounding boxes with low confidence
        t, _ = self.net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    def detect(self,frame):
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (self.inpWidth, self.inpHeight), [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.getOutputsNames(self.net))

        return self.getCoord(frame,outs)

