import cv2 as cv
import numpy as np

class ObjDector():
    def __init__(self, modelPath,cfgPath,pixel):
        self.confThreshold = 0.4  # Confidence threshold
        self.nmsThreshold = 0.7  # Non-maximum suppression threshold
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
                        label = '%s' % (self.classes[classId])
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

    def drawBox(self,items,frame,faceList):
        for i in items.keys():
            item = items[i]
            if item[7] == 0:
                continue
            coord = item[0]
            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]
            center_x = int(coord[0] * frameWidth)
            center_y = int(coord[1] * frameHeight)
            width = int(coord[2] * frameWidth)
            height = int(coord[3] * frameHeight)
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)

            box = [left, top, width, height]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            right = left + width
            bottom = top + height

            # Draw a bounding box.
            cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)


            # Get the label for the class name and its confidence
            label = '%s:%s' % (item[2],i)
            if item[4] is not None:
                label += ":Face:"+str(item[4])

            if item[5] is not None:
                label += ":Face:"+str(item[5])
            thecolor = (0,0,255)
            if item[6] == 0:
                thecolor = (0, 0, 255)
            elif item[6] == 1:
                thecolor = (34,139,34)
            elif item[6] == 2:
                thecolor = (255,0,0)            
            # Display the label at the top of the bounding box
            labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])),
                         (left + round(1.5 * labelSize[0]), top + baseLine),
                         (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, thecolor, 1)

        # Remove the bounding boxes with low confidence
        t, _ = self.net.getPerfProfile()
        
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5,color = (0,0,255))

    def detect(self,frame):
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (self.inpWidth, self.inpHeight), [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.getOutputsNames(self.net))

        return self.getCoord(frame,outs)

    def cropImage(self, frame, coord):
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        center_x = int(coord[0] * frameWidth)
        center_y = int(coord[1] * frameHeight)
        width = int(coord[2] * frameWidth)
        height = int(coord[3] * frameHeight)
        left = int(center_x - width / 2)
        top = int(center_y - height / 2)


        crop_img = frame[top:top+height,left:left+width]
        return crop_img
