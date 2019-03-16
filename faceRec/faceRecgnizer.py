import os
import cv2 as cv
import numpy as np
class FaceRecgnizer():
    def __init__(self):
        self.modelPath = "faceModel/faceModel.XML"
        self.model = cv.face.LBPHFaceRecognizer_create()
        if os.path.exists(self.modelPath):
            self.model.read(self.modelPath)
            print("success read model")
        else:
            print("No model, train initial one")
            faces, labels = self.prepare_training_data("initialFaces/")
            self.model.train(faces, np.array(labels))
            self.model.write("faceModel/faceModel.XML")
            print("initial model train finish")
    def predict(self,path):
        image = cv.imread(path)

        # detect face
        face, rect = self.detect_face(image)
        label,confidence = self.model.predict(face)
        print(label, confidence)
    def detect_face(self,img):
        # convert the test image to gray image as opencv face detector expects gray images
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # load OpenCV face detector, I am using LBP which is fast
        # there is also a more accurate but slow Haar classifier
        face_cascade = cv.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

        # let's detect multiscale (some images may be closer to camera than others) images
        # result is a list of faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

        # if no faces are detected then return original img
        if (len(faces) == 0):
            return None, None

        # under the assumption that there will be only one face,
        # extract the face area
        (x, y, w, h) = faces[0]

        # return only the face part of the image
        return gray[y:y + w, x:x + h], faces[0]

    def prepare_training_data(self,data_folder_path):

        # ------STEP-1--------
        # get the directories (one directory for each subject) in data folder
        dirs = os.listdir(data_folder_path)

        # list to hold all subject faces
        faces = []
        # list to hold labels for all subjects
        labels = []

        # let's go through each directory and read images within it
        for dir_name in dirs:

            # our subject directories start with letter 's' so
            # ignore any non-relevant directories if any
            if not dir_name.startswith("s"):
                continue;

            # ------STEP-2--------
            # extract label number of subject from dir_name
            # format of dir name = slabel
            # , so removing letter 's' from dir_name will give us label
            label = int(dir_name.replace("s", ""))

            # build path of directory containin images for current subject subject
            # sample subject_dir_path = "training-data/s1"
            subject_dir_path = data_folder_path + "/" + dir_name

            # get the images names that are inside the given subject directory
            subject_images_names = os.listdir(subject_dir_path)

            # ------STEP-3--------
            # go through each image name, read image, 
            # detect face and add face to list of faces
            for image_name in subject_images_names:
                print("Process: ",image_name)

                # ignore system files like .DS_Store
                if image_name.startswith("."):
                    continue;

                # build image path
                # sample image path = training-data/s1/1.pgm
                image_path = subject_dir_path + "/" + image_name

                # read image
                image = cv.imread(image_path)

                # detect face
                face, rect = self.detect_face(image)

                # ------STEP-4--------
                # for the purpose of this tutorial
                # we will ignore faces that are not detected
                if face is not None:
                    # add face to list of faces
                    faces.append(face)
                    # add label for this face
                    labels.append(label)


        return faces, labels


