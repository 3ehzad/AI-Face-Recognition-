#!/usr/local/bin/python
# coding: utf-8

# Face Recognition with OpenCV

# First we Detect the Faces using OpenCV then Label each Face by OpenCV

# ### Import Required Modules


# - **cv2:** is _OpenCV_ module for Python which we will use for face detection and face recognition.
# - **os:** We will use this Python module to read our training directories and file names.
# - **numpy:** We will use this module to convert Python lists to numpy arrays as OpenCV face recognizers accept numpy arrays.


# import OpenCV module
import cv2
from sklearn.datasets import fetch_lfw_people
# import os module for reading training data directories and paths
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import precision_recall_fscore_support
import lfw
import config
# import numpy to convert python lists to numpy arrays as
# it is needed by OpenCV face recognizers
import numpy as np
#import sys,getopt for passing arguments to script by terminal
import sys,getopt

# ### Training Data

# The more images used in training the better.
# Normally a lot of images are used for training a face recognizer so that it can learn different looks of the same person, for example with glasses, without glasses, laughing, sad, happy, crying, with beard, without beard etc.
#
# All training data is inside _`training-data`_ folder. _`training-data`_ folder contains one folder for each person
#  and **each folder is named with format `sLabel (e.g. s1, s2)` where label is actually the integer label assigned to that person**.
#  For example folder named s1 means that this folder contains images for person 1.
# The directory structure tree for training data is as follows:
#
# training-data
# |-------------- s1
# |               |-- 'picname'.jpg
# |               |-- ...

# |-------------- s2
# |               |-- 'picname'.jpg
# |               |-- ...
# .
# .
# .

# The _`test-data`_ folder contains images that we will use to test our face recognizer after it has been successfully trained.





# function to detect face using OpenCV
check=[0]
def detect_face(img,accuracy):
    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print img.shape
    # load OpenCV face detector, I am using LBP which is fast
    # there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    # face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')

    # let's detect multiscale (some images may be closer to camera than others) images
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    #if no faces are detected and accuracy high then try harder
    #if there is really no face in the image or there is but we couldn't find then watch dog counter will be greater than some special number
    watch_dog_counter=0
    #use following counter to change the minNeighbors in detectMultiScale
    min_Neighbors_counter=3
    while len(faces) != 1 and accuracy == "high":
        #print "trying for the ",watch_dog_counter,"times\n"
        if watch_dog_counter <=5 :
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=min_Neighbors_counter)
            min_Neighbors_counter+=1
        # elif 5 < watch_dog_counter <=10 :
        #     min_Neighbors_counter=3
        #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=min_Neighbors_counter)
        #     min_Neighbors_counter += 1
        else:
            break
        watch_dog_counter+=1
    # if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    # under the assumption that there will be only one face,
    # extract the face area
    (x, y, w, h) = faces[0]

    # return only the face part of the image
    return gray[y:y + w, x:x + h], faces[0]


#      Preparing data step can be further divided into following sub-steps.
#
#      1. Read all the folder names of subjects/persons provided in training data folder. So for example, in this tutorial we have folder names: `s1, s2`.
#      2. For each subject, extract label number. Folder names follow the format `sLabel` where `Label` is an integer representing the label we have assigned to that subject. So for example, folder name `s1` means that the subject has label 1, s2 means subject label is 2 and so on. The label extracted in this step is assigned to each face detected in the next step.
#      3. Read all the images of the subject, detect face from each image.
#      4. Add each face to faces vector with corresponding subject label (extracted in above step) added to labels vector.

# this function will read all persons' training images, detect face from each image
# and will return two lists of exactly same size, one list
# of faces and another list of labels for each face
def prepare_training_data(data_folder_path,accuracy):
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

            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;

            # build image path
            # sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)
            # display an image window to show the image
            cv2.imshow("Training on image...", cv2.resize(image, (500, 500)))
            cv2.waitKey(100)

            # detect face
            face, rect = detect_face(image,accuracy)

            # ------STEP-4--------
            # we will ignore faces that are not detected
            if face is not None:
                # add face to list of faces
                faces.append(face)
                # add label for this face
                labels.append(label)
            else:
                print "cant detect the face: ",image_path

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels


# Below are some utility functions that we will use for drawing bounding box (rectangle) around face and putting celeberity name near the face bounding box.
# First function `draw_rectangle` draws a rectangle on image based on passed rectangle coordinates. It uses OpenCV's built in function `cv2.rectangle(img, topLeftPoint, bottomRightPoint, rgbColor, lineWidth)` to draw rectangle. We will use it to draw a rectangle around the face detected in test image.
#
# Second function `draw_text` uses OpenCV's built in function `cv2.putText(img, text, startPoint, font, fontSize, rgbColor, lineWidth)` to draw text on image.


# function to draw rectangle on image
# according to given (x, y) coordinates and
# given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# function to draw text on give image starting from
# passed (x, y) coordinates.
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)





# this function will predict the name of test data images ,by matching each test image with the training data subjects
def predict(test_img,face_recognizer,flag_name,subjects,accuracy,confusion,classifier_true,classifier_pred):
    # make a copy of the image as we don't want to chang original image
    img = test_img[0].copy()
    # detect face from the image
    face, rect = detect_face(img,accuracy)

    # predict the image using our face recognizer
    if not face is None:
        label, confidence = face_recognizer.predict(face)
        if flag_name==False :
            label_text = "s" + str(label) #+","+test_img[1]

            # print "correct answer :",test_img[1]," => ","recognizer: ",label_text
            correct_test_label = test_img[1].split("_")[0]
            classifier_true.append(correct_test_label)
            classifier_pred.append(label_text)
            # if correct_test_label in confusion:
            #     confusion[correct_test_label].append(label_text)
            # else:
            #     confusion[correct_test_label]=[]
            #     confusion[correct_test_label].append(label_text)

        else:
            label_text = subjects[label]
        # draw a rectangle around face detected
        draw_rectangle(img, rect)
        # draw name of predicted person
        draw_text(img, label_text, rect[0]+3, rect[1] + 20 )

    # get name of respective label returned by face recognizer
  #
    #label_text = subjects[label]
#

    else :
        print "no face detected"

    return img


# this function will read all test images
# and will add them to test_imgs list
def prepare_test_data(test_folder_path,test_imgs):
    imgs = os.listdir(test_folder_path)
    for img_name in imgs:
        img_path=test_folder_path+"/"+img_name
        if img_name.startswith("."):
            continue;
        img=cv2.imread(img_path)
        test_imgs.append((img,img_name.split(".")[0]))

#this function will read the subjects real name from subjects.txt file
def subjects_name(subject_name_file_path):
    subjects=[]
    subjects.append("")
    file = open(subject_name_file_path, "r")
    all=file.readlines()
    for x in all:
        subjects.append(x.rstrip())
    file.close()
    return subjects

def plot_confusion_matrix(cm , classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def drow_confusion_matrix(true,pred):
    precision, recall, fscore, support = precision_recall_fscore_support(true, pred,average='weighted')
    print "average precision weighted by support: ", precision
    print "average recall weighted by support:", recall
    tit= "average precision: "+ str(precision)+"\n"+"average recall:"+str(recall)
    array=confusion_matrix(true, pred)
    counter=1
    class_names=[]
    for i in range(len(array)):
        class_names.append(i+1)
    plot_confusion_matrix(array, classes=class_names,title=tit)


def main(argv):


    #These four following line are defaul values
    #You can change them by passing args in terminal
    flag_name = config.flag_name
    training_data_path = config.training_data_path
    test_data_path = config.test_data_path
    subjects_path = config.subjects_path
    accuracy = config.accuracy
    show_predicted_images = config.show_predicted_images
    splitFactor = config.splitFactor
    minFace = config.minFace
    enableConfusionMatrix = config.enableConfusionMatrix

    #use following try except for passing argument through terminal
    try:
        opts, args = getopt.getopt(argv, "-train:-test:-name:-subjects:-accuracy:-split:-minface:-confm:-showimgs", ["training-data-path=", "test-data-path=", "name-flag=", "subjects-path=", "accuracy-level=","split-factor=","minface-face=","confm-confm=","showimgs=showtestimages"])
    except getopt.GetoptError:
        # print 'face.py --train <training data path> --test <test data path> --name <True or False> --subjects <subject file path> --accuracy <low or high>'
        # print 'face.py --train <training data path> --test <test data path> --name <True or False> --subjects <subject file path> --accuracy <low or high> --TTSplitFactor <number between 0 to 1 (0.3 is good)> --MinFacePerson <int number>'
        print "kir"
        sys.exit(2)
    for opt, arg in opts:
        print opt
        if opt == '-h':
            print "face.py --train <training data path> --test <test data path> --name <'True' or 'False'> --subjects <subject file path> --accuracy <'low' or 'high'> --split <number between 0 to 1 (0.3 is good)> --minface <int number> --confm <'ON' or 'OFF' --showimgs <'ON' or 'OFF'>"
            sys.exit()
        elif opt in ("-train", "--training-data-path"):
            training_data_path = arg
        elif opt in ("-test", "--test-data-path"):
            test_data_path = arg
        elif opt in ("-name", "--name-flag"):
            flag_name = arg
        elif opt in ("-subjects", "--subjects-path"):
            subjects_path = arg
        elif opt in ("-accuracy", "--accuracy-level"):
            accuracy = arg
        elif opt in ("-split","--split-factor"):
            splitFactor=float(arg)
        elif opt in ("-minface","--minface-face"):
            minFace=int(arg)
        elif opt in ("-confm","--confm-confm"):
            enableConfusionMatrix=arg
        elif opt in ("-showimgs","--showimgs-showtestimages"):
            show_predicted_images=arg

    # As OpenCV face recognizer accepts labels as integers so we need to define a mapping between integer labels and persons actual names so below I am defining a mapping of persons integer labels and their respective names.
    subjects = subjects_name(subjects_path)


    # OpenCV comes equipped with three face recognizers.
    #
    # 1. EigenFace Recognizer: This can be created with `cv2.face.createEigenFaceRecognizer()`
    # 2. FisherFace Recognizer: This can be created with `cv2.face.createFisherFaceRecognizer()`
    # 3. Local Binary Patterns Histogram (LBPH): This can be created with `cv2.face.LBPHFisherFaceRecognizer()`

    # create our LBPH face recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # or use EigenFaceRecognizer by replacing above line with
    # face_recognizer = cv2.face.EigenFaceRecognizer_create()

    # or use FisherFaceRecognizer by replacing above line with
    # face_recognizer = cv2.face.FisherFaceRecognizer_create()





    print("Preparing data...")
    #     Prepare training data
    #     OpenCV face recognizer accepts data in a specific format.
    #     It accepts two vectors, one vector is of faces of all the persons and the second vector is of integer labels for each face so that when processing a face the face recognizer knows which person that particular face belongs too.
    #     For example, if we had 2 persons and 2 images for each person.
    #     PERSON-1    PERSON-2
    #
    #     img1        img1
    #     img2        img2
    #     .           .
    #     .           .
    #     .           .
    #     Then the prepare data step will produce following face and label vectors.
    #     FACES                        LABELS
    #
    #     person1_img1_face              1
    #     person1_img2_face              1
    #     person2_img1_face              2
    #     person2_img2_face              2
    #     .
    #     .
    #     .

    if training_data_path != "lfw":
        faces, labels = prepare_training_data(training_data_path,accuracy)
    else :
        lfw.loadLFW("oldlfw","lfw_train","lfw_test",minFace,splitFactor)
        training_data_path = "lfw_train"
        test_data_path = "lfw_test"
        faces, labels = prepare_training_data(training_data_path, accuracy)

    print("Data prepared")

    # train our face recognizer of our training faces
    face_recognizer.train(faces, np.array(labels))


    # print total faces and labels
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))


    print("Predicting images...")
    test_imgs = []
    prepare_test_data(test_data_path, test_imgs)


    predicted_imgs = []
    confusion=dict()
    classifier_true = []
    classifier_pred = []

    for i in test_imgs:
        predicted_imgs.append(predict(i,face_recognizer,flag_name,subjects,accuracy,confusion,classifier_true,classifier_pred))
    print("Prediction complete")


    if enableConfusionMatrix == "ON":
        plt.figure()
        drow_confusion_matrix(classifier_true,classifier_pred)
        plt.show()
    else:
        precision, recall, fscore, support = precision_recall_fscore_support(classifier_true, classifier_pred, average='weighted')
        print "average precision weighted by support: ", precision
        print "average recall weighted by support:", recall

    # display images
    if show_predicted_images == "ON":
        for i in range(len(predicted_imgs)):
            cv2.imshow(str(i), cv2.resize(predicted_imgs[i], (500, 500)))

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
   main(sys.argv[1:])