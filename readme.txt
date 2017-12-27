How to use face.py :
-First you need to download and install python 2.7 and opencv(version 2) 
***We assume that you know how to install them***

You need to pass 5 arguments to the script 
—-train : train data directory
—-test : test data directory 
—-name : if you want your subject to have names set it True , else they just have ’s’ and number e.g. s1,s2,…
—-subjects : a path to a file that include the subject real name in exact order of the training data images
—accuracy : if you want to have higher accuracy but lower speed you should set this to “high” 

**training data images for each subject must be in a folder that its name start with an ’s’ e.g. s1. And all the images of that subject must be in that folder**

**test data can include any random ordered photos**

You can use ‘att_faces’ as train and ‘att_faces_test’ as test 
OR
You can use ’training_data’ as train and ‘test_data’ as test

To run the program open terminal and run: python2.7 face.py —-train <training data path> -—test <‘test data path’>  —-name <True or False> -—subjects <‘subject.txt file path’> —-accuracy <‘low or ‘high’>

All these arguments have default values so if you don’t want to change it just download ‘att_faces’ and ‘att_face_test’ and ‘subjects.txt’ and put them in the same folder az ‘face.py’ and just run the program like that: python2.7 face.py

When you run the program a visualize of the pictures show up (it may take a minute to finish) after that all the test images that the program could find their names will have a rectangle around the face and a label of their name.

****if you set the accuracy to high it may take some longer , about 5 min in my 5core system, but if you set it to high THERE WERE NO FACE IN THE DATASET THAT THE PROGRAM COULD NOT DETECT !!!!****

**you can use each of the two data set provided here**

**in some pictures the program couldn’t find a face in it , when the program is executing you may see some files name that the program couldn’t detect the face in it **

**********************Thank you for reading******************** 