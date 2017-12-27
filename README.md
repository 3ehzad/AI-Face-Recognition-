# AI-Face-Recognition-
you can read , readme.txt for more information
but if you don't have enough time just run one of following codes in terminal:
for low accuracy : python2 face.py --accuracy low
for high accuracy: python2 face.py --accuracy high


if you want to change the default training and test data set run:
python2 face.py --train 'address directory of your training dataset' --test 'address directory of your test dataset'


if you want to see the name of the subjects, pls add their names in the exact order of the training data set in 'subjects.txt'
and run the code below:
python2 face.py --subjects 'address of the subjects.txt' --name True
