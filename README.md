# AI-Face-Recognition-
you can read , readme.txt for more information

## use LFW DataSet
simply run this or see below for more details (first download http://vis-www.cs.umass.edu/lfw/lfw.tgz and name it "oldlfw" and move it in project directory) :

python2 face.py --train 'lfw' --split 0.3 --minface 10 --confm "OFF"

or this:

python2 face.py --train 'lfw' --split 0.3 --minface 100 --confm "ON"

***
if you want to use LFW data set first you need to download it from link here : http://vis-www.cs.umass.edu/lfw/lfw.tgz
then extract it and name it "oldlfw" and move it into your project directory.
then when you run you just need to write in terminal:


python2 face.py --train 'lfw'


***
if you want to change the split factor of test and train data (in this example make it 0.3)(more preferable to be under 0.5):
python2 face.py --train 'lfw' --split 0.3


***
if you want to change the minimum image per face (in this example just choose people with have more than 50 images):
python2 face.py --train 'lfw' --minface 50


*** 
if you want to see the confusion matrix (confusion matrix for more than 50 classes are not very useable!):

python2 face.py --train 'lfw' --confm "ON"

if not:

python2 face.py --train 'lfw' --confm "OFF"

***

for low accuracy : python2 face.py --accuracy low

for high accuracy: python2 face.py --accuracy high



## use custom data set



***
if you want to change the default training and test data set run:

python2 face.py --train 'address directory of your training dataset' --test 'address directory of your test dataset'


***
if you want to see the name of the subjects, pls add their names in the exact order of the training data set in 'subjects.txt'
and run the code below:

python2 face.py --subjects 'address of the subjects.txt' --name True

