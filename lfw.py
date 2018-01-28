import shutil
import os, os.path



def loadLFW(oldlfw_path, trainlfw_path, testlfw_path, min_Number_of_Image_per_Face, Train_Test_Split_Factor):
    try:
        os.makedirs(trainlfw_path)
    except:
        shutil.rmtree(trainlfw_path)
        os.makedirs(trainlfw_path)
    oldlfw_list=os.listdir(oldlfw_path)
    counter=1
    for person_name in oldlfw_list:
        if person_name.startswith("."):
            continue;
        person_path=oldlfw_path+"/"+person_name
        person_list=os.listdir(person_path)

        if len(person_list) >= min_Number_of_Image_per_Face:
            src=person_path
            dst=trainlfw_path+"/s"+str(counter)
            counter+=1
            try:
                shutil.copytree(src, dst)
            except:
                shutil.rmtree(dst)
                shutil.copytree(src, dst)

    try:
        os.makedirs(testlfw_path)
    except:
        shutil.rmtree(testlfw_path)
        os.makedirs(testlfw_path)

    newlfw_list=os.listdir(trainlfw_path)
    for person_name in newlfw_list:
        if person_name.startswith("."):
            continue;
        person_path = trainlfw_path + "/" + person_name
        person_list=os.listdir(person_path)

        number_of_test_images=int (len(person_list) * Train_Test_Split_Factor)
        counter=1
        for i in range(number_of_test_images):
            image_path=person_path+"/"+person_list[i]
            dst=testlfw_path
            name="_"+str(counter)
            counter+=1
            os.rename(image_path,person_path+name)
            shutil.move(person_path+name,dst)


# loadLFW("oldlfw","lfw_train","lfw_test",100,0.3)