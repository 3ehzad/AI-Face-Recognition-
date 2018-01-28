training_data_path ="lfw" #if you have a custom dataset change it to the address of your training data

test_data_path = "test_data" #if you're using LFW data set you don't need to change it
                            # but if you're using your custom dataset change it to the address of
                            # your test data

flag_name = False # if you want to see the NAME of each person instead of s1 or s2 or etc.
                    # change it to True but if you do that you need to make the subject.txt file
                    # and name all image in it


subjects_path = "subjects.txt" # if you make the above flag to True you have to make a file and name
                                # all your subjects in it and you have to write the path to it here

accuracy = "low"  # you can change it to "high"


show_predicted_images="OFF" #  if you want to see the test image (predicted) at the end of the execution
                            # you have to make it "ON"
                            #but if your test data is huge it would be VERYYY SLOWW


splitFactor=0.3 #number test / number of whole data  , (preferable under 0.5) (have to be between 0 and 1)
                # (for LFW dataset ONLY)

minFace=100 #minimum number of face that each person have to had (for LFW dataset ONLY)

enableConfusionMatrix="ON" #if you want to see the confusion matrix make it "ON" else make it "OFF"