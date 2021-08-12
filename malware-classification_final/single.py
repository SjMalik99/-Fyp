import os
import numpy as np
import cv2
import  matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

####### CODE FOR PREPARING DATASET FOR SVM STARTS HERE ###################
path = "E:\\malware-classification\\malimg_paper_dataset_imgs\\Autorun.K\\00a53b82a0ebbbea57289e26000f95ba.png"
# petimage = cv2.imread(path)
# features=cv2.resize(petimage,(64,64))
# cv2.imshow("image",features)
try:
    petimage = cv2.imread(path)
    features=cv2.resize(petimage,(64,64))
    IMG_SIZE = 64

    X = np.array(features).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    np.save('testfile.npy',X)
    # import pickle
    # print("Hello")
    #
    # pickle_out = open("testfile.pickle","wb")
    # pickle.dump(X,pickle_out)
    # pickle_out.close()

    #Image is 2D so converting it to 1D
except Exception as e:
            pass




