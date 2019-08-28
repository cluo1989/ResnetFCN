import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

files = os.listdir('./')

for f in files:
    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    if np.sum(img==11) != 0:
        print(f)
        #cv2.imshow('img', img)
        #cv2.waitKey()
        plt.imshow(img)
        plt.show()
