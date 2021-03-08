import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import IPython
import scipy
import sklearn

img = cv.imread('images/02_g.jpg')
plt.imshow(img)


def getSize(pic):
    width=pic.shape[0]
    height=pic.shape[1]
    return [pic,width,height]