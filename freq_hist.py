import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import IPython
import scipy

def avg_bgr(img) :
    dim = (1080,720)
    img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    avg = meanStd_nomask(img,find_mask(img))
    return avg
def find_mask(img) :
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,mask =cv.threshold(gray,10,255,cv.THRESH_BINARY)
    return mask

def meanStd_nomask(img,img_mask) :
    img_mean,img_std = cv.meanStdDev(img,mask = img_mask)
    return (img_mean[0][0],img_std[0][0])











