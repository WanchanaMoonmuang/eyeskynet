import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import IPython
import scipy
import sklearn
import extract_blood_vessel
from PIL import Image

img = cv.imread('images/02_g.jpg')
plt.imshow(img)


def get_size(img):
    width = img.shape[0]
    height = img.shape[1]
    return [width, height]


# ---------------------------------------------------------extract_blood_vessel---------------------------------------

def extract_bv(image):
    b, green_fundus, r = cv.split(image)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced_green_fundus = clahe.apply(green_fundus)

    # applying alternate sequential filtering (3 times closing opening)
    r1 = cv.morphologyEx(contrast_enhanced_green_fundus, cv.MORPH_OPEN,
                         cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)), iterations=1)
    R1 = cv.morphologyEx(r1, cv.MORPH_CLOSE, cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (5, 5)), iterations=1)
    r2 = cv.morphologyEx(R1, cv.MORPH_OPEN, cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (11, 11)), iterations=1)
    R2 = cv.morphologyEx(r2, cv.MORPH_CLOSE, cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (11, 11)), iterations=1)
    r3 = cv.morphologyEx(R2, cv.MORPH_OPEN, cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (23, 23)), iterations=1)
    R3 = cv.morphologyEx(r3, cv.MORPH_CLOSE, cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (23, 23)), iterations=1)
    f4 = cv.subtract(R3, contrast_enhanced_green_fundus)
    f5 = clahe.apply(f4)

    # removing very small contours through area parameter noise removal
    ret, f6 = cv.threshold(f5, 15, 255, cv.THRESH_BINARY)
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255
    contours, hierarchy = cv.findContours(
        f6.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv.contourArea(cnt) <= 200:
            cv.drawContours(mask, [cnt], -1, 0, -1)
    im = cv.bitwise_and(f5, f5, mask=mask)
    ret, fin = cv.threshold(im, 15, 255, cv.THRESH_BINARY_INV)
    newfin = cv.erode(fin, cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (3, 3)), iterations=1)

    # removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
    # vessels and also in an interval of area
    fundus_eroded = cv.bitwise_not(newfin)
    xmask = np.ones(img.shape[:2], dtype="uint8") * 255
    xcontours, xhierarchy = cv.findContours(
        fundus_eroded.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.04 * peri, False)
        if len(approx) > 4 and cv.contourArea(cnt) <= 3000 and cv.contourArea(cnt) >= 100:
            shape = "circle"
        else:
            shape = "veins"
        if(shape == "circle"):
            cv.drawContours(xmask, [cnt], -1, 0, -1)

    finimage = cv.bitwise_and(fundus_eroded, fundus_eroded, mask=xmask)
    blood_vessels = cv.bitwise_not(finimage)

    return blood_vessels


# ---------------------------------------------------------create bloodvessel img---------------------------------------------------------------------------------
blood = extract_bv(img)


# ----------------------------------------------------------delete_blood_vessel----------------------------------------

def delete_blood_vessel(img, blood):
    src1 = img
    src2 = blood

    src2 = cv.cvtColor(src2, cv.COLOR_GRAY2BGR)
    src2 = src2 / 255
    dst = src2 * src1

    Image.fromarray(dst.astype(np.uint8)).save(
        'test_out/numpy_image_alpha_blend.jpg')

    img2 = cv.imread('test_out/numpy_image_alpha_blend.jpg')

    plt.imshow(img2)


delete_blood_vessel(img, blood)
