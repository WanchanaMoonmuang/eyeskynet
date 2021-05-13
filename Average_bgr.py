import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import IPython



RES = (1080,720)


def main() :
    PATH ="D:\DataSet\pre"
    os.chdir(PATH)
    #filename = load_filename_from_folder(PATH)
    print("Loading images from PATH")
    #img_array = np.array(load_images_from_folder(PATH),dtype ='O')
    img_array = []
    img_array.append(cv.imread('15_g.jpg'))
    img_array.append(cv.imread('2423_left.jpg'))
    test_data = [["file name","most_f_b","most_f_g","most_f_r"]]
    count = 0
    saveto ="D:\Downloads\eyeskynet\Outputpic"
    os.chdir(saveto)
    
    for input_img in img_array :
        input_img = cv.resize(input_img,RES,interpolation = cv.INTER_AREA)
        hist,binh = np.histogram(input_img.ravel(),256,[0,256])
        hist = remove_zero_hist(hist)
        plt.figure()
        plt.plot(hist)
        plt.savefig("hist_bgr_"+str(count)+'.jpg')
        count+=1



def remove_zero_hist(hist) :
    while True :
        (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(hist)
        hist[np.where(hist >= maxVal-1)] = 0
        if np.argmax(hist) >= 20 :
            return hist
#Change dir
def load_filename_from_folder(folder):
    return [filename for filename in os.listdir(folder)]
#load image from folder as np array
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
def avg_bgr(img) :
    dim = (1080,720)
    img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    avg,std = meanStd_nomask(img,find_mask(img))
    return avg
def find_mask(img) :
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,mask =cv.threshold(gray,10,255,cv.THRESH_BINARY)
    return mask

def meanStd_nomask(img,img_mask) :
    img_mean,img_std = cv.meanStdDev(img,mask = img_mask)
    return (img_mean[0][0],img_std[0][0])

if __name__ == '__main__' :
    main()









