import os
import cv2 as cv
import IPython
import numpy as np
import matplotlib.pyplot as plt

#Normalize contrast by apply CLAHE
#clahe = cv.createCLAHE(clipLimit=3, tileGridSize=(10,10))
RES = (300,300)
th = 20
#Normal use th 17 25
#Fix use histogram

def main():
    
    PATH ="D:\eyeskynetdata\others"
    filename = load_filename_from_folder(PATH)
    print("Loading images from : "+PATH)
    img_array = np.array(load_images_from_folder(PATH),dtype ='O')
    saveto ="D:\Downloads\eyeskynet\DLpre300/others"
    try :
        os.mkdir(saveto)
        print("Create OUTPUT")
    except OSError :
        print("Fail to create OUTPUT folder or Already Exist")
    os.chdir(saveto)
    print("Change dir to OUTPUT")
    count = 0
    for input_img in img_array :
        
        roi = find_roi(input_img,find_mask(input_img,th)) #find_mask(input_img,find_th_mask(hist)) find_mask(input_img,th)
        new_img = cv.resize(roi,RES,interpolation = cv.INTER_AREA)
        cv.imwrite(filename[count],new_img)
        print("Processed No {} : {}".format(count+1,filename[count]))
        count+=1
def prepro(img,RES) :

    roi = find_roi(img,find_mask(img,th)) 
    new_img = cv.resize(roi,RES,interpolation = cv.INTER_AREA)
    return new_img



def find_roi(img,mask) :
    contours, hierarchy = cv.findContours(mask,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    cnt_mask = np.zeros(mask.shape[:2],np.uint8)
    img_size = img.shape[0] * img.shape[1]
    for cnt in contours:
        if cv.contourArea(cnt) > img_size * 0.1 :
            x,y,w,h = cv.boundingRect(cnt)
            radius = int(w/2) - 10
            center = (int(x+( w/2 )),int(y + (h/2)))
            cv.circle(cnt_mask,center,radius,(255,255,255),-1)
        
    put_mask = cv.bitwise_and(img,img,mask=cnt_mask)
    crop_img = put_mask[y:y+h, x:x+w]
    return crop_img

def find_mask(img,th) :
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,mask =cv.threshold(gray,th,255,cv.THRESH_BINARY)
    return mask

def cal_hist (img):
    hist,bina = np.histogram(img.ravel(),256,[0,256])
    hist = remove_zero_hist(hist)
    return hist

def remove_zero_hist(hist) :
    while True :
        (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(hist)
        hist[np.where(hist >= maxVal-1)] = 0
        if np.argmax(hist) >= 20 :
            return hist

def find_th_mask(img_hist) :
    peak1 = np.argmax(img_hist)
    (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(img_hist)
    img_hist[np.where(img_hist >= maxVal-1)] = 0
    peak2 = np.argmax(img_hist)
    th = int((peak1+peak2)/2)
    return th
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
    return images
if __name__ == "__main__":
    main()
