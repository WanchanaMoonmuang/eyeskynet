import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from CDR import *

clahe = cv.createCLAHE(clipLimit=3, tileGridSize=(10,10))
RES = (800,800)
INPUT = 'glaucoma'
def main():
    PATH = 'D:\eyeskynetdata'+'/TEST/'+INPUT
    filename = load_filename_from_folder(PATH)
    print("Loading images from : "+PATH)
    img_array = np.array(load_images_from_folder(PATH),dtype ='O')

    saveto ="D:\Downloads\Output/zoomed_"+INPUT
    try :
        os.mkdir(saveto)
        print("Create OUTPUT")
    except OSError :
        print("Fail to create OUTPUT folder or Already Exist")
    os.chdir(saveto)
    print("Change dir to OUTPUT")

    for i,img in enumerate(img_array) :
        try :
            print('processing no.',i+1)
            img = cv.resize(img,RES,interpolation = cv.INTER_AREA)
            img_mask = find_mask(img)
            img_b,img_g,img_r=cv.split(img)
            percent_img_cd = (img.size /3) * 0.0005

            img_cd = detect_cd(img_g,img_mask)
            tcd_center,tcd_r,tcd_cnt = find_circle(img_cd,percent_img_cd)
            findbright = cv.bitwise_and(img,img,mask=tcd_cnt)
            gray = cv.cvtColor(findbright, cv.COLOR_BGR2GRAY)
            (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)
            mostbright = maxLoc
        
            cd_center,cd_r,cnt_cd = find_circle_b(img_cd,mostbright,'cd',percent_img_cd)
            out_img = detect_roi(img,cd_center)
            cv.imwrite(filename[i+1],out_img)
        except:
            continue
    print("Done")



def detect_roi(img,center) : 
    cen_x,cen_y = center
    from_x = cen_x - 150
    to_x = cen_x + 150
    
    from_y = cen_y - 150
    to_y = cen_y + 150
    crop_img = img[from_y:to_y,from_x:to_x]
    return crop_img


if __name__ == '__main__' :
    main()
