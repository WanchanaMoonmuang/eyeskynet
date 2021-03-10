import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import IPython
import scipy
import sklearn

#Normalize contrast by apply CLAHE
clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(8,8))

def main() :
    folder="D:\DataSet\DataThatWeUse+++\Glaucoma\TRAIN"
    img_array = np.array(load_images_from_folder(folder))
    test_img = img_array[4]
    #test_img = cv.imread("2420_left.jpg")
    #plt.imshow(cv.cvtColor(test_img,cv.COLOR_BGR2RGB))
    test_img_nobv = remove_bv(test_img,ex_blood(test_img)) 
    od = detect_roi(test_img_nobv)

    out_img = test_img.copy()
    out_img = cv.circle(out_img, find_center(test_img_nobv),max(find_rad(test_img_nobv)) , (255, 0, 0), 5)

    plt.imshow(cv.cvtColor(out_img, cv.COLOR_BGR2RGB))

#Change dir

#load image from folder as np array
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images



def ex_blood (img) :
    
    img_b,img_g,img_r = cv.split(img) 
    img_g = cv.GaussianBlur(img_g,(5,5),0)
    img_g = clahe.apply(img_g)
    
    # LOOP is how many times open close morph and Amplify the different
    LOOP = 2
    open1 = cv.morphologyEx(img_g, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)), iterations = LOOP)
    close1 = cv.morphologyEx(open1, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)), iterations = LOOP)
    open2 = cv.morphologyEx(close1, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(11,11)), iterations = LOOP)
    close2 = cv.morphologyEx(open2, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(11,11)), iterations = LOOP)
    open3 = cv.morphologyEx(close2, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(23,23)), iterations = LOOP)
    close3 = cv.morphologyEx(open3, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(23,23)), iterations = LOOP)
    ampdif = cv.subtract(close3,img_g)
    amped_img = clahe.apply(ampdif)
    
# removing very small contours through area parameter noise removal
    th,img_bi = cv.threshold(amped_img,15,255,cv.THRESH_BINARY)
    mask = np.ones(amped_img.shape[:2], dtype="uint8") * 255
    temp_img_bi = img_bi.copy()
    contours, hierarchy = cv.findContours(temp_img_bi,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv.contourArea(cnt) <= 200:
            cv.drawContours(mask, [cnt], -1, 0, -1)
    temp_img = cv.bitwise_and(amped_img, amped_img, mask=mask)
    th,img_inv_bi = cv.threshold(temp_img,15,255,cv.THRESH_BINARY_INV)
    blood_ves = cv.erode(img_inv_bi, cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)), iterations=1)
    
    # removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
    #vessels and also in an interval of area
    fundus_eroded = cv.bitwise_not(blood_ves)	
    xmask = np.ones(img.shape[:2], dtype="uint8") * 255
    xcontours, xhierarchy = cv.findContours(fundus_eroded.copy(),cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)	
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.04 * peri, False)   				
        if len(approx) > 4 and cv.contourArea(cnt) <= 3000 and cv.contourArea(cnt) >= 100:
            shape = "circle"
        else:
            shape = "veins"
        if(shape=="circle"):
            cv.drawContours(xmask, [cnt], -1, 0, -1)

    finalimage = cv.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)	

    return finalimage

def remove_bv(img,blood_vs) :
    remove_ves = cv.inpaint(img,blood_vs,5,cv.INPAINT_TELEA)
    return remove_ves

#### CHANGE THRESHOLD FOR ROI HERE
def detect_roi(img) :
    """If u called remove_bv NO NEED TO NORMALIZE"""
    
    img_b,img_g,img_r = cv.split(img) #split to B-G-R
    #img = cv.GaussianBlur(img,(5,5),0)
    #img = clahe.apply(img_g)
    
    th, img_hpix = cv.threshold(img_g, 99, 255, cv.THRESH_BINARY)
    
    return img_hpix

def rev_od (img) :
    img_b,img_g,img_r = cv.split(img) #split to B-G-R
    th, img_hpix = cv.threshold(img_g, 99, 255, cv.THRESH_BINARY_INV)
    return img_hpix

def draw_center(brgimg) :
    
    out_img = brgimg.copy()
    
    gray = cv.cvtColor(brgimg, cv.COLOR_BGR2GRAY)
    
    (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)
    x_cen,y_cen = maxLoc
    max_y,max_x = gray.shape
    
    cv.circle(out_img, maxLoc, 20, (0, 0, 255), 10)
    
    cv.line(out_img, (x_cen, 0), (x_cen, max_y), (0, 255, 0), thickness=2)
    cv.line(out_img, (0, y_cen), (max_x, y_cen), (0, 255, 0), thickness=2)
    
    print("Center is :",maxLoc)
    
    return out_img

def find_center (brgimg) :
    gray = cv.cvtColor(brgimg, cv.COLOR_BGR2GRAY)
    (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)
    return maxLoc

#Return array of Distance from center by use roi from image to calculate radius"distance"
#Format of list : cen2left cen2right cen2top cen2bot

#INPUT no_bv img
def find_rad(brgimg):
    
    cen_x,cen_y = find_center(brgimg)
    bi_roi = detect_roi(brgimg)
    max_y,max_x = bi_roi.shape
    
    #Format is First left ,First right, First upper,First lower,
    from_center_2black = []
    count = 0
    #fix y run x from 0 (left)
    #FIRST LEFT
    for i in bi_roi[cen_y,cen_x::-1] :
        if i == 0 :
            from_center_2black.append(count)
            count = 0
            break
        count+=1
    #FIRST RIGHT
    for i in bi_roi[cen_y,cen_x:] :
        if i == 0 :
            from_center_2black.append(count)
            count = 0
            break
        count+=1
        
    #FIRST TOP
    for i in bi_roi[cen_y:,cen_x] :
        if i == 0 :
            from_center_2black.append(count)
            count = 0
            break
        count+=1

    
    #FIRST BOT
    for i in bi_roi[cen_y::-1,cen_x] :
        if i == 0 :
            from_center_2black.append(count)
            count = 0
            break
        count+=1
    
    dis_from_cen = np.array(from_center_2black)
    
    return dis_from_cen

if __name__ == "__main__":
    main()
