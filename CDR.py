import os
import cv2 as cv
import IPython
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import timeit


#Normalize contrast by apply CLAHE
clahe = cv.createCLAHE(clipLimit=3, tileGridSize=(10,10))
RES = (1080,720)


def main():
    PATH ="D:\DataSet\pre"
    filename = load_filename_from_folder(PATH)
    print("Loading images from : "+PATH)
    img_array = np.array(load_images_from_folder(PATH),dtype ='O')
    test_data = []
    count = 0
    saveto ="D:\Downloads\eyeskynet\Outputpic"
    try :
        os.mkdir(saveto)
        print("Create OUTPUT")
    except OSError :
        print("Fail to create OUTPUT folder or Already Exist")
    os.chdir(saveto)
    print("Change dir to OUTPUT")
    
    for input_img in img_array :
        etc=''
        num = count + 1
        start = timeit.default_timer()
        print("Processing Image No.{} : {}".format(num,filename[count]),end=" ...\n")

        try :
            cdr ,out_img= cal_cdr(input_img)
            print("RESULT CDR :",cdr)

            cv.imwrite(filename[count],out_img)
            
            

        except :
            cdr = 0
            print("ERROR file :",filename[count])
            etc = filename[count]
        data_format = [filename[count],str(cdr),etc]
        test_data.append(data_format)
        count+=1
        stop = timeit.default_timer()
        print("This Image run time :",stop-start)

    dataset = pd.DataFrame(test_data,columns = ['filename','CDR','etc'])
    dataset.to_csv('dataset.csv',index=False)


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

def find_mask(img) :
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,mask =cv.threshold(gray,10,255,cv.THRESH_BINARY)
    return mask

def ex_blood (img) :
    
    img_b,img_g,img_r = cv.split(img) 
    img_g = cv.GaussianBlur(img_g,(5,5),0)
    img_g = clahe.apply(img_g)
    
    # 3 times open close morph and Amplify the different
    LOOP = 3
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
    remove_ves = cv.GaussianBlur(remove_ves,(5,5),0)
    return remove_ves

def meanStd_nomask(img,img_mask) :
    img_mean,img_std = cv.meanStdDev(img,mask = img_mask)
    return (img_mean[0][0],img_std[0][0])

def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv.getRotationMatrix2D(center, angle, scale)
    rotated = cv.warpAffine(image, M, (w, h))
    return rotated

def rotate4(img,center) :
    out_img = img.copy()
    angle = (90,180,270)
    for x in angle :
        allrotate = rotate(out_img,x,center)
        out_img = cv.bitwise_or(out_img,allrotate)
    return out_img

def find_circle(odcd_img,percent = 1500,remove_l = False) :
    
    test_con = odcd_img
    if remove_l : #cv.countNonZero(test_con) >= 2500
        test_con = cv.morphologyEx(test_con, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(11,11)), iterations = 1)
        test_con = cv.morphologyEx(test_con, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(11,11)), iterations = 1)

    contours, hierarchy = cv.findContours(test_con,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    cnt_mask = np.zeros(test_con.shape,np.uint8)
    center = (0,0)
    radius = 0
    mul= 0.04
    for cnt in contours :
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, mul * peri, False)
        
        if  ( len(approx) > 4 ) and (test_con.size/5 > cv.contourArea(cnt) >= percent) : #Percent of image control? 20 10 5
            cv.drawContours(cnt_mask, [cnt], 0, 255,-1)
            (x,y),radius = cv.minEnclosingCircle(cnt)
            center = (int(x),int(y))

    if cv.countNonZero(cnt_mask) == 0 :
        percent = percent/2
        contours, hierarchy = cv.findContours(test_con,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
        for cnt in contours :
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, mul * peri, False)
            if  ( len(approx) > 4 ) and (test_con.size/5 > cv.contourArea(cnt) >= percent) : #Percent of image control? 20 10 5
                cv.drawContours(cnt_mask, [cnt], 0, 255,-1)
                (x,y),radius = cv.minEnclosingCircle(cnt)
                center = (int(x),int(y))
    if cv.countNonZero(cnt_mask) == 0 :
        percent = percent/2
        contours, hierarchy = cv.findContours(test_con,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
        for cnt in contours :
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, mul * peri, False)
            if  ( len(approx) > 4 ) and (test_con.size/5 > cv.contourArea(cnt) >= percent) : #Percent of image control? 20 10 5
                cv.drawContours(cnt_mask, [cnt], 0, 255,-1)
                (x,y),radius = cv.minEnclosingCircle(cnt)
                center = (int(x),int(y))
    if cv.countNonZero(cnt_mask) == 0 :
        percent = percent/2
        contours, hierarchy = cv.findContours(test_con,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
        for cnt in contours :
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, mul * peri, False)
            if  ( len(approx) > 4 ) and (test_con.size/5 > cv.contourArea(cnt) >= percent) : #Percent of image control? 20 10 5
                cv.drawContours(cnt_mask, [cnt], 0, 255,-1)
                (x,y),radius = cv.minEnclosingCircle(cnt)
                center = (int(x),int(y))
            
    return (center,radius,cnt_mask)

def find_circle_b(cdod_img,in_center,which,percent = 1500,rotate_st = False):
    
    test_con = cdod_img
    roi_mask = np.zeros(test_con.shape,np.uint8)
    cen_x,cen_y = in_center
    cv.rectangle(roi_mask, (cen_x - 75, cen_y + 75), (cen_x + 75, cen_y - 75), 255, -1)
    test_con = cv.bitwise_and(test_con,roi_mask)
    
    contours, hierarchy = cv.findContours(test_con,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
   # mask = []
    
    cnt_mask = np.zeros(test_con.shape,np.uint8)
    
    center = (0,0)
    radius = 0
    for cnt in contours :
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.04 * peri, False)
        
        if  ( len(approx) > 4 ) and (test_con.size/5 > cv.contourArea(cnt) >= percent) and cv.pointPolygonTest(cnt,in_center,True) > 0 : #Percent of image control? 20 10 5
            cv.drawContours(cnt_mask, [cnt], 0, 255,-1)
            (x,y),radius = cv.minEnclosingCircle(cnt)
            center = (int(x),int(y))
    
    
    if cv.countNonZero(cnt_mask) == 0 :
        percent = percent/2
        contours, hierarchy = cv.findContours(test_con,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
        for cnt in contours :
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.04 * peri, False)
            if  ( len(approx) > 4 ) and (test_con.size/5 > cv.contourArea(cnt) >= percent) and cv.pointPolygonTest(cnt,in_center,True) > 0: #Percent of image control? 20 10 5
                cv.drawContours(cnt_mask, [cnt], 0, 255,-1)
                (x,y),radius = cv.minEnclosingCircle(cnt)
                center = (int(x),int(y))
    if cv.countNonZero(cnt_mask) == 0 :
        percent = percent/2
        contours, hierarchy = cv.findContours(test_con,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
        for cnt in contours :
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.04 * peri, False)
            if  ( len(approx) > 4 ) and (test_con.size/5 > cv.contourArea(cnt) >= percent) and cv.pointPolygonTest(cnt,in_center,True) > 0: #Percent of image control? 20 10 5
                cv.drawContours(cnt_mask, [cnt], 0, 255,-1)
                (x,y),radius = cv.minEnclosingCircle(cnt)
                center = (int(x),int(y))
    if cv.countNonZero(cnt_mask) == 0 :
        percent = percent/2
        contours, hierarchy = cv.findContours(test_con,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
        for cnt in contours :
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.04 * peri, False)
            if  ( len(approx) > 4 ) and (test_con.size/5 > cv.contourArea(cnt) >= percent) and cv.pointPolygonTest(cnt,in_center,True) > 0: #Percent of image control? 20 10 5
                cv.drawContours(cnt_mask, [cnt], 0, 255,-1)
                (x,y),radius = cv.minEnclosingCircle(cnt)
                center = (int(x),int(y))
    
    if (cv.countNonZero(cnt_mask) < 3000 and which == 'od') or rotate_st:
        cnt_mask = rotate4(cnt_mask,in_center)
        contours, hierarchy = cv.findContours(cnt_mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
        for cnt in contours :
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.04 * peri, False)
            if  ( len(approx) > 4 ) and (test_con.size/5 > cv.contourArea(cnt) >= percent ):
                (x,y),radius = cv.minEnclosingCircle(cnt)
                center = (int(x),int(y))

    return (center,radius,cnt_mask)

def detect_cd(img_g,mask):
    
    img_g = clahe.apply(img_g)
    
    MAX_TH = 240
    #hist_b,hist_g,hist_r = cal_hist(img_b,img_g,img_r)
    #detect cd use GREEN channel
    percent_img_cd = img_g.size * 0.0015 #0.015 #0.002 0.0015
    #print("Percent CD limit :",percent_img_cd)
    cd_use = img_g

    cd_mean,cd_std = meanStd_nomask(cd_use,mask)
    th_cd = cd_mean + 6 * cd_std
    ret_cd,img_cd =cv.threshold(cd_use,th_cd,255,cv.THRESH_BINARY)
    
    if th_cd >= MAX_TH or cv.countNonZero(img_cd) < percent_img_cd :
        th_cd = cd_mean + 5.5 * cd_std #0.006125% from normal distribution
        ret_cd,img_cd =cv.threshold(cd_use,th_cd,255,cv.THRESH_BINARY)
        
    if th_cd >= MAX_TH or cv.countNonZero(img_cd) < percent_img_cd :
        th_cd = cd_mean + 5 * cd_std #0.025% from normal distribution
        ret_cd,img_cd =cv.threshold(cd_use,th_cd,255,cv.THRESH_BINARY)
    
    if th_cd >= MAX_TH or cv.countNonZero(img_cd) < percent_img_cd :
        th_cd = cd_mean + 4.5 * cd_std
        ret_cd,img_cd =cv.threshold(cd_use,th_cd,255,cv.THRESH_BINARY)
    
    if th_cd >= MAX_TH or cv.countNonZero(img_cd) < percent_img_cd :
        th_cd = cd_mean + 4 * cd_std
        ret_cd,img_cd =cv.threshold(cd_use,th_cd,255,cv.THRESH_BINARY)
    
    if th_cd >= MAX_TH or cv.countNonZero(img_cd) < percent_img_cd :
        th_cd = cd_mean + 3.5 * cd_std
        ret_cd,img_cd =cv.threshold(cd_use,th_cd,255,cv.THRESH_BINARY)
    
    if th_cd >= MAX_TH or cv.countNonZero(img_cd) < percent_img_cd :
        th_cd = cd_mean + 3 * cd_std
        ret_cd,img_cd =cv.threshold(cd_use,th_cd,255,cv.THRESH_BINARY)
    
    if th_cd >= MAX_TH or cv.countNonZero(img_cd) < percent_img_cd :
        th_cd = cd_mean + 2.5 * cd_std
        ret_cd,img_cd =cv.threshold(cd_use,th_cd,255,cv.THRESH_BINARY)
   
    if th_cd >= MAX_TH or cv.countNonZero(img_cd) < percent_img_cd :
        th_cd = cd_mean + 2 * cd_std
        ret_cd,img_cd =cv.threshold(cd_use,th_cd,255,cv.THRESH_BINARY)

    return img_cd

def detect_od(img_r,mask) :
    
    img_r = clahe.apply(img_r)
    od_use = img_r
    MAX_TH = 240
    percent_img_od = img_r.size * 0.01 #0.02
    
    od_mean,od_std = meanStd_nomask(od_use,mask)
    th_od = od_mean + 5 * od_std
    ret_od,img_od =cv.threshold(od_use,th_od,255,cv.THRESH_BINARY)

    if th_od >= MAX_TH or cv.countNonZero(img_od) <= percent_img_od :
        th_od = od_mean + 4.5 * od_std
        ret_od,img_od =cv.threshold(od_use,th_od,255,cv.THRESH_BINARY)
    
    if th_od >= MAX_TH or cv.countNonZero(img_od) <= percent_img_od :
        th_od = od_mean + 3.5 * od_std
        ret_od,img_od =cv.threshold(od_use,th_od,255,cv.THRESH_BINARY)
    
    if th_od >= MAX_TH or cv.countNonZero(img_od) <= percent_img_od :
        th_od = od_mean + 3 * od_std
        ret_od,img_od =cv.threshold(od_use,th_od,255,cv.THRESH_BINARY)
    
    if th_od >= MAX_TH or cv.countNonZero(img_od) <= percent_img_od :
        th_od = od_mean + 2.5 * od_std
        ret_od,img_od =cv.threshold(od_use,th_od,255,cv.THRESH_BINARY)
    
    if th_od >= MAX_TH or cv.countNonZero(img_od) <= percent_img_od :
        th_od = od_mean + 2 * od_std
        ret_od,img_od =cv.threshold(od_use,th_od,255,cv.THRESH_BINARY)
    
    if th_od >= MAX_TH or cv.countNonZero(img_od) <= percent_img_od :
        th_od = od_mean + 1.5 * od_std
        ret_od,img_od =cv.threshold(od_use,th_od,255,cv.THRESH_BINARY)
    
    if th_od >= MAX_TH or cv.countNonZero(img_od) <= percent_img_od :
        th_od = od_mean + od_std
        ret_od,img_od =cv.threshold(od_use,th_od,255,cv.THRESH_BINARY)
    
    return img_od

def cal_cdr (img) :

    img = cv.resize(img,RES,interpolation = cv.INTER_AREA)
    
    img_mask = find_mask(img)
    blood_img = ex_blood(img)
    blood_img = cv.bitwise_and(blood_img,blood_img,mask=img_mask)
    
    nobv_img = remove_bv(img,blood_img)
    nobv_img = cv.bitwise_and(nobv_img,nobv_img,mask=img_mask)
    
    img_b,img_g,img_r=cv.split(nobv_img)
    
    img_cd = detect_cd(img_g,img_mask)
    
    img_od = detect_od(img_r,img_mask)
    

    percent_img_cd = (img.size /3) * 0.0005 #0.0006 0.0005
    percent_img_od = (img.size /3) * 0.0015 # 0.0012 0.0015
    
    #Find CD
    tcd_center,tcd_r,tcd_cnt = find_circle(img_cd,percent_img_cd)
    #tod_center,tod_r,tod_cnt = find_circle(img_od,percent_img_od)
    
        
    findbright = cv.bitwise_and(nobv_img,nobv_img,mask=tcd_cnt)
    gray = cv.cvtColor(findbright, cv.COLOR_BGR2GRAY)
    (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)
    mostbright = maxLoc
    test_cx,test_cy = maxLoc
    
    
    #if  tod_cnt[test_cy][test_cx] == 0 :
        
    #    findbright = cv.bitwise_and(nobv_img,nobv_img,mask=tod_cnt)
    #    gray = cv.cvtColor(findbright, cv.COLOR_BGR2GRAY)
    #    (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)
    #    mostbright = maxLoc
        
    rotate_cd = False
    #Find OD
    cd_center,cd_r,cnt_cd = find_circle_b(img_cd,mostbright,'cd',percent_img_cd,rotate_st = rotate_cd)
    if cd_r == 0 :
        tcd_center,tcd_r,tcd_cnt = find_circle(img_cd,percent_img_cd,remove_l=True)
        findbright = cv.bitwise_and(nobv_img,nobv_img,mask=tcd_cnt)
        gray = cv.cvtColor(findbright, cv.COLOR_BGR2GRAY)
        (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)
        mostbright = maxLoc
        test_cx,test_cy = maxLoc

        cd_center,cd_r,cnt_cd = find_circle_b(img_cd,mostbright,'cd',percent_img_cd,rotate_st = rotate_cd)

    od_center,od_r,cnt_od = find_circle_b(img_od,mostbright,'od',percent_img_od)
    
    if cd_r > od_r :
        cd_center , od_center = od_center , cd_center
        cd_r , od_r = od_r , cd_r
    
    if cd_r == 0 or od_r == 0 :
        cdr = 0
        return cdr
    else :
        cdr = cd_r/od_r
    

    #img_cd = cv.cvtColor(img_cd, cv.COLOR_GRAY2BGR)
    #img_od = cv.cvtColor(img_od, cv.COLOR_GRAY2BGR)
    out_img = cv.circle(img, cd_center, int(cd_r), (255,0,0), 2)
    out_img = cv.circle(img, od_center, int(od_r), (0,255,0), 2)
    
    
    return cdr,out_img #blood_img to check
    
if __name__ == "__main__":
    main()
