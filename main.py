import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import IPython
import timeit
import pandas as pd

#Normalize contrast by apply CLAHE
clahe = cv.createCLAHE(clipLimit=3, tileGridSize=(10,10))
#Blur with cv.GaussianBlur(img(5,5),0) first
def main() :
    
    PATH ="D:\DataSet\DataThatWeUse+++\Others\ALLOTHERS\TRAIN"
    filename = load_filename_from_folder(PATH)
    print("Loading images from : "+PATH)
    img_array = np.array(load_images_from_folder(PATH),dtype ='O')
    test_data = []
    count = 0
    saveto ="D:\Downloads\eyeskynet\Output"
    try :
        os.mkdir(saveto)
        print("Create OUTPUT")
    except OSError :
        print("Fail to create OUTPUT folder or Already Exist")
    os.chdir(saveto)
    print("Change dir to OUTPUT")
    
    res = (1080,720)

    for input_img in img_array :
        
        start = timeit.default_timer()
        print("Processing Image No.{} : {}".format(count,filename[count]),end=" ...\n")
        mask = find_mask(input_img)

        
        resized_img = cv.resize(input_img,res,interpolation=cv.INTER_AREA)
        mean_BGR,std_BGR = meanStd_nomask(resized_img,find_mask(resized_img))
        print("Average BGR :",mean_BGR)

        #CDR
        nobv_img = remove_bv(input_img,ex_blood(input_img))
        center = find_center(nobv_img)
        #We use green for cd , red for od
        

        cd_r,od_r = detect_cd_od(nobv_img,mask)
        cdr = float(cd_r)/float(od_r)
        out_img = draw_cdr(input_img,center,cd_r,od_r)
        cv.imwrite(filename[count],out_img)
        print("CDR :",cdr)

        data_format = [filename[count],str(cdr),mean_BGR]
        test_data.append(data_format)
        count+=1
        stop = timeit.default_timer()
        print("This Image run time :",stop-start)

    dataset = pd.DataFrame(test_data,columns = ['filename','CDR','Average BGR'])
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

def ex_blood (img) :
    
    img_b,img_g,img_r = cv.split(img) 
    img_g = cv.GaussianBlur(img_g,(5,5),0)
    img_g = clahe.apply(img_g)
    
    # 3 times open close morph and Amplify the different
    LOOP = 1
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

def find_center (brgimg) :
    gray = cv.cvtColor(brgimg, cv.COLOR_BGR2GRAY)
    (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)
    return maxLoc

def detect_cd_od (img,mask) :
    #img_size = nobv_img.shape[0] * nobv_img.shape[1]
    #img = cv.GaussianBlur(img,(5,5),0)
    img_b,img_g,img_r = cv.split(img)

    img_b = clahe.apply(img_b)
    img_g = clahe.apply(img_g)
    img_r = clahe.apply(img_r)

    MAX_TH = 239
    #hist_b,hist_g,hist_r = cal_hist(img_b,img_g,img_r)
    #detect cd use GREEN channel
    center = find_center(img)
    cd_use = img_g

    cd_mean,cd_std = meanStd_nomask(cd_use,mask)

    th_cd = cd_mean + 5 * cd_std #0.006125% from normal distribution
    ret_cd,img_cd =cv.threshold(cd_use,th_cd,255,cv.THRESH_BINARY)
    if th_cd >MAX_TH or cv.countNonZero(img_cd) == 0 :
        th_cd = cd_mean + 4.5 * cd_std #0.0125% from normal distribution
        ret_cd,img_cd =cv.threshold(cd_use,th_cd,255,cv.THRESH_BINARY)
    if th_cd >MAX_TH or cv.countNonZero(img_cd) == 0 :
        th_cd = cd_mean + 4 * cd_std #0.025% from normal distribution
        ret_cd,img_cd =cv.threshold(cd_use,th_cd,255,cv.THRESH_BINARY)
    if th_cd >MAX_TH or cv.countNonZero(img_cd) == 0 :
        th_cd = cd_mean + 3.5 * cd_std
        ret_cd,img_cd =cv.threshold(cd_use,th_cd,255,cv.THRESH_BINARY)
    if th_cd >MAX_TH or cv.countNonZero(img_cd) == 0 :
        th_cd = cd_mean + 3 * cd_std
        ret_cd,img_cd =cv.threshold(cd_use,th_cd,255,cv.THRESH_BINARY)
    if th_cd >MAX_TH or cv.countNonZero(img_cd) == 0 :
        th_cd = cd_mean + 2.5 * cd_std
        ret_cd,img_cd =cv.threshold(cd_use,th_cd,255,cv.THRESH_BINARY)
    if th_cd >MAX_TH or cv.countNonZero(img_cd) == 0 :
        th_cd = cd_mean + 2 * cd_std
        ret_cd,img_cd =cv.threshold(cd_use,th_cd,255,cv.THRESH_BINARY)
    if th_cd >MAX_TH or cv.countNonZero(img_cd) == 0 :
        th_cd = cd_mean + 1.5 * cd_std
        ret_cd,img_cd =cv.threshold(cd_use,th_cd,255,cv.THRESH_BINARY)
    if th_cd >MAX_TH or cv.countNonZero(img_cd) == 0 :
        th_cd = cd_mean + cd_std
        ret_cd,img_cd =cv.threshold(cd_use,th_cd,255,cv.THRESH_BINARY)
    if th_cd >MAX_TH or cv.countNonZero(img_cd) == 0 :
        th_cd = cd_mean
        ret_cd,img_cd =cv.threshold(cd_use,th_cd,255,cv.THRESH_BINARY)

    
    
    img_r_mean,img_r_std = meanStd_nomask(img_r,mask)
    th_r = img_r_mean + 2 * img_r_std #Not likely normal dist. but try use 13.6%
    ret_r,img_r_test =cv.threshold(img_r,th_r,255,cv.THRESH_BINARY)
    if th_r >MAX_TH or cv.countNonZero(img_r_test) == 0 :
        th_r = img_r_mean + 1.5 * img_r_std
        ret_r,img_r_test =cv.threshold(img_r,th_r,255,cv.THRESH_BINARY)
        if th_r >MAX_TH or cv.countNonZero(img_r_test) == 0 :
            th_r = img_r_mean + img_r_std
            ret_r,img_r_test =cv.threshold(img_r,th_r,255,cv.THRESH_BINARY)
            if th_r >MAX_TH or cv.countNonZero(img_r_test) == 0 :
                th_r = img_r_mean + img_r_std * 0.5
                ret_r,img_r_test =cv.threshold(img_r,th_r,255,cv.THRESH_BINARY)
                if th_r >MAX_TH or cv.countNonZero(img_r_test) == 0 :
                    th_r = img_r_mean
                    ret_r,img_r_test =cv.threshold(img_r,th_r,255,cv.THRESH_BINARY)
    img_od = img_r_test.copy()

    cd_rotate = rotate15(img_cd,center)
    cd_r = find_rad(img,cd_rotate)
    od_r = find_rad(img,img_od)


    if od_r < cd_r :
        
        cd_r,od_r = od_r,cd_r
    #else : #Remain same cd
        #od_rotate = rotate8(img_od,center) #Always rotate od
        #od_r = find_rad(img,od_rotate)

    return (cd_r,od_r)

def find_mask(img) :
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,mask =cv.threshold(gray,10,255,cv.THRESH_BINARY)
    return mask

def meanStd_nomask(img,img_mask) :
    img_mean,img_std = cv.meanStdDev(img,mask = img_mask)
    return (img_mean[0][0],img_std[0][0])

def find_rad(brg_img,odcd_img):
    
    cen_x,cen_y = find_center(brg_img)
    bi_roi = odcd_img
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
    if len(dis_from_cen) == 0 :
        return 0
    else :
        return max(dis_from_cen)

def draw_cdr(img,center,cd_r,od_r) :

    test_img = img.copy()
    #Draw cd as BLUE
    test_draw_cd = cv.circle(test_img, center,cd_r , (255, 0, 0), 5)

    #Draw od as Green
    test_draw_od = cv.circle(test_img, center,od_r , (0, 255, 0), 5)

    return test_img

def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv.getRotationMatrix2D(center, angle, scale)
    rotated = cv.warpAffine(image, M, (w, h))
    return rotated

def rotate_circle(img,center) :
    out_img = img.copy()
    for x in range(361) :
        allrotate = rotate(out_img,x,center)
        out_img = cv.bitwise_or(out_img,allrotate)
    return out_img

def rotate8(img,center) :
    out_img = img.copy()
    angle = (45,90,135,180,225,270,315)
    for x in angle :
        allrotate = rotate(out_img,x,center)
        out_img = cv.bitwise_or(out_img,allrotate)
    return out_img

def rotate15(img,center) :
    out_img = img.copy()
    angle = (30,45,60,90,120,135,150,180,210,225,240,270,300,315,330)
    for x in angle :
        allrotate = rotate(out_img,x,center)
        out_img = cv.bitwise_or(out_img,allrotate)
    return out_img
    
if __name__ == "__main__":
    main()
