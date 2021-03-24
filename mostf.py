import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import csv
#Normalize contrast by apply CLAHE
clahe = cv.createCLAHE(clipLimit=3, tileGridSize=(10,10))

def main () :
    PATH ="D:\DataSet\DataThatWeUse+++\Others\ALLSOURCE"
    filename = load_filename_from_folder(PATH)
    print("Loading images from PATH")
    img_array = np.array(load_images_from_folder(PATH),dtype ='O')
    test_data = [["file name","most_f_b","most_f_g","most_f_r"]]
    count = 0
    saveto ="D:\Downloads\eyeskynet\Outputf"
    os.chdir(saveto)

    for input_img in img_array :
        img_b,img_g,img_r = normalize_bgr(input_img)
        hist_b,hist_g,hist_r = cal_hist(img_b,img_g,img_r)

        mostf_b = np.argmax(hist_b)
        mostf_g = np.argmax(hist_g)
        mostf_r = np.argmax(hist_r)

        data_format = [filename[count],str(mostf_b),str(mostf_g),str(mostf_r)]
        test_data.append(data_format)
        count+=1
    write_csv("Dataf_Others.csv",test_data)
def most_f(in_img) :
    img_b,img_g,img_r = normalize_bgr(in_img)
    hist_b,hist_g,hist_r = cal_hist(img_b,img_g,img_r)
    mostf_b = np.argmax(hist_b)
    mostf_g = np.argmax(hist_g)
    mostf_r = np.argmax(hist_r)
    return (mostf_b,mostf_g,mostf_r)

def cal_hist (img_b,img_g,img_r):
    hist_b,binb = np.histogram(img_b.ravel(),256,[0,256])
    hist_b = remove_zero_hist(hist_b)

    hist_g,bing = np.histogram(img_g.ravel(),256,[0,256])
    hist_g = remove_zero_hist(hist_g)

    hist_r,binr= np.histogram(img_r.ravel(),256,[0,256])
    hist_r = remove_zero_hist(hist_r)

    return (hist_b,hist_g,hist_r)

#Write csv
def write_csv(filename, data):
    csv_file = open(filename, "w", newline='')
    writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC, delimiter=",")
    for data in data:
        writer.writerow(data)
    csv_file.close()

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
def normalize_bgr (img) :
    img = cv.GaussianBlur(img,(5,5),0)
    img_b,img_g,img_r = cv.split(img)

    img_b = clahe.apply(img_b)

    img_g = clahe.apply(img_g)

    img_r = clahe.apply(img_r)

    return (img_b,img_g,img_r)

def remove_zero_hist(hist) :
    while True :
        (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(hist)
        hist[np.where(hist >= maxVal-1)] = 0
        if np.argmax(hist) >= 20 :
            return hist



if __name__ == "__main__":
    main()
