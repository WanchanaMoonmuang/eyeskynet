import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import IPython
import pandas as pd
from CDR import *
from Average_bgr import *
from mostf import *




def main() :

    dim = (1080,720)
   
    PATH ="D:\eyeskynetdata\TEST\others"
    filename = load_filename_from_folder(PATH)
    print("Loading images from : "+PATH)
    img_array = np.array(load_images_from_folder(PATH),dtype ='O')
    count = 0
    saveto ="D:\Downloads\eyeskynet\Output"
    try :
        os.mkdir(saveto)
        print("Creating Output")
    except OSError :
        print("Fail to create OUTPUT folder or Already Exist. Proceeding")
    os.chdir(saveto)
    print("Change dir to OUTPUT")
    TN = ['normal','glaucoma','others']
    FN = ['cdr','average_bgr','most_freq_b_px','most_freq_g_px','most_freq_r_px','num_blood_px']
    TG = 2
    print("------------------------- CURRENTLY CLASS : {} ---------------------------".format(TN[TG]))
    data_set = []
    feature_name = []
    file_name=[]
    target=[]
    target_name=[]


    for input_img in img_array :
        img = cv.resize(input_img, dim, interpolation = cv.INTER_AREA)
        num = count+1
        try :
            print("Processing Image No.{} : {}".format(num,filename[count]),end=" ...\n")
            cdr,out_img = cal_cdr(img)
        except :
            cdr = 0
        print("CDR :",cdr)
        average_bgr = avg_bgr(img)
        print("Average BGR :",average_bgr)
        most_b,most_g,most_r = most_f(img)
        print("Most B :",most_b)
        print("Most G :",most_g)
        print("Most R :",most_r)
        px_blood = cv.countNonZero(ex_blood(img))
        print("Total blood pixels :",px_blood)

        input_data=[cdr,average_bgr,most_b,most_g,most_r,px_blood]
        data_set.append(input_data)
        target.append(TG)
        target_name.append(TN[TG])
        feature_name.append(FN)
        file_name.append(filename[count])



        count += 1

    print("Completed process ...")
    print("Saving Data")

    df_set= {'data': data_set,
            'target': target,
            'target_name':target_name,
            'feature_name': feature_name,
            'filename' : file_name
            }

    df = pd.DataFrame(df_set, columns= ['data', 'target','target_name','feature_name','filename'])
    df.to_csv('CompDS.csv')
    print("Completed Data")

def mix_data():

    dfH = pd.read_csv('Healthy.csv')
    dfG = pd.read_csv('Glaucoma.csv')
    dfO = pd.read_csv('Other.csv')

    mix = pd.concat([dfH,dfG,dfO])
    mix.drop(mix.filter(regex="Unname"),axis=1, inplace=True)

    mix.to_csv('mix_data.csv',index = False)
if __name__ == "__main__":
    main()
