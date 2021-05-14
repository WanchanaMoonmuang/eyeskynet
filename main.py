import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import IPython
import pandas as pd
import joblib
from CDR import *
from Average_bgr import *
from mostf import *
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
import ast

#ex_data(PATH) ex_data('D:\eyeskynetdata\glaucoma')

CLASSIFIER_NAME = "Randomforest"
def main() :
    MODELPATH = 'D:\Downloads\eyeskynet\Output\ModelRandomForestFinal'
    os.chdir(MODELPATH)
    MODELFILE = 'RandomForest_fold4.pkl'
    model = joblib.load(MODELFILE)
    os.chdir("D:\Downloads\eyeskynet\Output\Eval")
    data = pd.read_csv("comp_ds.csv")
    X = data['data'].values
    y = np.array(data['target'].values)
    X = np.array([str2list(i) for i in X])
    eval_data(model,X,y)
    # os.chdir(MODELPATH)
    # dt_ada = joblib.load(MODELFILE)
    # IMGPATH = 'D:\eyeskynetdata\TEST\others'
    # IMGFILE = '1ffa961b-8d87-11e8-9daf-6045cb817f5b..jpg'
    # os.chdir(IMGPATH)
    # img = cv.imread(IMGFILE)
    # data = ex_test(img)
    # prob = dt_ada.predict_proba(np.array([data]))
    # print(prob)
    # pred = dt_ada.predict(np.array([data]))
    # print(pred)
    # result = np.argmax(prob,axis = 1)
    # print(result)
    # if result[0] == 0 :
    #     print("Classified as Normal")
    # elif result[0] == 1 :
    #     print("Classified as Glaucoma")
    # else :
    #     print("Classified as possible Others diseases")
    


    
def str2list (lstring) :
    nlist = ast.literal_eval(lstring)
    return nlist
def ex_test(input_img) :
    dim = (1080,720)
    img = cv.resize(input_img, dim, interpolation = cv.INTER_AREA)
    try :
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

    data=[cdr,average_bgr/255,most_b/255,most_g/255,most_r/255,px_blood/10000]
    return data

def ex_data(directory,saveto,label) :
    dim = (1080,720)
   
    PATH = directory
    filename = load_filename_from_folder(PATH)
    print("Loading images from : "+PATH)
    img_array = np.array(load_images_from_folder(PATH),dtype ='O')
    count = 0
    
    try :
        os.mkdir(saveto)
        print("Creating Output")
    except OSError :
        print("Fail to create OUTPUT folder or Already Exist. Proceeding")
    os.chdir(saveto)
    print("Change dir to OUTPUT")
    TN = ['normal','glaucoma','others']
    FN = ['cdr','average_bgr','most_freq_b_px','most_freq_g_px','most_freq_r_px','num_blood_px']
    TG = label
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

        input_data=[cdr,average_bgr/255,most_b/255,most_g/255,most_r/255,px_blood/10000]
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
    df.to_csv(TN[label]+'_ds.csv')
    print("Completed Data")

def mix_data():

    dfH = pd.read_csv('healthy_ds.csv')
    dfG = pd.read_csv('glaucoma_ds.csv')
    dfO = pd.read_csv('others_ds.csv')

    mix = pd.concat([dfH,dfG,dfO])
    mix.drop(mix.filter(regex="Unname"),axis=1, inplace=True)

    mix.to_csv('comp_ds.csv',index = False)
    print("Done")

def eval_data(classifier,X_test,y_test) :
    fold = 'testset'
    class_labels = [0,1,2]
    y_test = label_binarize(y_test, classes=class_labels)
    class_names = np.array(['Normal','Glaucoma','Others'])

    try :#SVM this
        y_score = classifier.decision_function(X_test) #decision_fn
    except : #KNN this
        y_score = classifier.predict_proba(X_test)

        

        #Confusion matrix TP TN FP FN
    y_pred = classifier.predict(X_test)
        
    y_test_t = np.argmax(y_test, axis=1)
        
    y_pred_t = np.argmax(y_pred, axis=1)
        
        #cm = multilabel_confusion_matrix(y_test, y_pred)
    cm = confusion_matrix(y_test_t,y_pred_t,labels = class_labels)
    print(cm)
    table_list = cm.ravel()
    tp_h = table_list[0]
    tp_g = table_list[4]
    tp_o = table_list[8]

    fp_h = table_list[3] + table_list[6]
    fp_g = table_list[1] + table_list[5]
    fp_o = table_list[5] + table_list[2]

    fn_h = table_list[1]+table_list[2]
    fn_g = table_list[3]+table_list[5]
    fn_o = table_list[6] + table_list[7]

    tn_h = table_list[4] + table_list[5] +table_list[7] + table_list[8]
    tn_g = table_list[0] + table_list[2] + table_list[6] + table_list[8]
    tn_o = table_list[0] + table_list[1] + table_list[3] + table_list [4]

    prec_h = tp_h/(tp_h+fp_h)
    sen_h = tp_h/(tp_h+fn_h)
    spe_h = tn_h/(tn_h+fp_h)
    acc_h = (tp_h+tn_h)/(tp_h+tn_h+fp_h+fn_h)

    prec_g = tp_g/(tp_g+fp_g)
    sen_g = tp_g/(tp_g+fn_g)
    spe_g = tn_g/(tn_g+fp_g)
    acc_g = (tp_g+tn_g)/(tp_g+tn_g+fp_g+fn_g)

    prec_o = tp_o/(tp_o+fp_o)
    sen_o = tp_o/(tp_o+fn_o)
    spe_o = tn_o/(tn_o+fp_o)
    acc_o = (tp_o+tn_o)/(tp_o+tn_o+fp_o+fn_o)

    cmtable = [{'TP':tp_h,'TN' : tn_h,'FP' : fp_h,'FN' : fn_h,'pre':prec_h,'sen':sen_h,'spe':spe_h,'acc' : acc_h},
                    {'TP':tp_g,'TN' : tn_g,'FP' : fp_g,'FN' : fn_g,'pre':prec_g,'sen':sen_g,'spe':spe_g,'acc' : acc_g},
                    {'TP':tp_o,'TN' : tn_o,'FP' : fp_o,'FN' : fn_o,'pre':prec_o,'sen':sen_o,'spe':spe_o,'acc' : acc_o}]
    raw_table = pd.DataFrame(cm)
    raw_table.to_csv('rawtable_fold{}.csv'.format(fold))
    table_df = pd.DataFrame(cmtable)
    table_df.to_csv('PredictvsTrue_fold{}.csv'.format(fold))

    print("Precision on H :",prec_h)
    print("Sensitivity on H :",sen_h)
    print("Specificity on H:",spe_h)
    print("Accuracy on H :",acc_h)

    print("Precision on G :",prec_g)
    print("Sensitivity on G :",sen_g)
    print("Specificity on G:",spe_g)
    print("Accuracy on G :",acc_g)

    print("Precision on O :",prec_o)
    print("Sensitivity on O :",sen_o)
    print("Specificity on O:",spe_o)
    print("Accuracy on O :",acc_o)
        
    joblib.dump(cmtable,CLASSIFIER_NAME+'_'+fold+'_table'+'.pkl')
        
        

    report = classification_report(y_test, y_pred, target_names=class_names)
    joblib.dump(report,CLASSIFIER_NAME+'_'+fold+'_report'+'.pkl')
    print(report)



        #ROC 1vsR
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(class_labels)):
        fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    mean_fpr = np.linspace(0, 1, 100)
# Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot of a ROC curve for a specific class
    interp_tpr = np.interp(mean_fpr,fpr[2],tpr[2])
    interp_tpr[0] = 0.0
    plt.figure()
    plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.4f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic {}'.format(fold))
    plt.legend(loc="lower right")

    plt.savefig("roc_auc_fold"+str(fold)+".jpg")

    area_under_curve = roc_auc[2]
    print("Area Under Curve :",area_under_curve)
        
        
        
    print("----------------------------------------------------------")

if __name__ == "__main__":
    main()
