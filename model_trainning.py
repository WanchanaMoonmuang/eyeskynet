
#Import classifier here
from sklearn.neighbors import KNeighborsClassifier #KNeighborsClassifier(n_neighbors=3)
from sklearn.svm import SVC #SVC(kernel='linear') 'rbf'  sigmoid
#
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import scipy
import ast
import joblib

from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
import time



#Data = [cdr,avg,mostb,mostg,mostr,bloodpx]
F1 = 0 #CDR always use
F2 = 1  #avg bgr
F3 = 2  #most
F4 = 3
F5 = 4
F6 = 5


TITLE = "Healthy - Glaucoma - Others"
FILENAME = 'HGO_ds.csv' #.csv
DIR = 'D:\Downloads\eyeskynet\Output'
CLASSIFIER_NAME = 'Knn7'  #'SVMLinear'



def main():
    
    seconds = time.time()
    local_time = time.ctime(seconds)
    local_time = local_time[11:13]+local_time[14:16]+local_time[17:19]
    Out_folder = DIR+"\Model"+CLASSIFIER_NAME+local_time
    

    print("Loading file {} at {}".format(FILENAME,DIR))
    os.chdir(DIR)
    data = pd.read_csv(FILENAME)
    X = data['data'].values #List of stringlist
    
    y = data['target'].values
    y = label_binarize(y, classes=[0,1,2])
    n_classes = y.shape[1]
    class_names = np.array(['Healthy','Glaucoma','Others'])
    class_labels = [0,1,2]

    X,X_test,y,y_test = train_test_split(X,y, test_size = 0.1)

    kf = KFold(n_splits=5)

    print("Classifier :",CLASSIFIER_NAME)
    print("--- Feature used ---")
    fname = str2list(data.iloc[0][3])
    print(fname[F1])
    print(fname[F2])
    print(fname[F3])
    print(fname[F4])
    print(fname[F5])
    print(fname[F6])
    try :
        os.mkdir(Out_folder)
    except OSError :
        print("Fail to create folder")
    os.chdir(Out_folder)
    #Classifier declare #KNeighborsClassifier(n_neighbors=3)
    clf = KNeighborsClassifier(n_neighbors=7)#SVC(kernel='linear',probability=True) #decision_function_shape = 'ovo'


    classifier = OneVsRestClassifier(clf) #ovr or ovo

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for fold,(train_index, val_index) in enumerate(kf.split(X)) :
        
        print("------------------------------------------STARTING FOLD no. :",fold)
        X_train = np.array([[str2list(X[i])[F1],str2list(X[i])[F2]/255,str2list(X[i])[F3]/255,str2list(X[i])[F4]/255,str2list(X[i])[F5]/255,str2list(X[i])[F6]/10000] for i in train_index])
        y_train = np.array(y[train_index])
        
        X_test = np.array([[str2list(X[i])[F1],str2list(X[i])[F2]/255,str2list(X[i])[F3]/255,str2list(X[i])[F4]/255,str2list(X[i])[F5]/255,str2list(X[i])[F6]/10000] for i in val_index])
        y_test = np.array(y[val_index])
    
        classifier.fit(X_train,y_train)
        

        try :#SVM this
            y_score = classifier.decision_function(X_test) #decision_fn
        except : #KNN this
            y_score = classifier.predict_proba(X_test)

        
        print("{} Fold no. {} ".format(CLASSIFIER_NAME,fold))

        #Save model
        joblib.dump(classifier,CLASSIFIER_NAME+'_fold'+str(fold)+'.pkl')

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
        sen_h = tn_h/(tn_h+fp_h)
        acc_h = (tp_h+tn_h)/(tp_h+tn_h+fp_h+fn_h)

        prec_g = tp_g/(tp_g+fp_g)
        sen_g = tn_g/(tn_g+fp_g)
        acc_g = (tp_g+tn_g)/(tp_g+tn_g+fp_g+fn_g)

        prec_o = tp_o/(tp_o+fp_o)
        sen_o = tn_o/(tn_o+fp_o)
        acc_o = (tp_o+tn_o)/(tp_o+tn_o+fp_o+fn_o)

        cmtable = [{'TP':tp_h,'TN' : tn_h,'FP' : fp_h,'FN' : fn_h,'pre':prec_h,'sen':sen_h,'acc' : acc_h},
                    {'TP':tp_g,'TN' : tn_g,'FP' : fp_g,'FN' : fn_g,'pre':prec_g,'sen':sen_g,'acc' : acc_g},
                    {'TP':tp_o,'TN' : tn_o,'FP' : fp_o,'FN' : fn_o,'pre':prec_o,'sen':sen_o,'acc' : acc_o}]

        table_df = pd.DataFrame(cmtable)
        table_df.to_csv('PredictvsTrue_fold{}.csv'.format(fold))

        print("Precision on H :",prec_h)
        print("Sensitivity on H :",sen_h)
        print("Accuracy on H :",acc_h)

        print("Precision on G :",prec_g)
        print("Sensitivity on G :",sen_g)
        print("Accuracy on G :",acc_g)

        print("Precision on O :",prec_o)
        print("Sensitivity on O :",sen_o)
        print("Accuracy on O :",acc_o)
        
        joblib.dump(cmtable,CLASSIFIER_NAME+'_fold'+str(fold)+'_table'+'.pkl')
        
        

        report = classification_report(y_test, y_pred, target_names=class_names)
        joblib.dump(report,CLASSIFIER_NAME+'_fold'+str(fold)+'_report'+'.pkl')
        print(report)



        #ROC 1vsR
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot of a ROC curve for a specific class
        interp_tpr = np.interp(mean_fpr,fpr[2],tpr[2])
        interp_tpr[0] = 0.0

        tprs.append(interp_tpr)
        aucs.append(roc_auc[2])
        plt.figure()
        plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.4f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic fold {}'.format(fold))
        plt.legend(loc="lower right")

        plt.savefig("roc_auc_fold"+str(fold)+".jpg")

        area_under_curve = roc_auc[2]
        print("Area Under Curve :",area_under_curve)
        
        
        
        print("----------------------------------------------------------")
    
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ %0.4f auc. std.' %(std_auc))

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver operating characteristic 5 folds",xlabel ='False Positive Rate',ylabel = 'True Positive Rate')
    
    ax.legend(loc="lower right")
    plt.savefig("roc_auc_allfold"+str(fold)+".jpg")
    plt.show()





def str2list (lstring) :
    nlist = ast.literal_eval(lstring)
    return nlist


if __name__ == "__main__":
    main()


