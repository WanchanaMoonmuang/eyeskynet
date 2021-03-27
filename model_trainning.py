
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

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
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
CLASSIFIER_NAME = 'SVMLinear ovr'  #'SVMLinear'
df_shape = 'ovo' #ovo / ovr



def main():
    
    seconds = time.time()
    local_time = time.ctime(seconds)
    local_time = local_time[11:13]+local_time[14:16]+local_time[17:19]
    Out_folder = DIR+"\Model"+local_time
    

    print("Loading file {} at {}".format(FILENAME,DIR))
    os.chdir(DIR)
    data = pd.read_csv(FILENAME)
    X = data['data'].values #List of stringlist
    
    y = data['target'].values
    y = label_binarize(y, classes=[0,1,2])
    n_classes = y.shape[1]
    
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
    clf = SVC(kernel='linear',probability=True) #decision_function_shape = 'ovo'


    classifier = OneVsRestClassifier(clf) #ovr or ovo

    

    for fold,(train_index, val_index) in enumerate(kf.split(X)) :
        
        print("------------------------------------------STARTING FOLD no. :",fold)
        X_train = np.array([[str2list(X[i])[F1],str2list(X[i])[F2]/255,str2list(X[i])[F3]/255,str2list(X[i])[F4]/255,str2list(X[i])[F5]/255,str2list(X[i])[F6]/10000] for i in train_index])
        y_train = np.array(y[train_index])
        
        X_test = np.array([[str2list(X[i])[F1],str2list(X[i])[F2]/255,str2list(X[i])[F3]/255,str2list(X[i])[F4]/255,str2list(X[i])[F5]/255,str2list(X[i])[F6]/10000] for i in val_index])
        y_test = np.array(y[val_index])
    


        classifier.fit(X_train,y_train)
        

        try :
            y_score = classifier.decision_function(X_test) #decision_fn
        except :
            plot_roc = False

        
        print("{} Fold no. {} ".format(CLASSIFIER_NAME,fold))

        #Save model
        joblib.dump(classifier,CLASSIFIER_NAME+'_fold'+str(fold)+'.pkl')

        
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
    
    

def str2list (lstring) :
    nlist = ast.literal_eval(lstring)
    return nlist


if __name__ == "__main__":
    main()


