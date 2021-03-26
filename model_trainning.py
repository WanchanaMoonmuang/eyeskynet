
#Import classifier here
from sklearn.neighbors import KNeighborsClassifier #KNeighborsClassifier(n_neighbors=3)
from sklearn.svm import SVC #SVC(kernel='linear') 'rbf'
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

#Data = [cdr,avg,mostb,mostg,mostr,bloodpx]
F1 = 0 #CDR always use
F2 = 1  #avg bgr
F3 = 3  #mostg

TITLE = "Healthy & Others"
FILENAME = 'HO_ds.csv' #.csv
DIR = 'D:\Downloads\eyeskynet\Output'
CLASSIFIER_NAME = 'SVMLinear'
n_classes = 2

def main():
    plot_roc=True

    print("Loading file {} at {}".format(FILENAME,DIR))
    os.chdir(DIR)
    data = pd.read_csv(FILENAME)
    data_set, test_set = train_test_split(data, test_size = 0.1)
    kf = KFold(n_splits=5)
    print("Classifier :",CLASSIFIER_NAME)
    print("--- Feature used ---")
    fname = str2list(data_set.iloc[0][3])
    print(fname[F1])
    print(fname[F2])
    print(fname[F3])

    #Classifier declare
    classifier = SVC(kernel='linear')
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()

    for fold,(train_index, val_index) in enumerate(kf.split(data_set)) :
        
        print("------------------------------------------STARTING FOLD no. :",fold)
        train_ds = [data_set.iloc[i] for i in train_index]
        val_ds = [data_set.iloc[i] for i in val_index]
    
        X = np.array([[str2list(i[0])[F1],str2list(i[0])[F2],str2list(i[0])[F3]] for i in train_ds]).astype(np.float)
        y = np.array([i[1] for i in train_ds])

        X_test = np.array([[str2list(i[0])[F1],str2list(i[0])[F2],str2list(i[0])[F3]] for i in val_ds]).astype(np.float) #[str2list(i[0]) for i in val_ds]
        y_test = np.array([i[1] for i in val_ds])
    
        classifier.fit(X,y)
        score = classifier.score(X_test, y_test)
        
        try :
            y_score = classifier.decision_function(X_test) #decision_fn
        except :
            plot_roc = False

        
        print("{} Fold no. {} Score : {}".format(CLASSIFIER_NAME,fold,score))

        #Save model
        joblib.dump(classifier,CLASSIFIER_NAME+'_fold'+str(fold)+'.pkl')
        preds = classifier.predict(X_test)

        # lr = LogisticRegression()
        # lr.fit(X,y)
        # preds = lr.predict(X_test)
        # probas = lr.predict_proba(X_test)[:, 1] #lr.predict_proba(X_test)[:, 1]

        #table = pd.DataFrame(confusion_matrix(y_test, preds), columns=['Predicted Healthy', "Predicted Glaucoma","Predicted Others"], index=['Actual Healthy', 'Actual Glaucoma' ,'Actual Others'])
        #table.to_csv(CLASSIFIER_NAME+'_fold'+str(fold)+'.csv')
        cm =confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        print(f'True Positives: {tp}')
        print(f'False Positives: {fp}')
        print(f'True Negatives: {tn}')
        print(f'False Negatives: {fn}')

        
        #plot cm 
        plt.clf()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
        classNames = ['Negative','Positive']
        plt.title(TITLE)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        s = [['TN','FP'], ['FN', 'TP']]
        for i in range(2):
            for j in range(2):
                plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
        plt.savefig("valtable_fold"+str(fold)+".jpg")
        plt.show()

        Recall = tp / (tp + fn)
        Precision = tp / (tp + fn)
        f_score = (2 * Recall * Precision) / (Recall + Precision)
        print("F-SCORE :",f_score)
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        plot_roc_curve(fold,fpr,tpr,roc_auc)
        
        ra_score = roc_auc_score(y_test, preds)
        print("ROC AUC SCORE :", ra_score)

        
        print("----------------------------------------------------------")
    
        

def str2list (lstring) :
    nlist = ast.literal_eval(lstring)
    return nlist

def plot_roc_curve(fold,fpr, tpr,roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
    lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    plt.savefig("rocauc_fold"+str(fold)+".jpg")
    plt.show()


if __name__ == "__main__":
    main()


