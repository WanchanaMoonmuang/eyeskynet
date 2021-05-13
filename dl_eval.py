import tensorflow as tf
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

import joblib
import pandas as pd
from sklearn.metrics import auc
from sklearn.metrics import roc_curve


#raw input is 1080,720
#B3 input should 300,300 dense 224
sq = 224
input_shape = (sq,sq,3)
RES = (sq,sq)
batch_size = 1 #32
#f2_78 = 82.7%
fold = 29 #dense36 b3e50 b0 f2/29
class_names = ['healthy', 'glaucoma', 'others']

def main() :
    PATH = "D:\Downloads\Output\DL_model_eval/B0f2epoch29"
    test_PATH =  "D:\Downloads\Output\DL_wzoom_test" #'D:\Downloads\Output\DLTEST'
    
    os.chdir(PATH) #D:\Downloads\Output\DL_model\ModelB3augmentedzoom60_000247\model\f1
    model_PATH = 'D:\Downloads\Output\DL_model\ModelB0augmentedzoom30_151929\model/f2'#'D:\Downloads\Output\Deeplearning_model\ModelEnetB3augmentedfix60_172511\model/f1'
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
         test_PATH,shuffle=False,
         image_size= RES,labels='inferred',label_mode='int',
         class_names=class_names,color_mode='rgb',
         batch_size=batch_size)

    test_images = list()
    test_labels = np.array([])
    for image_batch, labels_batch in test_ds:
        # img = image_batch.numpy()
        label = labels_batch.numpy()
        
        test_labels = np.append(test_labels,label)
        # test_images.append(img)
    
    test_labels = np.array(test_labels.astype(int),)
    
    print("labels :")
    print(test_labels)
    

    # temp = list()
    # for i in test_labels :
    #     if i == 0 :
    #         temp.append(np.array([1,0,0]))
    #     elif i == 1 :
    #         temp.append(np.array([0,1,0]))
    #     else :
    #         temp.append(np.array([0,0,1]))
    # test_labels = np.array(temp)
    
    


    #os.chdir(model_PATH)
    # nopimg = cv.imread("315_right.jpg")
    # pimg = cv.imread("387_left.jpg")

    model = create_model_B0()
    model.load_weights(model_PATH+'\cp-00'+str(fold)+'.ckpt')
    loss, acc = model.evaluate(test_ds, verbose=2)
    print('\nTest loss:', loss)
    print('\nTest accuracy:', acc)

    f = open("test%d.txt" %fold, "w")
    f.write('Test loss: {} \nTest accuracy: {}'.format(loss,acc))
    f.close()
    # img = tf.keras.preprocessing.image.load_img(
    # test_PATH, target_size=(RES))
    # img_array = tf.keras.preprocessing.image.img_to_array(img)
    # img_array = tf.expand_dims(img_array, 0) 
    
    prediction = model.predict(test_ds,batch_size=batch_size)
    
    score = tf.nn.softmax(prediction)
    y_pred_t = np.argmax(score,axis=1)
    print("Predicted :")
    print(y_pred_t)
    

    cm = tf.math.confusion_matrix(test_labels, y_pred_t)
    
    # score = tf.nn.softmax(prediction[0])
    # print(class_names[np.argmax(score)],100*np.max(score))
    

    n_classes = len(class_names)
    npcm = cm.numpy()
    print(npcm)
    table_list = npcm.ravel()
    #Confusion matrix TP TN FP FN
        
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
    raw_table = pd.DataFrame(npcm)
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
        

    y_score = score.numpy()
    
    temp = list()
    for i in test_labels :
        if i == 0 :
            temp.append(np.array([1,0,0]))
        elif i == 1 :
            temp.append(np.array([0,1,0]))
        else :
            temp.append(np.array([0,0,1]))
    test_bin = np.array(temp)
    y_test = test_bin
    
        #ROC 1vsR
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_score[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot of a ROC curve for a specific class
    mean_fpr = np.linspace(0, 1, 100)
    interp_tpr = np.interp(mean_fpr,fpr[2],tpr[2])
    interp_tpr[0] = 0.0

        
    plt.figure()
    plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.4f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic epoch {}'.format(fold))
    plt.legend(loc="lower right")

    plt.savefig("roc_auc_fold"+str(fold)+".jpg")

    area_under_curve = roc_auc[2]
    print("Area Under Curve :",area_under_curve)
        
def create_model_B3():
    model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255,input_shape=input_shape),
        tf.keras.applications.EfficientNetB3(
        include_top=True, weights= None, input_tensor=None,
        pooling=None, classes=len(class_names),
        classifier_activation='softmax')
        # tf.keras.applications.DenseNet201(
        # include_top=True, weights=None, input_tensor=None,
        # input_shape=None, pooling=None, classes=len(class_names))
    ])
    
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return model

def create_model_B0():
    model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255,input_shape=input_shape),
        tf.keras.applications.EfficientNetB0(
        include_top=True, weights= None, input_tensor=None,
        pooling=None, classes=len(class_names),
        classifier_activation='softmax')
        # tf.keras.applications.DenseNet201(
        # include_top=True, weights=None, input_tensor=None,
        # input_shape=None, pooling=None, classes=len(class_names))
    ])
    
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return model



if __name__ == "__main__":
    main()
