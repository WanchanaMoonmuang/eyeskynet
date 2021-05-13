import tensorflow as tf
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
import os

#from dl_model_train import create_model
from dl_model_train import augment_concate


square = 224
RES = (square,square) #B3 300,300 dense 224,224
input_shape = (square,square,3)
img_height = RES[1]
img_width = RES[0]
class_names = ['healthy', 'glaucoma', 'others']
num_classes = len(class_names)
batch_size = 16 # b0 use 32 else 16
CLASSIFIER_NAME = 'B0augmentedloadtrainzoomm30_'
more_epochs = 30
current_epochs = 30 #load epoch

def main() :
    continue_epoch = current_epochs + 1
    model_PATH = 'D:\Downloads\Output\DL_model\ModelB0augmentedzoom30_151929\model/f4'#"D:\Downloads\eyeskynet\Deeplearning_model\ModelEnetB3augmentedfix60_172511\model/f1" #'D:\Downloads\Output\DL_model\ModelB3augmentedloadzoom60from90_113503\model\e91'#

    data_PATH ="D:\Downloads\Output\zoom_train"
    print("Loading train images from : "+data_PATH)

    val_PATH = 'D:\Downloads\Output\DL_wzoom'
    print("Loading validate images from : "+val_PATH)
    
    SAVETO ="D:\Downloads\Output\DL_model"
    seconds = time.time()
    local_time = time.ctime(seconds)
    local_time = local_time[11:13]+local_time[14:16]+local_time[17:19]
    Out_folder = SAVETO+"\Model"+CLASSIFIER_NAME+local_time

    test_PATH = "D:\Downloads\Output\DL_wzoom_test"

    try :
        os.mkdir(Out_folder)
        print("Create OUTPUT folder")
    except OSError :
        print("Fail to create OUTPUT folder or Already Exist")
    os.chdir(Out_folder)
    print("Change dir to",Out_folder)
    print("Load test set")

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_PATH,shuffle=True,
        image_size= RES ,labels='inferred', label_mode='int',
        class_names=class_names, color_mode='rgb',
        batch_size=batch_size)

    model = create_model()
    model.load_weights(model_PATH+"/cp-00"+str(current_epochs)+".ckpt") #adjust 00 or 0

    
    
    histories=[]
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                    data_PATH,shuffle=True,
                    image_size= RES ,labels='inferred', label_mode='int',
                    class_names=class_names, color_mode='rgb',
                    batch_size=batch_size)
    # val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #                 val_PATH,shuffle=True,validation_split=0.2,subset='validation',
    #                 image_size= RES ,labels='inferred', label_mode='int',seed = int(local_time),
    #                 class_names=class_names, color_mode='rgb',
    #                 batch_size=batch_size)

    train_ds,val_ds = augment_concate(train_ds,split_data=True)
    print("+++++++++++++++++++++++++++++ continue epoch %d ++++++++++++++++++++++++++++" % continue_epoch)

    checkpoint_dir = "model/e%d" % continue_epoch
    checkpoint_path = checkpoint_dir + "/cp-{epoch:04d}.ckpt"
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path, 
                verbose=0, 
                save_weights_only=True)
    tb_callback = tf.keras.callbacks.TensorBoard(
                log_dir="./logs/e%d" % continue_epoch,
                update_freq="epoch") 

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=current_epochs+more_epochs,initial_epoch=current_epochs,
        callbacks=[cp_callback,tb_callback]
        )
        # probability_model = tf.keras.Sequential([
        #     model,tf.keras.layers.Softmax()
        #     ])
    print(model.summary())
    model.save_weights(checkpoint_dir)

    histories.append(history.history)
        
        #Test with test sets

        
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    print('\nTest loss:', test_loss)
    print('\nTest accuracy:', test_acc)

    f = open("test%d.txt" %continue_epoch, "w")
    f.write('Test loss: {} \nTest accuracy: {}'.format(test_loss,test_acc))
    f.close()
    print("========================================")

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255,input_shape=input_shape),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2,fill_mode = 'constant'),
        tf.keras.applications.EfficientNetB0(
        include_top=True, weights= None, input_tensor=None,
        pooling=None, classes=num_classes,
        classifier_activation='softmax')
        # tf.keras.applications.DenseNet201(
        # include_top=True, weights=None, input_tensor=None,
        # input_shape=None, pooling=None, classes=num_classes)
    ])
    
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return model
if __name__ == "__main__":
    main()