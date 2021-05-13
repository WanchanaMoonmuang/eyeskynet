import tensorflow as tf
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
from pprint import pprint
import os

#raw train img 1080,720, raw test img 1080,720

square = 224
RES = (square,square) #B3 300,300 dense 224,224
input_shape = (square,square,3)
img_height = RES[1]
img_width = RES[0]
class_names = ['healthy', 'glaucoma', 'others']
num_classes = len(class_names)
batch_size = 16 # b0 use 32 else 16
CLASSIFIER_NAME = 'B0augmentedzoom30_'
epochs = 30

def main() :
    PATH ="D:\Downloads\Output/DL_wzoom"
    print("Loading images from : "+PATH)
    
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

    
    
    histories=[]

    for fold in range(5) :

        fold +=1
        print("Load trainning set")
        trainval_ds = tf.keras.preprocessing.image_dataset_from_directory(
        PATH,shuffle=True,
        image_size= RES ,labels='inferred', label_mode='int',
        class_names=class_names, color_mode='rgb',
        batch_size=batch_size)

        train_ds , val_ds = augment_concate(trainval_ds)
        print("+++++++++++++++++++++++++++++ Fold %d ++++++++++++++++++++++++++++" % fold)

        model = create_model()

        checkpoint_dir = "model/f%d" % fold
        checkpoint_path = checkpoint_dir + "/cp-{epoch:04d}.ckpt"
    
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path, 
                verbose=0, 
                save_weights_only=True)
        tb_callback = tf.keras.callbacks.TensorBoard(
                log_dir="./logs/f%d" % fold,
                update_freq="epoch") 

        history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[cp_callback,tb_callback]
        )
        # probability_model = tf.keras.Sequential([
        #     model,tf.keras.layers.Softmax()
        #     ])
        print(model.summary())
        model.save_weights(checkpoint_dir)

        plot_acc(history,epochs,fold)

        histories.append(history.history)
        
        #Test with test sets

        print("============= Fold %d ==================" % fold)
        test_loss, test_acc = model.evaluate(test_ds, verbose=2)
        print('\nTest loss:', test_loss)
        print('\nTest accuracy:', test_acc)

        f = open("test%d.txt" %fold, "w")
        f.write('Test loss: {} \nTest accuracy: {}'.format(test_loss,test_acc))
        f.close()
        print("========================================")
        
    pprint(histories)

def augment_concate(dataset,split_data = True) :
    #loop batch
    images = list()
    labels = list()
    for img_batch,label_batch in dataset :
        for i in range(len(img_batch)) :
            images.append(img_batch[i].numpy().astype("uint8"))
            labels.append(label_batch[i].numpy().astype("uint8"))
    images = np.array(images)
    labels = np.array(labels)
    length = len(images)

    augment = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")
                              ,tf.keras.layers.experimental.preprocessing.RandomRotation(0.2,fill_mode = 'constant')]) #,tf.keras.layers.experimental.preprocessing.RandomRotation(0.2,fill_mode = 'constant')

    if split_data :
        train_img = images[:(round(length*0.8))]
        train_label = labels[:(round(length*0.8))]
    
        val_img = images[(round(length*0.8)):]
        val_label = labels[(round(length*0.8)):]
        print("Validation set :",len(val_img))
        val_ds = tf.data.Dataset.from_tensor_slices((val_img,val_label)).batch(batch_size)
        augmented = augment(train_img)
        new_train = np.concatenate((train_img,augmented), axis=0)
        new_train_label = np.concatenate((train_label,train_label),axis=0)
        print("Train after augmented :",len(new_train))
        train_ds = tf.data.Dataset.from_tensor_slices((new_train,new_train_label))
        train_ds = train_ds.shuffle(len(new_train), reshuffle_each_iteration=True).batch(batch_size)
        return train_ds,val_ds

    else :
        train_img = images
        train_label = labels
        augmented = augment(train_img)
        new_train = np.concatenate((train_img,augmented), axis=0)
        new_train_label = np.concatenate((train_label,train_label),axis=0)
        print("Train after augmented :",len(new_train))
        train_ds = tf.data.Dataset.from_tensor_slices((new_train,new_train_label))
        train_ds = train_ds.shuffle(len(new_train), reshuffle_each_iteration=True).batch(batch_size)
        
        return train_ds
    
    
    
    
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

def show_ds(train_ds,val_ds) :
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")
    plt.show()           
    plt.figure(figsize=(10, 10))
    for images, labels in val_ds.take(1):
        for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")
    plt.show()

def plot_acc(history,epochs,fold) :
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Training and Validation Accuracy : Fold '+str(fold))

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Training and Validation Loss : Fold '+str(fold))
    plt.savefig("plot_acc"+str(fold)+".jpg")



#Change dir
# def load_filename_from_folder(folder):
#     return [filename for filename in os.listdir(folder)]
# #load image from folder as np array

# def load_images_from_folder(folder):
#     images = []
#     for filename in os.listdir(folder):
#         img = cv.imread(os.path.join(folder, filename))
#         if img is not None:
#             images.append(img)
#     return images
if __name__ == "__main__":
    main()
