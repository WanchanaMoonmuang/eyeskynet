from fastapi import FastAPI, UploadFile, Form, File

import cv2 as cv
import numpy as np

import tensorflow as tf

from CDR import *
from Average_bgr import *
from mostf import *
import joblib
from main import ex_test

from dl_imgprepro import prepro
from dl_eval import create_model_B0
from dl_eval import create_model_B3

CTEST = 'DL'

DL_PATH = 'D:\Downloads\Output\DL_model\ModelB0augmentedzoom30_151929\model/f2/cp-0029.ckpt'#'D:\Downloads\Output\Deeplearning_model\ModelEnetB3augmentedfix60_172511\model/f1'
ML_PATH = 'D:\Downloads\Output\ML_Output\ModelRandomForestFinal/RandomForest_fold4.pkl'
class_names = ["normal", "glaucoma", "other"]
sq = 224
RES = (sq,sq)
ml_model = joblib.load(ML_PATH)
dl_model = create_model_B0()
dl_model.load_weights(DL_PATH)

def predict_ml(bgr_img):

    ml_data = ex_test(bgr_img)
    ml_prob = ml_model.predict_proba(np.array([ml_data],))
    pred_class = np.argmax(ml_prob)
    prob_class = ml_prob[0][pred_class]/np.sum(ml_prob)
    print(ml_prob)
    ml_message = 'Machine learning classify as : "{}" with proability {}%'.format(class_names[pred_class],prob_class)
    print(ml_message)

    return class_names[pred_class] , prob_class

def predict_dl(bgr_img):
    rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
    dl_pred = dl_model.predict(np.array([rgb_img],))
    score = np.array(tf.nn.softmax(dl_pred))
    print(score)
    dl_message = 'Deep learning classify as : "{}" with proability {}%'.format(class_names[np.argmax(score)],100*np.max(score))
    print(dl_message)

    return class_names[np.argmax(score)] , 100*np.max(score)


app = FastAPI()

@app.get("/")
async def helloworld():
    return {"Glaucoma screening": "Eyeskynet"}


@app.post("/api/fundus")
async def upload_image(nonce: str=Form(None, title="Query Text"), 
                       image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    
    
    
    if CTEST == 'DL':
        try:
            bgr_img = prepro(img,RES)
        except :
            bgr_img = cv.resize(img,RES, interpolation = cv.INTER_AREA)
        print("Image shape after prepro",bgr_img.shape)
        class_out, class_conf =predict_dl(bgr_img)
        
    elif CTEST == 'ML':
        class_out,class_conf = predict_ml(img)
    
    return {
        "nonce": nonce,
        "classification": class_out,
        "confidence_score": np.float(class_conf),
        "debug": {
            "image_size": dict(zip(["height", "width", "channels"], img.shape)),
        }
    }
