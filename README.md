# Glaucoma screening project Eyeskynet propose CN240
# Part 1 Machine learning
Feature extraction :

1. Average RGB value => Average_bgr.py

2. CDR calculation => CDR.py

3. Number of blood pixels => CDR.py @function ex_blood

4. Most frequent intensity of RGB => freq_hist.py , mostf.py

Trianning Part :
Algorithms tested DecisionTree , DecisionTree with Adaboost , KNN , SVM ,RandomForest

Result : RandomForest best for our dataset and features.

The most ROC & lowest STD.
![roc_auc_allfold4](https://user-images.githubusercontent.com/60337642/118156803-fe77ba80-b443-11eb-8715-452afce21a71.jpg)

Extract all feature data with => main.py

if run separately use => Mixcsv.py to compress data into single file

Train model with 5 fold cross validation => model_tranning.py

Evaluate with test set with => main.py

# Part 2 Deep learning
Algorithm tested DenseNet201 , EfficientNetB3 , EfficientNetB0

Result : EfficientNetB3 Most Accuracy but x3 more resources used and bigger model.

![roc_auc_fold50](https://user-images.githubusercontent.com/60337642/118156020-10a52900-b443-11eb-845c-17526daf5b42.jpg)

We use EfficientNetB0 with Augmented data result in light weight model.

![roc_auc_fold29](https://user-images.githubusercontent.com/60337642/118155930-f703e180-b442-11eb-8f9d-7d073ae8cf80.jpg)

Steps to create a model.

1. Preprocessing images ROI => dl_imgprepro.py

2. Augmentation generate zoom images => generate_zoom.py

3. Train model with Tensorflow => dl_model_train.py

4. Evaluate model with => dl_eval.py

*dl_load_train.py to further more trainning
