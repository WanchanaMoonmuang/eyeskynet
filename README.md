# Glaucoma screening project Eyeskynet propose CN240
# Part 1 Machine learning
Feature extraction :
1. Average RGB value => Average_bgr.py
2.CDR calculation => CDR.py
3.Number of blood pixels => CDR.py @function ex_blood
4.Most frequent intensity of RGB => freq_hist.py , mostf.py
Extract all feature data with => main.py
Train model with 5 fold cross validation => model_tranning.py
