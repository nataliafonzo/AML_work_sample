"""
Exercise description
--------------------

Description:
In the context of Mercadolibre's Marketplace an algorithm is needed to predict if an item listed in the markeplace is new or used.

Your tasks involve the data analysis, designing, processing and modeling of a machine learning solution 
to predict if an item is new or used and then evaluate the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k.jsonlines` and a function to read that dataset in `build_dataset`.

For the evaluation, you will use the accuracy metric in order to get a result of 0.86 as minimum. 
Additionally, you will have to choose an appropiate secondary metric and also elaborate an argument on why that metric was chosen.

The deliverables are:
--The file, including all the code needed to define and evaluate a model.
--A document with an explanation on the criteria applied to choose the features, 
  the proposed secondary metric and the performance achieved on that metrics. 
  Optionally, you can deliver an EDA analysis with other formart like .ipynb

"""

import json
import time

from feature_extraction import clean_nulls, extract_features
from classifier import ClassificationPredictor, train

from sklearn.metrics import accuracy_score, precision_score, recall_score


def build_dataset():
    data = [json.loads(x) for x in open("MLA_100k.jsonlines")]
    target = lambda x: x.get("condition")
    N = -10000
    X_train = data[:N]
    X_test = data[N:]
    y_train = [target(x) for x in X_train]
    y_test = [target(x) for x in X_test]
    for x in X_test:
        del x["condition"]
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    start = time.time()
    print("Loading dataset...")
    X_train, y_train, X_test, y_test = build_dataset()
    end = time.time()
    print("{} seconds to load datasets.".format(round(end - start,2)))
    
    start = time.time()
    print("Extracting features. Expect this to take around 9 minutes...")
    X_train = extract_features(clean_nulls(X_train))
    X_test = extract_features(clean_nulls(X_test))
    end = time.time()
    print("{} seconds to extract features.".format(round(end - start,2)))
    
    model = ClassificationPredictor()
    print("Initiating data pre-processing and training...")
    model = train(model,X_train,y_train)
    
    print("Training done. Initiating performance evaluation...")
    y_pred = model.predict(X_test)
    accuracy = round(accuracy_score(y_pred,y_test),2)
    precision = round(precision_score(y_pred,y_test,pos_label="used"),2)
    recall = round(recall_score(y_pred,y_test,pos_label="used"),2)
    print("""\nPerformance in test set \n\nAccuracy = {} \nPrecision = {} \nRecall = {}""".format(accuracy,precision,recall))
