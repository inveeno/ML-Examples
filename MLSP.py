###### Machine Learning Example
###### Decision Tree Classifier Model
__author__ = 'MOE ISSA'

import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse

def main():
    parser = argparse.ArgumentParser(description='Machine Learning Example for Decision Tree Classifier Model')
    parser.add_argument('-f','--file', help='Csv File',required=True)
    parser.add_argument('-l','--label',help='Label Column', required=True)
    args = parser.parse_args()
    file_name = str(args.file)
    label_name = str(args.label)

    try:
        df = pd.read_csv(file_name)
    except Exception, e:
        print ("Error in reading : %s" % args.file )
        print e
        return 0

    le = {}
    for col in df.columns:
    	le[col] = preprocessing.LabelEncoder()
        df[col] = le[col].fit_transform(df[col])
    
    labels = df[label_name].values
    
    features = df[[x for x in df.columns if label_name not in x]]

    train, test = train_test_split(df, test_size = 0.2) # split 20% of the datafile into Test dataset

    train_labels = train[label_name].values
    train_features = train[[x for x in train.columns if label_name not in x]]

    test_labels = test[label_name].values
    test_features = test[[x for x in test.columns if label_name not in x]]

    clf = tree.DecisionTreeClassifier()
    clf.fit(train_features,train_labels)

    print(le['class'].inverse_transform(clf.predict([[0,2,8,1,3,1,0,0,5,0,2,2,2,7,7,0,2,1,4,3,2,3]])))
    
    print("prediction Accuracy = ")
    print(str(accuracy_score(test_labels, clf.predict(test_features))))

if __name__ == '__main__':
    main()