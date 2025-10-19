## import libraries
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import gzip
import json
from datetime import datetime
import dateutil.parser
import random

def getMaxLen(dataset):
    # Find the longest review (number of characters)
    review_lens = [len(i['review_text']) for i in dataset]
    maxLen = max(review_lens)
    return maxLen
    
def featureQ1(datum, maxLen):
    # Feature vector for one data point
    feature = len(datum['review_text'])/maxLen
    return [1] + [feature]
    
def Q1(dataset):
    # get max len of 'review_text'
    maxLen = getMaxLen(dataset)

    # build feature(X) and target(y) lists
    X = [featureQ1(d,maxLen) for d in dataset]
    y = [d['rating'] for d in dataset]

    # calc least squares
    theta,residuals,rank,s = np.linalg.lstsq(X, y)

    # calc mean squared error
    MSE = residuals[0]/len(y)

    return theta, MSE

def featureQ2(datum, maxLen):
    # note: should be 1, length feature, day feature (list), month feature (list)
    # normailize length feature
    len_feature = len(datum['review_text'])/maxLen
    date_dt = datum['parsed_date']

    # weekday returns Mon: 0,...,Sun:6
    day = date_dt.weekday()

    # initalize day feature 
    day_feature = [0,0,0,0,0,0,0]
    
    # day is index and mark as true  for element reprsenting day of week
    day_feature[day] = 1

    # month returns Jan:1,...,Dec:12 (must account for index start at zero)
    month = date_dt.month

    # initalize month feature then insert into list
    month_feature = [0,0,0,0,0,0,0,0,0,0,0,0]

    # day is index and mark as true  for element reprsenting day of week
    day_feature[day] = 1
    
    # since counting starts at 0 Jan:0,...,Dec:11
    month_feature[month-1] = 1

    # build Q2 feature 
    # note: drop first element for month and day feature to prevent multicollinearity
    feature = np.concatenate((np.array([1]),np.array([len_feature]), np.array(day_feature[1:]), np.array(month_feature[1:])))

    return feature

def Q2(dataset):
    # get max len of 'review_text'
    maxLen = getMaxLen(dataset)

    # build features(X) and target(y) lists
    X = [featureQ2(d,maxLen) for d in dataset]
    y = [d['rating'] for d in dataset]
    
    # calc least squares
    theta,residuals,rank,s = np.linalg.lstsq(X, y)

    # calc mean squared error
    MSE = residuals/len(y)
    MSE2 = MSE[0]

    return X, y, MSE2

def featureQ3(datum, maxLen):
    # note: should be 1, length feature, day feature (int), month feature (int)
    # normailize length feature
    len_feature = len(datum['review_text'])/maxLen
    date_dt = datum['parsed_date']

    # weekday returns Mon: 0,...,Sun:6
    day = date_dt.weekday()

    # month returns Jan:1,...,Dec:12 (must account for index start at zero)
    month = date_dt.month

    # build Q2 feature
    feature = np.concatenate((np.array([1]),np.array([len_feature]), np.array([day]), np.array([month])))

    return feature

def Q3(dataset):
    # note: MSE should be a *number*, not e.g. an array of length 1
    # get max len of 'review_text'
    maxLen = getMaxLen(dataset)

    # build features(X) and target(y) lists
    X = [featureQ3(d,maxLen) for d in dataset]
    y = [d['rating'] for d in dataset]

    # calc least squares
    theta,residuals,rank,s = np.linalg.lstsq(X, y)

    #calc mean squared error
    MSE = residuals/len(y)
    MSE3 = MSE[0]
    
    return X, y, MSE3

def Q4(dataset):
    # get max len of 'review_text'
    maxLen = getMaxLen(dataset)

    # 50/50 split for train and test data
    half_way = len(dataset) // 2

    # train_data: 0 to half_way-1 and test_data: half_way to len(dataset)
    dataset_train = dataset[:half_way]
    dataset_test = dataset[half_way:] 

    # train/test for Q2 features
    X2_train = np.array([featureQ2(d, maxLen) for d in dataset_train])
    X2_test = np.array([featureQ2(d, maxLen) for d in dataset_test])
    
    # train/test for Q3 features
    X3_train = np.array([featureQ3(d, maxLen) for d in dataset_train])
    X3_test = np.array([featureQ3(d, maxLen) for d in dataset_test])
    
    # train/test for targets
    y_train = np.array([d['rating'] for d in dataset_train])
    y_test = np.array([d['rating'] for d in dataset_test])
    
    # initalize sklearn linear regression model
    lr = LinearRegression(fit_intercept=False)

    # fit q2 training data
    lr.fit(X2_train, y_train)
    
    # predict targets for test data
    y_pred = lr.predict(X2_test)

    # calc mean squared error
    test_mse2 = np.mean((y_test - y_pred)**2)

    # fit q3 training data
    lr.fit(X3_train, y_train)
    
    # predict targets for test data
    y_pred = lr.predict(X3_test)

    # calc mean squared error
    test_mse3 = np.mean((y_test - y_pred)**2)
    
    return test_mse2, test_mse3

def featureQ5(datum):
    # Feature vector for one data point
    feature = len(datum['review/text'])
    return [1] + [feature]

def Q5(dataset,feat_func):

    # build feature(X) and target(y) lists
    X = np.array([feat_func(d) for d in dataset])
    y = np.array([d['review/overall'] >=4 for d in dataset])

    # initalize sklearn logistic regression model
    log_reg = LogisticRegression(class_weight='balanced')

    # fit logistic regression model
    log_reg.fit(X,y)

    # make predictions using log_reg model
    y_pred = log_reg.predict(X)

    # define true/false conditions for data/predictions True:1 and False:0
    y_true = y.astype(int) 
    y_pred_int = y_pred.astype(int)

    # find true positives
    true_positives = np.sum(y_true * y_pred_int)

    #find true negatives
    true_negatives = np.sum((1 - y_true) * (1 - y_pred_int))

    # find false positives
    false_positives = np.sum((1 - y_true) * y_pred_int)

    # find false negatives
    false_negatives = np.sum(y_true * (1 - y_pred_int))

    # Calculate balanced error rate (1-accuracy)
    bal_accuracy = balanced_accuracy_score(y, y_pred)
    bal_error_rate = 1 - bal_accuracy
    return true_positives, true_negatives, false_positives, false_negatives, bal_error_rate

def Q6(dataset):
    def featureQ5(datum):
        # Feature vector for one data point
        feature = len(datum['review/text'])
        return [1] + [feature]

    # build feature(X) and target(y) lists
    X = np.array([featureQ5(d) for d in dataset])
    y = np.array([d['review/overall'] >=4 for d in dataset])

    # initalize sklearn logistic regression model
    log_reg = LogisticRegression(class_weight='balanced')

    # fit logistic regression model
    log_reg.fit(X,y)

    # grab col 2 probablities since intrested where 'review/overall' >=4
    probs = log_reg.predict_proba(X)[:, 1]

    # sort in decending order based on probability from log_reg
    sorted_indices = np.argsort(probs)[::-1]
    
    # sort true targets based on probs from log_reg
    sorted_y = y[sorted_indices]

    # define k values to test
    k_vals = np.array([1,100,1000,10000])
    
    # calc sum of correct predictions 
    true_counts = np.cumsum(sorted_y)
    
    # calc precision for each k using numpy linear algebra
    precs = true_counts[k_vals - 1] / k_vals
    precs.tolist()
    return precs

def featureQ7(datum):
    # Implement (any feature vector which improves performance over Q5)
    appearance = datum['review/appearance']
    palate = datum['review/palate']
    taste = datum['review/taste']
    aroma = datum['review/aroma']
    abv = datum.get('beer/ABV', 0) 
    
    # calc weighted sum using results from PCA
    yum_factor = (0.41 * appearance + 
                     0.47 * palate + 
                     0.48 * taste + 
                     0.47 * aroma + 
                     0.32 * abv)
    
    return [1] + [yum_factor]


