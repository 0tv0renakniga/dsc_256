## import libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import gzip
import json
from datetime import datetime
import dateutil.parser
import random
"""
def getMaxLen(dataset):
    # Find the longest review (number of characters)
    #return maxLen

def featureQ1(datum, maxLen):
    # Feature vector for one data point
"""

def Q1(dataset):
    import numpy as np
    # Implement...
    def getMaxLen(dataset):
        # Find the longest review (number of characters)
        review_lens = [len(i['review_text']) for i in dataset]
        maxLen = max(review_lens)
        return maxLen
    
    def featureQ1(datum, maxLen):
    # Feature vector for one data point
        feature = len(datum['review_text'])/maxLen
        return [1] + [feature]
    
    maxLen = getMaxLen(dataset)
    X = [featureQ1(d) for d in dataset]
    y = [d['review'] for d in dataset]

    theta,residuals,rank,s = np.linalg.lstsq(X, y)
    MSE = residuals/len(y)
    return theta, MSE


