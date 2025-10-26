from collections import defaultdict
from sklearn import linear_model
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import math
import itertools
import scipy

def feat(d, catID, maxLength, includeCat = True, includeReview = True, includeLength = True):
    feat = []
    # add bias term
    feat.append([1])
    if includeCat:
        # initalize zeros -> [0,0,..,0]
        OHE = [0 for i in range(0,len(catID))]
        # check if beer style > 1000 occurances in set
        if d['beer/style'] in catID.keys():
            OHE[catID[d['beer/style']]] = 1
            feat.append(OHE)
        else:
            feat.append(OHE)
        
    if includeReview:
        #float 'review' keys
        review_keys = ['review/appearance',
                       'review/palate',
                       'review/taste',
                       'review/overall',
                       'review/aroma']
        review_feat = [float(d[i]) for i in review_keys]
        feat.append(review_feat)
    if includeLength:
        # normalized len of review 
        len_feature = [len(d['review/text'])/maxLength]
        feat.append(len_feature)
    features = list(itertools.chain.from_iterable(feat))
    
    return features 
def pipeline(reg, catID, dataTrain, dataValid, dataTest, includeCat=True, includeReview=True, includeLength=True):
    mod = linear_model.LogisticRegression(C=reg, class_weight='balanced')

    maxLength = max([len(d['review/text']) for d in dataTrain])
    
    Xtrain = [feat(d, catID, maxLength, includeCat, includeReview, includeLength) for d in dataTrain]
    Xvalid = [feat(d, catID, maxLength, includeCat, includeReview, includeLength) for d in dataValid]
    Xtest = [feat(d, catID, maxLength, includeCat, includeReview, includeLength) for d in dataTest]
    
    yTrain = [d['beer/ABV'] > 7 for d in dataTrain]
    yValid = [d['beer/ABV'] > 7 for d in dataValid]
    yTest = [d['beer/ABV'] > 7 for d in dataTest]
    
    # (1) Fit the model on the training set
    mod.fit(Xtrain,yTrain)
    # (2) Compute validation BER
    y_pred = mod.predict(Xvalid)
    bal_accuracy = balanced_accuracy_score(yValid, y_pred)
    vBER = 1 - bal_accuracy
    # (3) Compute test BER
    y_pred = mod.predict(Xtest)
    bal_accuracy = balanced_accuracy_score(yTest, y_pred)
    tBER = 1 - bal_accuracy

    return mod, vBER, tBER


def Q1(catID, dataTrain, dataValid, dataTest):
    # No need to modify this if you've implemented the functions above
    mod, validBER, testBER = pipeline(10, catID, dataTrain, dataValid, dataTest, True, False, False)

    return mod, validBER, testBER

def Q2(catID, dataTrain, dataValid, dataTest):
    mod, validBER, testBER = pipeline(10, catID, dataTrain, dataValid, dataTest, True, True, True)

    return mod, validBER, testBER

def Q3(catID, dataTrain, dataValid, dataTest):
    # Your solution here...
    # initalize list to hold best results
    # [mod,validBER,testBER]
    best = [0,0,0] 
    for c in [0.001, 0.01, 0.1, 1, 10]:
        # run model
        mod, validBER, testBER = pipeline(c, catID, dataTrain, dataValid, dataTest, True, True, True)
        # check results are better than last itteration
        if best[0] == 0 or validBER < best[1]:
            best = [mod, validBER, testBER]
    # Return the validBER and testBER for the model that works best on the validation set
    best_mod,best_validBER,best_testBER = best

    return best_mod,best_validBER,best_testBER


def Q4(C, catID, dataTrain, dataValid, dataTest):
    mod, validBER, testBER_noCat = pipeline(C, catID, dataTrain, dataValid, dataTest, False, True, True)
    mod, validBER, testBER_noReview = pipeline(C, catID, dataTrain, dataValid, dataTest, True, False, True)
    mod, validBER, testBER_noLength = pipeline(C, catID, dataTrain, dataValid, dataTest, True, True, False)
    return testBER_noCat, testBER_noReview, testBER_noLength

def Jaccard(s1, s2):
    # Implement |s1 & s2|/ |s1 V s2|
    common = len(s1 & s2)
    both = len(s1 | s2)
    try:
        result = common/both
    except:
        print("sets s1 and s2 empty")
        result = 0
    return result

def mostSimilar(i, N, usersPerItem):
    # initalize list to store similarity results (similarity, itemID)
    similarities = []
    s1 = usersPerItem[i]
    for item in usersPerItem.keys():
        s2 = usersPerItem[item]
        if s1 == s2:
            continue
        jac = Jaccard(s1,s2)
        similarities.append((jac,item))
    # sort similarites
    similarities.sort(reverse=True)
    return(similarities[:N])

def MSE(y, ypred):
    y_test = np.array(y)
    y_pred = np.array(ypred)
    mse = np.mean((y_test - y_pred)**2)
    return(mse)

def getMeanRating(dataTrain):
    mean_rating = np.mean(np.array([d['star_rating'] for d in dataTrain]))
    return(mean_rating)

def getUserAverages(itemsPerUser, ratingDict):
    # Implement (should return a dictionary mapping users to their averages)
    #return userAverages
    userAverages = {}
    for user in itemsPerUser.keys():
        ratings = [ratingDict[(user, item)] for item in itemsPerUser[user]]
        userAverages[user] = np.mean(ratings)
    return(userAverages)

def getItemAverages(usersPerItem, ratingDict):
    # Implement...
    itemAverages = {}
    for item in usersPerItem.keys():
        ratings = [ratingDict[(user, item)] for user in usersPerItem[item]]
        itemAverages[item] = np.mean(ratings)
    return(itemAverages)

def predictRating(user,item,ratingMean,reviewsPerUser,usersPerItem,itemsPerUser,userAverages,itemAverages):
    """
    r(u, i) = R_i_bar + [ SUM(j in I_u) ( (R_u,j - R_j_bar) * Sim(i, j) ) ] / 
                      [ SUM(j in I_u) ( Sim(i, j) ) ]
    """
    ratings = []
    sims = []
    # loop over reviews by user user_review_i
    for user_review_i in reviewsPerUser[user]:
        item_id = user_review_i["product_id"]
        # don't include comparision of item against itself
        if item_id == item:
            continue
        # calc (R_u,j - R_j_bar)
        rating_diff = user_review_i['star_rating'] - itemAverages[item_id]
        ratings.append(rating_diff)
        # calc Sim(i, j)
        sim = Jaccard(usersPerItem[item],usersPerItem[item_id])
        sims.append(sim)

    # if no similar items then return item avg
    if len(sims) == 0: 
        final_rating = itemAverages.get(item,ratingMean)
        return(final_rating)
    # calc sums for non empty case
    else:
        ratings_arr = np.array(ratings)
        sims_arr = np.array(sims)
        sims_sum = np.sum(sims_arr)
        # can't divide by zero
        if sims_sum >0:
            final_rating = itemAverages.get(item,ratingMean) + np.dot(ratings_arr, sims_arr)/sims_sum
            return(final_rating)
        else:
            final_rating = itemAverages.get(item,ratingMean)
            return(final_rating)

def predictRatingQ7(user,item,ratingMean,reviewsPerUser,usersPerItem,itemsPerUser,userAverages,itemAverages):
    # Your solution here
    k=5
    mu = ratingMean
    
    # calc user bias
    num_user_reviews = len(reviewsPerUser.get(user, []))
    user_avg = userAverages.get(user, mu)
    b_u_raw = user_avg - mu
    b_u = b_u_raw * (num_user_reviews / (num_user_reviews + k))
    
    # calc item bias
    num_item_raters = len(usersPerItem.get(item, set()))
    item_avg = itemAverages.get(item, mu)
    b_i_raw = item_avg - mu
    b_i = b_i_raw * (num_item_raters / (num_item_raters + k))
    
    # calc final rating
    final_rating = mu + b_u + b_i
    
    return(final_rating)
