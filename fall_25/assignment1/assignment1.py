#====================================================================
# rating prediction                                                 =
#====================================================================
import pandas as pd
import numpy as np
from tqdm import tqdm

def predictRating(user, item, ratingMean, reviewsPerUser, usersPerItem, 
                  itemsPerUser, userAverages, itemAverages):
    
    #prediction = mu + b_u + b_i
    
    # rating centered around mean and shifts via user/book bais
    k = 5 
    mu = ratingMean
    
    # calc user bias (b_u)
    num_user_reviews = len(reviewsPerUser.get(user, []))
    user_avg = userAverages.get(user, mu)
    b_u_raw = user_avg - mu
    b_u = b_u_raw * (num_user_reviews / (num_user_reviews + k))
    
    # calc item bias (b_i)
    num_item_raters = len(usersPerItem.get(item, set()))
    item_avg = itemAverages.get(item, mu)
    b_i_raw = item_avg - mu
    b_i = b_i_raw * (num_item_raters / (num_item_raters + k))
    
    # calc final rating
    final_rating = mu + b_u + b_i
    
    return final_rating

# read training data
print("read train_Interactions.csv...")
df = pd.read_csv("train_Interactions.csv")

# make features from training Set
print("calc statistics ...")

# calc average rating
ratingMean = df['rating'].mean()

# get average rating feature for each user and item
userAverages = df.groupby('userID')['rating'].mean().to_dict()
itemAverages = df.groupby('bookID')['rating'].mean().to_dict()

# get interaction counts/lists for regularization
reviewsPerUser = df.groupby('userID')['bookID'].apply(list).to_dict()
usersPerItem = df.groupby('bookID')['userID'].apply(set).to_dict()

# This is in your signature, so we compute it
itemsPerUser = df.groupby('userID')['bookID'].apply(set).to_dict()

print("Statistics calculation complete.")

# read test data
print("read predictions_Rating.csv...")
test_df = pd.read_csv('predictions_Rating.csv')

# make predictions on test data
print("make prediction on test data...")
predictions = []

# loop over test data
for row in tqdm(test_df.itertuples(), total=len(test_df)):
    user = row.userID
    item = row.bookID
    
    # call function with all features
    pred = predictRating(
        user, item, 
        ratingMean, 
        reviewsPerUser, usersPerItem, 
        itemsPerUser, 
        userAverages, itemAverages
    )
    predictions.append(pred)

# add predictions to the dataframe
test_df['prediction'] = predictions

# clip predictions to the valid 0-5 range
test_df['prediction'] = test_df['prediction'].clip(0, 5)

# save csv
output_cols = ['userID', 'bookID', 'prediction']
test_df[output_cols].to_csv('predictions_Rating.csv', index=False)

print("\nrating prediction done")
#====================================================================
#                                                                   =
#====================================================================

#====================================================================
# read prediction                                                   =
#====================================================================
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import numpy as np

# read training data
df = pd.read_csv('train_Interactions.csv')

# create binary label: 1 if rating > 0 (read), 0 otherwise (not read)
df['read'] = (df['rating'] > 0).astype(int)

# calc user-level features
user_stats = df.groupby('userID').agg(
    user_read_rate=('read', 'mean'),
    user_num_books=('bookID', 'count')
).reset_index()

# calc book-level features
book_stats = df.groupby('bookID').agg(
    book_read_rate=('read', 'mean'),
    book_num_users=('userID', 'count')
).reset_index()

# put features back into the dataframe
df = df.merge(user_stats, on='userID')
df = df.merge(book_stats, on='bookID')

# split into train and test sets
train, test = train_test_split(df, test_size=0.2, random_state=42)

# define features
features = ['user_read_rate', 'user_num_books', 'book_read_rate', 'book_num_users']

# prep training data
X_train = train[features]
y_train = train['read']
X_train = sm.add_constant(X_train) 

# train logistic regression model
model = sm.Logit(y_train, X_train)
result = model.fit()

# print model summary for interpretation
print(result.summary())

# evaluate on test set
X_test = test[features]
y_test = test['read']
X_test = sm.add_constant(X_test)
preds = result.predict(X_test)
acc = ((preds > 0.5).astype(int) == y_test).mean()
print('Test Accuracy:', acc)

baseline_acc = (y_test == 1).mean()
print('baseline accuracy:', baseline_acc)

# prediction function for a given userID and bookID
def predict_read(user_id, book_id, result, df, user_stats, book_stats):
    # get user features and default to global means if not seen
    user_data = user_stats[user_stats['userID'] == user_id]
    if user_data.empty:
        user_read_rate = df['read'].mean()
        user_num_books = 0
    else:
        user_read_rate = user_data['user_read_rate'].values[0]
        user_num_books = user_data['user_num_books'].values[0]
    
    # get book features and default to global means if not seen
    book_data = book_stats[book_stats['bookID'] == book_id]
    if book_data.empty:
        book_read_rate = df['read'].mean()
        book_num_users = 0
    else:
        book_read_rate = book_data['book_read_rate'].values[0]
        book_num_users = book_data['book_num_users'].values[0]
    
    # prep input array (with constant/intercept)
    x = np.array([[1.0, user_read_rate, user_num_books, book_read_rate, book_num_users]])
    
    # predict probability and threshold
    prob = result.predict(x)[0]
    return int(prob > 0.5)

# read predictions_Rating.csv since it has all the pairs read data
print("Loading predictions_Rating.csv...")
test_df = pd.read_csv('predictions_Read.csv')

# make predictions
print("making predictions...")
predictions = []

for row in tqdm(test_df.itertuples(), total=len(test_df)):
    user = row.userID
    item = row.bookID
    
    # make predictions
    pred = predict_read(user, item, result, df, user_stats, book_stats)
    predictRating(
        user, item, 
        ratingMean, 
        reviewsPerUser, usersPerItem, 
        itemsPerUser, 
        userAverages, itemAverages
    )
    predictions.append(pred)

# add predictions to the dataframe and save as csv
test_df['prediction'] = predictions
test_df.to_csv('predictions_Read.csv', index=False)
print("\nread prediction done")
#====================================================================
#                                                                   =
#====================================================================

#====================================================================
# cat prediction                                                    =
#====================================================================
import pandas as pd
import re

# rewrote json as csv so wouldn't have to deal with it
df = pd.read_csv('test_cat_predictions.csv')

# define expanded category keywords 
keywords = {
    'children': [
        'children', 'child', 'kid', 'kids', 'picture book', 'bedtime', 'toddler',
        'baby', 'preschool', 'kindergarten', 'illustrations', 'board book',
        'middle grade', 'scary', 'kinderbuch', 'enfant', 'niños', 'bilderbuch',
        'classic', 'fairy', 'sweet story','beautiful','wonderful'
    ],
    'comics_graphic': [
        'comic', 'comics', 'graphic novel', 'manga', 'anime', 'panel', 'artwork',
        'illustrator', 'volume', 'visual', 'art', 'strip', 'superhero', 'bande dessinee', 
        'bd', 'Wolverine','marvel','batman','volumes','graphic'
    ],
    'fantasy_paranormal': [
        'fantasy', 'paranormal', 'magic', 'wizard', 'witch', 'dragon', 'vampire',
        'werewolf', 'supernatural', 'fae', 'elf', 'curse', 'ghost', 'demon',
        'god', 'goddess', 'mythology', 'kingdom', 'prophecy', 'immortal', 
        'fantasia', 'fantasie', 'sorcerer', 'shifter','novella','dark',
    ],
    'mystery_thriller_crime': [
        'mystery', 'thriller', 'crime', 'murder', 'detective', 'police', 'investigation',
        'suspense', 'killer', 'suspect', 'spy', 'noir', 'whodunit', 'agent',
        'case', 'lawyer', 'fbi', 'forensic', 'krimi', 'misterio', 'mystere', 'policier',
        'creepy','gross','scary','woman','death','action'
    ],
    'young_adult': [
        'young adult', 'ya', 'teen', 'teenager', 'high school', 'romance', 'coming of age',
        'dystopian', 'dystopia', 'angst', 'love triangle', 'sixteen', 'seventeen',
        'boyfriend', 'girlfriend', 'crush', 'prom', 'jugendbuch','Webber','Austen',
        'hype','kiss'
    ]
}

def predict_category(text):
    if not isinstance(text, str):
        return 'fantasy_paranormal' 

    text = text.lower()
    scores = {cat: 0 for cat in keywords}

    # count keyword occurrences
    for cat, words in keywords.items():
        for word in words:
            if len(word) <= 3:
                 if re.search(r'\b' + re.escape(word) + r'\b', text):
                     scores[cat] += 1
            else:
                if word in text:
                    scores[cat] += 1

    # find category with the highest score
    max_cat = max(scores, key=scores.get)
    max_val = scores[max_cat]

    # fallback to most frequent in dataset
    if max_val == 0:
        return 'fantasy_paranormal'
    
    return max_cat

# predict_category uses the expanded freq keyword dictionary 
df['prediction'] = df['review_text'].apply(predict_category)
catDict = {
  "children": 0,
  "comics_graphic": 1,
  "fantasy_paranormal": 2,
  "mystery_thriller_crime": 3,
  "young_adult": 4
}
# save csv
output_df = df[['user_id', 'review_id', 'prediction']]
output_df['prediction'] = output_df['prediction'].map(catDict)
output_df.to_csv('predictions_Category.csv', index=False)

print("Done.. woooo yeah!!!!")
#====================================================================
#                                                                   =
#====================================================================
