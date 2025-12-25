import gzip
from collections import defaultdict

def readGz(path):
  for l in gzip.open(path, 'rt'):
    yield eval(l)

def readCSV(path):
  f = gzip.open(path, 'rt')
  f.readline()
  for l in f:
    yield l.strip().split(',')


### Category prediction baseline: Just consider some of the most common words from each category
### note the Performance will be measured in terms of the fraction of correct classifications
### baseline performance:(this from the code below): 0.3582
catDict = {
  "children": 0,
  "comics_graphic": 1,
  "fantasy_paranormal": 2,
  "mystery_thriller_crime": 3,
  "young_adult": 4
}

predictions = open("predictions_Category.csv", 'w')
predictions.write("userID,reviewID,prediction\n")
for l in readGz("test_Category.json.gz"):
  cat = catDict['fantasy_paranormal'] # If there's no evidence, just choose the most common category in the dataset
  words = l['review_text'].lower()
  if 'children' in words:
    cat = catDict['children']
  if 'comic' in words:
    cat = catDict['comics_graphic']
  if 'fantasy' in words:
    cat = catDict['fantasy_paranormal']
  if 'mystery' in words:
    cat = catDict['mystery_thriller_crime']
  if 'love' in words:
    cat = catDict['young_adult']
  predictions.write(l['user_id'] + ',' + l['review_id'] + "," + str(cat) + "\n")

predictions.close()

# train_Category.json.gz context
"""
contents of training data: 100000 observations
below is a subset of the train data
{'user_id': 'u75242413', 'review_id': 'r45843137', 'rating': 4, 'review_text': "a clever book with a deeply troubling premise and an intriguing protagonist. Thompson's clean, sparse prose style kept each page feeling light even as some rather heavy existential questions dropped upon them. I enjoyed it. \n and that cover design is boom-pow gorgeous.", 'n_votes': 1, 'genre': 'mystery_thriller_crime', 'genreID': 3}
{'user_id': 'u72358746', 'review_id': 'r38427923', 'rating': 2, 'review_text': "A little too much retconning for me, to be honest. Wolverine's past has mostly been a mystery and for the most part, I am content with that. Saying he formed a proto-X-Men group doesn't feel right, and neither does the part Xavier plays so far (I didn't think he really established a school before he was crippled) .", 'n_votes': 0, 'genre': 'comics_graphic', 'genreID': 1}
{'user_id': 'u55827211', 'review_id': 'r97393610', 'rating': 5, 'review_text': "So glad I finally got around to reading this book. Tammara Webber, you are officially my most favorite author...well, next to Jane Austen, of course. :) I'm reading the rest of BTL series now!", 'n_votes': 0, 'genre': 'young_adult', 'genreID': 4}
{'user_id': 'u15021470', 'review_id': 'r76296012', 'rating': 4, 'review_text': 'I would classify this more as erotic paranormal. I almost stopped reading at the whole needing to save her kid part but then it redeemed it self when it revealed this was a fake memory a few sentences later. It was a quick, entertaining read.', 'n_votes': 0, 'genre': 'fantasy_paranormal', 'genreID': 2}
{'user_id': 'u37264352', 'review_id': 'r09884372', 'rating': 4, 'review_text': "Loved it! So very Butcher, although I still think that Dresden fits his writing style better. This was a little too informally written for an epic fantasy. The characters are great though and the story has gotten off to a great start. The plot line seems a little obvious but I don't really mind that.", 'n_votes': 0, 'genre': 'fantasy_paranormal', 'genreID': 2}
"""

# test_Category.json.gz context
"""
contents of training data: 10000 observations
below is a subset of the test data

{'user_id': 'u77355739', 'review_id': 'r70788666', 'rating': 4, 'review_text': "love it \n it's that kind of crime-mystery novel that anyone can : read , love and understand . \n my favorite part is in the first chapter when Mr. quin help the old man( i forget his name) to solve a long time closed mystery , and that is only the beginning", 'n_votes': 0}
{'user_id': 'u93892904', 'review_id': 'r58778896', 'rating': 4, 'review_text': "This latest book from S.J. Bolton, features two characters from a previous book D.C. Lacey Flint and D.I. Joesbury (Now You See Me). She has decided maybe to write a series, as opposed to her previous stand-alone books. Joesbury tells Lacey she has an assignment to go to Cambridge and pose as an undercover student, to look into a higher than usual suicide rate of female students. The methods of suicide are also extremely unusual and particularly brutal. She has only one contact there, Dr. Evi Oliver, who is suspicious about the girls deaths and is having problems of her own, i.e. a stalker, who is getting in and out of her home, unseen. It's an interesting story, quite unnerving and creepy, but all is not what is seems and Lacey soon in danger and targeted herself. She and Joesbury have a relationship of sorts from their previous case, he is obsessed by her and she has feelings for him, but they have not acted on them yet. Now, it is a bit of a far-fetched tale, but yet, it is a gripper and a page turner. I liked it, because S.J. Bolton has a way of weaving a plot to keep you intrigued and she has a way with her style of writing to keep you interested. I like Flint and Joesbury and can see a future for them in a series, if Ms. Bolton decided to go down that route.", 'n_votes': 2}
{'user_id': 'u53246589', 'review_id': 'r24585482', 'rating': 4, 'review_text': "Maybe I'm dumb, but after reading two of these books, I'm still confused. What is Mary Poppins? Is she a star? A goddess? I don't get it! But what I do get that is that the stories about her are wonderful and fun, although vastly different from anything starring Julie Andrews, that's for darn sure! \n Reread to the kids, spring 2014. Really love the Nellie-Rubina chapter, which was totally lost on my kids.", 'n_votes': 1}
{'user_id': 'u51256488', 'review_id': 'r66832354', 'rating': 5, 'review_text': "This is a re-read for me in anticipation of The Last Star coming out. I did the audio this time. A co-worker was reading this and talked to me about it and I couldn't remember it. Actually, I was confusing it with another book so I thought I'd better refresh my memory. I liked it even more the second time.", 'n_votes': 0}
{'user_id': 'u37667882', 'review_id': 'r26215995', 'rating': 5, 'review_text': "Book: The Beast (Black Dagger Brotherhood #14) \n Author: J.R. Ward \n Publication Date: 4/5/2016 \n Reviewed by: Tammy Payne- Book Nook Nuts \n My Rating: 5 Stars \n REVIEW \n As with all books in this series, I absolutely enjoyed every single word. I love all the brothers but Rhage was always a favorite and this book reminds me why. \n So much is happening in this story, and it's explosive. We get many of those who we love in this book. Layla and Quinn and Blay experience something that leaves us all wondering for a bit. Rhage and Mary are blessed beyond words or will it be ripped away? V has news for Payne and the others. Lassiter also appears and is funny as ever. Our doctor Mannie may be in for a shocker soon. We meet Bitty someone who is becoming very special to the brothers but to one even more so. And Assail well he has now become one of my favorites also. \n This book had my emotions all over the place but oh how I enjoyed it. I would suggest you begin with book one in this series, so you learn who everyone is. \n I bought the audible of this book as well as borrowed the book version from my local library.", 'n_votes': 1}
"""

# pairs_Category.csv context
"""
contents of prediction data: 10000 observations
below is a subset of prediction data
userID,reviewID,prediction
u77355739,r70788666
u93892904,r58778896
u53246589,r24585482
u51256488,r66832354
"""
