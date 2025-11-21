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
{'user_id': 'u75242413', 'review_id': 'r45843137', 'rating': 4, 'review_text': "a clever book with a deeply troubling premise and an intriguing protagonist. Thompson's clean, sparse prose style kept each page feeling light even as some rather heavy existential questions dropped upon them. I enjoyed it. \n and that cover design is boom-pow gorgeous.", 'n_votes': 1, 'genre': 'mystery_thriller_crime', 'genreID': 3}
{'user_id': 'u72358746', 'review_id': 'r38427923', 'rating': 2, 'review_text': "A little too much retconning for me, to be honest. Wolverine's past has mostly been a mystery and for the most part, I am content with that. Saying he formed a proto-X-Men group doesn't feel right, and neither does the part Xavier plays so far (I didn't think he really established a school before he was crippled) .", 'n_votes': 0, 'genre': 'comics_graphic', 'genreID': 1}
{'user_id': 'u55827211', 'review_id': 'r97393610', 'rating': 5, 'review_text': "So glad I finally got around to reading this book. Tammara Webber, you are officially my most favorite author...well, next to Jane Austen, of course. :) I'm reading the rest of BTL series now!", 'n_votes': 0, 'genre': 'young_adult', 'genreID': 4}
{'user_id': 'u15021470', 'review_id': 'r76296012', 'rating': 4, 'review_text': 'I would classify this more as erotic paranormal. I almost stopped reading at the whole needing to save her kid part but then it redeemed it self when it revealed this was a fake memory a few sentences later. It was a quick, entertaining read.', 'n_votes': 0, 'genre': 'fantasy_paranormal', 'genreID': 2}
{'user_id': 'u37264352', 'review_id': 'r09884372', 'rating': 4, 'review_text': "Loved it! So very Butcher, although I still think that Dresden fits his writing style better. This was a little too informally written for an epic fantasy. The characters are great though and the story has gotten off to a great start. The plot line seems a little obvious but I don't really mind that.", 'n_votes': 0, 'genre': 'fantasy_paranormal', 'genreID': 2}
