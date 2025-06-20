import matplotlib.pyplot as plt

import gzip
tr = []
time = None
rating = None
# URL in assignment 1 slides
for l in gzip.open("Beeradvocate.txt.gz"):
  if l.startswith("review/overall"):
    rating = float(l.split(": ")[1])
  if l.startswith("review/time"):
    time = int(l.split(": ")[1])
    tr.append((time,rating))

len(tr)

X = [x[0] for x in tr[:10000]]
Y = [x[1] for x in tr[:10000]]

plt.plot(X, Y)

plt.show()

tr.sort()
sliding = []

wSize = 10000

tSum = sum([x[0] for x in tr[:wSize]])
rSum = sum([x[1] for x in tr[:wSize]])

for i in range(wSize,len(tr)-1):
  tSum += tr[i][0] - tr[i-wSize][0]
  rSum += tr[i][1] - tr[i-wSize][1]
  sliding.append((tSum*1.0/wSize,rSum*1.0/wSize))

X = [x[0] for x in sliding]
Y = [x[1] for x in sliding]
