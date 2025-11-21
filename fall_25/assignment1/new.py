import csv
from collections import defaultdict

# 1. Train the model (Count popularity)
bookCount = defaultdict(int)
totalRead = 0

# Note: Ensure the filename matches your local file (e.g., removed .gz if unzipped)
print("Training...")
with open("train_Interactions.csv", 'r') as f:
    reader = csv.reader(f)
    header = next(reader) # Skip header
    for row in reader:
        user, book, rating = row
        bookCount[book] += 1
        totalRead += 1

# 2. Rank books
mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

# 3. Build the "Popular" set with an OPTIMIZED threshold
return1 = set()
count = 0
# A threshold of ~0.70 to 0.75 (70-75%) often performs better than 0.50
threshold_ratio = 0.73 
print(f"Building popular set with threshold: {threshold_ratio}")

for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead * threshold_ratio: 
        break

# 4. Make Predictions
print("Predicting...")
predictions = open("predictions_Read.csv", 'w')
# Write header manually to match the required format
predictions.write("userID,bookID,prediction\n") 

with open("pairs_Read.csv", 'r') as f:
    reader = csv.reader(f)
    header = next(reader) # Skip input header
    
    for row in reader:
        u, b = row[0], row[1] # safely get user and book
        
        if b in return1:
            predictions.write(f"{u},{b},1\n")
        else:
            predictions.write(f"{u},{b},0\n")

predictions.close()
print("Done! Saved to predictions_Read.csv")
