# imports
import pandas as pd
from sklearn.model_selection import train_test_split

# defining a csv path which can be changed later with real datasets.
csv_path = "data/intents.csv"

# creating a pandas dataframe
df = pd.read_csv(csv_path)

# vocabulary over complete data
vocabulary = set()
for sentence in df["text"]:
    words = sentence.split()
    vocabulary.update(words)
vocabulary_list = sorted(vocabulary)


print(f"\nThe original vocabulary created using complete dataset contains: {len(vocabulary)} unique words.\n")
print(vocabulary_list)
print("-------------------------------------------")

# vocabulary over training data only
dataframe, test_dataframe = train_test_split(df, test_size=0.25, random_state=42, stratify=df["intent"])

train_vocabulary = set()
for sentence in dataframe["text"]:
    words =  sentence.split()
    train_vocabulary.update(words)
train_vocabulary_list = sorted(train_vocabulary)


print(f"\nThe training vocabulary created using training dataset contains: {len(train_vocabulary)} unique words.\n")
print(sorted(train_vocabulary))
print("-------------------------------------------")

print("\nThe words which are there in vocabulary created using original dataset but not in the vocabulary created by training set are: \n")
print(vocabulary - train_vocabulary)
print("-------------------------------------------")

# why we created "train_vocabulary"
# leakage vs memorixation
# memorization is the model learning the training set too well.
# leakage is when information from the test set sneaks into the training process, so the exam isn’t a surprise any more.