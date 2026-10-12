import string
import pandas as pd
from sklearn.model_selection import train_test_split

csv_path = "data/intents.csv"

df = pd.read_csv(csv_path)
dataframe, test_dataframe = train_test_split(df, test_size=0.25, random_state=42, stratify=df["intent"])

def tokenize(text):
    text = text.replace('-', ' ')
    text = text.replace('’', "'").replace('‘', "'")
    text = text.replace('“', '"').replace('”', '"')
    words = text.split()
    cleaned_words = []
    for word in words:
        cleaned = word.translate(str.maketrans('', '', string.punctuation)).lower()
        if cleaned and any(char.isalnum() for char in cleaned):
            cleaned_words.append(cleaned)
    return cleaned_words

vocab = set()
for sentence in dataframe["text"]:
    words = tokenize(sentence)
    vocab.update(words)
vocab_list = sorted(vocab)

def sentence_to_vector(sentence, vocab_list):
    """
    Turns a sentence into bag of words vector.

    args:
    sentence - a list of tokenized words that needs to be vectorized
    vocab_list - a vocabulary list on which the model is trained

    returns:
    list - vectorized form of the sentence
    """

    vector = []

    for word in vocab_list:
        count = sentence.count(word)
        vector.append(count)

    return vector

s1 = tokenize("Hello, Jarvis!")
print(s1)
v1 = sentence_to_vector(s1, vocab_list)
print(v1)
print(len(v1))
print(sum(v1))
print("-------------------")

s2 = tokenize("What's Up, JarvIs, H-e-L-L-O, I'm kinda mad.")
print(s2)
v2 = sentence_to_vector(s2, vocab_list)
print(v2)
print(len(v2))
print(sum(v2))
print("-------------------")

s3 = tokenize("open-browser")
print(s3)
v3 = sentence_to_vector(s3, vocab_list)
print(v3)
print(len(v3))
print(sum(v3))
print("-------------------")

s4 = tokenize("what’s the time")
print(s4)
v4 = sentence_to_vector(s4, vocab_list)
print(v4)
print(len(v4))
print(sum(v4))
print("-------------------")