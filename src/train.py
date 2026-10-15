import pandas as pd
from features import build_vocabulary, sentence_to_vector, tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

csv_path = "data/intents.csv"
split_seed = 42

df = pd.read_csv(csv_path)

train_dataframe, test_dataframe = train_test_split(df, random_state=split_seed, stratify=df["intent"], test_size=0.25)

vocabulary = build_vocabulary(train_dataframe["text"])

X_train = []
for sentence in train_dataframe["text"]:
    vector = sentence_to_vector(tokenize(sentence), vocabulary)
    X_train.append(vector)

X_test = []
for sentence in test_dataframe["text"]:
    vector = sentence_to_vector(tokenize(sentence), vocabulary)
    X_test.append(vector)

y_train = train_dataframe["intent"].tolist()
y_test = test_dataframe["intent"].tolist()

train_rows = len(X_train)
train_cols = len(X_train[0])
test_rows = len(X_test)
test_cols = len(X_test[0])

# Sanity Check 
disagreements = 0
for (sentence, vector) in zip(train_dataframe["text"], X_train):
    tokenized_len = len(tokenize(sentence))
    vector_sum = sum(vector)
    if vector_sum != tokenized_len:
        disagreements += 1

print(f"Sanity check - rows that disagree: {disagreements}")
print("--------------------------------------------------")

model = LogisticRegression(random_state=split_seed)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy_list = []

for (sentence, prediction, actual) in zip(test_dataframe["text"], predictions, test_dataframe["intent"]):

    is_correct = prediction == actual
    accuracy_list.append(is_correct)

    print(f"Sentence                     : {sentence}")
    print(f"What intent model predicted  : {prediction}") 
    print(f"What actual intent was       : {actual}")

    if not is_correct:
        tokens = tokenize(sentence)
        vocab_tokens = [token for token in tokens if token in vocabulary]
        print(
            f"--> [MISCLASSIFICATION] Tokens in vocabulary: {vocab_tokens}"
        )
    print("-------------------------------------------") 

accuracte_answers = sum(accuracy_list)
test_length = len(test_dataframe)

accuracy = accuracte_answers / test_length * 100

print(f"Split speed: {split_seed}")
print(f"The accuracy of model is: {accuracy:.2f}%")

