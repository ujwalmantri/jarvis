import string
import pandas as pd
from sklearn.model_selection import train_test_split

csv_path = "data/intents.csv"

df = pd.read_csv(csv_path)
dataframe, test_dataframe = train_test_split(df, test_size=0.25, random_state=42, stratify=df["intent"])

total_intents = len(test_dataframe["intent"])

def check_accuracy(guess):
    correct_list = (test_dataframe["intent"] == guess)
    correct_predictions = correct_list.sum()
    accuracy = (correct_predictions / total_intents) * 100
    ## can use .mean() too but i like .sum()

    print(f"The total number of intents in testing set: {total_intents}")
    print("-----------------------------------------------")

    print(f"The total number of correct predictions: {correct_predictions}")
    print("-----------------------------------------------")

    print(f"The accuracy of the model predicting intent == {guess} is: {accuracy:.2f}%")
    print("-----------------------------------------------")
    return accuracy

check_accuracy("greeting")
check_accuracy("time")
check_accuracy("open_app")