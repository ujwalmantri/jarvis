import pandas as pd 
from sklearn.model_selection import train_test_split

csv_path = "data/intents.csv"

df = pd.read_csv(csv_path)
dataframe, test_dataframe = train_test_split(df, test_size=0.25, random_state=42, stratify=df["intent"])

print("---------------------------")
print("The data was split into test and train splits.")
print(f"The original had: {len(df)!r} rows")
print(f"The intent count of original data was:")
print(df["intent"].value_counts())

print("---------------------------")
print(f"The training split has: {len(dataframe)!r} rows")
print(f"The intent count of training data was:")
print(dataframe["intent"].value_counts())

print("---------------------------")
print(f"The testing split has: {len(test_dataframe)!r} rows")
print(f"The intent count of testing data was:")
print(test_dataframe["intent"].value_counts())