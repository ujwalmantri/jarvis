import pandas as pd
df = pd.read_csv("data/intents.csv")
print(df.shape)
print(df["intent"].value_counts())
print(df.tail(3))
