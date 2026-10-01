import pandas as pd

df = pd.read_csv("hotel_bookings.csv")
print(df.shape)
print(df.head())
print(df["adr"].describe())