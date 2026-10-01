import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("hotel_bookings.csv")
print("Before cleaning:", df.shape)

# Remove impossible prices (zero or negative) and extreme outliers
df = df[(df["adr"] > 0) & (df["adr"] < 1000)]
print("After cleaning:", df.shape)
print(df["adr"].describe())

# Put months in calendar order
months = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
df["arrival_date_month"] = pd.Categorical(df["arrival_date_month"], categories=months, ordered=True)

# Average price by hotel type and by month
print(df.groupby("hotel")["adr"].mean().round(2))
monthly = df.groupby("arrival_date_month", observed=True)["adr"].mean()
print(monthly.round(2))

monthly.plot(kind="bar", title="Average price by month")
plt.ylabel("Average daily rate")
plt.tight_layout()
plt.savefig("price_by_month.png")
plt.show()