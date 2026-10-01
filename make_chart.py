import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

df = pd.read_csv("hotel_bookings.csv")
df = df[(df["adr"] > 0) & (df["adr"] < 1000)]
df["children"] = df["children"].fillna(0)
df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
df["total_guests"] = df["adults"] + df["children"] + df["babies"]
df = df.drop(columns=["reservation_status", "reservation_status_date",
                      "assigned_room_type", "company", "agent"])

train = df[df["arrival_date_year"] < 2017]
test = df[df["arrival_date_year"] == 2017]
X_train, y_train = train.drop(columns=["adr"]), train["adr"]
X_test, y_test = test.drop(columns=["adr"]), test["adr"]

cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
prep = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), cat_cols)],
    remainder="passthrough",
)
pipe = Pipeline([("prep", prep), ("model", HistGradientBoostingRegressor(max_iter=300, random_state=42))])
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred, alpha=0.1, s=6)
plt.plot([0, 400], [0, 400], "r--", label="Perfect prediction")
plt.xlim(0, 400)
plt.ylim(0, 400)
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Actual vs predicted (2017 test set)")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
print("Saved actual_vs_predicted.png")