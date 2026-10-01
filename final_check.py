import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("hotel_bookings.csv")
df = df[(df["adr"] > 0) & (df["adr"] < 1000)]
df["children"] = df["children"].fillna(0)
df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
df["total_guests"] = df["adults"] + df["children"] + df["babies"]
df = df.drop(columns=["reservation_status", "reservation_status_date",
                      "assigned_room_type", "company", "agent"])

# Time-based split: learn from 2015-2016, test on 2017
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
print("Train: 2015-2016 | Test: 2017")
print("Average error (MAE):", round(mean_absolute_error(y_test, pred), 2))
print("R2 score:", round(r2_score(y_test, pred), 3))

# Demo: 8 real bookings, actual vs predicted
sample = X_test.sample(8, random_state=1)
demo = sample[["hotel", "arrival_date_month", "total_nights", "total_guests", "reserved_room_type"]].copy()
demo["actual_price"] = y_test.loc[sample.index].round(1)
demo["predicted_price"] = pipe.predict(sample).round(1)
print(demo.to_string())