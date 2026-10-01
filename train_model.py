import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("hotel_bookings.csv")

# Clean
df = df[(df["adr"] > 0) & (df["adr"] < 1000)]
df["children"] = df["children"].fillna(0)

# New features
df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
df["total_guests"] = df["adults"] + df["children"] + df["babies"]

# Drop columns that would leak the answer or are mostly empty
df = df.drop(columns=["reservation_status", "reservation_status_date",
                      "assigned_room_type", "company", "agent"])

y = df["adr"]
X = df.drop(columns=["adr"])
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

# Split: 80% to learn from, 20% to test on
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

prep = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), cat_cols)],
    remainder="passthrough",
)

models = {
    "Ridge (simple baseline)": Ridge(),
    "Gradient Boosting": HistGradientBoostingRegressor(max_iter=300, random_state=42),
}

for name, model in models.items():
    pipe = Pipeline([("prep", prep), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print(name)
    print("  Average error (MAE):", round(mean_absolute_error(y_test, pred), 2))
    print("  R2 score:", round(r2_score(y_test, pred), 3))