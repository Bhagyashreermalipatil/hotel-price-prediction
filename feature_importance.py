import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

df = pd.read_csv("hotel_bookings.csv")
df = df[(df["adr"] > 0) & (df["adr"] < 1000)]
df["children"] = df["children"].fillna(0)
df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
df["total_guests"] = df["adults"] + df["children"] + df["babies"]
df = df.drop(columns=["reservation_status", "reservation_status_date",
                      "assigned_room_type", "company", "agent"])

y = df["adr"]
X = df.drop(columns=["adr"])
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

prep = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), cat_cols)],
    remainder="passthrough",
)
pipe = Pipeline([("prep", prep), ("model", HistGradientBoostingRegressor(max_iter=300, random_state=42))])
pipe.fit(X_train, y_train)

# Which features matter most?
sample = X_test.sample(3000, random_state=0)
result = permutation_importance(pipe, sample, y_test.loc[sample.index], n_repeats=5,
                                random_state=0, scoring="neg_mean_absolute_error")
importance = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)
print(importance.head(10).round(2))

importance.head(10).iloc[::-1].plot(kind="barh", title="Top 10 features driving price")
plt.xlabel("Increase in error if this feature is scrambled")
plt.tight_layout()
plt.savefig("feature_importance.png")

joblib.dump(pipe, "price_model.joblib")
print("Saved price_model.joblib and feature_importance.png")