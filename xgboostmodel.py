import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, median_absolute_error
from sklearn.preprocessing import OneHotEncoder
import ast
import joblib

from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Example
df = pd.DataFrame({
    'Rating': ['G', 'PG', 'PG-13', 'R', 'Not Rated', 'Unrated', 'NC-17', 'PG']
})

# Encode
le = LabelEncoder()
df['rating_encoded'] = le.fit_transform(df['Rating'])

# Print result
print(df)
print("\nLabel mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
joblib.dump(le,"rating_encoder.pkl")
# Load and preprocess data
# df = pd.read_csv("possible_encoding.csv")
df = pd.read_csv("final.csv")
df["rating_encoded"] = le.transform(df['Rating'])
df = df.drop(["Rating"],axis=1)
# print(df['Rating'].unique())
# exit(1)
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Columns to remove (adjusted for XGBoost's better handling of categoricals)
# colstoremove = [
#     'movieId', 'imdb_id', 'original_title', 'production_companies', 
#     'production_countries', 'release_date', 'title', 'director', 
#     'Unnamed', 'vote_count', 'vote_average', 'popularity', "director_encoded", "main_actor_encoded"
# ]

colstoremove = [
    'movieId', 'imdb_id', 'original_title', 'production_companies', 
    'production_countries', 'release_date', 'title', 'director', 
    'Unnamed', 'vote_count', 'vote_average', 'popularity', "Unnamed: 0.1"
]
df = df.drop(columns=colstoremove, errors='ignore')
df = df.dropna()
print(df.columns)
# Target engineering: Log-transform revenue (critical for box office)
y = np.log1p(df["revenue"])  # Using log1p to handle zeros
X = df.drop("revenue", axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost with tuned hyperparameters
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.5,
    reg_lambda=0.5,
    random_state=42,
    early_stopping_rounds=50,
    eval_metric='rmse'
)

# Train with early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=10
)

# Predictions (exponentiate to return to original scale)
y_pred = np.expm1(model.predict(X_test))
y_test_exp = np.expm1(y_test)

# Metrics
print("\n=== Evaluation Metrics ===")
print("RMSE:", np.sqrt(mean_squared_error(y_test_exp, y_pred)))
print("R^2 Score:", r2_score(y_test_exp, y_pred))
print("MAE:", mean_absolute_error(y_test_exp, y_pred))
print("Median Absolute Error:", median_absolute_error(y_test_exp, y_pred))
print("Explained Variance Score:", explained_variance_score(y_test_exp, y_pred))

# Feature importance
print("\n=== Feature Importance ===")
feature_importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

for feature, importance in feature_importance.items():
    print(f"{feature:<40} {importance:.4f}")

# Save model
joblib.dump(model, "box_office_predictor_xgboost.pkl")