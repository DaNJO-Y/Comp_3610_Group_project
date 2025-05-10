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
import pandas as pd

# Assuming df is your DataFrame
# Remove columns not suitable for ML

df= pd.read_parquet("final_dataset.parquet")
columns_to_drop = [
    # Identifiers or duplicates
    'Unnamed: 0', 'movieId', 'imdb_id', 'original_title', 'title',
    
    # Raw text, lists, or dicts
    'production_companies', 'production_countries', 'genres', 'lead_cast',
    'experienced_genres_x', 'experienced_genres_y',

    # Raw dates
    'release_date', 'last_appearance_x', 'last_appearance_y', 
    "year",

    # Categorical strings (to be encoded separately if needed)
    'director', 'main_actor', 'production_company', 
    # "budget",
    "log_budget",
    # 'Rating',
    # 'release_season', 
    # 'runtime_category', 
    # 'budget_category',
    # "runtime",
    'main_country', 'actor_name', 'director_name',
    # "director_experience", "actor_experience",
    "total_movies_y", "total_movies_x",
    "genre_diversity",
    "is_sequel",
    "has_top_actor"
]

df = df.drop(columns=columns_to_drop, errors='ignore')
# df.dropna(inplace=True)
# Optional: also drop any other columns with object or list types, if unsure
# df = df.select_dtypes(exclude=['object', 'list'])

# Display cleaned DataFrame
# print(df.head())

# Target engineering: Log-transform revenue (critical for box office)
y = np.log1p(df["revenue"])  # Using log1p to handle zeros
X = df.drop("revenue", axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test , X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Initialize XGBoost with tuned hyperparameters
model = XGBRegressor(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.5,
    reg_lambda=0.5,
    random_state=42,
    early_stopping_rounds=50,
    eval_metric='rmse'
)

# model = XGBRegressor(
#     n_estimators=100,
#     max_depth=6,
#     learning_rate=0.1,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     objective='reg:squarederror',
#     eval_metric='rmse',
#     early_stopping_rounds=10
# )

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
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
print("\n=== Feature Importance (All) ===")
feature_importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# # Scatter plot of predicted vs actual revenues
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=y_test_exp, y=y_pred)

# # Add a line y = x for reference
# max_val = max(max(y_test_exp), max(y_pred))
# plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')

# plt.xlabel("Actual Revenue")
# plt.ylabel("Predicted Revenue")
# plt.title("Gradient Boosting: Predicted vs Actual Box Office Revenue")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter

# Convert revenues to hundreds of millions
y_test_exp_m = y_test_exp / 1e8
y_pred_m = y_pred / 1e8

# Formatter function to show plain numbers
formatter = FuncFormatter(lambda x, _: f'{x:.0f}')

# Scatter plot of predicted vs actual revenues
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_exp_m, y=y_pred_m)

# Add a line y = x for reference
max_val = max(max(y_test_exp_m), max(y_pred_m))
plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')

# Apply formatter to axes
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)

plt.xlabel("Actual Revenue (in 100 Millions USD)")
plt.ylabel("Predicted Revenue (in 100 Millions USD)")
plt.title("XGBoost: Predicted vs Actual Box Office Revenue")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


top_10_features = feature_importance.head(10)

# Plot
plt.figure(figsize=(10, 6))
top_10_features.plot(kind='barh', color='skyblue')
plt.gca().invert_yaxis()  # Highest importance at the top

plt.title('Top 10 Feature Importances (Gradient Boosting)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()


# Loop through all features and print
for feature, importance in feature_importance.items():
    print(f"{feature}: {importance:.5f}")

# Assume X is your feature DataFrame and model is your trained model
# print(list(df.columns))


joblib.dump(model, "newxgboost.pkl")

# from sklearn.inspection import PartialDependenceDisplay

# features = ['budget', 'Rating', 'average_revenue_y',
# 'average_revenue_x',
# 'genre_Animation',
# "family_friendly"]  # replace with your top features

# PartialDependenceDisplay.from_estimator(model, X_train, features)
# plt.tight_layout()  # optional: prevents label overlap
# plt.show()