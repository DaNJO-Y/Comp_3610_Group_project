import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, median_absolute_error
import joblib

# Load and preprocess data (same as before)
df = pd.read_parquet("finally.parquet")
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

# Target engineering: Log-transform revenue
y = np.log1p(df["revenue"])  # Using log1p to handle zeros
X = df.drop("revenue", axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Initialize Random Forest with tuned hyperparameters
model = RandomForestRegressor(
    n_estimators=150,          # Number of trees in the forest
    max_depth=10,              # Maximum depth of each tree
    min_samples_split=8,       # Minimum number of samples required to split a node
    min_samples_leaf=5,        # Minimum number of samples required at each leaf node
    # max_features=40,     # Number of features to consider at each split
    bootstrap=True,            # Whether bootstrap samples are used
    random_state=42,
    n_jobs=-1                 # Use all available cores
)

# Fit the model
model.fit(X_train, y_train)

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
# print(feature_importance)
for feature, importance in feature_importance.items():
    print(f"{feature}: {importance:.5f}")



# data = {
#     # --- Top Features (Your Original Decimals -> Actual Values) ---
#     "average_revenue_y": [1200000000],  # Directors' avg (Zootopia + Tangled)
#     "average_revenue_x": [80000000],    # Voice actors' avg
#     "genre_Animation": [1],             # Binary (1 = True)
#     "family_friendly": [1],             # PG-rated
#     "company_Walt Disney Pictures": [1],# Replaced "Hollywood Pictures"
#     "Rating": ["PG"],                   # MPAA rating
#     "genre_Drama": [0],                 # Binary (0 = False)
#     "genre_Foreign": [0],               # Not foreign
#     "company_Touchstone Pictures": [0], # Not involved
#     "company_BBC Films": [0],           # Not involved
#     "company_United Artists": [0],      # Not involved
#     "director_experience": [15],        # Years directing (Byron Howard)
#     "genre_Adventure": [1],             # Binary (1 = True)
#     "budget": [150000000],              # Actual budget
#     "genre_Family": [1],                # Binary (1 = True)
#     "years_since_last_release_x": [5],  # Howard's last film (2016)
#     "company_other": [0],               # Not other
#     "company_Paramount Pictures": [0],  # Not involved
#     "country_Australia": [0],           # Not primary country
#     "country_United Kingdom": [0],      # Not primary country
#     "country_United States of America": [1], # Primary country
#     "runtime_category": ["Medium"],     # 102 mins = Medium
#     "years_since_last_release_y": [3],  # Disney's last animated (2018)
#     "budget_category": ["High"],        # $150M = High
#     "company_Gaumont": [0],             # Not involved
#     "company_New Line Cinema": [0],     # Not involved
#     "actor_experience": [10],           # Avg voice actor experience (yrs)
#     "genre_TV Movie": [0],              # Not a TV movie
#     "company_Metro-Goldwyn-Mayer (MGM)": [0], # Not involved
#     "genre_War": [0],                   # Not war genre
#     "runtime": [102],                   # Minutes
#     "country_Japan": [0],               # Not primary country
#     "avg_genre_revenue": [500000000],   # Avg animation revenue (est.)
#     "country_other": [0],               # Not other
#     "company_Columbia Pictures Corporation": [0], # Not involved
#     "company_Lions Gate Films": [0],    # Not involved
#     "company_Regency Enterprises": [0], # Not involved
#     "genre_Western": [0],               # Not western
#     "genre_Horror": [0],                # Not horror
#     "country_France": [0],              # Not primary country
#     "genre_Documentary": [0],           # Not documentary
#     "genre_Music": [1],                 # Musical film (1 = True)
#     "country_Germany": [0],             # Not primary country
#     "company_Canal+": [0],              # Not involved
#     "company_Miramax Films": [0],       # Not involved
#     "release_dayofweek": ["Friday"],    # Actual release day
#     "genre_Mystery": [0],               # Not mystery
#     "genre_Romance": [0],               # Not romance
#     "country_Italy": [0],               # Not primary country
#     "genre_Action": [0],                # Not action
#     "company_Fine Line Features": [0],  # Not involved
#     "company_Twentieth Century Fox Film Corporation": [0], # Not involved
#     "company_Columbia Pictures": [0],   # Not involved
#     "release_season": ["Holiday"],      # Thanksgiving release
#     "genre_Thriller": [0],              # Not thriller
#     "company_Universal Pictures": [0],  # Not involved
#     "genre_Crime": [0],                 # Not crime
#     "genre_Science Fiction": [0],       # Not sci-fi
#     "genre_Comedy": [1],                # Has comedic elements (1 = True)
#     "country_Canada": [0],              # Not primary country
#     "company_Fox Searchlight Pictures": [0], # Not involved
#     "genre_Fantasy": [1],               # Fantasy elements (1 = True)
#     "holiday_release": [1],             # Released near Thanksgiving (1 = True)
#     "country_India": [0],               # Not primary country
#     "genre_History": [0],               # Not historical
#     "company_Channel Four Films": [0],  # Not involved
#     "company_Warner Bros.": [0],        # Not involved
#     "company_France 2 Cinéma": [0],     # Not involved
#     "country_Spain": [0],               # Not primary country
#     "company_The Weinstein Company": [0], # Not involved
#     "company_TriStar Pictures": [0],    # Not involved
#     "company_Orion Pictures": [0],      # Not involved
#     "company_DreamWorks SKG": [0],      # Not involved
#     "company_Village Roadshow Pictures": [0], # Not involved
#     "company_Summit Entertainment": [0],# Not involved
#     "company_Lionsgate": [0],           # Not involved
#     "company_Imagine Entertainment": [0], # Not involved
# }
# dff = pd.DataFrame(data)
# prediction = np.expm1(model.predict(dff))

# print(prediction)

# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np


# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=y_test_exp, y=y_pred)


# max_val = max(max(y_test_exp), max(y_pred))
# plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')

# plt.xlabel("Actual Revenue")
# plt.ylabel("Predicted Revenue")
# plt.title("Random Forest: Predicted vs Actual Box Office Revenue")
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
plt.title("Random Forest: Predicted vs Actual Box Office Revenue")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Get top 10 features
top_10_features = feature_importance.head(10)

# Plot
plt.figure(figsize=(10, 6))
top_10_features.plot(kind='barh', color='skyblue')
plt.gca().invert_yaxis()  # Highest importance at the top

plt.title('Top 10 Feature Importances (Random Forest)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# Optional: Save the model
# joblib.dump(model, 'random_forest_model.pkl')