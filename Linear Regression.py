import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_error, explained_variance_score, median_absolute_error
from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec
import ast


df = pd.read_csv("Movies_no_revenue_zero.csv") 


df = df.dropna(subset=["budget", "runtime", "revenue", "popularity", "vote_average", "vote_count", "genres", "production_countries", "lead_cast", "director", "production_companies"])


def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except:
        return []

def extract_company_names(companies):
    try:
        companies_list = ast.literal_eval(companies)  
        return [company['name'] for company in companies_list] 
    except:
        return []

def extract_country_names(countries):
    try:
        countries_list = ast.literal_eval(countries)  
        return [country['name'] for country in countries_list]  
    except:
        return []

def extract_list_items(lst):
    try:
        return ast.literal_eval(lst) if isinstance(lst, str) else lst
    except:
        return []

df['genres'] = df['genres'].apply(extract_list_items)
df['production_countries'] = df['production_countries'].apply(extract_country_names)
df['lead_cast'] = df['lead_cast'].apply(safe_literal_eval)
df['director'] = df['director'].apply(lambda x: [x])  
df['production_companies'] = df['production_companies'].apply(extract_company_names)


genres_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
genres_encoded = genres_ohe.fit_transform(df['genres'].apply(lambda x: ','.join(x)).values.reshape(-1, 1))

countries_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
production_countries_encoded = countries_ohe.fit_transform(df['production_countries'].apply(lambda x: ','.join(x)).values.reshape(-1, 1))


all_text = df['lead_cast'].tolist() + df['director'].tolist() + df['production_companies'].tolist()
w2v_model = Word2Vec(sentences=all_text, vector_size=10, window=5, min_count=1, workers=4)

def get_w2v_vector(words, model, size=10):
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(size)


df['lead_cast_vec'] = df['lead_cast'].apply(lambda x: get_w2v_vector(x, w2v_model))
df['director_vec'] = df['director'].apply(lambda x: get_w2v_vector(x, w2v_model))
df['production_companies_vec'] = df['production_companies'].apply(lambda x: get_w2v_vector(x, w2v_model))


X = np.hstack([
    df[['budget', 'runtime', 'popularity', 'vote_average', 'vote_count']].values,
    genres_encoded,
    production_countries_encoded,
    np.vstack(df['lead_cast_vec']),
    np.vstack(df['director_vec']),
    np.vstack(df['production_companies_vec'])
])
y = df['revenue'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R^2 Score:", r2_score(y_test, y_pred))


mae = mean_absolute_error(y_test, y_pred)


mse = mean_squared_error(y_test, y_pred)


rmse = np.sqrt(mse)


evs = explained_variance_score(y_test, y_pred)


medae = median_absolute_error(y_test, y_pred)


print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"Explained Variance Score: {evs}")
print(f"Median Absolute Error: {medae}")
