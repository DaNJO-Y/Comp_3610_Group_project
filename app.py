import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
from datetime import datetime
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = joblib.load("box_office_predictor_xgboost.pkl")
actor_encoder = joblib.load("actor_encoder.pkl")
director_encoder = joblib.load("director_encoder.pkl")
rating_encoder = joblib.load("rating_encoder.pkl")
# List of model feature columns
model_features = ['runtime', 'budget', 'country_United States of America', 'country_France',
 'country_United Kingdom', 'country_Germany', 'country_Japan',
 'country_Canada', 'country_India', 'country_Italy', 'country_Spain',
 'country_South Korea', 'country_other', 'company_Paramount Pictures',
 'company_Universal Pictures', 'company_Twentieth Century Fox Film Corporation',
 'company_Columbia Pictures', 'company_Warner Bros.',
 'company_New Line Cinema', 'company_Metro-Goldwyn-Mayer (MGM)',
 'company_Walt Disney Pictures', 'company_Miramax Films',
 'company_United Artists', 'company_TriStar Pictures',
 'company_Columbia Pictures Corporation', 'company_Touchstone Pictures',
 'company_Orion Pictures', 'company_France 2 Cinéma',
 'company_Fox Searchlight Pictures', 'company_BBC Films',
 'company_Village Roadshow Pictures', 'company_Gaumont',
 'company_DreamWorks SKG', 'company_Channel Four Films',
 'company_Regency Enterprises', 'company_Canal+', 'company_Lions Gate Films',
 'company_Hollywood Pictures', 'company_StudioCanal', 'company_Rai Cinema',
 'company_Toho Company', 'company_Imagine Entertainment',
 'company_CJ Entertainment', 'company_other', 'genre_Science Fiction',
 'genre_TV Movie', 'genre_Western', 'genre_Animation', 'genre_Adventure',
 'genre_Crime', 'genre_Family', 'genre_Music', 'genre_Thriller',
 'genre_Action', 'genre_War', 'genre_Fantasy', 'genre_Foreign',
 'genre_Documentary', 'genre_History', 'genre_Romance', 'genre_Mystery',
 'genre_Comedy', 'genre_Horror', 'genre_Drama', 'director_encoded',
 'release_year', 'release_month', 'release_day_of_week', 'main_actor_encoded', 'rating_encoded']

# model_features = ['runtime', 'budget', 'country_United States of America', 'country_France',
#  'country_United Kingdom', 'country_Germany', 'country_Japan',
#  'country_Canada', 'country_India', 'country_Italy', 'country_Spain',
#  'country_South Korea', 'country_other', 'company_Paramount Pictures',
#  'company_Universal Pictures', 'company_Twentieth Century Fox Film Corporation',
#  'company_Columbia Pictures', 'company_Warner Bros.',
#  'company_New Line Cinema', 'company_Metro-Goldwyn-Mayer (MGM)',
#  'company_Walt Disney Pictures', 'company_Miramax Films',
#  'company_United Artists', 'company_TriStar Pictures',
#  'company_Columbia Pictures Corporation', 'company_Touchstone Pictures',
#  'company_Orion Pictures', 'company_France 2 Cinéma',
#  'company_Fox Searchlight Pictures', 'company_BBC Films',
#  'company_Village Roadshow Pictures', 'company_Gaumont',
#  'company_DreamWorks SKG', 'company_Channel Four Films',
#  'company_Regency Enterprises', 'company_Canal+', 'company_Lions Gate Films',
#  'company_Hollywood Pictures', 'company_StudioCanal', 'company_Rai Cinema',
#  'company_Toho Company', 'company_Imagine Entertainment',
#  'company_CJ Entertainment', 'company_other', 'genre_Science Fiction',
#  'genre_TV Movie', 'genre_Western', 'genre_Animation', 'genre_Adventure',
#  'genre_Crime', 'genre_Family', 'genre_Music', 'genre_Thriller',
#  'genre_Action', 'genre_War', 'genre_Fantasy', 'genre_Foreign',
#  'genre_Documentary', 'genre_History', 'genre_Romance', 'genre_Mystery',
#  'genre_Comedy', 'genre_Horror', 'genre_Drama', 'release_year', 'release_month', 'release_day_of_week']

# # Dummy encoders (you must replace with your actual mappings)
# director_encoder = {'Steven Spielberg': 1, 'Christopher Nolan': 2}  # example
# actor_encoder = {'Tom Hanks': 1, 'Leonardo DiCaprio': 2}  # example

def process_form_data(form, director, lead_actor):
    # Start with all zeros
    data = {col: 0 for col in model_features}
    
    # Simple numeric fields
    data['runtime'] = float(form['runtime'])
    data['budget'] = float(form['budget'])

    # Handle date
    release_date = datetime.strptime(form['date'], '%Y-%m-%d')
    data['release_year'] = release_date.year
    data['release_month'] = release_date.month
    data['release_day_of_week'] = release_date.weekday()

    # One-hot for country
    country_col = f"country_{form['country']}" if f"country_{form['country']}" in data else 'country_other'
    data[country_col] = 1
    data['country_United States of America'] = 1 if form['country'] == 'United States' else 0

    # One-hot for company
    company_col = f"company_{form['company']}" if f"company_{form['company']}" in data else 'company_other'
    data[company_col] = 1

    # One-hot for genres (assumed comma-separated)
    genres = [g.strip() for g in form['genres'].split(',')]
    for genre in genres:
        genre_col = f"genre_{genre}"
        if genre_col in data:
            data[genre_col] = 1

    # Encoded director and actor
    # director_name = form.get('director', '')
    # if director_name:
    #     try:
    #         # Transform using the target encoder
    #         data['director_encoded'] = director_encoder.transform([director_name])[0]
    #     except KeyError:
    #         data['director_encoded'] = 0

    # # Handle main actor encoding
    # actor_name = form.get('lead_actor', '')
    # if actor_name:
    #     try:
    #         data['main_actor_encoded'] = actor_encoder.transform([actor_name])[0]
    #     except KeyError:
    #         data['main_actor_encoded'] = 0
    # print(data['main_actor_encoded'])
    dat = {"director": [director]}
    wow = pd.DataFrame(dat)
    x = director_encoder.transform(wow)

    dat = {"main_actor": [lead_actor]}
    wow = pd.DataFrame(dat)
    y = actor_encoder.transform(wow)
    print(y['main_actor'][0])

    print(x)
    data['director_encoded'] = x['director'][0]
    data['main_actor_encoded'] = y['main_actor'][0]
    # wow = pd.DataFrame({"director": [director]})
    # encoded_director = int(director_encoder.transform(wow)[0])  # extract scalar
    # data['director_encoded'] = [encoded_director]  # assign as list to make a valid DataFrame column

    # # Encode main actor
    # wow = pd.DataFrame({"main_actor": [lead_actor]})
    # encoded_actor = int(actor_encoder.transform(wow)[0])  # extract scalar
    # data['main_actor_encoded'] = [encoded_actor]
    # print(type(data["main_actor_encoded"]))
    # # Convert to DataFrame
    return pd.DataFrame([data])

@app.route('/')
def index():
    return render_template('model.html')
# Define the route to get the form and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    form = request.form
    print(form)
    # print(form['director'])
    # print(form['lead_actor'])
    df = process_form_data(form, form['director'], form['lead_actor'] )
    print(df.columns)
    # Get the feature names used during training
    feature_names = model.get_booster().feature_names
    print(feature_names)
    # Ensure the dataframe columns are in the same order as the training feature names
    df = df[feature_names]
    print(df.columns)
    # Make prediction
    prediction = model.predict(df)
    prediction = np.expm1(prediction)
    # Return the prediction result
    return jsonify({'prediction': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
