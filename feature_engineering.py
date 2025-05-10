import pandas as pd
import holidays
from datetime import datetime
import numpy as np



def is_holiday_release(date, country='US', window_days=10):
    """
    Check if a movie release date is near a major holiday using the `holidays` library.
    
    Args:
        date (datetime): Release date.
        country (str): Country code (e.g., 'US', 'IN', 'CA'). Default: 'US'.
        window_days (int): Days before/after a holiday to flag. Default: 7.
    
    Returns:
        bool: True if near a holiday, False otherwise.
    """
    year = date.year
    country_holidays = holidays.CountryHoliday(country, years=year)
    
    # Check if date is within ±window_days of any holiday
    for holiday_date in country_holidays.keys():
        delta = (date.date() - holiday_date).days
        if abs(delta) <= window_days:
            return True
    return False

def map_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df = pd.read_csv("hopefully.csv")




# print(df.info())


df = df.dropna()
df.drop(['Original Id', 'vote_count', 'vote_average', 'popularity'], axis=1, inplace=True)
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['year'] = df['release_date'].dt.year
# df['release_month'] = df['release_date'].dt.month
# df['release_day'] = df['release_date'].dt.day
df['release_dayofweek'] = df['release_date'].dt.dayofweek  # Monday=0, Sunday=6
# df['release_quarter'] = df['release_date'].dt.quarter
df['release_season'] = df['release_date'].dt.month.apply(map_season)
# Apply the function row-wise to flag near-holiday releases
df['holiday_release'] = df['release_date'].apply(
    lambda x: is_holiday_release(x, country='US') if pd.notnull(x) else False
)
df['family_friendly'] = df['Rating'].apply(
    lambda x: x in ['G', 'PG', 'PG-13']
)


mean_budget = df[df['budget'] > 0]['budget'].mean()

# Define the standard deviation for adding variance (you can adjust this value)
std_dev = 10000000  # You can adjust the level of variance you want

# Add random noise (variance) to the imputed values
noise = np.random.normal(0, std_dev, size=(df['budget'] == 0).sum())  # Generate noise only for zero values

# Replace zero budgets with the mean plus random noise using boolean indexing
df.loc[df['budget'] == 0, 'budget'] = mean_budget + noise

# Categorize the budgets into Low, Medium, High based on the quantiles
df['log_budget'] = np.log1p(df['budget'])
df['budget_category'] = pd.qcut(df['log_budget'], q=3, labels=['Low', 'Medium', 'High'])





df['main_country'] = df['production_countries'].apply(
    lambda x: eval(x)[0]['name'] if x and len(eval(x)) > 0 else 'Unknown'
)

# Get top 10 most common countries excluding 'United States of America' and 'Unknown'
top_countries = df[df['main_country'] != 'Unknown']['main_country'].value_counts().nlargest(10).index.tolist()
top_countries = [country for country in top_countries if country != 'United States of America']

# Create binary flags
df['country_United States of America'] = df['main_country'].apply(lambda x: 1 if x == 'United States of America' else 0)

# Create binary flags for top countries
for country in top_countries:
    df[f'country_{country}'] = df['main_country'].apply(lambda x: 1 if x == country else 0)

# Assign 'Unknown' entries to 'country_other'
df['country_other'] = df['main_country'].apply(lambda x: 1 if x not in top_countries and x != 'United States of America' and x != 'Unknown' else 0)






# Assume production_companies is a list of dicts
df['production_company'] = df['production_companies'].apply(
    lambda x: eval(x)[0]['name'] if x and len(eval(x)) > 0 else 'Unknown'
)

# Get top 10-20 most common production companies (excluding 'Unknown')
top_companies = df[df['production_company'] != 'Unknown']['production_company'].value_counts().nlargest(30).index.tolist()

# Create binary flags for top 10-20 production companies
for company in top_companies:
    df[f'company_{company}'] = df['production_company'].apply(lambda x: 1 if x == company else 0)

# Assign all other companies (not in the top 10-20) to 'company_other'
df['company_other'] = df['production_company'].apply(lambda x: 1 if x not in top_companies and x != 'Unknown' else 0)

import ast


# 1. Convert string representation of list to actual list
df['lead_cast'] = df['lead_cast'].fillna('[]')
df['lead_cast'] = df['lead_cast'].apply(ast.literal_eval)

# 2. Extract first actor (or 'None' if empty)
df['main_actor'] = df['lead_cast'].apply(lambda x: x[0] if len(x) > 0 else 'None')

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Auto-detect terminal width
pd.set_option('display.max_colwidth', None)  # Show full column content



df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Create binary flags for each unique genre
unique_genres = set([genre for genres_list in df['genres'] for genre in genres_list])

# Create one-hot encoded columns for each genre
for genre in unique_genres:
    df[f'genre_{genre}'] = df['genres'].apply(lambda x: 1 if genre in x else 0)

df['runtime_category'] = pd.cut(
    df['runtime'],
    bins=[0, 80, 120, 1000],
    labels=['Short', 'Medium', 'Long'],
    include_lowest=True
)


import pandas as pd
import ast
from collections import defaultdict
from datetime import datetime

# Ensure correct data types
df['release_date'] = pd.to_datetime(df['release_date'])

# Initialize containers
actor_data = defaultdict(lambda: {'revenues': [], 'dates': [], 'genres': set()})
director_data = defaultdict(lambda: {'revenues': [], 'dates': [], 'genres': set()})

# Populate stats
for _, row in df.iterrows():
    release_date = row['release_date']
    revenue = row['revenue']
    genres = row['genres']
    
    # Actors
    for actor in row['lead_cast']:
        actor_data[actor]['revenues'].append(revenue)
        actor_data[actor]['dates'].append(release_date)
        actor_data[actor]['genres'].update(genres)

    # Director
    director = row['director']
    if pd.notna(director):
        director_data[director]['revenues'].append(revenue)
        director_data[director]['dates'].append(release_date)
        director_data[director]['genres'].update(genres)

# Convert to DataFrames
def build_stats_df(data_dict, role_name):
    stats = []
    current_time = datetime.now()
    for name, info in data_dict.items():
        last_date = max(info['dates'])
        years_since_last = round((current_time - last_date).days / 365)
        stats.append({
            f'{role_name}_name': name,
            'last_appearance': last_date,
            'years_since_last_release': years_since_last,
            'average_revenue': sum(info['revenues']) / len(info['revenues']),
            'total_movies': len(info['revenues']),
            'experienced_genres': list(info['genres'])
        })
    return pd.DataFrame(stats)

# Build stats dataframes
actor_stats = build_stats_df(actor_data, 'actor')
director_stats = build_stats_df(director_data, 'director')

# Merge actor and director stats with main dataframe
df = df.merge(actor_stats, "inner", left_on="main_actor", right_on="actor_name")
df = df.merge(director_stats, "inner", left_on="director", right_on="director_name")




# Calculate average actor statistics (averaging all actors' stats)
avg_actor_stats = pd.DataFrame({
    'years_since_last_release': [round(actor_stats['years_since_last_release'].mean())],
    'average_revenue': [round(actor_stats['average_revenue'].mean())],
    'total_movies': [round(actor_stats['total_movies'].mean())]
})

# Calculate average director statistics (averaging all directors' stats)
avg_director_stats = pd.DataFrame({
    'years_since_last_release': [round(director_stats['years_since_last_release'].mean())],
    'average_revenue': [round(director_stats['average_revenue'].mean())],
    'total_movies': [round(director_stats['total_movies'].mean())]
})

from collections import Counter

def get_majority_genres(series):
    genre_counts = Counter()
    total = len(series)

    for genres in series:
        if isinstance(genres, list):
            genre_counts.update(genres)
        elif isinstance(genres, str):
            genre_counts.update(genres.split('|'))

    # Filter genres that appear in more than half of the records
    majority_genres = [genre for genre, count in genre_counts.items() if count > total / 2]
    return majority_genres

# Get majority genres
actor_majority_genres = get_majority_genres(actor_stats['experienced_genres'])
director_majority_genres = get_majority_genres(director_stats['experienced_genres'])

# Add to stats
avg_actor_stats['experienced_genres'] = [", ".join(actor_majority_genres)]
avg_director_stats['experienced_genres'] = [", ".join(director_majority_genres)]
avg_director_stats['last_appearance'] = pd.to_datetime( (25 - avg_director_stats['years_since_last_release']).astype(str) + "-01-01", yearfirst=True)
avg_actor_stats['last_appearance'] = pd.to_datetime((25-avg_actor_stats['years_since_last_release']).astype(str) + "-01-01", yearfirst=True)
avg_actor_stats['actor_name'] = "average"
avg_director_stats['director_name'] = "average"


actor_stats = pd.concat([actor_stats, avg_actor_stats[actor_stats.columns]], ignore_index=True)
director_stats = pd.concat([director_stats, avg_director_stats[director_stats.columns]], ignore_index=True)

actor_stats.to_csv("actor.csv", index=False)
director_stats.to_csv("director.csv", index=False)


# Initialize an empty set to store unique director-actor collaborations
collaborations = set()

# Loop over each movie to generate director-actor pairs
for _, row in df.iterrows():
    director = row['director']
    actors = row['lead_cast']  # Assuming 'lead_cast' is already a list of actors

    # For each actor in the 'lead_cast', create a director-actor collaboration pair
    for actor in actors:
        collaborations.add((director, actor))

# Convert the set of collaborations to a DataFrame
collaborations_df = pd.DataFrame(list(collaborations), columns=['director', 'actor'])

# Step 1: Convert the collaboration pairs to a set for fast lookup
collaboration_set = set(tuple(x) for x in collaborations_df[['director', 'actor']].values)

# Step 2: Define a function to check if (director, main_actor) exists in the set
def has_collaborated(row):
    return int((row['director'], row['main_actor']) in collaboration_set)




genre_columns = [col for col in df.columns if col.startswith('genre_')]

# Calculate mean revenue per genre
genre_avg_revenue = {}
for genre_col in genre_columns:
    genre_name = genre_col.replace('genre_', '')
    avg_rev = df[df[genre_col] == 1]['revenue'].mean()
    genre_avg_revenue[genre_name] = avg_rev

# Convert to DataFrame for better visualization
avg_revenue_df = pd.DataFrame(list(genre_avg_revenue.items()), columns=['Genre', 'Avg_Revenue'])

# Step 1: Compute average revenue per genre (as before)
genre_columns = [col for col in df.columns if col.startswith('genre_')]
genre_avg_revenue = {}
for genre_col in genre_columns:
    genre_name = genre_col.replace('genre_', '')
    genre_avg_revenue[genre_name] = df[df[genre_col] == 1]['revenue'].mean()

# Step 2: For each movie, compute the mean of its genre averages
def get_movie_genre_avg_revenue(genres_list):
    relevant_avgs = [genre_avg_revenue[g] for g in genres_list if g in genre_avg_revenue]
    return sum(relevant_avgs) / len(relevant_avgs) if relevant_avgs else None

# Assuming `df['genres']` is a list of genres (e.g., ['Action', 'Comedy'])
df['avg_genre_revenue'] = df['genres'].apply(get_movie_genre_avg_revenue)
pd.set_option('display.max_columns', None)



# Define allowed/standard ratings
valid_ratings = {
    'G': 'G',
    'PG': 'PG',
    'PG-13': 'PG-13',
    'R': 'R',
    'NC-17': 'NC-17',
    'Unrated': 'Unrated',
    'Not Rated': 'Unrated',
    'Approved': 'Unrated',
    'Passed': 'Unrated',
    'X': 'NC-17',
    'M': 'PG',
    'GP': 'PG',
    'M/PG': 'PG',
    'TV-G': 'G',
    'TV-PG': 'PG',
    'TV-14': 'PG-13',
    'TV-MA': 'R',
    'TV-Y': 'G',
    'TV-Y7': 'G',
    'TV-Y7-FV': 'G',
    'TV-13': 'PG-13',
    '13+': 'PG-13',
    '16+': 'R',
    '18+': 'R'
}

# Apply mapping and set anything not in map to 'Unknown'
df['Rating'] = df['Rating'].map(valid_ratings).fillna('Unknown')



df = df[df['Rating'] != "Unknown"]
budget_map = {'Low': 0, 'Medium': 1, 'High': 2}
runtime_map = {'Short': 0, 'Medium': 1, 'Long': 2}
season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
rating_map = {'G': 0, 'PG': 1, 'PG-13': 2, 'R': 3, 'NC-17': 4}  # Adjust based on your data

# Apply mappings
df['budget_category'] = df['budget_category'].map(budget_map)
df['runtime_category'] = df['runtime_category'].map(runtime_map)
df['release_season'] = df['release_season'].map(season_map)
df['Rating'] = df['Rating'].map(rating_map)


actor_experience = df['main_actor'].value_counts().rename('actor_experience')


director_experience = df['director'].value_counts().rename('director_experience')


df = df.merge(actor_experience, left_on='main_actor', right_index=True, how='left')
df = df.merge(director_experience, left_on='director', right_index=True, how='left')


df['actor_experience'] = df['actor_experience'].fillna(0)
df['director_experience'] = df['director_experience'].fillna(0)

df["genre_diversity"] = df["genres"].apply(
    lambda x: len(x)/len(unique_genres)
)

def is_sequel_by_title(title):
    sequel_keywords_expanded = [
    ' 2 ', ' 3 ', ' 4 ', ' 5 ',
    'Part 2', 'Part 3', 'Part 4', 'Part 5', 'Part 6', 'Part 7', 'Part 8', # More "Part"
    'Part II', 'Part III', 'Part IV', 'Part V', 'Part VI', 'Part VII', 'Part VIII', # More Roman Numerals
    'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', # More Roman Numerals (standalone)
    'returns', 'strikes back', 'rises', 'awakens', 'legacy', 'next generation', 'a new beginning',
    'episode i', 'episode ii', 'episode iii', 'episode iv', 'episode v', 'episode vi', 'episode vii', 'episode viii', 'episode ix', # "Episode"
    'episode 1', 'episode 2', 'episode 3', 'episode 4', 'episode 5', 'episode 6', 'episode 7', 'episode 8', 'episode 9', # "Episode" with numbers
    'the sequel', 'the next chapter', 'the final chapter', 'the conclusion' # "The" variations
]
    if isinstance(title, str) and any(keyword in title.lower() for keyword in sequel_keywords_expanded):
        return 1
    return 0

df['is_sequel'] = df['original_title'].apply(is_sequel_by_title)


top_actor = pd.read_csv("Top_actors.csv")
top_actors_set = set(top_actor['Actor_Name'].str.strip())
def check_top_actor(cast_list, top_actors_set):
    # Normalize top actor names
    normalized_top_actors = {actor.lower().strip() for actor in top_actors_set}

    if isinstance(cast_list, list):
        # Normalize cast names
        normalized_cast = [actor.lower().strip() for actor in cast_list if isinstance(actor, str)]
    else:
        return 0

    for actor in normalized_cast:
        if actor in normalized_top_actors:
            return 1
    return 0


df['has_top_actor'] = df['main_actor'].apply(lambda x: check_top_actor([x], top_actors_set))
df.to_csv("final_dataset.csv")
df.to_parquet("final_dataset.parquet")
