import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('songs.csv')

df = df.drop_duplicates(subset='Song-Title')

df['Song-Title'] = df['Song-Title'].str.lower()
df['Artist'] = df['Artist'].str.lower().str.replace(' ', '')
df['Genre'] = df['Genre'].str.lower()

df['combined_features'] = df['Song-Title'] + ' ' + df['Artist'] + ' ' + df['Genre']

vectorizer = CountVectorizer()
vectorized_features = vectorizer.fit_transform(df['combined_features'])

cosine_sim = cosine_similarity(vectorized_features)

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = df[df['Song-Title'] == title].index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    song_indices = [i[0] for i in sim_scores]

    return df['Song-Title'].iloc[song_indices]

input_song = 'shape of you'
recommendations = get_recommendations(input_song)
print(recommendations)
