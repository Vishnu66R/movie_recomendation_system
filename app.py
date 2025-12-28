import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import ast

#PAGE TITLE
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

#DATA FROM CSV
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    movies = movies[['title', 'overview', 'genres',
                     'production_companies', 'release_date', 'runtime', 'vote_average']]
    movies.dropna(subset=['title', 'overview'], inplace=True)
    movies.reset_index(drop=True, inplace=True)
    return movies

#PROCESSING GENRE
def preprocess_genres(movies):
    def parse_genres(genres_str):
        try:
            genres_list = ast.literal_eval(genres_str)
            names = [g['name'] for g in genres_list]
            return "|".join(names)
        except:
            return ""
    movies['genres'] = movies['genres'].apply(parse_genres)
    return movies

movies = load_data()
movies = preprocess_genres(movies)

#FINDING RECOMENDATION KEYWORDS
@st.cache_resource
def build_similarity(movies):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    vectors = tfidf.fit_transform(movies['overview'])
    similarity = cosine_similarity(vectors)
    return similarity

similarity = build_similarity(movies)

#RECOMEND BY TITLE / GENRE
def recommend_by_movie(movie_name, n):
    index = movies[movies['title'] == movie_name].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:n+1]
    return [
        {
            'title': movies.iloc[i[0]].title,
            'genres': movies.iloc[i[0]].genres,
            'overview': movies.iloc[i[0]].overview,
            'production_companies': movies.iloc[i[0]].production_companies,
            'release_date': movies.iloc[i[0]].release_date,
            'runtime': movies.iloc[i[0]].runtime,
            'vote_average': movies.iloc[i[0]].vote_average
        }
        for i in movie_list
    ]

def recommend_by_genre(genre, n):
    genre_movies = movies[movies['genres'].str.contains(genre, case=False)]
    return [
        {
            'title': row['title'],
            'genres': row['genres'],
            'overview': row['overview'],
            'production_companies': row['production_companies'],
            'release_date': row['release_date'],
            'runtime': row['runtime'],
            'vote_average': row['vote_average']
        }
        for i, row in genre_movies.head(n).iterrows()
    ]

#PAGE HEADING
st.markdown("<h1 style='text-align:center;'>Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>We recomend similar movies by title, genre, or are you ready for a surprise pick now!!!</p>", unsafe_allow_html=True)

#movie image
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
try:
    image = Image.open("movie.jpg")
    st.image(image, width=450)
except:
    pass
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

#SEARCH BY TITLE OR GENRE
search_type = st.radio(" Choose search method:", ["Movie Title", "Genre"], horizontal=True)

#RECOMENDATION SLIDER
num_movies = st.slider("Number of recommendations", min_value=1, max_value=15, value=5)
st.markdown("---")

#SELECTION
if search_type == "Movie Title":
    selected_movie = st.selectbox("Select a movie", movies['title'].values)

    if st.button("🎬 Get Recommendations"):
        results = recommend_by_movie(selected_movie, num_movies)
        st.subheader("⭐ Recommended Movies")
        for movie in results:
            st.markdown(f"### 🎬 {movie['title']}")
            st.write(f"**Genres:** {movie['genres']}")
            st.write(f"**Overview:** {movie['overview']}")
            st.write(f"**Production Companies:** {movie.get('production_companies','N/A')}")
            st.write(f"**Release Date:** {movie.get('release_date','N/A')}")
            st.write(f"**Runtime:** {movie.get('runtime','N/A')} minutes")
            st.write(f"**Vote Average:** {movie.get('vote_average','N/A')}")
            st.markdown("---")

else:
    all_genres = sorted(set(genre.strip() for genres in movies['genres'] for genre in genres.split('|')))
    selected_genre = st.selectbox("Select a genre", all_genres)

    if st.button("🎭 Get Recommendations"):
        results = recommend_by_genre(selected_genre, num_movies)
        st.subheader(f"⭐ Movies in {selected_genre}")
        for movie in results:
            st.markdown(f"### 🎬 {movie['title']}")
            st.write(f"**Genres:** {movie['genres']}")
            st.write(f"**Overview:** {movie['overview']}")
            st.write(f"**Production Companies:** {movie.get('production_companies','N/A')}")
            st.write(f"**Release Date:** {movie.get('release_date','N/A')}")
            st.write(f"**Runtime:** {movie.get('runtime','N/A')} minutes")
            st.write(f"**Vote Average:** {movie.get('vote_average','N/A')}")
            st.markdown("---")

#SURPRISE ME
st.markdown("---")
st.subheader("🎲 Surprise Me")
if st.button("Show Random Recommendations"):
    random_movie = random.choice(movies['title'].values)
    results = recommend_by_movie(random_movie, num_movies)
    st.subheader(f"🎉 Because you might like: **{random_movie}**")
    for movie in results:
        st.markdown(f"### 🎬 {movie['title']}")
        st.write(f"**Genres:** {movie['genres']}")
        st.write(f"**Overview:** {movie['overview']}")
        st.write(f"**Production Companies:** {movie.get('production_companies','N/A')}")
        st.write(f"**Release Date:** {movie.get('release_date','N/A')}")
        st.write(f"**Runtime:** {movie.get('runtime','N/A')} minutes")
        st.write(f"**Average Ratings:** {movie.get('vote_average','N/A')}")
        st.markdown("---")

st.markdown("---")
st.caption("Built using Streamlit app by vr2048")