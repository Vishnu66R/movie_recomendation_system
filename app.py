import streamlit as st
import pandas as pd
import random
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# PAGE NAME
st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide"
)

# GET DATA FROM CSV FILE
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    movies = movies[
        [
            'title', 'overview', 'genres',
            'production_companies', 'release_date',
            'runtime', 'vote_average'
        ]
    ]
    movies.dropna(subset=['title', 'overview'], inplace=True)
    movies.reset_index(drop=True, inplace=True)
    return movies

# GENRES
def preprocess_genres(movies):
    def parse_genres(x):
        try:
            g = ast.literal_eval(x)
            return "|".join([i['name'] for i in g])
        except:
            return ""
    movies['genres'] = movies['genres'].apply(parse_genres)
    return movies

# COMPANIES
def preprocess_companies(movies):
    def parse_companies(x):
        try:
            c = ast.literal_eval(x)
            return ", ".join([i['name'] for i in c])
        except:
            return "N/A"
    movies['production_companies'] = movies['production_companies'].apply(parse_companies)
    return movies

movies = load_data()
movies = preprocess_genres(movies)
movies = preprocess_companies(movies)

# RECOMMENDATION KEYWORDS
@st.cache_resource
def build_similarity(data):
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    vectors = tfidf.fit_transform(data['overview'])
    return cosine_similarity(vectors)

similarity = build_similarity(movies)

# BY TITLE OR GENRE
def recommend_by_movie(movie_name, n):
    idx = movies[movies['title'] == movie_name].index[0]
    scores = similarity[idx]
    movies_list = sorted(
        list(enumerate(scores)),
        key=lambda x: x[1],
        reverse=True
    )[1:n+1]
    return movies.iloc[[i[0] for i in movies_list]]

def recommend_by_genre(genre, n):
    return movies[movies['genres'].str.contains(genre, case=False)].head(n)

# RATINGS
def render_stars(vote):
    if pd.isna(vote):
        return "N/A"
    stars = round(vote / 2)
    return "⭐" * stars + f" ({vote}/10)"

# PAGE HEADING
st.markdown(
    "<h1 style='text-align:center;'>Movie Recommendation System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;font-size:18px;'>We recommend movies by title, genre or a surprise pick 🍿</p>",
    unsafe_allow_html=True
)

# MOVIE IMAGE
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    try:
        st.image(Image.open("movie.jpg"), use_container_width=True)
    except:
        pass

st.markdown("---")

# RADIO BUTTON
search_type = st.radio(
    "🔍 Choose search method",
    ["Movie Title", "Genre"],
    horizontal=True
)

# SLIDER
num_movies = st.slider(
    "🎯 Number of recommendations",
    1, 15, 5
)

st.markdown("---")

# SEARCH BY TITLE OR GENRE
if search_type == "Movie Title":
    selected_movie = st.selectbox(
        "Select a movie",
        movies['title'].values
    )

    if st.button("🎬 Get Recommendations"):
        results = recommend_by_movie(selected_movie, num_movies)
        st.subheader("⭐ Recommended Movies")
        for _, movie in results.iterrows():
            st.markdown(f"### 🎬 {movie['title']}")
            st.write(f"**Genres:** {movie['genres']}")
            st.write(f"**Overview:** {movie['overview']}")
            st.write(f"**Production Companies:** {movie['production_companies']}")
            st.write(f"**Release Date:** {movie['release_date']}")
            st.write(f"**Runtime:** {movie['runtime']} mins")
            st.write(f"**Rating:** {render_stars(movie['vote_average'])}")
            st.markdown("---")

else:
    all_genres = sorted(
        set(
            g.strip()
            for x in movies['genres']
            for g in x.split('|')
        )
    )

    selected_genre = st.selectbox(
        "Select genre",
        all_genres
    )

    if st.button("🎭 Get Recommendations"):
        results = recommend_by_genre(selected_genre, num_movies)
        st.subheader(f"⭐ Movies in {selected_genre}")
        for _, movie in results.iterrows():
            st.markdown(f"### 🎬 {movie['title']}")
            st.write(f"**Genres:** {movie['genres']}")
            st.write(f"**Overview:** {movie['overview']}")
            st.write(f"**Production Companies:** {movie['production_companies']}")
            st.write(f"**Release Date:** {movie['release_date']}")
            st.write(f"**Runtime:** {movie['runtime']} mins")
            st.write(f"**Rating:** {render_stars(movie['vote_average'])}")
            st.markdown("---")

# SURPRISE ME
st.markdown("---")
st.subheader("🎲 Surprise Me")

if st.button("Show Random Movies"):
    random_movie = random.choice(movies['title'].values)
    results = recommend_by_movie(random_movie, num_movies)
    st.subheader(f"🎉 Because you liked **{random_movie}**")
    for _, movie in results.iterrows():
        st.markdown(f"### 🎬 {movie['title']}")
        st.write(f"**Genres:** {movie['genres']}")
        st.write(f"**Overview:** {movie['overview']}")
        st.write(f"**Production Companies:** {movie['production_companies']}")
        st.write(f"**Release Date:** {movie['release_date']}")
        st.write(f"**Runtime:** {movie['runtime']} mins")
        st.write(f"**Rating:** {render_stars(movie['vote_average'])}")
        st.markdown("---")

st.markdown("---")
st.caption("Built with Streamlit by VR2048")