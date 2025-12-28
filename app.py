import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# PAGE CONFIGURATION
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

#LOADING CSV FILE
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    movies = movies[['title', 'overview', 'genres', 'homepage']]
    movies.dropna(subset=['title', 'overview'], inplace=True)
    movies.reset_index(drop=True, inplace=True)
    return movies

@st.cache_resource
def build_similarity(movies):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    vectors = tfidf.fit_transform(movies['overview'])
    similarity = cosine_similarity(vectors)
    return similarity

movies = load_data()
similarity = build_similarity(movies)

# RECOMEND BY MOVIE NAME AND MOVIE GENRE
def recommend_by_movie(movie_name, n):
    index = movies[movies['title'] == movie_name].index[0]
    distances = similarity[index]
    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:n+1]
    return [(movies.iloc[i[0]].title, movies.iloc[i[0]].homepage) for i in movie_list]

def recommend_by_genre(genre, n):
    genre_movies = movies[movies['genres'].str.contains(genre, case=False)]
    return [(row['title'], row['homepage']) for i, row in genre_movies.head(n).iterrows()]

# HEADING
st.markdown(
    "<h1 style='text-align:center;'>🎬 Movie Recommendation System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style='text-align:center; font-size:18px;'>
    Discover movies you’ll love based on similarity, genre, or a surprise pick 🍿  
    Choose your preference below and get instant recommendations.
    </p>
    """,
    unsafe_allow_html=True
)

# MOVIE IMAGE
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
image = Image.open("movie.jpg")
st.image(image, width=450)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# TO SELECT NO OF MOVIES
num_movies = st.slider(
    "🎯 Number of recommendations",
    min_value=1,
    max_value=15,
    value=5
)

st.markdown("---")

# SEARCH BY TITLE OR GENRE
search_type = st.radio(
    "🔍 Choose search method:",
    ["Movie Title", "Genre"],
    horizontal=True
)

# SEARCH MOVIE
if search_type == "Movie Title":
    selected_movie = st.selectbox(
        "Select a movie",
        movies['title'].values
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎬 Get Recommendations"):
            results = recommend_by_movie(selected_movie, num_movies)
            st.subheader("⭐ Recommended Movies")
            for title, homepage in results:
                st.write("•", title)
                if pd.notna(homepage) and homepage.strip() != "":
                    st.markdown(f"[Visit Homepage]({homepage})", unsafe_allow_html=True)

    with col2:
        if st.button("🎲 Surprise Me"):
            random_movie = random.choice(movies['title'].values)
            results = recommend_by_movie(random_movie, num_movies)
            st.subheader(f"🎉 Because you might like: **{random_movie}**")
            for title, homepage in results:
                st.write("•", title)
                if pd.notna(homepage) and homepage.strip() != "":
                    st.markdown(f"[Visit Homepage]({homepage})", unsafe_allow_html=True)

# SEARCH BY GENRE
else:
    all_genres = sorted(
        set(
            genre.strip()
            for genres in movies['genres']
            for genre in genres.split('|')
        )
    )

    selected_genre = st.selectbox(
        "Select a genre",
        all_genres
    )

    if st.button("🎭 Get Recommendations"):
        results = recommend_by_genre(selected_genre, num_movies)
        st.subheader(f"⭐ Movies in {selected_genre}")
        for title, homepage in results:
            st.write("•", title)
            if pd.notna(homepage) and homepage.strip() != "":
                st.markdown(f"[Visit Homepage]({homepage})", unsafe_allow_html=True)

st.markdown("---")
st.caption("Built using Streamlit and Machine Learning")