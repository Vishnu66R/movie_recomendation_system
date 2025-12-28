# 🎬 Movie Recommendation System

This is a **Movie Recommendation Web App** built using **Python, Streamlit, and Machine Learning**.  
It allows users to discover movies based on **movie similarity**, **genre preference**, or a **surprise pick**. 🍿

---

## **Features**

- **Search by Movie Title:** Get recommendations similar to your favorite movie.
- **Search by Genre:** Find top movies in your favorite genre.
- **Surprise Me Button:** Discover random movies to watch.
- **Adjustable Recommendations:** Use the slider to choose how many movies to get (1–15).
- **Homepage Links:** Visit the official website of the movie if available.
- **Interactive and Clean UI:** Simple, modern, and user-friendly interface built using Streamlit.

---

## **How It Works**

- **Content-based Recommendation:**  
  Uses the **overview** of movies and **TF-IDF vectorization** to find similar movies.  
- **Cosine Similarity:**  
  Measures how similar movies are to each other based on their descriptions.  
- **Genre Filtering:**  
  Allows users to filter movies based on selected genres.

---

## **Dataset**

- Source: TMDB 5000 Movies Dataset (from Kaggle)  
- Columns used: `title`, `overview`, `genres`, `homepage`  
- Preprocessed to remove unnecessary columns and duplicates.

---

## **How to Use**

1. Go to the **live Streamlit app**: [Insert your Streamlit Cloud URL here]
2. Choose a **search method**: Movie Title or Genre
3. Select the movie or genre
4. Adjust the **number of recommendations** using the slider
5. Click **Get Recommendations** or **Surprise Me**
6. See recommended movies with links to their homepage

---

## **Technologies Used**

- Python 3  
- Streamlit  
- Pandas  
- Scikit-learn (TF-IDF & Cosine Similarity)  
- PIL (for images)

---

## **Author**

- Name: Vishnu R 
- College: College of Engineering Perumon 
- Year/Semester: 3rd/6th 

---
