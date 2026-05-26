
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CineMatch AI", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0F1117;
    color: white;
}
.movie-card {
    background-color: #1E1E2F;
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

movies = pd.read_csv("data/movies.csv")

movies["tags"] = movies["genre"] + " " + movies["overview"]

vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(movies["tags"])

similarity = cosine_similarity(vectors)

def recommend(movie_name):
    movie_name = movie_name.lower()
    index = movies[movies["title"].str.lower() == movie_name].index

    if len(index) == 0:
        return []

    idx = index[0]
    distances = list(enumerate(similarity[idx]))
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

    recommended = []
    for i in movies_list:
        recommended.append(movies.iloc[i[0]].title)

    return recommended

st.title("🎬 CineMatch AI")
st.subheader("Machine Learning Movie Recommendation System")

movie_list = movies["title"].values
selected_movie = st.selectbox("Search Movie", movie_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    if recommendations:
        st.success("Recommended Movies:")
        for movie in recommendations:
            st.markdown(f"<div class='movie-card'>🍿 {movie}</div>", unsafe_allow_html=True)
    else:
        st.error("Movie not found.")
