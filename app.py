import pickle
import streamlit as st
import numpy as np
import math

st.header('Book Recommender System Using Machine Learning')
model = pickle.load(open('artifacts/model.pkl', 'rb'))
book_names = pickle.load(open('artifacts/book_names.pkl', 'rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))


def fetch_poster(suggestion):
    """Fetches book posters (image URLs) based on book suggestions."""
    poster_url = []
    for book_id in suggestion:
        book_name = book_pivot.index[book_id]
        idx = np.where(final_rating['title'] == book_name)[0][0]
        url = final_rating.iloc[idx]['image_url']
        poster_url.append(url)
    return poster_url


def recommend_book(book_name):
    """Generates book recommendations based on the input book."""
    books_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    poster_url = fetch_poster(suggestion[0])  

    for i in range(len(suggestion[0])):  
        books_list.append(book_pivot.index[suggestion[0][i]])

    return books_list, poster_url


selected_books = st.selectbox(
    "Type or select a book from the dropdown",
    book_names
)

if st.button('Show Recommendation'):
    if selected_books:
        recommended_books, poster_url = recommend_book(selected_books)

        if len(recommended_books) == len(poster_url):
            rows = math.ceil(len(recommended_books) / 4)
            
            for i in range(rows):
                cols = st.columns(4)  
                
                for j in range(4):
                    idx = i * 4 + j
                    if idx < len(recommended_books):  
                        with cols[j]:
                            st.text(recommended_books[idx]) 
                            st.image(poster_url[idx])  
        else:
            st.warning("Something went wrong with fetching the recommendations.")
    else:
        st.warning("Please select a book to get recommendations.")
