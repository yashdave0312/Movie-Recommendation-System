# 🎬 Movie Recommendation System

A **Movie Recommendation System** built using **Natural Language Processing (NLP)** and **Machine Learning**. The system recommends movies similar to a movie selected by the user by analyzing information such as genres, keywords, overview, cast, and other movie-related features.

## 📌 Project Overview

The goal of this project is to build a recommendation system that can identify movies with similar characteristics.

The project uses **NLP techniques** to process textual movie information and convert it into numerical features. **Cosine Similarity** is then used to measure the similarity between movies and generate recommendations.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-learn
* Natural Language Processing
* Machine Learning

## 🧠 Concepts Used

* Data Cleaning
* Text Preprocessing
* Tokenization
* Stopword Removal
* Stemming / Lemmatization
* Feature Extraction
* Bag of Words / TF-IDF
* Cosine Similarity
* Content-Based Recommendation

## ⚙️ How It Works

1. Load the movie dataset.
2. Select useful movie features such as:

   * Title
   * Genres
   * Keywords
   * Overview
   * Cast
   * Director
3. Clean and preprocess the text data.
4. Combine the important textual features.
5. Convert the text into numerical vectors using NLP techniques.
6. Calculate similarity between movies using **Cosine Similarity**.
7. When the user selects a movie, find the movies most similar to it.
8. Display the recommended movies.

## 📊 Recommendation Approach

This project follows a **Content-Based Filtering** approach.

The system recommends movies based on the similarity of their content rather than relying on ratings or preferences from other users.

### Example

If the user selects:

**Interstellar**

The system may recommend movies such as:

* The Martian
* Gravity
* Inception
* Arrival
* 2001: A Space Odyssey

The recommendations are based on similarities in the movie's available textual features.

```

## 🚀 Future Improvements

* Create a Streamlit web interface.
* Add movie posters and additional information.
* Improve recommendations using multiple NLP techniques.
* Add user-based recommendations.
* Deploy the recommendation system online.

## 🎯 Learning Outcome

Through this project, I learned how **NLP and Machine Learning can be combined to build a real-world recommendation system**, including text preprocessing, feature extraction, vectorization, similarity calculation, and content-based filtering.
