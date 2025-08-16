# Enhanced Movie Recommender System

## Project Overview
This project implements a **Movie Recommendation System** using **Python, Pandas, Scikit-learn, and SQL-style data handling**.  
It supports **two recommendation approaches**:
1. **Content-based Filtering** – Finds similar movies using **TF-IDF vectorization** and **cosine similarity** on movie metadata (titles + genres).
2. **Popularity-based Filtering** – Recommends movies based on **average ratings** and **number of votes**, ensuring reliability by filtering out low-rated or rarely rated movies.

---

## Workflow
1. **Data Loading & Validation**  
   - Reads `movies.csv` and `ratings.csv` with validation for missing columns, duplicates, and errors.  

2. **Content-based Pipeline**  
   - Combines movie titles + genres.  
   - Builds a **TF-IDF matrix** for textual similarity.  
   - Uses **cosine similarity** to recommend the most similar movies to a user’s input.  

3. **Popularity-based Pipeline**  
   - Aggregates ratings to compute **average rating** and **rating count**.  
   - Filters movies with at least `MIN_RATING_COUNT`.  
   - Calculates a **popularity score** (weighted mix of rating and rating count).  

4. **User Interaction**  
   - CLI-based interface where users can choose between **content-based** or **popularity-based** recommendations.  
   - Handles invalid inputs gracefully and provides fallback options.  

---

## Features
- Robust **error handling** for file issues and invalid inputs.  
- **Fuzzy matching** to handle slight misspellings in movie titles.  
- Performance logging for TF-IDF computation and data loading.  
- Flexible **top-N recommendations**.  
- Clean separation of **content-based** and **popularity-based** recommendation pipelines.  

---

## Outcomes
- Provides **personalized recommendations** when a user searches for a movie.  
- Suggests **globally popular movies** for users without a specific preference.  
- Demonstrates strong **data handling, NLP, and ML fundamentals** in Python.  

---

## Myself:
**Muthu Sanjay P - CE22B076**  
IIT Madras • Data/Analytics/ML Enthusiast  
Mail: ce22b076@smail.iitm.ac.in • [LinkedIn](https://www.linkedin.com/in/muthusanjay/)  • [GitHub](https://github.com/SanjuIIT)  
