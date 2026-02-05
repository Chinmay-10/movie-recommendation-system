# Movie Recommendation System (Hybrid)

A **hybrid movie recommendation system** that combines **content-based filtering** and **collaborative filtering**, deployed as an interactive **Streamlit web application**.

This project demonstrates **end-to-end recommender system design**, clean modular architecture, and practical machine learning integration suitable for real-world applications.

---

## 🚀 Key Features

- Hybrid recommendation engine (Content-Based + Collaborative)
- Cold-start handling using content similarity
- Weighted hybrid scoring mechanism
- Interactive Streamlit web interface
- Visual score breakdown for explainability
- Clean, modular, production-ready code structure

---

## 🧠 System Overview

### 1. Content-Based Filtering
- Uses movie metadata (genres)
- Recommends movies similar to a selected reference movie

### 2. Collaborative Filtering
- Uses historical user ratings
- Recommends movies based on similar user behavior

### 3. Hybrid Strategy
Final recommendation score is calculated as:

```

hybrid_score = 0.6 × content_score + 0.4 × collaborative_score

```

Results from both models are merged, deduplicated, and ranked.

---

## 🖥️ Tech Stack

- Python 3
- Pandas & NumPy
- Scikit-learn
- Streamlit
- Matplotlib

---

## 📁 Project Structure

```

RecommendationSystem/
│
├── app.py                     # Streamlit application
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
├── .gitignore                 # Git ignore rules
│
├── data/
│   ├── movies.csv
│   └── ratings.csv
│
└── src/
├── preprocess.py          # Data loading & preprocessing
├── content_model.py       # Content-based recommender
├── collaborative_model.py # Collaborative filtering logic
├── hybrid.py              # Hybrid recommender engine
└── explainability.py      # Visualization utilities

````

---

## ▶️ How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/Chinmay-10/movie-recommendation-system.git
cd movie-recommendation-system
````

### 2. Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📊 Application Output

* Personalized movie recommendations
* Genre details for each movie
* Hybrid score visualization
* Transparent contribution of content vs collaborative filtering

---

## 🎯 What This Project Demonstrates

* Practical recommender system implementation
* Hybrid ML system design
* Clean software engineering practices
* Deployable machine learning application
* Real-world data handling and ranking logic

---

## 📌 Future Improvements

* User authentication
* Persistent recommendation storage
* Advanced similarity models (TF-IDF / embeddings)
* Evaluation metrics (Precision, Recall)
* Cloud deployment

---

## 👤 Author

**Chinmay Patil**
AI & Data Science Undergraduate
Focused on Machine Learning Systems and Applied AI

```