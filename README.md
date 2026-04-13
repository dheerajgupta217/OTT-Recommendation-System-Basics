# OTT Recommendation System (Regression + Collaborative Filtering)

## Overview

This project implements a hybrid recommendation system for an OTT platform using:

* **Regression Model** → Predict user ratings
* **Collaborative Filtering** → Recommend shows based on similar users

---

##  Features

* Predict user preferences using ML
* Recommend shows using user similarity
* File-based dataset (CSV)
* Scalable architecture for real-world systems

---

## Datasets

### 1. Regression Dataset

`ott_user_data.csv`

### 2. Collaborative Filtering Dataset

`ott_ratings.csv`

---

##  Models Used

### Regression

* Linear Regression
* Predicts rating score

###  Collaborative Filtering

* User-Based Filtering
* Cosine Similarity

---

##  Tech Stack

* Python
* Pandas
* Scikit-learn

---

##  Workflow

### Regression

1. Load dataset
2. Train model
3. Predict ratings

### Collaborative Filtering

1. Build user-item matrix
2. Compute similarity
3. Recommend unseen shows

---

##  Example Output

* Predicted rating: 4.2 ⭐
* Recommended shows: [104, 105]

---

##  Future Improvements

* Item-based collaborative filtering
* Matrix factorization (SVD)
* Deep learning recommender systems
* Real-time recommendation API
* UI with Streamlit

---

##  Use Cases

* OTT platforms (Netflix, Prime)
* E-commerce recommendations
* Music streaming apps

---

##  Contributing

Pull requests are welcome!

---

## License

MIT License
