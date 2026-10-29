"""
CineMetrics – Backend (DSBD Processing & ML Pipeline)
All data wrangling, NLP, regression, and MapReduce logic.
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, r2_score, mean_absolute_error, mean_squared_error,
)
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & Clean
# ─────────────────────────────────────────────────────────────────────────────
def load_and_clean_data(filepath="indian_ott_movies.csv"):
    raw_df = pd.read_csv(filepath)

    # --- audit: raw stats ---
    initial_shape = raw_df.shape
    null_counts = raw_df.isnull().sum()
    pre_skew = raw_df.skew(numeric_only=True)

    clean_df = raw_df.copy()

    # --- median imputation ---
    impute_cols = ["Budget_Cr", "Marketing_Spend_Cr", "Runtime_Min"]
    nulls_fixed = {}
    for col in impute_cols:
        n_miss = clean_df[col].isnull().sum()
        nulls_fixed[col] = int(n_miss)
        if n_miss > 0:
            clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    # --- IQR outlier capping ---
    cap_cols = ["Budget_Cr", "Box_Office_Est_Cr"]
    outliers_capped = {}
    for col in cap_cols:
        q1 = clean_df[col].quantile(0.25)
        q3 = clean_df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (clean_df[col] < lower) | (clean_df[col] > upper)
        outliers_capped[col] = int(mask.sum())
        clean_df[col] = clean_df[col].clip(lower, upper)

    post_skew = clean_df.skew(numeric_only=True)

    audit = {
        "initial_shape": initial_shape,
        "null_counts": null_counts,
        "nulls_fixed": nulls_fixed,
        "outliers_capped": outliers_capped,
        "pre_skew": pre_skew,
        "post_skew": post_skew,
    }
    return raw_df, clean_df, audit


# ─────────────────────────────────────────────────────────────────────────────
# 2. NLP – Sentiment Classification
# ─────────────────────────────────────────────────────────────────────────────
def train_nlp_model(df):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=800)
    X = vectorizer.fit_transform(df["User_Review_Text"])
    y = df["Sentiment_Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="Positive")
    rec = recall_score(y_test, y_pred, pos_label="Positive")
    f1 = f1_score(y_test, y_pred, pos_label="Positive")
    cm = confusion_matrix(y_test, y_pred, labels=["Positive", "Negative"])

    # top words per class
    feature_names = np.array(vectorizer.get_feature_names_out())
    class_labels = model.classes_
    top_words = {}
    for i, label in enumerate(class_labels):
        log_probs = model.feature_log_prob_[i]
        top_idx = log_probs.argsort()[-12:][::-1]
        top_words[label] = list(feature_names[top_idx])

    return {
        "vectorizer": vectorizer,
        "model": model,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "top_positive_words": top_words.get("Positive", []),
        "top_negative_words": top_words.get("Negative", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Regression – Box Office Prediction
# ─────────────────────────────────────────────────────────────────────────────
def train_regression_model(df):
    features = ["Budget_Cr", "Marketing_Spend_Cr", "Runtime_Min", "Critic_Score"]
    target = "Box_Office_Est_Cr"

    data = df[features + [target, "Sentiment_Label"]].dropna()
    X = data[features].copy()
    X["Sentiment_Flag"] = (data["Sentiment_Label"] == "Positive").astype(int)
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred.round(2)})

    return {
        "model": model,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "results_df": results_df,
        "coefficients": dict(zip(list(features) + ["Sentiment_Flag"], model.coef_)),
        "intercept": model.intercept_,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. MapReduce Word Count Simulation
# ─────────────────────────────────────────────────────────────────────────────
def run_mapreduce_wordcount(df, genre_filter=None):
    subset = df.copy()
    if genre_filter and genre_filter != "All":
        subset = subset[subset["Genre"] == genre_filter]

    stop = ENGLISH_STOP_WORDS
    pattern = re.compile(r"\b[a-zA-Z]{4,}\b")

    # --- MAP ---
    mapped = []
    for text in subset["User_Review_Text"].dropna():
        tokens = pattern.findall(text.lower())
        for t in tokens:
            if t not in stop:
                mapped.append((t, 1))

    map_sample = mapped[:20]

    # --- SHUFFLE ---
    unique_keys = len(set(k for k, _ in mapped))

    # --- REDUCE ---
    counter = Counter()
    for word, count in mapped:
        counter[word] += count

    top15 = counter.most_common(15)
    reduce_df = pd.DataFrame(top15, columns=["Word", "Count"])

    return {
        "map_sample": map_sample,
        "shuffle_unique_keys": unique_keys,
        "reduce_df": reduce_df,
        "total_tokens": len(mapped),
    }
