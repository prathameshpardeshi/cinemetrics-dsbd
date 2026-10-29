# CineMetrics // DSBD Analytics Dashboard

CineMetrics is an end-to-end Data Science & Big Data Analytics (DSBD) platform focused on Indian cinema and OTT releases. The platform showcases a production-ready data pipeline, statistical inference, natural language processing, predictive regression modeling, and a distributed MapReduce simulation.

---

## Features & Modules

### 01 / Pipeline & Data Wrangling
- **Audit & Ingestion**: Automated ingestion of raw cinema data (`indian_ott_movies.csv`).
- **Missing Value Imputation**: Median imputation strategy for missing numerical fields (`Budget_Cr`, `Marketing_Spend_Cr`, `Runtime_Min`).
- **IQR Outlier Capping**: Detection and clipping using $[Q_1 - 1.5 \times \text{IQR},\, Q_3 + 1.5 \times \text{IQR}]$ for extreme anomalies in financial indicators.
- **Visual Validation**: Side-by-side Plotly boxplots comparing distributions before and after outlier removal.

### 02 / Exploratory Statistical Inference
- **Genre Grouping**: Parametric aggregations including Mean, Median, Standard Deviation, Variance, and Skewness.
- **Correlation Analysis**: Interactive Pearson correlation matrix heatmap (`text_auto=".2f"`).
- **Multivariate Scatter**: Budget vs. Box Office collections partitioned by Genre and sized by Critic Score.

### 03 / NLP Sentiment & Naive Bayes
- **TF-IDF Vectorization**: Text feature extraction on audience reviews using Scikit-Learn's `TfidfVectorizer`.
- **Classification**: Multinomial Naive Bayes (`MultinomialNB`) classification for audience sentiment.
- **Diagnostic Metrics**: Confusion matrix heatmap, model accuracy, precision, recall, and F1-score.
- **Feature Importance**: Horizontal rank visualization of top predictive tokens for positive and negative sentiment.
- **Pipeline Inspector**: Tokenizer and vectorizer inspector for individual movies in the dataset.

### 04 / Box Office Regression & Simulator
- **Predictive Engine**: Linear regression modeling box office revenue from budget, marketing spend, runtime, critic score, and sentiment flags.
- **Evaluation**: $R^2$ score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) against actual collections with diagonal parity plots.
- **Greenlight Simulator**: Interactive release scenario simulator with real-time box office predictions, projected ROI %, and sentiment classification confidence.

### 05 / MapReduce WordCount Engine
- **Three-Phase Simulation**:
  - **Map Phase**: Emits intermediate key-value pairs `(word, 1)` across audience reviews.
  - **Shuffle & Sort**: Groups tokens by unique keys and calculates reduction metrics.
  - **Reduce Phase**: Aggregates token counts and displays the top 15 most frequent keywords per genre.

---

## Project Structure

```text
cinemetrics-dsbd/
├── app.py                  # Streamlit dark-themed dashboard (5 views)
├── backend.py              # DSBD data pipeline, ML models, & MapReduce engine
├── generate_data.py        # Synthetic dataset generator with injected anomalies
├── indian_ott_movies.csv   # 2,500 records dataset
├── requirements.txt        # Python library dependencies
├── .gitignore              # Git ignore configuration
└── README.md               # Project documentation
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/cinemetrics-dsbd.git
cd cinemetrics-dsbd
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Regenerate the dataset
```bash
python generate_data.py
```

### 4. Run the application
```bash
streamlit run app.py
```
Or with Python module syntax:
```bash
python -m streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Design System
- **Theme**: Editorial Letterboxd / IMDb Pro Dark
- **Palette**: Dark background (`#0b0d10`), card containers (`#14181d`), amber accents (`#e5a93c`), and teal accents (`#2dd4bf`).
- **Typography**: Inter for UI, JetBrains Mono for metrics and numerical tables.
