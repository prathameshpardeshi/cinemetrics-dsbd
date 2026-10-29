"""
CineMetrics – Dataset Generator
Generates indian_ott_movies.csv with 2,500 rows of realistic Indian cinema data.
"""

import numpy as np
import pandas as pd
import random

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

N = 2500

# ── 1. Movie IDs ─────────────────────────────────────────────────────────────
movie_ids = [f"IND-{1001 + i}" for i in range(N)]

# ── 2. Titles ─────────────────────────────────────────────────────────────────
title_words_1 = [
    "Kaala", "Andha", "Raat", "Khoon", "Zindagi", "Maut", "Chehra",
    "Raaz", "Aakhri", "Pehli", "Doosri", "Andheri", "Naagin", "Qaid",
    "Jazbaat", "Siyaah", "Laal", "Safed", "Toofan", "Bijli", "Aag",
    "Paani", "Zameen", "Aasman", "Roshni", "Andhera", "Khamoshi",
    "Awaaz", "Parchhain", "Saya", "Neela", "Peela", "Bura", "Achha",
    "Bhaari", "Gehri", "Pakki", "Kacchi", "Toot", "Jud", "Daag",
    "Dhuaan", "Aahad", "Tasveer", "Kagaz", "Zubaan", "Muqaam",
    "Khauf", "Darpan", "Ujala", "Dost", "Dushman", "Raat", "Din",
    "Sach", "Jhooth", "Raz", "Parda", "Gehraai", "Pathar",
]
title_words_2 = [
    "Ka Khel", "Ki Kahani", "Ka Raaz", "Ki Raat", "Ka Saya",
    "Ki Aawaz", "Ka Aakhri", "Ki Zindagi", "Ka Andha", "Ki Maut",
    "Ka Toofan", "Ki Parchhain", "Ka Chehra", "Ki Khamoshi",
    "Ka Dushman", "Ki Sach", "Ka Safar", "Ki Dastaan", "Ka Waqt",
    "Ki Tasveer", "Ka Nishaan", "Ki Khoj", "Ka Muqaam", "Ki Gehraai",
    "Ka Dhuaan", "Ki Aag", "Ka Jhooth", "Ki Awaaz", "Ka Raz",
    "Ki Roshni",
]
suffixes = [
    "", "", "", "",                      # Most titles have no suffix
    ": Chapter 1", ": Chapter 2", ": Part 1", ": Part 2",
    "(2019)", "(2020)", "(2021)", "(2022)", "(2023)", "(2024)",
    "- Reloaded", "- Unleashed", "- Returns",
]

south_titles = [
    "Vikram", "Pushpa", "RRR", "KGF", "Master", "Beast", "Thalapathy",
    "Varisu", "Leo", "Jawan", "Pathaan", "Animal", "Kalki", "Salaar",
    "Devara", "Jailer", "Vettaiyan", "Coolie", "Thug Life", "GOAT",
    "Lucky Baskhar", "Singham Returns", "Stree 2", "Bhool Bhulaiyaa 3",
    "War 2", "Sky Force", "Fateh", "Deva", "Game Changer", "Sankranthiki",
    "Sikandar", "Bigg Boss", "Nag Ashwin Film", "Mahesh Babu Film",
    "Pawan Kalyan Film", "Jr NTR Film", "Ram Charan Film", "Vijay Film",
    "Rajinikanth Film", "Kamal Haasan Film", "Mohanlal Film",
    "Mammootty Film", "Fahadh Film", "Dulquer Film", "Tovino Film",
    "Nivin Pauly Film", "Prithviraj Film", "Dileep Film", "Asif Ali Film",
]

def make_title():
    if random.random() < 0.35:
        base = random.choice(south_titles)
    else:
        base = random.choice(title_words_1) + " " + random.choice(title_words_2)
    suffix = random.choice(suffixes)
    return (base + " " + suffix).strip()

titles = [make_title() for _ in range(N)]

# ── 3. Genre ─────────────────────────────────────────────────────────────────
genres = ["Mystery-Thriller", "Crime-Noir", "Action-Thriller", "Psychological Drama", "Political Thriller"]
genre_weights = [0.22, 0.20, 0.25, 0.18, 0.15]
genre_col = random.choices(genres, weights=genre_weights, k=N)

# ── 4. Release Platform ───────────────────────────────────────────────────────
platforms = ["Theatrical", "Netflix India", "Prime Video", "JioCinema", "Disney+ Hotstar"]
platform_weights = [0.30, 0.22, 0.20, 0.16, 0.12]
platform_col = random.choices(platforms, weights=platform_weights, k=N)

# ── 5. Original Language ──────────────────────────────────────────────────────
languages = [
    "Hindi",
    "Tamil [Hindi Dub]",
    "Telugu [Hindi Dub]",
    "Malayalam [Hindi Dub]",
    "Kannada [Hindi Dub]",
]
lang_weights = [0.30, 0.25, 0.25, 0.12, 0.08]
language_col = random.choices(languages, weights=lang_weights, k=N)

# ── 6. Budget & Marketing ─────────────────────────────────────────────────────
budget_raw = np.random.uniform(8.0, 250.0, N).round(2)
marketing_raw = np.random.uniform(2.0, 60.0, N).round(2)

# ── 7. Runtime ────────────────────────────────────────────────────────────────
runtime_raw = np.random.randint(95, 186, N)

# ── 8. Critic Score ───────────────────────────────────────────────────────────
critic_raw = np.random.uniform(1.5, 9.8, N).round(1)

# ── 9. Sentiment & Reviews ───────────────────────────────────────────────────
# 60% positive, 40% negative
sentiment_col = random.choices(["Positive", "Negative"], weights=[0.60, 0.40], k=N)

positive_templates = [
    "An absolute masterpiece of suspense! The plot twists kept me on the edge of my seat throughout. The background score elevated every scene and the climax was jaw-dropping. Highly recommend!",
    "What a brilliant screenplay! The director has crafted every scene with incredible precision. The lead performance was haunting and the cinematography was breathtaking. Must watch.",
    "One of the finest thrillers I've seen from Indian cinema. The pacing was near-perfect and the screenplay had layers I didn't expect. The dubbing quality was surprisingly good for a Tamil film.",
    "Genuinely gripping from start to finish. The mystery unravels beautifully and the climax left me speechless. Background score is phenomenal – stayed in my head for days.",
    "Exceptional storytelling. Every character had depth and the plot never felt forced. The action sequences were choreographed brilliantly. This is what Indian cinema should look like.",
    "A psychological thriller done right. The screenplay keeps you second-guessing till the very end. Outstanding performances all around and the background music added so much tension.",
    "Blew my mind completely. The crime drama is layered, nuanced and deeply engaging. The cinematography is dark and moody – perfectly matching the tone. Watch it with subtitles!",
    "Brilliant direction and a haunting background score. The political thriller elements were handled with great maturity. The climax was both shocking and satisfying. 10/10.",
    "This film restored my faith in Indian thrillers. The plot twists are organic, not forced. The dubbing of the Telugu original is seamless and the acting is world-class.",
    "Absolutely loved every minute. The screenplay is tightly written with no loose ends. The background score during the climax gave me chills. One of the year's best releases.",
    "A rare gem that combines great storytelling with top-notch execution. The mystery thriller had me theorizing wrong answers till the end. The pacing is flawless. Superb!",
    "Terrific crime noir atmosphere. The film drips with tension and the performances are riveting. The ending is unexpected and deeply satisfying. A landmark in Indian cinema.",
    "Magnificent filmmaking. The action thriller sequences were intense without being over-the-top. The moral ambiguity of characters felt very real. Brilliant climax sequence.",
    "What a ride! The psychological drama messes with your head in the best way possible. The background score is haunting and the screenplay is original and inventive. Love it.",
    "Standing ovation worthy. The political thriller is bold, intelligent and extremely well-acted. The director doesn't hold back and the result is a cinematic triumph. Watch now!",
    "Simply stunning. The mystery elements were woven into the narrative so naturally. I didn't see the plot twist coming at all. The dubbing from Malayalam was excellent quality.",
    "A thrilling experience from the first frame to the last. The crime noir visuals are gorgeous and dark. The performances, especially the antagonist, are unforgettable. 5 stars.",
    "Incredible screenplay and direction. Every scene serves a purpose. The background score is a character in itself. The climax rewards patient viewers beautifully. Loved it!",
    "One of the most intelligent thrillers produced in recent years. The action-thriller blend works perfectly. No boring moments, great dialogues and a satisfying conclusion. Wow!",
    "A dark, gripping, beautifully crafted film. The psychological drama stays with you long after the credits roll. The dubbing was handled well for a Kannada film. Impressive work.",
    "The kind of thriller that makes you forget to breathe. Incredible background score that builds dread perfectly. The screenplay is airtight. The climax is absolutely worth the wait.",
    "Superb direction and outstanding cast. The film's atmosphere is thick with tension from the very first scene. Plot twists were genuinely surprising. Indian cinema at its finest!",
    "Loved the gritty, realistic approach to the crime drama. The screenplay doesn't pander to the audience and trusts their intelligence. A must-watch for thriller fans. Phenomenal.",
    "Riveting from start to finish. The mystery unravels at the perfect pace. Background score by the composer is career-best work. The climax scene made the whole theatre erupt.",
    "Exceptional attention to detail in every frame. The political thriller angle is handled with rare maturity for Indian cinema. The Hindi dubbing was smooth and well-done. Superb!",
    "A cinematic experience unlike anything else this year. The action sequences feel real and consequential. The screenplay has genuine emotional weight. Absolutely loved this film.",
    "The best thriller I've watched this decade. Plot twists that actually make sense in retrospect. The background score is hauntingly beautiful. The climax will be talked about forever.",
    "Gripping, original, and deeply satisfying. The psychological drama peels back layers of the human psyche in fascinating ways. Direction is masterful. A true cinematic achievement.",
    "Jaw-dropping sequences balanced with emotional depth. The crime noir setting is evocative and immersive. Each character is three-dimensional. The climax is absolutely unforgettable.",
    "Pure cinematic gold. The mystery thriller format is used brilliantly to explore deeper themes. Background score and cinematography work in perfect harmony. Loved every second.",
]

negative_templates = [
    "Extremely disappointing. The plot had so many holes it was hard to take seriously. The climax felt rushed and the pacing in the second half completely fell apart. Wasted 3 hours.",
    "Terrible execution of what could have been a great premise. The screenplay is a mess and the plot twists felt forced and illogical. The dubbing was also very poor. Don't bother.",
    "Boring and predictable. I guessed the twist in the first 20 minutes and nothing surprised me after that. The background score was intrusive and over the top. Very disappointing.",
    "A complete waste of time. The action sequences were repetitive and the story went nowhere. The climax was laughably bad. Poor screenplay with zero character development. Avoid.",
    "I wanted to like this but couldn't. The pacing is dreadful – the second half drags endlessly. The dubbing quality of this Tamil film was abysmal. Terrible background score too.",
    "Overlong and self-indulgent. The director needed a tighter editor. The mystery elements are poorly set up and the resolution makes no sense. Very disappointing for a thriller.",
    "Not worth your time. The crime drama is clichéd and derivative. The plot twists are telegraphed from miles away. Poor performances compound an already weak screenplay.",
    "A convoluted mess masquerading as a psychological drama. Too many characters, too many subplots and no coherent direction. The climax was the worst part of an already bad film.",
    "Dreadful. The film takes itself way too seriously and delivers nothing. The background score was overbearing. The Hindi dubbing felt disconnected from the emotions on screen. Terrible.",
    "Hugely underwhelming. The political thriller premise had promise but the execution was shallow and preachy. The pacing was off from the very beginning. Disappointed.",
    "The first half shows promise but the second half collapses completely. Plot holes everywhere and the climax was a massive letdown. The background score felt recycled. Skip it.",
    "I've seen better thrillers made on a fraction of the budget. The screenplay is lazy and full of convenient coincidences. The dubbing from Telugu was jarring and amateur. Avoid.",
    "Disappointing on every level. The mystery thriller premise is squandered. Characters behave irrationally to serve the plot. The climax insults the audience's intelligence. Terrible.",
    "One of the worst films I've seen this year. The action sequences feel fake and bloated. The screenplay meanders without purpose. The climax is incoherent. Total waste of money.",
    "Painfully boring. The so-called psychological drama is shallow and predictable. No real tension, no real surprises. The background score was the only decent thing in this film.",
    "Poorly made crime noir. The visual style tries to compensate for a terrible script but fails. The plot twists feel arbitrary. The Malayalam dubbing was particularly awful. Bad film.",
    "An embarrassing thriller. The mystery is resolved in the most unsatisfying way possible. The performances felt wooden and the direction was uninspired. Avoid at all costs.",
    "Nothing works in this film. The pacing is wrong, the screenplay is derivative and the climax is a shambles. The background score is irritating rather than tense. Huge disappointment.",
    "Tedious from start to finish. The action thriller promised excitement but delivered boredom. The plot twists come out of nowhere and don't make sense. Poor dubbing quality too.",
    "Absolutely terrible. The film tries to be clever but ends up being confusing and incoherent. The background score drowns out the dialogue. The climax is laughably predictable. Awful.",
    "A slog to get through. The political thriller has no nuance and the characters are cardboard cutouts. The second half is painfully slow. The climax resolves nothing. Very bad.",
    "The screenplay is full of plot holes that the director ignores hoping you won't notice. The dubbing was poor and distracted from the already weak story. Disappointing waste.",
    "Dull, derivative and dreadfully paced. The psychological drama has no real insight. The background score is repetitive and annoying. The climax felt like the writers gave up. Skip.",
    "A disaster of a crime drama. Overacted, poorly scripted and with a climax that makes no logical sense. The Kannada dubbing was terrible. One of the worst films in recent memory.",
    "Starts with an interesting premise and squanders it completely. The plot goes in circles without purpose. The background score is the worst thing about an already bad film. Terrible.",
    "Boring mystery thriller that mistakes complexity for intelligence. The plot twist is neither surprising nor satisfying. The pacing is tortuous. A complete disappointment from start to finish.",
    "Poor execution of a tired formula. The action sequences are unconvincing and the screenplay is lazy. The climax resolves the story in the most clichéd way possible. Very average.",
    "Waste of a talented cast. The screenplay gives them nothing to work with. The psychological drama elements are surface-level at best. The background score is overbearing. Avoid.",
    "Genuinely baffling filmmaking. The plot decisions make no sense and the climax contradicts everything that came before. The dubbing quality was the cherry on top of a bad experience.",
    "Tedious, predictable and poorly crafted. The crime noir aesthetic can't hide a hollow story. No plot twists worthy of the name. The background score was the only decent element.",
]

def pick_review(sentiment):
    if sentiment == "Positive":
        return random.choice(positive_templates)
    else:
        return random.choice(negative_templates)

review_col = [pick_review(s) for s in sentiment_col]

# ── 10. Box Office (linear regression target) ─────────────────────────────────
# BOE ≈ a*Budget + b*Marketing + c*Critic + d*Sentiment_flag + noise
# Tuned so R² ≈ 0.82–0.89
sentiment_flag = np.array([1 if s == "Positive" else 0 for s in sentiment_col])
noise = np.random.normal(0, 45, N)   # tuned -> R2 ~0.84-0.87

box_office_raw = (
    1.35  * budget_raw
    + 1.80  * marketing_raw
    + 12.50 * critic_raw
    + 22.00 * sentiment_flag
    - 15.00                      # base intercept (some films lose money)
    + noise
).round(2)
box_office_raw = np.clip(box_office_raw, 0.5, None)   # no negative collections

# ── 11. Inject ~4% missing values ─────────────────────────────────────────────
def inject_nans(arr, rate=0.04):
    arr = arr.astype(float)
    idx = np.random.choice(len(arr), size=int(rate * len(arr)), replace=False)
    arr[idx] = np.nan
    return arr

budget_col   = inject_nans(budget_raw.copy())
marketing_col = inject_nans(marketing_raw.copy())
runtime_col  = inject_nans(runtime_raw.astype(float).copy())

# ── 12. Inject 18 extreme outliers ───────────────────────────────────────────
def inject_outliers(arr, n=9, low=850, high=1400):
    idx = np.random.choice(np.where(~np.isnan(arr))[0], size=n, replace=False)
    arr[idx] = np.random.uniform(low, high, n).round(2)
    return arr

budget_col      = inject_outliers(budget_col,      n=9, low=850,  high=1400)
box_office_col  = inject_outliers(box_office_raw.copy(), n=9, low=1100, high=2800)

# ── 13. Assemble DataFrame ────────────────────────────────────────────────────
df = pd.DataFrame({
    "Movie_ID":           movie_ids,
    "Title":              titles,
    "Genre":              genre_col,
    "Release_Platform":   platform_col,
    "Original_Language":  language_col,
    "Budget_Cr":          budget_col,
    "Marketing_Spend_Cr": marketing_col,
    "Runtime_Min":        runtime_col,
    "Critic_Score":       critic_raw,
    "User_Review_Text":   review_col,
    "Sentiment_Label":    sentiment_col,
    "Box_Office_Est_Cr":  box_office_col,
})

# ── 14. Save ──────────────────────────────────────────────────────────────────
out_path = "indian_ott_movies.csv"
df.to_csv(out_path, index=False)
print(f"[OK] Dataset saved -> {out_path}")
print(f"     Shape : {df.shape}")
print("\n-- Column null counts --")
print(df.isnull().sum())
print("\n-- Budget_Cr outliers (>500 Cr) --")
print(df[df["Budget_Cr"] > 500][["Movie_ID", "Title", "Budget_Cr"]])
print("\n-- Box_Office_Est_Cr outliers (>1000 Cr) --")
print(df[df["Box_Office_Est_Cr"] > 1000][["Movie_ID", "Title", "Box_Office_Est_Cr"]])

# ── 15. Quick regression check ────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

check_df = df.dropna(subset=["Budget_Cr", "Marketing_Spend_Cr", "Runtime_Min"])
check_df = check_df[check_df["Budget_Cr"] < 500]           # exclude outliers for check
check_df = check_df[check_df["Box_Office_Est_Cr"] < 1000]

X = check_df[["Budget_Cr", "Marketing_Spend_Cr", "Critic_Score"]].copy()
X["Sentiment_Flag"] = (check_df["Sentiment_Label"] == "Positive").astype(int)
y = check_df["Box_Office_Est_Cr"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression().fit(X_train, y_train)
r2 = r2_score(y_test, lr.predict(X_test))
print(f"\n-- Linear Regression R2 (clean data) : {r2:.4f}")
print("   Target range: 0.82 to 0.89")
