
# CAP_3764_2026_Spring_Team_2

# Miami Flights Operational Analysis ✈️

## 📌 Project Overview

This project analyzes flight operations departing from **Miami International Airport (MIA)**
using machine learning to predict flight delay risk and discover operational patterns.

The dataset covers **January 1 – February 29, 2024** (winter season only).
All findings reflect winter operational patterns and should not be generalized to the full year.

Key objectives:
* Predict whether a flight will experience a significant delay (>15 minutes)
* Identify which operational factors contribute most to delay risk
* Discover hidden flight risk profiles through unsupervised clustering
* Provide actionable insights to support airport operational decision-making

---

## 🎯 Business Problem

Flight delays at MIA generate cascading disruptions — a single delayed aircraft
affects subsequent flights, crew scheduling, and passenger connections.
This project builds predictive models to flag high-risk flights before departure,
enabling proactive resource allocation by ground operations teams.

---

## 📂 Repository Structure

```
CAP_3764_2026_Spring_Team_2/
│
├── data/
│   ├── raw/
│   │   └── flight_data_2024.csv          # Original raw dataset (unmodified)
│   └── processed/
│       └── mia_flights_clean.csv         # Cleaned and filtered dataset (19,396 flights)
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb            # Data preparation and preprocessing
│   ├── 02_eda.ipynb                      # Exploratory data analysis
│   ├── 03_tree_based_model.ipynb         # Decision Tree, Random Forest, XGBoost
│   └── 04_clustering_model.ipynb         # PCA + KMeans + DBSCAN clustering
│
├── my_modules/
│   └── model_utils.py                    # Reusable data loading, preprocessing, and evaluation functions
│
├── outputs/                              # Generated figures, CSVs, and model outputs
├── mia_flights_env.yml                   # Conda environment with pinned dependencies
└── README.md                             # Project documentation
```

---

## 🛠️ Environment Setup

To reproduce this project in a clean environment:

```bash
conda env create -f mia_flights_env.yml
conda activate mia-flights
jupyter lab
```

The environment includes:
* Python 3.11
* pandas 3.0.0
* numpy 2.4.2
* matplotlib 3.10.8
* seaborn 0.13.2
* scikit-learn 1.3.2
* xgboost
* shap
* jupyterlab 4.5.3

---

## 📥 Dataset

This project uses the 2024 Flight Delay and Cancellation dataset.

Download from:
https://www.kaggle.com/datasets/nalisha/flight-delay-and-cancellation-data-1-million-2024

> **Note:** The full dataset covers all U.S. airports in 2024.
> This project filters for MIA origin flights only (January–February 2024).
> The processed file `data/processed/mia_flights_clean.csv` contains 19,396 flights.

---

## 📊 Dataset Description

**Coverage:** January 1, 2024 – February 29, 2024 (winter season only)
**Scope:** Flights departing from Miami International Airport (MIA)
**Records:** 19,396 flights after cleaning and filtering

### Numerical Variables
* Departure time, taxi out / taxi in time, air time, distance
* Weather delay, late aircraft delay

### Categorical Variables
* Day of week, month, origin airport, cancellation status

### Derived Features
* `dep_hour` — departure hour extracted from departure time
* `is_peak` — 1 if departure is during peak hours (6–8 AM or 4–6 PM)
* `is_weekend` — 1 if flight departs on Saturday or Sunday
* `is_early_morning` — 1 if departure before 8 AM
* `is_delayed` — binary target: 1 if late aircraft delay > 15 minutes

---

## 🧹 Data Preparation — `01_data_cleaning.ipynb`

* Raw dataset preserved as original copy before any transformation
* Removed 7,401 duplicate records
* Date conversion to datetime format
* Missing value analysis by cancellation status
* Delay columns filled with 0 (missing = no delay recorded)
* Incomplete non-cancelled records dropped
* Filtered for MIA origin flights only
* Cleaned dataset exported to `data/processed/`

---

## 📈 Exploratory Data Analysis — `02_eda.ipynb`

* Summary statistics for all numerical variables
* Cancellation rate by month and day of week (Tuesday: 2.23% — highest)
* Delay type comparison: late aircraft (15.18%) vs weather (1.11%)
* Time-of-day risk patterns by departure hour
* Correlation analysis: distance vs delay variables
* Outlier analysis using IQR method and boxplots

**Key EDA finding:** Cancellation rate (1.07%) was too imbalanced for reliable
prediction, leading to reframing the target as delay prediction (>15 min, 11.42% positive rate).

---

## 🤖 Modeling

### Tree-Based Models — `03_tree_based_model.ipynb`

**Target:** `is_delayed` (flight delayed >15 minutes)
**Class imbalance strategy:** `class_weight='balanced'` (RF) and `scale_pos_weight=7.76` (XGBoost)

| Model | ROC-AUC (Test) | ROC-AUC (CV) | Recall | F1 |
|---|---|---|---|---|
| Decision Tree | 0.6151 | 0.6401 | 20.3% | 0.2651 |
| Random Forest | 0.7501 | 0.7689 | 60.1% | 0.3250 |
| Tuned Random Forest | 0.7575 | 0.7897 | 8.1% | 0.1440 |
| **XGBoost** | **0.7891** | **0.8079** | **64.3%** | **0.3587** |

**XGBoost** achieved the highest discrimination (ROC-AUC = 0.7891, CV = 0.8079).
**Random Forest** is recommended for operational deployment due to its balance of
recall and interpretability. The Tuned Random Forest, despite high CV AUC, collapsed
recall to 8.1% — operationally unsuitable for delay detection.

Feature importance via MDI and **SHAP** consistently identifies `dep_hour` and `distance`
as the strongest predictors. Evening departures accumulate higher delay risk due to
propagation effects throughout the day.

### Clustering — `04_clustering_model.ipynb`

Unsupervised analysis using 7 operationally justified features across three dimensions:
- **Temporal:** `dep_hour`
- **Route characteristics:** `distance`, `air_time`
- **Delay behavior:** `taxi_out`, `taxi_in`, `weather_delay`, `late_aircraft_delay`

| Method | Result |
|---|---|
| KMeans (k=4) | 4 operational clusters, Silhouette = 0.237 |
| DBSCAN | 205 anomalous flights — avg weather delay 61 min, avg late aircraft delay 184 min |

---

## 📦 Custom Package — `my_modules/model_utils.py`

The `my_modules` package provides reusable components used across notebooks:

* **Data loading** — standardized CSV loading with dtype enforcement and path resolution
* **Preprocessing** — feature engineering pipeline (dep_hour, is_peak, is_weekend, is_early_morning, distance_bucket), missing value handling, and train/test splitting with stratification
* **Evaluation utilities** — classification report formatting, ROC-AUC computation, confusion matrix display, and decile/lift table generation
* **SHAP helpers** — wrapper functions for TreeExplainer computation and summary plot generation

Import example:
```python
from my_modules.model_utils import load_clean_data, build_features, evaluate_model
```

---

## 🔎 Key Findings

* **XGBoost** is the best overall model (ROC-AUC = 0.7891, Recall = 64.3%, F1 = 0.3587)
* **Random Forest** recommended for operations — strong recall (60.1%) with interpretable feature importance
* **dep_hour** and **distance** are the strongest delay predictors confirmed by both MDI and SHAP
* **Late aircraft delay** is the most operationally impactful delay type (15.18% of flights)
* **DBSCAN** identified 205 extreme-disruption flights with avg delays >180 min — potential targets for dedicated contingency protocols
* **Targeting top 10% highest-risk flights** yields 3.4x lift (XGBoost) over random selection

---

## ⚠️ Limitations

* Dataset covers only January–February 2024 — winter season patterns only
* External variables (crew availability, air traffic control, gate conflicts) not included
* Results are specific to MIA and may not generalize to other airports
* XGBoost recall of 64.3% means ~36% of delayed flights are still missed

---

## 🚀 Future Improvements

* Extend dataset to full calendar year for seasonal analysis
* Add airline carrier, destination airport, and real-time weather forecast features
* Deploy model as a real-time risk scoring API using FastAPI + Streamlit
* Threshold tuning to optimize the precision-recall tradeoff for specific operational needs

---

## 👥 Team Collaboration

This project was developed collaboratively by **Team 2** using:
* GitHub for version control with feature branches per task
* Pull requests for peer review before merging to `main`
* Conda environment (`mia_flights_env.yml`) with pinned dependencies for reproducibility
* Balanced contributions across data cleaning, EDA, modeling, and clustering

---

## 🎥 Deliverables

| File | Description |
|---|---|
| `notebooks/01_data_cleaning.ipynb` | Data preparation and cleaning |
| `notebooks/02_eda.ipynb` | Exploratory data analysis |
| `notebooks/03_tree_based_model.ipynb` | Tree-based delay prediction (DT, RF, XGBoost + SHAP) |
| `notebooks/04_clustering_model.ipynb` | Unsupervised operational clustering |
| `data/processed/mia_flights_clean.csv` | Cleaned dataset |
| `outputs/` | All figures and model outputs |
| `Executive_Summary_Report.docx` | Executive summary with key findings |

---

## 📬 Contact

For questions regarding this project, please refer to the repository or contact the project team.