
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

CAP_3764_2026_Spring_Team_2/
│
├── data/
│   ├──flight_data_2024.csv # Raw dataset
│   ├──mia_flights.clean.csv # Clean dataset
├── notebooks/
│   ├── 01_data_cleaning.ipynb  # Data preparation and preprocessing
│   ├── 02_eda.ipynb            # Exploratory data analysis
│   ├── 03_tree_based_model.ipynb  # Decision Tree + Random Forest
│   └── 04_clustering.ipynb     # PCA + KMeans + DBSCAN clustering
│
├── my_modules/                 # Custom data loading and preprocessing functions
├── outputs/                    # Generated plots, CSVs, and model exports
├── environment.yml             # Conda environment with pinned dependencies
└── README.md                   # Project documentation

## 🛠️ Environment Setup

To reproduce this project in a clean environment:
```bash
conda env create -f environment.yml
conda activate flight-cancellation-project
jupyter lab
```

The environment includes:
* Python 3.11
* pandas 3.0.0
* numpy 2.4.2
* matplotlib 3.10.8
* seaborn 0.13.2
* scikit-learn 1.3.2
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
prediction, leading to reframing the target as delay prediction (>15 min, 11.42%).

---

## 🤖 Modeling

### Tree-Based Models — `03_tree_based_model.ipynb`

Target: `is_delayed` (flight delayed >15 minutes)
Approach: `class_weight='balanced'` to handle 11.42% class imbalance

| Model | ROC-AUC (Test) | ROC-AUC (CV) | Recall | F1 |
|---|---|---|---|---|
| Decision Tree | 0.615 | 0.640 | 20.3% | 0.265 |
| Random Forest | 0.750 | 0.769 | 60.0% | 0.325 |

**Winner: Random Forest** — significantly better discrimination and recall.
Feature importance shows `dep_hour`, `distance`, and `day_of_week`
as the strongest predictors of delay risk.

### Clustering — `04_clustering.ipynb`

Unsupervised analysis to discover operational risk profiles.

* **PCA** — dimensionality reduction for visualization
* **KMeans (k=4)** — operational flight clusters, Silhouette: 0.237
* **DBSCAN** — density-based clustering, identified 205 anomalous flights
  with avg weather delay of 61 min and late aircraft delay of 184 min

---

## 🔎 Key Findings

* **Feature engineering** significantly improved model performance and interpretability
* **Random Forest** achieves ROC-AUC of 0.743 — meaningfully above random baseline
* **Late aircraft delay** is the most operationally impactful delay type (15.18% of flights)
* **Tuesday** shows the highest cancellation rate (2.23%)
* **Flights departing later in the day** accumulate higher delay risk
* **DBSCAN anomaly cluster** identifies extreme-disruption flights with avg delays >180 min
* **Distance** has near-zero correlation with delays — disruptions are operationally driven

---

## ⚠️ Limitations

* Dataset covers only January–February 2024 — winter season patterns only
* Seasonal trends and summer operations cannot be evaluated
* External variables (crew availability, air traffic control, gate conflicts) not included
* Random Forest recall of 55.5% means ~45% of delayed flights are still missed
* Results are specific to MIA and may not generalize to other airports

---

## 🚀 Future Improvements

* Extend dataset to full calendar year for seasonal analysis
* Add airline, destination, and weather forecast features
* Explore Gradient Boosting models (XGBoost, LightGBM)
* Deploy model as a real-time risk scoring API using FastAPI + Streamlit

---

## 👥 Team Collaboration

This project was developed collaboratively by Team 2 using:
* GitHub for version control with feature branches per task
* Pull requests for peer review before merging to main
* Conda environment with pinned dependencies for reproducibility
* Balanced contributions across data cleaning, EDA, modeling, and clustering

---

## 🎥 Deliverables

* `01_data_cleaning.ipynb` — Data preparation
* `02_eda.ipynb` — Exploratory analysis
* `03_tree_based_model.ipynb` — Tree-based delay prediction
* `04_clustering.ipynb` — Unsupervised operational clustering
* `data/mia_flights_clean.csv` — Cleaned dataset
* `outputs/` — All figures and model exports
* Executive Summary Report
* GitHub repository with full version history

---

## 📬 Contact

For questions regarding this project, please refer to the repository or contact the project team.