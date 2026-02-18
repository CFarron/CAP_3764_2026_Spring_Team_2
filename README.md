# CAP_3764_2026_Spring_Team_2

# Miami Flights Operational Analysis ✈️

## 📌 Project Overview

This project analyzes flight operations departing from **Miami International Airport (MIA)** with the goal of identifying patterns related to:

* Flight cancellations
* Weather-related delays
* Late aircraft delays
* Operational patterns by time and day

The objective is to generate actionable insights that could support operational decision-making and reduce disruption risk.

---

## 🎯 Project Goals

1. Analyze which days of the week show higher cancellation rates.
2. Examine how departure time influences weather and aircraft delays.
3. Identify operational patterns that increase disruption risk.
4. Provide data-driven insights with business value.

This project focuses on **descriptive analytics (EDA)** to uncover meaningful trends in the data.

---

## 📂 Repository Structure

```
CAP_3764_2026_Spring_Team_2/
│
├── data/
│   ├── raw/                # Original dataset
│   └── processed/          # Cleaned dataset
│
├── notebooks/              # Jupyter notebooks (EDA & analysis)
├── my_modules/             # Data cleaning and helper functions
├── outputs/                # Generated plots and exports
├── environment.yml         # Conda environment configuration
└── README.md               # Project documentation
```

---

## 🛠️ Environment Setup

To reproduce this project environment:

```bash
conda env create -f environment.yml
conda activate flights
jupyter lab
```

The environment includes:

* Python 3.11
* pandas
* numpy
* matplotlib
* seaborn
* jupyterlab

---

## Dataset

This project uses the 2024 Flight Delay and Cancellation dataset.

Download it from:
https://www.kaggle.com/datasets/nalisha/flight-delay-and-cancellation-data-1-million-2024

Place the file in:

data/flight_data_2024.csv

## 📊 Dataset Description

The dataset contains flight-level operational data including:

### Numerical Variables

* departure time
* taxi out / taxi in time
* air time
* distance
* weather delay
* late aircraft delay

### Categorical Variables

* day of week
* month
* origin airport
* cancellation status

The dataset currently covers:

**January 1, 2024 – February 29, 2024**

---

## 🧹 Data Preparation

The data cleaning process includes:

* Date conversion to datetime format
* Handling missing values
* Duplicate verification
* Data type validation
* Creation of derived features (e.g., departure hour)
* Export of cleaned dataset to `data/processed/`

---

## 📈 Exploratory Data Analysis (EDA)

The EDA includes:

* Summary statistics for all numerical variables
* Distribution analysis of delay variables
* Cancellation rate by month
* Cancellation rate by day of week
* Weather delay patterns by departure hour
* Outlier exploration using histograms and boxplots

Each visualization is accompanied by written insights explaining operational implications.

---

## 🔎 Key Insights

* Certain days of the week show higher cancellation frequency.
* Flights departing later in the day exhibit increased delay risk.
* Weather-related delays vary significantly depending on departure time.
* Late aircraft delays suggest cumulative operational disruptions throughout the day.

These insights highlight opportunities for improved scheduling and operational risk management.

---

## ⚠️ Limitations

* The dataset only covers two months (January–February 2024).
* Seasonal trends cannot be evaluated.
* External operational variables (crew availability, air traffic congestion) are not included.

---

## 🚀 Future Improvements

* Extend analysis to a full-year dataset.
* Build a predictive model for cancellation risk.
* Develop a dashboard for operational monitoring.

---

## 👥 Team Collaboration

This project was developed collaboratively using:

* GitHub for version control
* Feature branches for task management
* Pull requests for code review
* Conda environment for reproducibility

---

## 🎥 Deliverables

* Jupyter Notebook (EDA & analysis)
* Cleaned dataset
* GitHub repository
* 10-minute presentation video
* Slide deck summarizing insights

---

## 📬 Contact

For questions regarding this project, please refer to the repository or contact the project team.
