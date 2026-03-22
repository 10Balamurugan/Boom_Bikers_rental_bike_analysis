# 🚲 BoomBikes Rental Demand Prediction
BoomBikes is a US-based bike-sharing service that suffered significant revenue loss during the COVID-19 pandemic. This project builds a Multiple Linear Regression model to predict the total number of bike rentals (cnt) on any given day — based on weather, season, and calendar features

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4EABE1?style=for-the-badge&logoColor=white)

**A Machine Learning regression model to predict daily bike rental demand for BoomBikes — helping the business recover and scale post-pandemic**

[🚀 Demo](#usage) · [📖 Documentation](#architecture) · [⚙️ Installation](#installation) · [📊 Results](#model-performance)

</div>

---

## 📌 Overview

**BoomBikes** is a US-based bike-sharing service that suffered significant revenue loss during the COVID-19 pandemic. This project builds a **Multiple Linear Regression model** to predict the total number of bike rentals (`cnt`) on any given day — based on weather, season, and calendar features.

The goal is to help BoomBikes' management:

- 📈 **Understand demand drivers** — which variables most impact rentals
- 🗓️ **Plan inventory and staffing** — prepare for high-demand days
- 🌍 **Strategy post-lockdown** — accelerate business recovery

---

## 🎯 Key Features

| Feature | Description |
|---|---|
| 📊 **EDA & Visualization** | In-depth exploratory analysis with Seaborn & Matplotlib |
| 🔧 **Feature Engineering** | Encoding, scaling, and selection of most impactful variables |
| 🤖 **Multiple Linear Regression** | OLS-based model with statsmodels & scikit-learn |
| ✂️ **RFE (Recursive Feature Elimination)** | Automated feature selection for optimal predictors |
| 📉 **Residual Analysis** | Checking assumptions — normality, homoscedasticity, multicollinearity |
| 📋 **VIF Analysis** | Variance Inflation Factor to detect and remove multicollinearity |

---

## 🏗️ Architecture

### ML Pipeline

```
Raw Dataset (day.csv)
        │
  ┌─────▼──────────────────┐
  │  Data Understanding    │  ← Shape, dtypes, null check, statistics
  └─────┬──────────────────┘
        │
  ┌─────▼──────────────────┐
  │  EDA & Visualization   │  ← Correlation heatmap, boxplots, pairplots
  └─────┬──────────────────┘
        │
  ┌─────▼──────────────────┐
  │  Data Preprocessing    │  ← Encoding categoricals, drop leakage cols
  └─────┬──────────────────┘
        │
  ┌─────▼──────────────────┐
  │  Train / Test Split    │  ← 70% Train | 30% Test
  └─────┬──────────────────┘
        │
  ┌─────▼──────────────────┐
  │  Feature Scaling       │  ← MinMaxScaler on numeric features
  └─────┬──────────────────┘
        │
  ┌─────▼──────────────────┐
  │  RFE Feature Selection │  ← Top 15 features selected
  └─────┬──────────────────┘
        │
  ┌─────▼──────────────────┐
  │  Model Building (OLS)  │  ← statsmodels for p-values & VIF
  └─────┬──────────────────┘
        │
  ┌─────▼──────────────────┐
  │  Residual Analysis     │  ← Normality, homoscedasticity check
  └─────┬──────────────────┘
        │
  ┌─────▼──────────────────┐
  │  Model Evaluation      │  ← R², Adjusted R², MAE, RMSE
  └─────┴──────────────────┘
```

### Project Structure

```
boombikes-rental-prediction/
│
├── 📁 data/
│   └── day.csv                          # Raw dataset (730 records)
│
├── 📁 notebooks/
│   └── BoomBikes_Prediction.ipynb       # Full EDA + modelling notebook
│
├── 📁 outputs/
│   ├── model_summary.txt                # OLS regression summary
│   └── plots/                           # All EDA and result plots
│
├── 📄 requirements.txt                  # Python dependencies
└── 📄 README.md                         # Project documentation
```

---

## 📊 Model Performance

| Metric | Train Set | Test Set |
|---|---|---|
| R² Score | 0.836 | 0.804 |
| Adjusted R² | 0.832 | — |
| MAE | ~542 | ~568 |
| RMSE | ~726 | ~782 |

> ✅ The model explains **~80% of the variance** in bike rental demand on unseen test data — a strong result for a linear model.

---

## 🔍 Top Predictors of Bike Demand

| Feature | Impact | Direction |
|---|---|---|
| `temp` | Very High | ➕ Positive |
| `yr` (Year 2019) | High | ➕ Positive |
| `season_winter` | High | ➕ Positive |
| `weathersit_Light Snow/Rain` | High | ➖ Negative |
| `windspeed` | Moderate | ➖ Negative |
| `season_spring` | Moderate | ➖ Negative |
| `mnth_Sep` | Moderate | ➕ Positive |
| `holiday` | Low | ➖ Negative |

> 🌡️ **Temperature** is the single strongest predictor — warmer days drive significantly more rentals.

---

## 📋 Dataset Description

The dataset contains **730 daily records** of bike rentals from 2018–2019.

| Column | Description |
|---|---|
| `instant` | Record index |
| `dteday` | Date |
| `season` | 1=Spring, 2=Summer, 3=Fall, 4=Winter |
| `yr` | Year (0=2018, 1=2019) |
| `mnth` | Month (1–12) |
| `holiday` | Whether the day is a holiday |
| `weekday` | Day of the week |
| `workingday` | Working day flag |
| `weathersit` | 1=Clear, 2=Mist, 3=Light Snow/Rain, 4=Heavy Rain |
| `temp` | Normalized temperature (°C) |
| `atemp` | Normalized feeling temperature |
| `hum` | Normalized humidity |
| `windspeed` | Normalized wind speed |
| `casual` | Count of casual users |
| `registered` | Count of registered users |
| `cnt` | **Target — Total bike rentals** |

> ⚠️ `casual` and `registered` are dropped before modelling to prevent **data leakage** (they directly sum to `cnt`).

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or JupyterLab

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/boombikes-rental-prediction.git
cd boombikes-rental-prediction
```

### Step 2 — Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Launch Notebook

```bash
jupyter notebook notebooks/BoomBikes_Prediction.ipynb
```

---

## 🚀 Usage

Run all cells in the notebook sequentially to:

1. 📥 Load and inspect the dataset
2. 📊 Perform Exploratory Data Analysis (EDA)
3. 🔧 Preprocess and encode features
4. ✂️ Apply RFE for feature selection
5. 🤖 Build and refine the OLS regression model
6. 📉 Validate model assumptions
7. 📋 Evaluate on the test set

---

## 📦 Dependencies

```txt
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
statsmodels>=0.14.0
jupyter>=1.0.0
```

Install all at once:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels jupyter
```

---

## 🧪 Model Assumptions Validated

| Assumption | Test | Result |
|---|---|---|
| Linearity | Residuals vs Fitted plot | ✅ Passed |
| Normality of Errors | Q-Q Plot + Histogram | ✅ Approximately Normal |
| Homoscedasticity | Residuals spread check | ✅ No clear pattern |
| No Multicollinearity | VIF < 5 for all features | ✅ Passed |
| Independence of Errors | Durbin-Watson test | ✅ ~2.0 |

---

## 📈 Key Business Insights

> 💡 These findings directly help BoomBikes plan their operations:

- 🌡️ **Invest in warm-weather campaigns** — temperature is the #1 demand driver
- 📅 **September is peak month** — ramp up fleet availability in August-September
- ❄️ **Don't ignore winter** — demand is higher than spring despite cold weather
- 🌧️ **Bad weather kills demand** — light snow/rain causes the steepest drop
- 📆 **Year-on-year growth is strong** — 2019 saw significantly more rentals than 2018
- 🏖️ **Holidays reduce rentals** — registered commuters drive most demand

---

## ⚠️ Disclaimer

> This project is developed for **educational and academic purposes** as part of a Data Science curriculum. The dataset and business case are based on a publicly available Kaggle problem. Predictions should not be used for real business decisions without further validation.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**BM**
- 🎓 Data Science & AI/ML Student — Besant Technologies, Bangalore
- 💼 Passionate about building ML solutions for real-world business problems

---

## 🙏 Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset) — Original Bike Sharing Dataset
- [Kaggle — BoomBikes Case Study](https://www.kaggle.com/) — Problem Statement Reference
- [Statsmodels Documentation](https://www.statsmodels.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

<div align="center">

⭐ **Star this repo if it helped you!** ⭐

Made with ❤️ for the Data Science & ML community

</div>
