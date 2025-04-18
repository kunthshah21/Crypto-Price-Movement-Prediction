# Crypto Price Moevement Prediction 
*A Multi‑Horizon RNN‑Powered Streamlit Dashboard*  

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-ff4b4b?logo=streamlit)](https://crypto-price-movement-prediction-kunthshah.streamlit.app)&nbsp;&nbsp;&nbsp;[![GitHub Repo](https://img.shields.io/badge/Source-GitHub-181717?logo=github)](https://github.com/kunthshah21/Crypto-Price-Movement-Prediction)

---

## 1  | Quick Start / Installation

> **Prerequisites:** Python ≥ 3.9 and a working internet connection for first‑time model download.

```bash
# 1. Clone the repo
git clone https://github.com/kunthshah21/Crypto-Price-Movement-Prediction.git
cd Crypto-Price-Movement-Prediction

# 2. (Optional) create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # PowerShell: venv\Scripts\Activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit app
streamlit run app.py
```

The dashboard opens at **`http://localhost:8501`** by default.

---

## 2  | Project Features

| Horizon  | Model Pipeline                   | Dashboard Output                         |
|----------|----------------------------------|------------------------------------------|
| **Daily**   | 2‑layer LSTM                      | Up / Down + probability                  |
| **Monthly** | 1‑D CNN → Deep LSTM               | Up / Down + probability                  |
| **Yearly**  | 3‑stacked LSTMs                   | Up / Down + probability                  |

---

## 3  | Tools & Technologies Used

| Layer        | Technology            | Purpose                               |
|--------------|-----------------------|---------------------------------------|
| Deep‑Learning| **TensorFlow / Keras**| Model implementation                  |
| Data         | **Pandas, NumPy**     | Cleaning & feature engineering        |
| Visualization| **Matplotlib, Seaborn**| Exploratory data analysis             |
| Deployment   | **Streamlit**         | Interactive web UI                    |
| Environment  | **venv, pip**         | Dependency & environment management   |

---

## 4  | Repository Structure

```
├── app.py                  # Streamlit front‑end
├── models/
│   ├── daily_model.keras
│   ├── monthly_model.keras
│   └── yearly_model.keras
├── data/
│   └── crypto_prices.csv
├── notebooks/              # Jupyter/Colab notebooks for EDA & experiments
└── requirements.txt
```

---

## 5  | Usage

1. Choose **Daily**, **Monthly**, or **Yearly** in the sidebar.  
2. (Optional) upload a custom CSV with the required columns (see `/data` sample).  
3. Click **Predict** – the dashboard returns:  
   * Binary prediction (**Up** / **Down**)  
   * Probability score (0 – 1)  
4. Inspect historical plots and probability trends.

---

## 6 | Project Report

### 6.1 Introduction

The rapid growth of cryptocurrencies and digital assets worldwide has significantly transformed traditional financial landscapes, driven by increasing digitisation, the shift towards paperless transactions, and instability in conventional fiat currencies such as the US dollar. This surge in cryptocurrency adoption, characterised by extreme price volatility and market dynamism, presents substantial challenges for investors, traders, and financial institutions aiming to effectively predict price movements and make informed investment decisions.

This project proposes a predictive modelling approach utilizing recurrent neural networks (RNNs) to forecast cryptocurrency price fluctuations across three distinct horizons—daily, monthly, and yearly—specifically predicting whether prices will rise (1) or fall (0). The models incorporate historical price data, trading volume, market indicators, and external features such as social media sentiment.

### 6.2 Methodology

#### Dataset Overview
The dataset comprises 50,000 observations with the following columns:

| Column           | Type     |
|------------------|----------|
| Date             | object   |
| Open_Price       | float64  |
| Close_Price      | float64  |
| High_Price       | float64  |
| Low_Price        | float64  |
| Price_Change     | float64  |
| Volume           | int64    |
| MA_5             | float64  |
| MA_10            | float64  |
| RSI              | float64  |
| Volatility       | float64  |
| Sentiment_Score  | float64  |
| Global_Economy   | int64    |
| Event_Impact     | float64  |
| Price_Movement   | int64    |

#### Data Separation
The data was aggregated and separated into three independent datasets for daily, monthly, and yearly modelling. Aggregation included averaging sentiment scores, economic indicators, and summing volumes.

#### Data Preprocessing
Key preprocessing steps:
- Handling null values
- Converting date columns to DateTime
- Ensuring boolean target variables
- Removing data leakage by dropping dependent variables (`Close_Price`, `Price_Change`)

#### Target Analysis
Class distribution was balanced, requiring no additional sampling techniques such as SMOTE.

#### Trend Analysis
Trend analysis showed:
- No sustained upward or downward drift.
- No clear seasonality patterns.
- Stable variance over time.

#### Stationarity Testing (ADF Test)
Augmented Dickey-Fuller tests confirmed all features were stationary (p-values near 0).

#### Autocorrelation Analysis (ACF and PACF)
- Autocorrelation diminished quickly for most features.
- `MA_5` significant up to lag 4; `MA_10` up to lag 8.
- Window sizes around 10–20 were considered sufficient.

#### Feature Engineering
Features created:
- Lagged moving averages (`MA_5`, `MA_10`)
- Long-term sentiment (`sentiment_delta`)
- Total features engineered: 26

#### Feature Selection
Feature importance assessed using Cohen's D and Mutual Information (MI). Selected features:
- Open_Price, High_Price, Low_Price, MA_5, MA_10, RSI, MA_5_lag_2, MA_10_lag_2, MA_10_lag_7

A StandardScaler was applied for normalization.

### 6.3 Modelling

Models were split into train-validation-test sets (70-15-15), preserving chronological order.

#### Model Comparison
Several models were tested with minimal performance differences:

| Model                                    | Validation Accuracy |
|------------------------------------------|---------------------|
| Attention Layer on top of LSTM           | 50.7%               |
| CNN with LSTM                            | 51%                 |
| Temporal Convolutional Network (TCN)     | 50.9%               |
| Attention + CNN + Bi-directional LSTM    | 52%                 |

Simpler models were preferred due to minimal accuracy gains from complexity.

#### Daily Model
- **Architecture:** Two-layer LSTM with Dense layers, Dropout regularization, Adam optimizer.
- **Validation Accuracy:** ~50–51%

#### Monthly Model
- **Architecture:** Hybrid CNN (1D Conv) and Deep LSTM layers with Dropout and Batch Normalization.
- **Window size:** 60 days
- **Validation Accuracy:** ~53–54%

#### Yearly Model
- **Architecture:** Three stacked LSTM layers, Dense layers with Dropout.
- **Window size:** 10 years
- **Validation Accuracy:** ~52–55%

### 6.4 Results

#### Daily Model Results
- **Test Accuracy:** ~50%
- **Confusion Matrix:**

- **Classification Report:**


#### Monthly Model Results
- **Test Accuracy:** 54%
- **Classification Report:**


#### Yearly Model Results
- **Test Accuracy:** ~49%
- **Confusion Matrix:**


### 6.5 Model Analysis and Usability
- **Daily Model:** Suitable for immediate trades; limited by volatility.
- **Monthly Model:** Moderate improvement; useful for tactical planning.
- **Yearly Model:** Limited accuracy due to macro-event influence; provides strategic insights.

Overall, multi-horizon forecasts provide layered insights, but accuracy remains moderate. Future improvements may arise from richer macroeconomic data integration and cross-market indicators.

### 6.6 Business Application
Potential uses include:
- **Portfolio Management:** Monthly and yearly models assist with asset allocation and risk management.
- **Algorithmic Trading:** Daily predictions integrated into automated trading strategies.
- **Financial Advisory:** Providing actionable data-driven insights across multiple investment horizons.
- **Retail Investment Apps:** Combining short-term alerts with long-term strategies for retail users.

### 6.7 Model Deployment
Models are deployed via Streamlit for accessibility and ease of use:
- Users can select **Daily**, **Monthly**, or **Yearly** models.
- The application displays binary predictions ("Up"/"Down") alongside probability scores.
- Interactive design facilitates quick interpretation and comparison across different timescales.

**Streamlit App:** [Crypto Price Movement Prediction](https://crypto-price-movement-prediction-kunthshah.streamlit.app)

**GitHub Repository:** [GitHub Project](https://github.com/kunthshah21/Crypto-Price-Movement-Prediction)

--- 


---

## 7  | Contributing

Pull requests are welcome — please open an issue first to discuss proposed changes.
