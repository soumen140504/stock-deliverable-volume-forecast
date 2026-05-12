# Stock Current-Day Deliverable Volume Prediction Project

This project predicts **current-day Deliverable Volume** using ADANIPORTS stock market data.

## Formula

Estimated Deliverable Amount = Predicted Deliverable Volume × Close Price

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Regressor
- Joblib
- Streamlit
- Matplotlib

## Why Current-Day Prediction?

Future forecasting gave poor performance because stock data is highly volatile. Current-day Deliverable Volume Prediction is more stable, more accurate, and better for a fresher portfolio project.

## Features Used

- Prev Close
- Open
- High
- Low
- Last
- Close
- VWAP
- Volume
- Turnover

## Target

Deliverable Volume

## How to Run

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Interview Summary

This project predicts current-day Deliverable Volume of ADANIPORTS stock using machine learning. I used Random Forest Regressor because stock market relationships are nonlinear. The Streamlit app lets users enter stock values and receive predicted deliverable volume and estimated deliverable amount.
