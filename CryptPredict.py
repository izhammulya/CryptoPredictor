import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# Streamlit app title
st.title("Cryptocurrency Direction Prediction")
st.write("Predict the direction of cryptocurrencies using oscillator features and machine learning")

# Sidebar for user inputs
st.sidebar.header("Model Parameters")

# 1. Date range input
st.sidebar.subheader("Date Range")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

# 2. Cryptocurrency selection
st.sidebar.subheader("Cryptocurrency Selection")
crypto_options = ["ETH-USD", "LINK-USD", "INJ-USD", "BTC-USD", "ADA-USD", "DOT-USD", "SOL-USD"]
driver_symbol = st.sidebar.selectbox("Select Driver Cryptocurrency", crypto_options, index=0)
target_symbol = st.sidebar.selectbox("Select Target Cryptocurrency", crypto_options, index=2)

# Additional parameters
st.sidebar.subheader("Model Settings")
window = st.sidebar.slider("Rolling Window (days)", min_value=10, max_value=60, value=30)
prediction_horizon = st.sidebar.slider("Prediction Horizon (days)", min_value=1, max_value=7, value=3)
use_oscillator_features = st.sidebar.checkbox("Use Oscillator Features", value=True)
train_frac = st.sidebar.slider("Training Fraction", min_value=0.5, max_value=0.9, value=0.7, step=0.05)

# Convert dates to string format for yfinance
start = start_date.strftime("%Y-%m-%d")
end = end_date.strftime("%Y-%m-%d")

def fit_oscillator_on_window(tvals, driver_vals, target_vals):
    dt = (tvals[1] - tvals[0]).days if hasattr(tvals[0], 'day') else (tvals[1] - tvals[0])
    t_rel = np.arange(len(tvals)) * dt
    
    def simulate(alpha, omega, k):
        def driver_interp(tt):
            return np.interp(tt, t_rel, driver_vals)
        def fun(tt, x):
            y, yd = x
            ydd = k * driver_interp(tt) - 2*alpha*yd - omega**2 * y
            return [yd, ydd]
        sol = solve_ivp(fun, (t_rel[0], t_rel[-1]), [target_vals[0], 0.0], t_eval=t_rel, rtol=1e-4, atol=1e-6, max_step=dt)
        return sol.y[0]
    
    def residuals(p):
        alpha, omega, k = p
        if alpha < 0 or omega <= 0:
            return 1e6 * np.ones_like(target_vals)
        ypred = simulate(alpha, omega, k)
        return (ypred - target_vals)
    
    try:
        driver_demean = driver_vals - np.mean(driver_vals)
        fft = np.fft.rfft(driver_demean)
        freqs = np.fft.rfftfreq(len(driver_vals), d=dt)
        idx = np.argmax(np.abs(fft[1:])) + 1
        f_dom = freqs[idx] if freqs[idx] > 0 else 0.01
        omega0 = max(0.01, 2*np.pi*f_dom)
    except Exception:
        omega0 = 1.0
    
    x0 = [0.1 * omega0, omega0, 1.0]
    try:
        res = least_squares(residuals, x0, bounds=([1e-6,1e-6,-np.inf],[np.inf,np.inf,np.inf]), max_nfev=200, xtol=1e-6)
        return res.x
    except Exception:
        return (np.nan, np.nan, np.nan)

def main():
    # Add a run button
    if st.sidebar.button("Run Prediction"):
        
        with st.spinner("Downloading data and running prediction..."):
            
            # 1) fetch data
            try:
                df_driver = yf.download(driver_symbol, start=start, end=end, progress=False)
                df_target = yf.download(target_symbol, start=start, end=end, progress=False)
                
                if df_driver.empty or df_target.empty:
                    st.error("Failed to download data - check internet connection or ticker symbols.")
                    return
                    
            except Exception as e:
                st.error(f"Error downloading data: {e}")
                return

            # 2) align and build price series
            data = pd.DataFrame(index=df_driver.index)
            data['driver_close'] = df_driver['Close']
            data['target_close'] = df_target['Close']
            data = data.dropna().copy()

            if len(data) == 0:
                st.error("No overlapping data found between the selected cryptocurrencies.")
                return

            # Display basic info
            st.subheader("Data Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"{driver_symbol} Current Price", f"${data['driver_close'].iloc[-1]:.2f}")
            with col2:
                st.metric(f"{target_symbol} Current Price", f"${data['target_close'].iloc[-1]:.2f}")
            
            st.write(f"Data range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            st.write(f"Total observations: {len(data)}")

            # 3) basic features for driver and target
            data['driver_ret_1'] = data['driver_close'].pct_change(1)
            data['driver_ret_3'] = data['driver_close'].pct_change(3)
            data['driver_ret_7'] = data['driver_close'].pct_change(7)
            data['driver_logret'] = np.log(data['driver_close']).diff()
            data['driver_vol_14'] = data['driver_logret'].rolling(14).std()

            data['target_ret_1'] = data['target_close'].pct_change(1)
            data['target_ret_3'] = data['target_close'].pct_change(3)
            data['target_logret'] = np.log(data['target_close']).diff()
            data['target_vol_14'] = data['target_logret'].rolling(14).std()

            # Calculate historical daily volatility for the target cryptocurrency
            daily_volatility_target = data['target_logret'].dropna().std()

            data = data.tail(60) # Limit data to the latest 60 rows for faster processing

            # cross features
            data['target_driver_ratio'] = data['target_close'] / data['driver_close']
            data['target_minus_driver_ret'] = data['target_ret_1'] - data['driver_ret_1']
            data['rolling_corr_30'] = data['target_logret'].rolling(30).corr(data['driver_logret'])

            # 4) optional: rolling oscillator fit
            if use_oscillator_features:
                alphas, omegas, ks = [], [], []
                if len(data) >= window:
                    idx_list = list(range(window, len(data)))
                    dates = data.index
                    progress_bar = st.progress(0)
                    for i, idx in enumerate(idx_list):
                        window_idx = slice(idx - window, idx)
                        tvals = dates[window_idx]
                        driver_vals = data['driver_close'].iloc[window_idx].values
                        target_vals = data['target_close'].iloc[window_idx].values
                        try:
                            alpha, omega, kparam = fit_oscillator_on_window(tvals, driver_vals, target_vals)
                        except Exception:
                            alpha, omega, kparam = (np.nan, np.nan, np.nan)
                        alphas.append(alpha)
                        omegas.append(omega)
                        ks.append(kparam)
                        progress_bar.progress((i + 1) / len(idx_list))
                
                data['alpha_fit'] = np.nan
                data['omega_fit'] = np.nan
                data['k_fit'] = np.nan
                
                if alphas:
                    data.iloc[window:, data.columns.get_loc('alpha_fit')] = alphas
                    data.iloc[window:, data.columns.get_loc('omega_fit')] = omegas
                    data.iloc[window:, data.columns.get_loc('k_fit')] = ks

            # Check if enough data
            if len(data) < window + prediction_horizon:
                st.error(f"Not enough data to calculate all features and make prediction. Need at least {window + prediction_horizon} rows. Found {len(data)}.")
                return

            # Prepare feature columns
            feature_cols = [
                col for col in data.columns if col.startswith('driver_') or col.startswith('target_') or col.startswith('rolling_')
            ]
            if use_oscillator_features:
                feature_cols += ['alpha_fit','omega_fit','k_fit']

            # Extract latest features for forecasting
            current_date_for_prediction = data.index[-1]
            features_for_forecast = data.iloc[-1][feature_cols].values.reshape(1,-1)
            forecast_target_date = current_date_for_prediction + pd.Timedelta(days=prediction_horizon)

            # Create target variable
            data['target_close_future'] = data['target_close'].shift(-prediction_horizon)
            data['target_up'] = (data['target_close_future'] > data['target_close']).astype(int)
            data = data.dropna().copy()

            if len(data) == 0:
                st.error("Not enough valid data after feature engineering and target creation for training.")
                return

            # Prepare X and y
            X = data[feature_cols].values
            y = data['target_up'].values

            # Train-test split
            n = len(data)
            split = int(n * train_frac)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            dates_test = data.index[split:]

            # Train model
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(class_weight='balanced', max_iter=200, random_state=42))
            ])
            pipeline.fit(X_train, y_train)
            y_pred_proba = pipeline.predict_proba(X_test)[:,1]
            y_pred = (y_pred_proba >= 0.5).astype(int)

            # Make forecast
            p_up_forecast = pipeline.predict_proba(features_for_forecast)[0,1]
            label_forecast = 'BULL (up)' if p_up_forecast >= 0.5 else 'BEAR (down)'

            # Display results
            st.subheader("Model Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
            with col2:
                st.metric("Training Samples", len(X_train))
            with col3:
                st.metric("Test Samples", len(X_test))

            st.subheader("Prediction Results")
            st.info(f"**Prediction for {forecast_target_date.strftime('%Y-%m-%d')}**")
            st.info(f"Based on data up to {current_date_for_prediction.strftime('%Y-%m-%d')}")
            
            # Color-coded prediction
            if p_up_forecast >= 0.5:
                st.success(f"🟢 **{label_forecast}** - P(up next {prediction_horizon} days) = {p_up_forecast:.3f}")
            else:
                st.error(f"🔴 **{label_forecast}** - P(up next {prediction_horizon} days) = {p_up_forecast:.3f}")

            # Plot results
            st.subheader("Probability Timeline")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Historical predictions on test set
            ax.plot(dates_test, y_pred_proba, label='P(up) Test Set', color='blue')
            ax.plot(dates_test, y_test, alpha=0.6, label='Actual Up (Test Set)', color='green', linestyle=':')

            # Extend to forecast date
            all_dates_pred = list(dates_test) + [forecast_target_date]
            all_p_up = list(y_pred_proba) + [p_up_forecast]
            ax.plot(all_dates_pred, all_p_up, 'b--', alpha=0.7, label='P(up) + Forecast')
            ax.plot(forecast_target_date, p_up_forecast, 'ro', markersize=8, label=f'Forecast ({label_forecast})')

            ax.axhline(0.5, color='k', linestyle='--', alpha=0.4)
            ax.legend()
            ax.set_xlabel('Date')
            ax.set_ylabel('Probability of Up')
            ax.set_title(f'Predicted probability of {target_symbol} next-{prediction_horizon}-day up movement')
            ax.grid(True)
            plt.tight_layout()
            
            st.pyplot(fig)

            # Show classification report
            st.subheader("Detailed Classification Report")
            report = classification_report(y_test, y_pred, digits=4, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

            # Show confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, 
                               index=['Actual Down', 'Actual Up'], 
                               columns=['Predicted Down', 'Predicted Up'])
            st.dataframe(cm_df)

    else:
        st.info("👈 Configure the parameters in the sidebar and click 'Run Prediction' to start analysis.")

if __name__ == "__main__":
    main()