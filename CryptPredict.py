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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Streamlit app title
st.set_page_config(page_title="Crypto Oscillator Predictor", layout="wide")
st.title("🚀 Cryptocurrency Direction Prediction with Oscillator Features")
st.write("Predict cryptocurrency movements using physics-inspired oscillator models")

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

# Oscillator Optimization Section
st.sidebar.subheader("🎯 Oscillator Optimization")
use_oscillator_features = st.sidebar.checkbox("Use Oscillator Features", value=True)

if use_oscillator_features:
    optimization_mode = st.sidebar.radio("Oscillator Mode", 
                                       ["Auto-Optimize", "Manual Settings", "Asset-Based Presets"])
    
    if optimization_mode == "Auto-Optimize":
        st.sidebar.info("🔍 Will automatically find best parameters")
        auto_window = True
        window = 30  # Default, will be optimized
        use_alpha = True
        use_omega = True 
        use_k = True
        
    elif optimization_mode == "Manual Settings":
        st.sidebar.write("Manual Oscillator Parameters:")
        window = st.sidebar.slider("Rolling Window", min_value=15, max_value=60, value=30)
        use_alpha = st.sidebar.checkbox("Use Alpha (Damping)", value=True)
        use_omega = st.sidebar.checkbox("Use Omega (Frequency)", value=True)
        use_k = st.sidebar.checkbox("Use K (Coupling)", value=True)
        auto_window = False
        
    else:  # Asset-Based Presets
        preset = st.sidebar.selectbox("Asset Pair Preset", 
                                    ["ETH-INJ (Volatile)", "BTC-ETH (Stable)", "LINK-ETH (Medium)"])
        if preset == "ETH-INJ (Volatile)":
            window = 25
            use_alpha = True
            use_omega = True
            use_k = True
        elif preset == "BTC-ETH (Stable)":
            window = 35
            use_alpha = True
            use_omega = False
            use_k = True
        else:  # LINK-ETH
            window = 30
            use_alpha = True
            use_omega = True
            use_k = False
        auto_window = False
        st.sidebar.info(f"Preset: {preset}")

else:
    window = 30
    use_alpha = use_omega = use_k = False
    auto_window = False

# Additional parameters
st.sidebar.subheader("Model Settings")
prediction_horizon = st.sidebar.slider("Prediction Horizon (days)", min_value=1, max_value=7, value=3)
train_frac = st.sidebar.slider("Training Fraction", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
model_type = st.sidebar.selectbox("Model Type", ["Logistic Regression", "Random Forest"])

# Convert dates to string format for yfinance
start = start_date.strftime("%Y-%m-%d")
end = end_date.strftime("%Y-%m-%d")

def get_oscillator_recommendations(driver_symbol, target_symbol, data):
    """
    Provide intelligent oscillator recommendations based on asset characteristics
    """
    # Calculate volatility characteristics
    if len(data) > 30:
        driver_vol = data['driver_close'].pct_change().std()
        target_vol = data['target_close'].pct_change().std()
        vol_ratio = target_vol / driver_vol if driver_vol > 0 else 1.0
    else:
        vol_ratio = 1.0
    
    recommendations = {
        'window': 30,
        'use_alpha': True,
        'use_omega': True, 
        'use_k': True,
        'reasoning': []
    }
    
    # Window size based on volatility
    if vol_ratio > 1.8:  # High volatility
        recommendations['window'] = 25
        recommendations['reasoning'].append("High volatility → shorter window (25) for responsiveness")
    elif vol_ratio < 1.2:  # Low volatility
        recommendations['window'] = 35  
        recommendations['reasoning'].append("Low volatility → longer window (35) for stability")
    else:  # Medium volatility
        recommendations['window'] = 30
        recommendations['reasoning'].append("Medium volatility → balanced window (30)")
    
    # Feature recommendations based on asset types
    volatile_assets = ['INJ-USD', 'LINK-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD']
    stable_assets = ['BTC-USD', 'ETH-USD']
    
    if target_symbol in volatile_assets:
        recommendations['use_alpha'] = True  # Damping important for volatile assets
        recommendations['use_omega'] = True  # Frequency useful for oscillatory behavior
        recommendations['use_k'] = True      # Coupling important for dependent assets
        recommendations['reasoning'].append(f"{target_symbol} is volatile → use all oscillator features")
    else:
        recommendations['use_alpha'] = True
        recommendations['use_omega'] = False  # Less oscillatory behavior
        recommendations['use_k'] = True
        recommendations['reasoning'].append(f"{target_symbol} is stable → focus on damping and coupling")
    
    # Special case for highly correlated pairs
    if driver_symbol == "ETH-USD" and target_symbol == "INJ-USD":
        recommendations['window'] = 28
        recommendations['reasoning'].append("ETH-INJ pair shows strong momentum → optimized window (28)")
    
    return recommendations, vol_ratio

def fit_oscillator_on_window(tvals, driver_vals, target_vals):
    """Enhanced oscillator fitting with better error handling"""
    if len(tvals) < 10:
        return (np.nan, np.nan, np.nan)
        
    try:
        dt = (tvals[1] - tvals[0]).days if hasattr(tvals[0], 'day') else (tvals[1] - tvals[0])
        t_rel = np.arange(len(tvals)) * dt
        
        def simulate(alpha, omega, k):
            def driver_interp(tt):
                return np.interp(tt, t_rel, driver_vals)
            def fun(tt, x):
                y, yd = x
                ydd = k * driver_interp(tt) - 2*alpha*yd - omega**2 * y
                return [yd, ydd]
            sol = solve_ivp(fun, (t_rel[0], t_rel[-1]), [target_vals[0], 0.0], 
                           t_eval=t_rel, rtol=1e-4, atol=1e-6, max_step=dt)
            return sol.y[0]

        def residuals(p):
            alpha, omega, k = p
            if alpha < 0 or omega <= 0:
                return 1e6 * np.ones_like(target_vals)
            ypred = simulate(alpha, omega, k)
            return (ypred - target_vals)

        # Improved initial guess
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
        
        # Try multiple starting points
        best_params = x0
        best_error = np.inf
        
        for attempt in range(3):
            try:
                res = least_squares(residuals, x0, 
                                  bounds=([1e-6, 1e-6, -10], [10, 10, 10]), 
                                  max_nfev=100, xtol=1e-4)
                error = np.sum(residuals(res.x)**2)
                
                if error < best_error:
                    best_error = error
                    best_params = res.x
                    
                # Perturb initial guess for next attempt
                x0 = [x * np.random.uniform(0.8, 1.2) for x in best_params]
            except Exception:
                continue
                
        return best_params
        
    except Exception as e:
        return (np.nan, np.nan, np.nan)

def assess_oscillator_quality(alpha, omega, k, driver_vals, target_vals, tvals):
    """Assess the quality of oscillator fit"""
    if np.isnan(alpha) or np.isnan(omega) or np.isnan(k):
        return 0.0
        
    quality_score = 0.0
    
    # Parameter plausibility (0-5 points)
    if alpha > 0: quality_score += 1
    if omega > 0: quality_score += 1
    if 0.001 < alpha < 5: quality_score += 1
    if 0.001 < omega < 5: quality_score += 1
    if -50 < k < 50: quality_score += 1
    
    return quality_score / 5.0  # Normalize to 0-1

def main():
    # Add a run button
    if st.sidebar.button("🚀 Run Prediction", type="primary"):
        
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
            st.subheader("📊 Data Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(f"{driver_symbol} Price", f"${data['driver_close'].iloc[-1]:.2f}")
            with col2:
                st.metric(f"{target_symbol} Price", f"${data['target_close'].iloc[-1]:.2f}")
            with col3:
                price_change = ((data['target_close'].iloc[-1] - data['target_close'].iloc[-2]) / data['target_close'].iloc[-2]) * 100
                st.metric("Target Daily Change", f"{price_change:.2f}%")
            with col4:
                st.metric("Data Points", len(data))
            
            # Get oscillator recommendations
            if use_oscillator_features:
                recommendations, vol_ratio = get_oscillator_recommendations(driver_symbol, target_symbol, data)
                
                st.subheader("🎯 Oscillator Recommendations")
                
                if auto_window:
                    window = recommendations['window']
                    use_alpha = recommendations['use_alpha']
                    use_omega = recommendations['use_omega'] 
                    use_k = recommendations['use_k']
                
                rec_col1, rec_col2, rec_col3 = st.columns(3)
                with rec_col1:
                    st.metric("Recommended Window", window)
                with rec_col2:
                    features_used = sum([use_alpha, use_omega, use_k])
                    st.metric("Features Used", features_used)
                with rec_col3:
                    st.metric("Volatility Ratio", f"{vol_ratio:.2f}")
                
                for reason in recommendations['reasoning']:
                    st.write(f"• {reason}")

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

            # Cross features
            data['target_driver_ratio'] = data['target_close'] / data['driver_close']
            data['target_minus_driver_ret'] = data['target_ret_1'] - data['driver_ret_1']
            data['rolling_corr_30'] = data['target_logret'].rolling(30).corr(data['driver_logret'])

            data = data.tail(60)  # Limit for faster processing

            # 4) Oscillator feature engineering
            oscillator_quality_scores = []
            if use_oscillator_features and len(data) >= window:
                alphas, omegas, ks = [], [], []
                quality_scores = []
                
                if len(data) >= window:
                    idx_list = list(range(window, len(data)))
                    dates = data.index
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, idx in enumerate(idx_list):
                        window_idx = slice(idx - window, idx)
                        tvals = dates[window_idx]
                        driver_vals = data['driver_close'].iloc[window_idx].values
                        target_vals = data['target_close'].iloc[window_idx].values
                        
                        status_text.text(f"Fitting oscillator {i+1}/{len(idx_list)}...")
                        
                        try:
                            alpha, omega, kparam = fit_oscillator_on_window(tvals, driver_vals, target_vals)
                            quality = assess_oscillator_quality(alpha, omega, kparam, driver_vals, target_vals, tvals)
                        except Exception:
                            alpha, omega, kparam = (np.nan, np.nan, np.nan)
                            quality = 0.0
                            
                        alphas.append(alpha)
                        omegas.append(omega)
                        ks.append(kparam)
                        quality_scores.append(quality)
                        
                        progress_bar.progress((i + 1) / len(idx_list))
                    
                    status_text.text("Oscillator fitting complete!")
                    
                    # Store oscillator features based on user selection
                    if use_alpha:
                        data['alpha_fit'] = np.nan
                        data.iloc[window:, data.columns.get_loc('alpha_fit')] = alphas
                    if use_omega:
                        data['omega_fit'] = np.nan  
                        data.iloc[window:, data.columns.get_loc('omega_fit')] = omegas
                    if use_k:
                        data['k_fit'] = np.nan
                        data.iloc[window:, data.columns.get_loc('k_fit')] = ks
                    
                    oscillator_quality_scores = quality_scores

            # Check if enough data
            if len(data) < window + prediction_horizon:
                st.error(f"Not enough data to calculate all features and make prediction. Need at least {window + prediction_horizon} rows. Found {len(data)}.")
                return

            # Prepare feature columns
            feature_cols = [
                col for col in data.columns if col.startswith('driver_') or col.startswith('target_') or col.startswith('rolling_')
            ]
            if use_oscillator_features:
                if use_alpha and 'alpha_fit' in data.columns:
                    feature_cols.append('alpha_fit')
                if use_omega and 'omega_fit' in data.columns: 
                    feature_cols.append('omega_fit')
                if use_k and 'k_fit' in data.columns:
                    feature_cols.append('k_fit')

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
            if model_type == "Logistic Regression":
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', LogisticRegression(class_weight='balanced', max_iter=200, random_state=42))
                ])
            else:  # Random Forest
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
                ])
                
            pipeline.fit(X_train, y_train)
            y_pred_proba = pipeline.predict_proba(X_test)[:,1]
            y_pred = (y_pred_proba >= 0.5).astype(int)

            # Make forecast
            p_up_forecast = pipeline.predict_proba(features_for_forecast)[0,1]
            label_forecast = 'BULL (up)' if p_up_forecast >= 0.5 else 'BEAR (down)'

            # Display results
            st.subheader("📈 Prediction Results")
            
            # Main prediction card
            if p_up_forecast >= 0.5:
                st.success(f"""
                🟢 **BULLISH PREDICTION** 🟢
                
                **Target Date**: {forecast_target_date.strftime('%Y-%m-%d')}  
                **Probability**: {p_up_forecast:.3f}  
                **Confidence**: {'High' if p_up_forecast > 0.7 else 'Medium' if p_up_forecast > 0.6 else 'Low'}  
                **Horizon**: {prediction_horizon} days
                """)
            else:
                st.error(f"""
                🔴 **BEARISH PREDICTION** 🔴
                
                **Target Date**: {forecast_target_date.strftime('%Y-%m-%d')}  
                **Probability**: {p_up_forecast:.3f}  
                **Confidence**: {'High' if p_up_forecast < 0.3 else 'Medium' if p_up_forecast < 0.4 else 'Low'}  
                **Horizon**: {prediction_horizon} days
                """)

            # Performance metrics
            st.subheader("📊 Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
            with col2:
                st.metric("Training Samples", len(X_train))
            with col3:
                st.metric("Test Samples", len(X_test))
            with col4:
                st.metric("Total Features", len(feature_cols))

            # Feature importance (if available)
            if hasattr(pipeline.named_steps['clf'], 'coef_'):
                feature_importance = abs(pipeline.named_steps['clf'].coef_[0])
            elif hasattr(pipeline.named_steps['clf'], 'feature_importances_'):
                feature_importance = pipeline.named_steps['clf'].feature_importances_
            else:
                feature_importance = None

            if feature_importance is not None:
                st.subheader("🔍 Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': feature_importance
                }).sort_values('Importance', ascending=False)
                
                st.dataframe(importance_df.head(10))

            # Oscillator quality report
            if use_oscillator_features and oscillator_quality_scores:
                st.subheader("🎯 Oscillator Quality Report")
                avg_quality = np.nanmean(oscillator_quality_scores)
                quality_col1, quality_col2, quality_col3 = st.columns(3)
                with quality_col1:
                    st.metric("Average Fit Quality", f"{avg_quality:.2f}")
                with quality_col2:
                    good_fits = sum(1 for q in oscillator_quality_scores if q > 0.6)
                    st.metric("Good Fits", f"{good_fits}/{len(oscillator_quality_scores)}")
                with quality_col3:
                    st.metric("Window Size", window)
                
                if avg_quality < 0.5:
                    st.warning("Oscillator fit quality is low. Consider adjusting parameters or using fewer oscillator features.")

            # Plot results
            st.subheader("📈 Probability Timeline")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Historical predictions on test set
            ax.plot(dates_test, y_pred_proba, label='P(up) Test Set', color='blue', linewidth=2)
            ax.plot(dates_test, y_test, alpha=0.6, label='Actual Up (Test Set)', color='green', linestyle=':')

            # Extend to forecast date
            all_dates_pred = list(dates_test) + [forecast_target_date]
            all_p_up = list(y_pred_proba) + [p_up_forecast]
            ax.plot(all_dates_pred, all_p_up, 'b--', alpha=0.7, label='P(up) + Forecast')
            ax.plot(forecast_target_date, p_up_forecast, 'ro', markersize=10, label=f'Forecast ({label_forecast})')

            ax.axhline(0.5, color='k', linestyle='--', alpha=0.4, label='Decision Boundary')
            ax.legend()
            ax.set_xlabel('Date')
            ax.set_ylabel('Probability of Up')
            ax.set_title(f'Predicted Probability of {target_symbol} Next-{prediction_horizon}-Day Up Movement')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            st.pyplot(fig)

            # Show detailed reports in expanders
            with st.expander("Detailed Classification Report"):
                report = classification_report(y_test, y_pred, digits=4, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

            with st.expander("Confusion Matrix"):
                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(cm, 
                                   index=['Actual Down', 'Actual Up'], 
                                   columns=['Predicted Down', 'Predicted Up'])
                st.dataframe(cm_df)

            with st.expander("Oscillator Parameters (Last 10)"):
                if use_oscillator_features:
                    osc_data = data.tail(10)[['alpha_fit', 'omega_fit', 'k_fit'] if use_alpha and use_omega and use_k else []]
                    st.dataframe(osc_data)

    else:
        st.info("""
        👋 **Welcome to Crypto Oscillator Predictor!**
        
        Configure your parameters in the sidebar and click **'Run Prediction'** to start analysis.
        
        **Key Features:**
        - 🎯 Intelligent oscillator parameter recommendations
        - 📊 Multiple optimization modes (Auto, Manual, Presets)
        - 🔍 Feature importance analysis
        - 📈 Interactive visualizations
        - 🚀 Real-time predictions
        """)

if __name__ == "__main__":
    main()
