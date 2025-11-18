import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

try:
    import yfinance as yf
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from scipy.integrate import solve_ivp
    from scipy.optimize import least_squares
except ImportError as e:
    st.error(f"Missing required packages: {e}")
    st.stop()

# Streamlit app configuration
st.set_page_config(
    page_title="Crypto Predictor with Oscillator & Monte Carlo",
    page_icon="🚀",
    layout="wide"
)

# App title and description
st.title("🚀 Advanced Crypto Predictor with Oscillator Analysis")
st.markdown("""
**Enhanced Features:**
- **Oscillator Parameter Visualization** (Alpha, Omega, K)
- **Monte Carlo Simulation** for price prediction intervals
- **Prediction Horizon Positioning** with future dates
- **Confidence Bounds** (Lower/Upper bounds)
- **Oscillator Dynamics Analysis**
""")

# Sidebar configuration
st.sidebar.header("Configuration")

# Date range input
st.sidebar.subheader("📅 Date Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
with col2:
    end_date = st.date_input("End Date", value=pd.to_datetime("today"))

# Cryptocurrency selection
st.sidebar.subheader("💰 Cryptocurrencies")
crypto_options = {
    "ETH-USD": "Ethereum",
    "BTC-USD": "Bitcoin", 
    "LINK-USD": "Chainlink",
    "INJ-USD": "Injective",
    "ADA-USD": "Cardano",
    "DOT-USD": "Polkadot",
    "SOL-USD": "Solana"
}

driver_symbol = st.sidebar.selectbox(
    "Driver Cryptocurrency",
    options=list(crypto_options.keys()),
    format_func=lambda x: f"{x} ({crypto_options[x]})",
    index=0
)

target_symbol = st.sidebar.selectbox(
    "Target Cryptocurrency", 
    options=list(crypto_options.keys()),
    format_func=lambda x: f"{x} ({crypto_options[x]})",
    index=3
)

# Oscillator Configuration
st.sidebar.subheader("🎯 Oscillator Settings")
use_oscillator = st.sidebar.checkbox("Use Oscillator Features", value=True)

if use_oscillator:
    oscillator_window = st.sidebar.slider(
        "Oscillator Window Size",
        min_value=20,
        max_value=60,
        value=30,
        help="Window for fitting oscillator parameters"
    )
    
    st.sidebar.info("""
    **Oscillator Parameters:**
    - **Alpha (α)**: Damping coefficient
    - **Omega (ω)**: Natural frequency  
    - **K**: Coupling strength
    """)

# Rolling Window Configuration
st.sidebar.subheader("🔄 Rolling Window Settings")
window_size = st.sidebar.slider("Main Window (days)", 7, 90, 30)
short_window = st.sidebar.slider("Short Window (days)", 5, 20, 7)
medium_window = st.sidebar.slider("Medium Window (days)", 15, 45, 20)

# Model parameters
st.sidebar.subheader("⚙️ Model Parameters")
prediction_horizon = st.sidebar.slider("Prediction Horizon (days)", 1, 30, 7)
model_type = st.sidebar.selectbox("Model Type", ["Logistic Regression", "Random Forest"], index=0)

# Monte Carlo Simulation Parameters
st.sidebar.subheader("🎲 Monte Carlo Simulation")
num_simulations = st.sidebar.slider("Number of Simulations", 100, 5000, 1000)
confidence_level = st.sidebar.slider("Confidence Level (%)", 80, 99, 95)

# Oscillator Functions
def fit_oscillator_on_window(tvals, driver_vals, target_vals):
    """Fit oscillator parameters on a rolling window"""
    if len(tvals) < 10:
        return (np.nan, np.nan, np.nan)
        
    try:
        # Convert dates to numerical values
        if hasattr(tvals[0], 'timestamp'):
            t_numeric = np.array([t.timestamp() for t in tvals])
        else:
            t_numeric = np.arange(len(tvals))
        
        dt = t_numeric[1] - t_numeric[0] if len(t_numeric) > 1 else 1
        
        def oscillator_model(t, x, alpha, omega, k, driver_vals, t_numeric):
            y, yd = x
            # Interpolate driver value at time t
            driver_val = np.interp(t, t_numeric, driver_vals)
            ydd = k * driver_val - 2 * alpha * yd - omega**2 * y
            return [yd, ydd]
        
        def simulate_oscillator(alpha, omega, k):
            sol = solve_ivp(
                oscillator_model, 
                [t_numeric[0], t_numeric[-1]], 
                [target_vals[0], 0.0],
                args=(alpha, omega, k, driver_vals, t_numeric),
                t_eval=t_numeric,
                rtol=1e-6,
                atol=1e-8
            )
            return sol.y[0] if sol.success else np.full_like(target_vals, np.nan)
        
        def residuals(params):
            alpha, omega, k = params
            if alpha <= 0 or omega <= 0:
                return np.full_like(target_vals, 1e6)
            y_pred = simulate_oscillator(alpha, omega, k)
            return y_pred - target_vals
        
        # Initial parameter guess
        x0 = [0.1, 1.0, 1.0]
        
        # Fit parameters
        res = least_squares(
            residuals, 
            x0, 
            bounds=([1e-6, 1e-6, -10], [10, 10, 10]),
            max_nfev=50
        )
        
        return res.x if res.success else (np.nan, np.nan, np.nan)
        
    except Exception as e:
        return (np.nan, np.nan, np.nan)

def calculate_oscillator_features(data, window):
    """Calculate oscillator features for the entire dataset"""
    if len(data) < window:
        return data
    
    alphas, omegas, ks = [], [], []
    dates = data.index
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(window, len(data)):
        window_slice = slice(i - window, i)
        tvals = dates[window_slice]
        driver_vals = data['driver_close'].iloc[window_slice].values
        target_vals = data['target_close'].iloc[window_slice].values
        
        alpha, omega, k = fit_oscillator_on_window(tvals, driver_vals, target_vals)
        alphas.append(alpha)
        omegas.append(omega)
        ks.append(k)
        
        if i % 10 == 0:  # Update progress every 10 iterations
            progress = (i - window) / (len(data) - window)
            progress_bar.progress(progress)
            status_text.text(f"Fitting oscillators... {i-window}/{len(data)-window}")
    
    # Add oscillator features to dataframe
    data['oscillator_alpha'] = np.nan
    data['oscillator_omega'] = np.nan
    data['oscillator_k'] = np.nan
    
    data.iloc[window:, data.columns.get_loc('oscillator_alpha')] = alphas
    data.iloc[window:, data.columns.get_loc('oscillator_omega')] = omegas
    data.iloc[window:, data.columns.get_loc('oscillator_k')] = ks
    
    progress_bar.progress(1.0)
    status_text.text("Oscillator fitting complete!")
    
    return data

# Monte Carlo Simulation Functions
def monte_carlo_simulation(current_price, volatility, prediction_horizon, num_simulations):
    """Run Monte Carlo simulation for price prediction"""
    simulations = np.zeros((prediction_horizon, num_simulations))
    simulations[0] = current_price
    
    for day in range(1, prediction_horizon):
        random_returns = np.random.normal(0, volatility, num_simulations)
        simulations[day] = simulations[day-1] * np.exp(random_returns)
    
    return simulations

def calculate_prediction_intervals(simulations, confidence_level):
    """Calculate prediction intervals from Monte Carlo simulations"""
    final_prices = simulations[-1, :]
    
    alpha = (100 - confidence_level) / 2
    lower_percentile = alpha
    upper_percentile = 100 - alpha
    
    lower_bound = np.percentile(final_prices, lower_percentile)
    upper_bound = np.percentile(final_prices, upper_percentile)
    median_price = np.median(final_prices)
    mean_price = np.mean(final_prices)
    
    return {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'median_price': median_price,
        'mean_price': mean_price,
        'final_prices': final_prices
    }

# Main function
def main():
    if st.sidebar.button("🎯 Run Advanced Prediction", type="primary"):
        run_advanced_prediction()

def run_advanced_prediction():
    """Main prediction function with Oscillator and Monte Carlo"""
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Download data
        status_text.text("📥 Downloading cryptocurrency data...")
        progress_bar.progress(10)
        
        df_driver = yf.download(driver_symbol, start=start_str, end=end_str, progress=False)
        df_target = yf.download(target_symbol, start=start_str, end=end_str, progress=False)
        
        if df_driver.empty or df_target.empty:
            st.error("❌ Failed to download data.")
            return
        
        # Step 2: Process data
        status_text.text("🔄 Processing data...")
        progress_bar.progress(20)
        
        data = process_basic_data(df_driver, df_target, window_size, short_window, medium_window)
        
        if data is None:
            st.error("❌ Not enough data for analysis.")
            return
        
        # Step 3: Calculate oscillator features
        if use_oscillator:
            status_text.text("🎯 Calculating oscillator parameters...")
            progress_bar.progress(40)
            data = calculate_oscillator_features(data, oscillator_window)
        else:
            # Add placeholder oscillator columns
            data['oscillator_alpha'] = np.nan
            data['oscillator_omega'] = np.nan
            data['oscillator_k'] = np.nan
        
        # Step 4: Prepare features and target
        status_text.text("🔧 Preparing features...")
        progress_bar.progress(60)
        
        feature_cols = prepare_features(data, use_oscillator)
        X, y, features_for_forecast, dates_clean = create_features_target(data, feature_cols, prediction_horizon)
        
        if X.shape[0] == 0:
            st.error("❌ No valid data for training.")
            return
        
        # Step 5: Train model
        status_text.text("🤖 Training model...")
        progress_bar.progress(70)
        
        classification_results = train_and_predict(X, y, features_for_forecast, model_type, dates_clean)
        
        # Step 6: Monte Carlo Simulation
        status_text.text("🎲 Running Monte Carlo simulations...")
        progress_bar.progress(85)
        
        current_price = data['target_close'].iloc[-1]
        volatility = data['target_vol_medium'].iloc[-1] if not np.isnan(data['target_vol_medium'].iloc[-1]) else data['target_ret_1'].std()
        
        simulations = monte_carlo_simulation(current_price, volatility, prediction_horizon, num_simulations)
        prediction_intervals = calculate_prediction_intervals(simulations, confidence_level)
        
        results = {
            **classification_results,
            'monte_carlo': {
                'simulations': simulations,
                'intervals': prediction_intervals,
                'current_price': current_price,
                'volatility': volatility
            },
            'oscillator_data': data
        }
        
        # Step 7: Display results
        status_text.text("📊 Generating results...")
        progress_bar.progress(95)
        
        display_advanced_results(data, results, driver_symbol, target_symbol, prediction_horizon)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def process_basic_data(df_driver, df_target, main_window, short_window, medium_window):
    """Process basic price data"""
    
    data = pd.DataFrame(index=df_driver.index)
    data['driver_close'] = df_driver['Close']
    data['target_close'] = df_target['Close']
    data = data.dropna()
    
    if len(data) < main_window + 10:
        return None
    
    # Basic features
    data['driver_ret_1'] = data['driver_close'].pct_change(1)
    data['target_ret_1'] = data['target_close'].pct_change(1)
    
    data['driver_vol_medium'] = data['driver_ret_1'].rolling(medium_window).std()
    data['target_vol_medium'] = data['target_ret_1'].rolling(medium_window).std()
    
    data['price_ratio'] = data['target_close'] / data['driver_close']
    data['rolling_corr_medium'] = data['driver_ret_1'].rolling(medium_window).corr(data['target_ret_1'])
    
    return data.tail(100).copy()

def prepare_features(data, use_oscillator):
    """Prepare feature columns including oscillator features"""
    
    base_features = [
        'driver_ret_1', 'target_ret_1', 'driver_vol_medium', 
        'target_vol_medium', 'price_ratio', 'rolling_corr_medium'
    ]
    
    if use_oscillator:
        oscillator_features = ['oscillator_alpha', 'oscillator_omega', 'oscillator_k']
        # Only add oscillator features that have valid data
        for feature in oscillator_features:
            if feature in data.columns and not data[feature].isna().all():
                base_features.append(feature)
    
    feature_cols = [col for col in base_features if col in data.columns and not data[col].isna().all()]
    
    return feature_cols

def create_features_target(data, feature_cols, prediction_horizon):
    """Create features and target variable"""
    
    data['future_price'] = data['target_close'].shift(-prediction_horizon)
    data['target_up'] = (data['future_price'] > data['target_close']).astype(int)
    
    data_clean = data.dropna().copy()
    
    if len(data_clean) == 0:
        raise ValueError("No valid data after preprocessing")
    
    X = data_clean[feature_cols].values
    y = data_clean['target_up'].values
    dates_clean = data_clean.index
    
    features_for_forecast = data_clean.iloc[-1][feature_cols].values.reshape(1, -1)
    
    return X, y, features_for_forecast, dates_clean

def train_and_predict(X, y, features_for_forecast, model_type, dates):
    """Train model and make predictions"""
    
    min_samples = 10
    if len(X) < min_samples:
        raise ValueError(f"Not enough samples for training.")
    
    split_idx = max(1, int(len(X) * 0.7))
    
    if len(X) - split_idx < 3:
        split_idx = max(1, len(X) - 3)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates[split_idx:]
    
    if model_type == "Logistic Regression":
        classifier = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    else:
        classifier = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    forecast_proba = pipeline.predict_proba(features_for_forecast)[0, 1]
    
    return {
        'pipeline': pipeline,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'forecast_proba': forecast_proba,
        'X_test': X_test,
        'dates_test': dates_test,
        'split_idx': split_idx
    }

def display_advanced_results(data, results, driver_symbol, target_symbol, prediction_horizon):
    """Display all results with oscillator visualization"""
    
    current_price = data['target_close'].iloc[-1]
    mc_results = results['monte_carlo']
    intervals = mc_results['intervals']
    
    # Generate future dates
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, prediction_horizon + 1)]
    prediction_date = future_dates[-1]
    
    # Display main predictions
    st.header("🎯 Advanced Prediction Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"{target_symbol} Current", f"${current_price:.2f}")
    with col2:
        direction = "BULLISH" if results['forecast_proba'] >= 0.5 else "BEARISH"
        st.metric("Direction", f"{direction}")
    with col3:
        st.metric("Probability", f"{results['forecast_proba']:.3f}")
    with col4:
        change_pct = ((intervals['median_price'] - current_price) / current_price) * 100
        st.metric("Expected Change", f"{change_pct:+.1f}%")
    
    # OSCILLATOR VISUALIZATION SECTION
    if use_oscillator and not data['oscillator_alpha'].isna().all():
        st.subheader("🎯 Oscillator Parameter Analysis")
        
        # Create oscillator visualization
        fig_osc, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Alpha (Damping) over time
        valid_alpha = data['oscillator_alpha'].dropna()
        if len(valid_alpha) > 0:
            ax1.plot(valid_alpha.index, valid_alpha.values, color='red', linewidth=2)
            ax1.set_title('Oscillator Alpha (Damping Coefficient)')
            ax1.set_ylabel('Alpha Value')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Omega (Frequency) over time
        valid_omega = data['oscillator_omega'].dropna()
        if len(valid_omega) > 0:
            ax2.plot(valid_omega.index, valid_omega.values, color='blue', linewidth=2)
            ax2.set_title('Oscillator Omega (Natural Frequency)')
            ax2.set_ylabel('Omega Value')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: K (Coupling) over time
        valid_k = data['oscillator_k'].dropna()
        if len(valid_k) > 0:
            ax3.plot(valid_k.index, valid_k.values, color='green', linewidth=2)
            ax3.set_title('Oscillator K (Coupling Strength)')
            ax3.set_ylabel('K Value')
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Current oscillator state
        current_alpha = data['oscillator_alpha'].iloc[-1] if not np.isnan(data['oscillator_alpha'].iloc[-1]) else 0
        current_omega = data['oscillator_omega'].iloc[-1] if not np.isnan(data['oscillator_omega'].iloc[-1]) else 0
        current_k = data['oscillator_k'].iloc[-1] if not np.isnan(data['oscillator_k'].iloc[-1]) else 0
        
        parameters = ['Alpha (Damping)', 'Omega (Frequency)', 'K (Coupling)']
        values = [current_alpha, current_omega, current_k]
        colors = ['red', 'blue', 'green']
        
        bars = ax4.bar(parameters, values, color=colors, alpha=0.7)
        ax4.set_title('Current Oscillator Parameters')
        ax4.set_ylabel('Parameter Value')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig_osc)
        
        # Oscillator Interpretation
        st.subheader("📊 Oscillator State Interpretation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not np.isnan(current_alpha):
                if current_alpha > 0.5:
                    st.info("**High Damping (Alpha)**: System returns to equilibrium quickly")
                else:
                    st.info("**Low Damping (Alpha)**: System has persistent oscillations")
        
        with col2:
            if not np.isnan(current_omega):
                if current_omega > 1.0:
                    st.info("**High Frequency (Omega)**: Rapid price oscillations")
                else:
                    st.info("**Low Frequency (Omega)**: Slow, smooth price movements")
        
        with col3:
            if not np.isnan(current_k):
                if abs(current_k) > 0.5:
                    st.info("**Strong Coupling (K)**: High dependency between assets")
                else:
                    st.info("**Weak Coupling (K)**: Low dependency between assets")
    
    # Monte Carlo Results
    st.subheader("📈 Prediction Timeline with Monte Carlo")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Classification probabilities with future
    dates_test = results['dates_test']
    y_pred_proba = results['y_pred_proba']
    
    all_dates = list(dates_test) + [prediction_date]
    all_probabilities = list(y_pred_proba) + [results['forecast_proba']]
    
    ax1.plot(all_dates, all_probabilities, label='Prediction Probability', color='blue', linewidth=2, marker='o')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Boundary')
    ax1.plot(prediction_date, results['forecast_proba'], 'ro', markersize=10, label='Future Prediction')
    ax1.axvline(x=prediction_date, color='green', linestyle=':', alpha=0.7, label='Prediction Date')
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Probability of Price Increase')
    ax1.set_title(f'{target_symbol} Direction Prediction Timeline')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Monte Carlo Distribution
    final_prices = intervals['final_prices']
    ax2.hist(final_prices, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    ax2.axvline(intervals['lower_bound'], color='red', linestyle='--', linewidth=2, 
                label=f'Lower (${intervals["lower_bound"]:.2f})')
    ax2.axvline(intervals['median_price'], color='green', linestyle='-', linewidth=2, 
                label=f'Median (${intervals["median_price"]:.2f})')
    ax2.axvline(intervals['upper_bound'], color='red', linestyle='--', linewidth=2, 
                label=f'Upper (${intervals["upper_bound"]:.2f})')
    ax2.axvline(current_price, color='blue', linestyle='-', linewidth=2, 
                label=f'Current (${current_price:.2f})')
    
    ax2.set_xlabel('Predicted Price')
    ax2.set_ylabel('Probability Density')
    ax2.set_title(f'Monte Carlo: {target_symbol} Price Distribution after {prediction_horizon} days')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Prediction Intervals
    st.subheader("📊 Prediction Intervals")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Lower Bound", f"${intervals['lower_bound']:.2f}", 
                 f"{((intervals['lower_bound'] - current_price) / current_price * 100):+.1f}%")
    with col2:
        st.metric("Median", f"${intervals['median_price']:.2f}", 
                 f"{((intervals['median_price'] - current_price) / current_price * 100):+.1f}%")
    with col3:
        st.metric("Upper Bound", f"${intervals['upper_bound']:.2f}", 
                 f"{((intervals['upper_bound'] - current_price) / current_price * 100):+.1f}%")
    with col4:
        st.metric("Prediction Date", prediction_date.strftime('%Y-%m-%d'))

# Run the app
if __name__ == "__main__":
    main()

# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from datetime import datetime, timedelta

# try:
#     import yfinance as yf
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.pipeline import Pipeline
#     from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# except ImportError as e:
#     st.error(f"Missing required packages: {e}")
#     st.stop()

# # Streamlit app configuration
# st.set_page_config(
#     page_title="Crypto Predictor with Monte Carlo",
#     page_icon="🚀",
#     layout="wide"
# )

# # App title and description
# st.title("🚀 Advanced Crypto Predictor with Monte Carlo Simulation")
# st.markdown("""
# **Enhanced Features:**
# - **Monte Carlo Simulation** for price prediction intervals
# - **Prediction Horizon Positioning** with future dates
# - **Confidence Bounds** (Lower/Upper bounds)
# - **Probability Distribution** of future prices
# """)

# # Sidebar configuration
# st.sidebar.header("Configuration")

# # Date range input
# st.sidebar.subheader("📅 Date Range")
# col1, col2 = st.sidebar.columns(2)
# with col1:
#     start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
# with col2:
#     end_date = st.date_input("End Date", value=pd.to_datetime("today"))

# # Cryptocurrency selection
# st.sidebar.subheader("💰 Cryptocurrencies")
# crypto_options = {
#     "ETH-USD": "Ethereum",
#     "BTC-USD": "Bitcoin", 
#     "LINK-USD": "Chainlink",
#     "INJ-USD": "Injective",
#     "ADA-USD": "Cardano",
#     "DOT-USD": "Polkadot",
#     "SOL-USD": "Solana"
# }

# driver_symbol = st.sidebar.selectbox(
#     "Driver Cryptocurrency",
#     options=list(crypto_options.keys()),
#     format_func=lambda x: f"{x} ({crypto_options[x]})",
#     index=0
# )

# target_symbol = st.sidebar.selectbox(
#     "Target Cryptocurrency", 
#     options=list(crypto_options.keys()),
#     format_func=lambda x: f"{x} ({crypto_options[x]})",
#     index=3
# )

# # Rolling Window Configuration
# st.sidebar.subheader("🔄 Rolling Window Settings")

# window_size = st.sidebar.slider(
#     "Rolling Window Size (days)",
#     min_value=7,
#     max_value=90,
#     value=30
# )

# short_window = st.sidebar.slider("Short Window (days)", 5, 20, 7)
# medium_window = st.sidebar.slider("Medium Window (days)", 15, 45, 20)
# long_window = st.sidebar.slider("Long Window (days)", 30, 90, 50)

# # Model parameters
# st.sidebar.subheader("⚙️ Model Parameters")
# prediction_horizon = st.sidebar.slider(
#     "Prediction Horizon (days)",
#     min_value=1,
#     max_value=30,
#     value=7,
#     help="Number of days ahead to predict"
# )

# model_type = st.sidebar.selectbox(
#     "Model Type",
#     ["Logistic Regression", "Random Forest"],
#     index=0
# )

# # Monte Carlo Simulation Parameters
# st.sidebar.subheader("🎲 Monte Carlo Simulation")
# num_simulations = st.sidebar.slider(
#     "Number of Simulations",
#     min_value=100,
#     max_value=5000,
#     value=1000,
#     help="More simulations = more accurate confidence intervals"
# )

# confidence_level = st.sidebar.slider(
#     "Confidence Level (%)",
#     min_value=80,
#     max_value=99,
#     value=95,
#     help="Confidence interval for prediction bounds"
# )

# # Monte Carlo Simulation Functions
# def monte_carlo_simulation(current_price, volatility, prediction_horizon, num_simulations):
#     """
#     Run Monte Carlo simulation for price prediction
#     """
#     # Calculate daily returns and volatility
#     simulations = np.zeros((prediction_horizon, num_simulations))
#     simulations[0] = current_price
    
#     for day in range(1, prediction_horizon):
#         # Geometric Brownian Motion
#         random_returns = np.random.normal(0, volatility, num_simulations)
#         simulations[day] = simulations[day-1] * np.exp(random_returns)
    
#     return simulations

# def calculate_prediction_intervals(simulations, confidence_level):
#     """
#     Calculate prediction intervals from Monte Carlo simulations
#     """
#     final_prices = simulations[-1, :]
    
#     # Calculate percentiles based on confidence level
#     alpha = (100 - confidence_level) / 2
#     lower_percentile = alpha
#     upper_percentile = 100 - alpha
    
#     lower_bound = np.percentile(final_prices, lower_percentile)
#     upper_bound = np.percentile(final_prices, upper_percentile)
#     median_price = np.median(final_prices)
#     mean_price = np.mean(final_prices)
    
#     return {
#         'lower_bound': lower_bound,
#         'upper_bound': upper_bound,
#         'median_price': median_price,
#         'mean_price': mean_price,
#         'final_prices': final_prices
#     }

# # Main function
# def main():
#     if st.sidebar.button("🎯 Run Advanced Prediction", type="primary"):
#         run_advanced_prediction()

# def run_advanced_prediction():
#     """Main prediction function with Monte Carlo"""
    
#     # Convert dates to string format
#     start_str = start_date.strftime("%Y-%m-%d")
#     end_str = end_date.strftime("%Y-%m-%d")
    
#     # Progress tracking
#     progress_bar = st.progress(0)
#     status_text = st.empty()
    
#     try:
#         # Step 1: Download data
#         status_text.text("📥 Downloading cryptocurrency data...")
#         progress_bar.progress(10)
        
#         df_driver = yf.download(driver_symbol, start=start_str, end=end_str, progress=False)
#         df_target = yf.download(target_symbol, start=start_str, end=end_str, progress=False)
        
#         if df_driver.empty or df_target.empty:
#             st.error("❌ Failed to download data. Please check your internet connection and try again.")
#             return
        
#         # Step 2: Process data with rolling windows
#         status_text.text("🔄 Processing data with rolling windows...")
#         progress_bar.progress(30)
        
#         data = process_data_with_rolling_windows(df_driver, df_target, window_size, short_window, medium_window, long_window)
        
#         if data is None:
#             st.error("❌ Not enough data for analysis. Try a longer date range.")
#             return
        
#         # Step 3: Prepare features and target
#         status_text.text("🔧 Preparing features...")
#         progress_bar.progress(50)
        
#         feature_cols = prepare_rolling_features(data)
#         X, y, features_for_forecast, dates_clean = create_features_target(data, feature_cols, prediction_horizon)
        
#         if X.shape[0] == 0:
#             st.error("❌ No valid data for training after preprocessing.")
#             return
        
#         # Step 4: Train model and predict
#         status_text.text("🤖 Training model...")
#         progress_bar.progress(60)
        
#         classification_results = train_and_predict(X, y, features_for_forecast, model_type, dates_clean)
        
#         # Step 5: Monte Carlo Simulation
#         status_text.text("🎲 Running Monte Carlo simulations...")
#         progress_bar.progress(80)
        
#         # Get current price and volatility for Monte Carlo
#         current_price = data['target_close'].iloc[-1]
#         volatility = data['target_vol_medium'].iloc[-1] if not np.isnan(data['target_vol_medium'].iloc[-1]) else data['target_ret_1'].std()
        
#         # Run Monte Carlo simulation
#         simulations = monte_carlo_simulation(current_price, volatility, prediction_horizon, num_simulations)
#         prediction_intervals = calculate_prediction_intervals(simulations, confidence_level)
        
#         # Combine results
#         results = {
#             **classification_results,
#             'monte_carlo': {
#                 'simulations': simulations,
#                 'intervals': prediction_intervals,
#                 'current_price': current_price,
#                 'volatility': volatility
#             }
#         }
        
#         # Step 6: Display results
#         status_text.text("📊 Generating advanced results...")
#         progress_bar.progress(90)
        
#         display_advanced_results(data, results, driver_symbol, target_symbol, prediction_horizon)
        
#         progress_bar.progress(100)
#         status_text.text("✅ Advanced analysis complete!")
        
#     except Exception as e:
#         st.error(f"❌ An error occurred: {str(e)}")
#         import traceback
#         st.code(traceback.format_exc())

# def process_data_with_rolling_windows(df_driver, df_target, main_window, short_window, medium_window, long_window):
#     """Process data with multiple rolling windows"""
    
#     # Align data
#     data = pd.DataFrame(index=df_driver.index)
#     data['driver_close'] = df_driver['Close']
#     data['target_close'] = df_target['Close']
#     data = data.dropna()
    
#     if len(data) < main_window + 10:
#         return None
    
#     # Returns with different windows
#     data['driver_ret_1'] = data['driver_close'].pct_change(1)
#     data['driver_ret_short'] = data['driver_close'].pct_change(short_window)
#     data['driver_ret_medium'] = data['driver_close'].pct_change(medium_window)
    
#     # Rolling volatility
#     data['driver_vol_short'] = data['driver_ret_1'].rolling(short_window).std()
#     data['driver_vol_medium'] = data['driver_ret_1'].rolling(medium_window).std()
#     data['driver_vol_long'] = data['driver_ret_1'].rolling(long_window).std()
    
#     # Rolling averages
#     data['driver_ma_short'] = data['driver_close'].rolling(short_window).mean()
#     data['driver_ma_medium'] = data['driver_close'].rolling(medium_window).mean()
#     data['driver_ma_long'] = data['driver_close'].rolling(long_window).mean()
    
#     # Target features
#     data['target_ret_1'] = data['target_close'].pct_change(1)
#     data['target_ret_short'] = data['target_close'].pct_change(short_window)
#     data['target_ret_medium'] = data['target_close'].pct_change(medium_window)
    
#     data['target_vol_short'] = data['target_ret_1'].rolling(short_window).std()
#     data['target_vol_medium'] = data['target_ret_1'].rolling(medium_window).std()
#     data['target_vol_long'] = data['target_ret_1'].rolling(long_window).std()
    
#     data['target_ma_short'] = data['target_close'].rolling(short_window).mean()
#     data['target_ma_medium'] = data['target_close'].rolling(medium_window).mean()
#     data['target_ma_long'] = data['target_close'].rolling(long_window).mean()
    
#     # Cross-asset features
#     data['rolling_corr_short'] = data['driver_ret_1'].rolling(short_window).corr(data['target_ret_1'])
#     data['rolling_corr_medium'] = data['driver_ret_1'].rolling(medium_window).corr(data['target_ret_1'])
#     data['rolling_beta'] = data['rolling_corr_medium'] * (data['target_vol_medium'] / data['driver_vol_medium'])
    
#     data['price_ratio'] = data['target_close'] / data['driver_close']
#     data['price_ratio_ma'] = data['price_ratio'].rolling(medium_window).mean()
#     data['price_ratio_std'] = data['price_ratio'].rolling(medium_window).std()
    
#     data['ret_spread'] = data['target_ret_1'] - data['driver_ret_1']
#     data['ret_spread_ma'] = data['ret_spread'].rolling(medium_window).mean()
    
#     # Momentum features
#     data['driver_momentum'] = (data['driver_close'] - data['driver_ma_short']) / data['driver_vol_short']
#     data['target_momentum'] = (data['target_close'] - data['target_ma_short']) / data['target_vol_short']
    
#     data['driver_trend'] = (data['driver_ma_short'] - data['driver_ma_long']) / data['driver_vol_medium']
#     data['target_trend'] = (data['target_ma_short'] - data['target_ma_long']) / data['target_vol_medium']
    
#     # Use only recent data for performance
#     data = data.tail(150).copy()
    
#     return data

# def prepare_rolling_features(data):
#     """Prepare rolling feature column list"""
    
#     return_features = [
#         'driver_ret_1', 'driver_ret_short', 'driver_ret_medium',
#         'target_ret_1', 'target_ret_short', 'target_ret_medium'
#     ]
    
#     volatility_features = [
#         'driver_vol_short', 'driver_vol_medium', 'driver_vol_long',
#         'target_vol_short', 'target_vol_medium', 'target_vol_long'
#     ]
    
#     moving_average_features = [
#         'driver_ma_short', 'driver_ma_medium', 'driver_ma_long',
#         'target_ma_short', 'target_ma_medium', 'target_ma_long'
#     ]
    
#     cross_asset_features = [
#         'rolling_corr_short', 'rolling_corr_medium', 'rolling_beta',
#         'price_ratio', 'price_ratio_ma', 'price_ratio_std',
#         'ret_spread', 'ret_spread_ma'
#     ]
    
#     momentum_features = [
#         'driver_momentum', 'target_momentum',
#         'driver_trend', 'target_trend'
#     ]
    
#     all_features = (return_features + volatility_features + moving_average_features + 
#                    cross_asset_features + momentum_features)
    
#     feature_cols = [col for col in all_features if col in data.columns and not data[col].isna().all()]
    
#     return feature_cols

# def create_features_target(data, feature_cols, prediction_horizon):
#     """Create features and target variable"""
    
#     data['future_price'] = data['target_close'].shift(-prediction_horizon)
#     data['target_up'] = (data['future_price'] > data['target_close']).astype(int)
    
#     data_clean = data.dropna().copy()
    
#     if len(data_clean) == 0:
#         raise ValueError("No valid data after preprocessing")
    
#     X = data_clean[feature_cols].values
#     y = data_clean['target_up'].values
#     dates_clean = data_clean.index
    
#     features_for_forecast = data_clean.iloc[-1][feature_cols].values.reshape(1, -1)
    
#     return X, y, features_for_forecast, dates_clean

# def train_and_predict(X, y, features_for_forecast, model_type, dates):
#     """Train model and make predictions"""
    
#     min_samples = 20
#     if len(X) < min_samples:
#         raise ValueError(f"Not enough samples for training. Need at least {min_samples}, got {len(X)}")
    
#     split_idx = max(1, int(len(X) * 0.7))
    
#     if len(X) - split_idx < 5:
#         split_idx = max(1, len(X) - 5)
    
#     X_train, X_test = X[:split_idx], X[split_idx:]
#     y_train, y_test = y[:split_idx], y[split_idx:]
#     dates_test = dates[split_idx:]
    
#     if len(X_test) == 0:
#         raise ValueError("No test samples available.")
    
#     if model_type == "Logistic Regression":
#         classifier = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
#     else:
#         classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('classifier', classifier)
#     ])
    
#     pipeline.fit(X_train, y_train)
    
#     y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
#     y_pred = (y_pred_proba >= 0.5).astype(int)
    
#     forecast_proba = pipeline.predict_proba(features_for_forecast)[0, 1]
    
#     return {
#         'pipeline': pipeline,
#         'y_test': y_test,
#         'y_pred': y_pred,
#         'y_pred_proba': y_pred_proba,
#         'forecast_proba': forecast_proba,
#         'X_test': X_test,
#         'dates_test': dates_test,
#         'split_idx': split_idx
#     }

# def display_advanced_results(data, results, driver_symbol, target_symbol, prediction_horizon):
#     """Display all results with Monte Carlo and prediction intervals"""
    
#     current_price = data['target_close'].iloc[-1]
#     mc_results = results['monte_carlo']
#     intervals = mc_results['intervals']
    
#     # Generate future dates for prediction horizon
#     last_date = data.index[-1]
#     future_dates = [last_date + timedelta(days=i) for i in range(1, prediction_horizon + 1)]
#     prediction_date = future_dates[-1]
    
#     # Display main prediction cards
#     st.header("🎯 Advanced Prediction Results")
    
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.metric(f"{target_symbol} Current", f"${current_price:.2f}")
    
#     with col2:
#         direction = "BULLISH" if results['forecast_proba'] >= 0.5 else "BEARISH"
#         st.metric("Direction", f"{direction}")
    
#     with col3:
#         st.metric("Probability", f"{results['forecast_proba']:.3f}")
    
#     with col4:
#         change_pct = ((intervals['median_price'] - current_price) / current_price) * 100
#         st.metric("Expected Change", f"{change_pct:+.1f}%")
    
#     # Monte Carlo Prediction Intervals
#     st.subheader("📊 Monte Carlo Prediction Intervals")
    
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.metric(
#             "Lower Bound", 
#             f"${intervals['lower_bound']:.2f}",
#             f"{((intervals['lower_bound'] - current_price) / current_price * 100):+.1f}%"
#         )
    
#     with col2:
#         st.metric(
#             "Median Price", 
#             f"${intervals['median_price']:.2f}",
#             f"{((intervals['median_price'] - current_price) / current_price * 100):+.1f}%"
#         )
    
#     with col3:
#         st.metric(
#             "Upper Bound", 
#             f"${intervals['upper_bound']:.2f}",
#             f"{((intervals['upper_bound'] - current_price) / current_price * 100):+.1f}%"
#         )
    
#     with col4:
#         st.metric(
#             "Prediction Date", 
#             prediction_date.strftime('%Y-%m-%d')
#         )
    
#     # Enhanced Prediction Timeline with Future Horizon
#     st.subheader("📈 Advanced Prediction Timeline")
    
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
#     # Plot 1: Classification probabilities with future prediction
#     dates_test = results['dates_test']
#     y_pred_proba = results['y_pred_proba']
    
#     # Extend timeline to include future prediction
#     all_dates = list(dates_test) + [prediction_date]
#     all_probabilities = list(y_pred_proba) + [results['forecast_proba']]
    
#     ax1.plot(all_dates, all_probabilities, label='Prediction Probability', color='blue', linewidth=2, marker='o')
#     ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Boundary')
    
#     # Highlight future prediction
#     ax1.plot(prediction_date, results['forecast_proba'], 'ro', markersize=10, label=f'Future Prediction')
#     ax1.axvline(x=prediction_date, color='green', linestyle=':', alpha=0.7, label='Prediction Date')
    
#     # Add confidence interval for future prediction
#     ax1.fill_between([prediction_date, prediction_date], 
#                      results['forecast_proba'] - 0.1, 
#                      results['forecast_proba'] + 0.1, 
#                      alpha=0.2, color='red', label='Prediction Uncertainty')
    
#     ax1.set_xlabel('Date')
#     ax1.set_ylabel('Probability of Price Increase')
#     ax1.set_title(f'{target_symbol} Direction Prediction Timeline (Including Future Horizon)')
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
#     plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
#     # Plot 2: Monte Carlo Simulation Results
#     final_prices = intervals['final_prices']
    
#     # Plot histogram of final prices
#     ax2.hist(final_prices, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
#     ax2.axvline(intervals['lower_bound'], color='red', linestyle='--', linewidth=2, label=f'Lower Bound (${intervals["lower_bound"]:.2f})')
#     ax2.axvline(intervals['median_price'], color='green', linestyle='-', linewidth=2, label=f'Median (${intervals["median_price"]:.2f})')
#     ax2.axvline(intervals['upper_bound'], color='red', linestyle='--', linewidth=2, label=f'Upper Bound (${intervals["upper_bound"]:.2f})')
#     ax2.axvline(current_price, color='blue', linestyle='-', linewidth=2, label=f'Current Price (${current_price:.2f})')
    
#     ax2.set_xlabel('Predicted Price')
#     ax2.set_ylabel('Probability Density')
#     ax2.set_title(f'Monte Carlo Simulation: {target_symbol} Price Distribution after {prediction_horizon} days\n'
#                  f'{confidence_level}% Confidence Interval: [${intervals["lower_bound"]:.2f}, ${intervals["upper_bound"]:.2f}]')
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     st.pyplot(fig)
    
#     # Monte Carlo Path Visualization
#     st.subheader("🎲 Monte Carlo Simulation Paths")
    
#     fig2, ax3 = plt.subplots(figsize=(12, 6))
    
#     # Plot a subset of simulation paths for clarity
#     num_paths_to_plot = min(100, num_simulations)
#     simulation_dates = [last_date + timedelta(days=i) for i in range(prediction_horizon)]
    
#     for i in range(num_paths_to_plot):
#         ax3.plot(simulation_dates, mc_results['simulations'][:, i], alpha=0.1, color='blue')
    
#     # Plot confidence intervals
#     percentiles = np.percentile(mc_results['simulations'], [5, 25, 50, 75, 95], axis=1)
#     ax3.plot(simulation_dates, percentiles[2], 'g-', linewidth=3, label='Median Path')
#     ax3.fill_between(simulation_dates, percentiles[0], percentiles[4], alpha=0.3, color='red', label='90% Confidence Interval')
#     ax3.fill_between(simulation_dates, percentiles[1], percentiles[3], alpha=0.3, color='orange', label='50% Confidence Interval')
    
#     ax3.axhline(y=current_price, color='black', linestyle='--', linewidth=2, label=f'Current Price (${current_price:.2f})')
#     ax3.set_xlabel('Date')
#     ax3.set_ylabel('Price')
#     ax3.set_title(f'Monte Carlo Simulation Paths for {target_symbol} ({num_simulations} simulations)')
#     ax3.legend()
#     ax3.grid(True, alpha=0.3)
#     plt.xticks(rotation=45)
#     plt.tight_layout()
    
#     st.pyplot(fig2)
    
#     # Risk Analysis
#     st.subheader("📉 Risk Analysis")
    
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         probability_down = np.mean(final_prices < current_price) * 100
#         st.metric("Probability of Decline", f"{probability_down:.1f}%")
    
#     with col2:
#         var_95 = np.percentile(final_prices, 5)
#         var_loss = ((var_95 - current_price) / current_price) * 100
#         st.metric("VaR (95%)", f"{var_loss:.1f}%")
    
#     with col3:
#         expected_shortfall = np.mean(final_prices[final_prices <= var_95])
#         es_loss = ((expected_shortfall - current_price) / current_price) * 100
#         st.metric("Expected Shortfall", f"{es_loss:.1f}%")
    
#     with col4:
#         best_case = np.max(final_prices)
#         best_gain = ((best_case - current_price) / current_price) * 100
#         st.metric("Best Case", f"{best_gain:+.1f}%")
    
#     # Detailed Report
#     with st.expander("📋 Detailed Analysis Report"):
#         st.write("### Classification Performance")
#         accuracy = accuracy_score(results['y_test'], results['y_pred'])
#         st.metric("Model Accuracy", f"{accuracy:.3f}")
        
#         st.write("### Monte Carlo Statistics")
#         stats_data = {
#             'Statistic': ['Current Price', 'Mean Prediction', 'Median Prediction', 'Std Deviation', 
#                          'Skewness', 'Kurtosis', 'Confidence Interval'],
#             'Value': [
#                 f"${current_price:.2f}",
#                 f"${intervals['mean_price']:.2f}",
#                 f"${intervals['median_price']:.2f}",
#                 f"${np.std(final_prices):.2f}",
#                 f"{pd.Series(final_prices).skew():.3f}",
#                 f"{pd.Series(final_prices).kurtosis():.3f}",
#                 f"[${intervals['lower_bound']:.2f}, ${intervals['upper_bound']:.2f}]"
#             ]
#         }
#         st.table(pd.DataFrame(stats_data))

# # Run the app
# if __name__ == "__main__":
#     main()
