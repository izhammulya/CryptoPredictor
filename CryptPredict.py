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
    page_title="Crypto Prediction Suite with Oscillators",
    page_icon="🚀",
    layout="wide"
)

# App title and description
st.title("🚀 Crypto Prediction Suite: Oscillators + 3 Modeling Approaches")
st.markdown("""
**Complete Feature Set:**
- **Driver-Target Cryptocurrency Pairs** with oscillator analysis
- **🎲 Monte Carlo**: Price simulation with confidence intervals
- **📊 Logistic Regression**: Direction prediction (Up/Down)  
- **🌲 Random Forest**: Ensemble learning for price direction
- **🎯 Oscillator Parameters**: Alpha, Omega, K visualization
""")

# Sidebar configuration
st.sidebar.header("🎯 Prediction Configuration")

# Model Selection
st.sidebar.subheader("🤖 Select Prediction Method")
prediction_method = st.sidebar.radio(
    "Choose Model:",
    ["Monte Carlo Simulation", "Logistic Regression", "Random Forest"],
    index=0
)

# Cryptocurrency selection with DRIVER-TARGET pairs
st.sidebar.subheader("💰 Cryptocurrency Pairs")
crypto_options = {
    "ETH-USD": "Ethereum",
    "BTC-USD": "Bitcoin", 
    "LINK-USD": "Chainlink",
    "INJ-USD": "Injective",
    "ADA-USD": "Cardano",
    "SOL-USD": "Solana",
    "DOT-USD": "Polkadot"
}

driver_symbol = st.sidebar.selectbox(
    "Driver Cryptocurrency",
    options=list(crypto_options.keys()),
    index=0,
    help="The cryptocurrency that influences the target"
)

target_symbol = st.sidebar.selectbox(
    "Target Cryptocurrency", 
    options=list(crypto_options.keys()),
    index=3,
    help="The cryptocurrency we want to predict"
)

# Date range input
st.sidebar.subheader("📅 Date Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
with col2:
    end_date = st.date_input("End Date", value=pd.to_datetime("today"))

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

# Method-specific parameters
if prediction_method == "Monte Carlo Simulation":
    st.sidebar.subheader("🎲 Monte Carlo Settings")
    prediction_horizon = st.sidebar.slider("Prediction Horizon (days)", 1, 90, 30)
    num_simulations = st.sidebar.slider("Number of Simulations", 100, 5000, 1000)
    confidence_level = st.sidebar.slider("Confidence Level", 80, 99, 95)
    show_paths = st.sidebar.checkbox("Show Simulation Paths", value=True)
    
elif prediction_method == "Logistic Regression":
    st.sidebar.subheader("📊 Logistic Regression Settings")
    prediction_horizon = st.sidebar.slider("Prediction Horizon (days)", 1, 30, 7)
    train_test_split = st.sidebar.slider("Train/Test Split", 0.6, 0.9, 0.8)
    
elif prediction_method == "Random Forest":
    st.sidebar.subheader("🌲 Random Forest Settings")
    prediction_horizon = st.sidebar.slider("Prediction Horizon (days)", 1, 30, 7)
    n_estimators = st.sidebar.slider("Number of Trees", 50, 500, 100)
    max_depth = st.sidebar.slider("Max Tree Depth", 3, 20, 10)

# OSCILLATOR FUNCTIONS
def fast_oscillator_approximation(driver_prices, target_prices):
    """
    Fast approximation of oscillator parameters without ODE solving
    """
    if len(driver_prices) < 10:
        return 0.1, 1.0, 1.0
    
    try:
        # Calculate returns
        driver_rets = np.diff(np.log(driver_prices))
        target_rets = np.diff(np.log(target_prices))
        
        # Alpha: Damping from autocorrelation (simplified)
        if len(target_rets) > 1:
            autocorr = np.corrcoef(target_rets[:-1], target_rets[1:])[0,1]
            alpha = max(0.01, 0.5 * (1 - autocorr))
        else:
            alpha = 0.1
        
        # Omega: Frequency from volatility (simplified)
        if len(target_rets) > 0:
            volatility = np.std(target_rets)
            omega = max(0.1, 2 * np.pi * volatility * 10)  # Scale factor
        else:
            omega = 1.0
        
        # K: Coupling from correlation (simplified)
        if len(driver_rets) > 0 and len(target_rets) > 0:
            min_len = min(len(driver_rets), len(target_rets))
            correlation = np.corrcoef(driver_rets[:min_len], target_rets[:min_len])[0,1]
            k = correlation * 2.0  # Scale factor
        else:
            k = 1.0
        
        return alpha, omega, k
        
    except:
        return 0.1, 1.0, 1.0

def calculate_fast_oscillator_features(data, window):
    """Calculate oscillator features quickly"""
    if len(data) < window:
        return data
    
    # Use every 3rd data point to speed up calculation
    step = 2
    indices = list(range(window, len(data), step))
    
    alphas, omegas, ks = [], [], []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, idx in enumerate(indices):
        window_slice = slice(idx - window, idx)
        driver_vals = data['driver_close'].iloc[window_slice].values
        target_vals = data['target_close'].iloc[window_slice].values
        
        alpha, omega, k = fast_oscillator_approximation(driver_vals, target_vals)
        alphas.append(alpha)
        omegas.append(omega)
        ks.append(k)
        
        if i % 5 == 0:  # Update progress less frequently
            progress = (i + 1) / len(indices)
            progress_bar.progress(progress)
            status_text.text(f"Calculating oscillators... {i+1}/{len(indices)}")
    
    # Add oscillator features to dataframe
    data['oscillator_alpha'] = np.nan
    data['oscillator_omega'] = np.nan
    data['oscillator_k'] = np.nan
    
    # Assign values only at calculated indices
    for i, idx in enumerate(indices):
        if idx < len(data):
            data.iloc[idx, data.columns.get_loc('oscillator_alpha')] = alphas[i]
            data.iloc[idx, data.columns.get_loc('oscillator_omega')] = omegas[i]
            data.iloc[idx, data.columns.get_loc('oscillator_k')] = ks[i]
    
    # Forward fill missing values
    data['oscillator_alpha'] = data['oscillator_alpha'].ffill()
    data['oscillator_omega'] = data['oscillator_omega'].ffill()
    data['oscillator_k'] = data['oscillator_k'].ffill()
    
    progress_bar.progress(1.0)
    status_text.text("Oscillator calculation complete!")
    
    return data

# MONTE CARLO FUNCTIONS
def monte_carlo_simulation(current_price, historical_returns, horizon, n_simulations):
    """Enhanced Monte Carlo simulation"""
    mean_return = np.mean(historical_returns)
    std_return = np.std(historical_returns)
    
    # Generate random returns
    random_returns = np.random.normal(mean_return, std_return, (horizon, n_simulations))
    
    # Calculate price paths
    price_paths = np.zeros((horizon + 1, n_simulations))
    price_paths[0] = current_price
    
    for day in range(1, horizon + 1):
        price_paths[day] = price_paths[day-1] * (1 + random_returns[day-1])
    
    return price_paths

def calculate_prediction_intervals(price_paths, confidence_level):
    """Calculate prediction intervals for all horizons"""
    horizon_predictions = {}
    
    for day in range(1, price_paths.shape[0]):
        prices_at_horizon = price_paths[day]
        
        alpha = (100 - confidence_level) / 2
        lower_bound = np.percentile(prices_at_horizon, alpha)
        upper_bound = np.percentile(prices_at_horizon, 100 - alpha)
        median_price = np.median(prices_at_horizon)
        mean_price = np.mean(prices_at_horizon)
        
        horizon_predictions[day] = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'median_price': median_price,
            'mean_price': mean_price,
            'prices': prices_at_horizon
        }
    
    return horizon_predictions

# MACHINE LEARNING FUNCTIONS
def prepare_ml_features(data, horizon, use_oscillator):
    """Prepare features for machine learning models"""
    df = data.copy()
    
    # Basic price features for both driver and target
    df['driver_returns_1d'] = df['driver_close'].pct_change(1)
    df['driver_returns_5d'] = df['driver_close'].pct_change(5)
    df['driver_volatility_10d'] = df['driver_returns_1d'].rolling(10).std()
    
    df['target_returns_1d'] = df['target_close'].pct_change(1)
    df['target_returns_5d'] = df['target_close'].pct_change(5)
    df['target_volatility_10d'] = df['target_returns_1d'].rolling(10).std()
    
    # Cross-asset features
    df['price_ratio'] = df['target_close'] / df['driver_close']
    df['returns_correlation_20d'] = df['driver_returns_1d'].rolling(20).corr(df['target_returns_1d'])
    df['returns_spread'] = df['target_returns_1d'] - df['driver_returns_1d']
    
    # Momentum features
    df['target_momentum_5d'] = df['target_close'] / df['target_close'].shift(5) - 1
    df['driver_momentum_5d'] = df['driver_close'] / df['driver_close'].shift(5) - 1
    
    # Target variable (price direction)
    df['future_price'] = df['target_close'].shift(-horizon)
    df['price_up'] = (df['future_price'] > df['target_close']).astype(int)
    
    # Add oscillator features if enabled
    if use_oscillator:
        oscillator_features = ['oscillator_alpha', 'oscillator_omega', 'oscillator_k']
        for feature in oscillator_features:
            if feature in df.columns:
                df[feature] = df[feature].ffill()  # Ensure no NaN values
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def train_ml_model(data, horizon, model_type, use_oscillator, **kwargs):
    """Train machine learning model (Logistic Regression or Random Forest)"""
    # Prepare features
    df = prepare_ml_features(data, horizon, use_oscillator)
    
    # Feature columns
    base_features = [
        'driver_returns_1d', 'driver_returns_5d', 'driver_volatility_10d',
        'target_returns_1d', 'target_returns_5d', 'target_volatility_10d',
        'price_ratio', 'returns_correlation_20d', 'returns_spread',
        'target_momentum_5d', 'driver_momentum_5d'
    ]
    
    # Add oscillator features if available
    if use_oscillator:
        oscillator_features = ['oscillator_alpha', 'oscillator_omega', 'oscillator_k']
        for feature in oscillator_features:
            if feature in df.columns:
                base_features.append(feature)
    
    feature_cols = [col for col in base_features if col in df.columns]
    
    X = df[feature_cols].values
    y = df['price_up'].values
    
    # Train-test split
    test_size = kwargs.get('test_size', 0.2)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = df.index[split_idx:]
    
    # Create pipeline based on model type
    if model_type == "Logistic Regression":
        classifier = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    else:  # Random Forest
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', 10)
        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            class_weight='balanced'
        )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Current prediction
    current_features = df[feature_cols].iloc[-1].values.reshape(1, -1)
    current_prediction = pipeline.predict_proba(current_features)[0, 1]
    
    # Feature importance
    if model_type == "Logistic Regression":
        feature_importance = np.abs(pipeline.named_steps['classifier'].coef_[0])
    else:
        feature_importance = pipeline.named_steps['classifier'].feature_importances_
    
    return {
        'pipeline': pipeline,
        'feature_cols': feature_cols,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'dates_test': dates_test,
        'current_prediction': current_prediction,
        'feature_importance': feature_importance,
        'feature_names': feature_cols
    }

# Main function
def main():
    if st.sidebar.button("🎯 Run Prediction", type="primary"):
        run_prediction()

def run_prediction():
    """Main prediction function"""
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Download BOTH driver and target data
        status_text.text("📥 Downloading cryptocurrency data...")
        progress_bar.progress(20)
        
        df_driver = yf.download(driver_symbol, start=start_str, end=end_str, progress=False)
        df_target = yf.download(target_symbol, start=start_str, end=end_str, progress=False)
        
        if df_driver.empty or df_target.empty:
            st.error("❌ Failed to download data.")
            return
        
        # Step 2: Align and process data
        status_text.text("🔄 Processing data...")
        progress_bar.progress(40)
        
        # Create aligned dataframe
        data = pd.DataFrame(index=df_driver.index)
        data['driver_close'] = df_driver['Close']
        data['target_close'] = df_target['Close']
        data = data.dropna()
        
        current_target_price = data['target_close'].iloc[-1]
        current_driver_price = data['driver_close'].iloc[-1]
        
        # Step 3: Calculate oscillator features if enabled
        if use_oscillator:
            status_text.text("🎯 Calculating oscillator parameters...")
            progress_bar.progress(60)
            data = calculate_fast_oscillator_features(data, oscillator_window)
        else:
            # Add placeholder oscillator columns
            data['oscillator_alpha'] = np.nan
            data['oscillator_omega'] = np.nan
            data['oscillator_k'] = np.nan
        
        # Step 4: Run selected prediction method
        status_text.text(f"🤖 Running {prediction_method}...")
        progress_bar.progress(80)
        
        if prediction_method == "Monte Carlo Simulation":
            results = run_monte_carlo(data, current_target_price, prediction_horizon, num_simulations, confidence_level)
        else:
            # For ML models, pass additional parameters
            ml_params = {}
            if prediction_method == "Random Forest":
                ml_params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'test_size': 1-train_test_split}
            else:
                ml_params = {'test_size': 1-train_test_split}
                
            results = train_ml_model(data, prediction_horizon, prediction_method, use_oscillator, **ml_params)
        
        # Step 5: Display results
        status_text.text("📊 Generating results...")
        progress_bar.progress(95)
        
        display_results(data, results, prediction_method, driver_symbol, target_symbol, 
                       current_target_price, current_driver_price, prediction_horizon)
        
        progress_bar.progress(100)
        status_text.text("✅ Prediction complete!")
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def run_monte_carlo(data, current_price, horizon, n_simulations, confidence_level):
    """Run Monte Carlo simulation"""
    returns = data['target_close'].pct_change().dropna()
    
    price_paths = monte_carlo_simulation(current_price, returns, horizon, n_simulations)
    predictions = calculate_prediction_intervals(price_paths, confidence_level)
    
    return {
        'price_paths': price_paths,
        'predictions': predictions,
        'current_price': current_price,
        'returns': returns
    }

def display_results(data, results, method, driver_symbol, target_symbol, 
                   current_target_price, current_driver_price, horizon):
    """Display results based on selected method"""
    
    st.header(f"🎯 {method} Results: {driver_symbol} → {target_symbol}")
    
    # Display current prices
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"{driver_symbol} (Driver)", f"${current_driver_price:.2f}")
    with col2:
        st.metric(f"{target_symbol} (Target)", f"${current_target_price:.2f}")
    
    # Display oscillator results if enabled
    if use_oscillator and not data['oscillator_alpha'].isna().all():
        display_oscillator_analysis(data, driver_symbol, target_symbol)
    
    # Display method-specific results
    if method == "Monte Carlo Simulation":
        display_monte_carlo_results(data, results, driver_symbol, target_symbol, horizon)
    else:
        display_ml_results(data, results, method, driver_symbol, target_symbol, horizon)

def display_oscillator_analysis(data, driver_symbol, target_symbol):
    """Display oscillator parameter analysis"""
    
    st.subheader("🎯 Oscillator Parameter Analysis")
    
    # Get current oscillator values
    current_alpha = data['oscillator_alpha'].iloc[-1] if not np.isnan(data['oscillator_alpha'].iloc[-1]) else 0
    current_omega = data['oscillator_omega'].iloc[-1] if not np.isnan(data['oscillator_omega'].iloc[-1]) else 0
    current_k = data['oscillator_k'].iloc[-1] if not np.isnan(data['oscillator_k'].iloc[-1]) else 0
    
    # Create oscillator visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Alpha (Damping) over time
    valid_alpha = data['oscillator_alpha'].dropna()
    if len(valid_alpha) > 0:
        ax1.plot(valid_alpha.index, valid_alpha.values, color='red', linewidth=2)
        ax1.set_title('Oscillator Alpha (Damping Coefficient)')
        ax1.set_ylabel('Alpha Value')
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Omega (Frequency) over time
    valid_omega = data['oscillator_omega'].dropna()
    if len(valid_omega) > 0:
        ax2.plot(valid_omega.index, valid_omega.values, color='blue', linewidth=2)
        ax2.set_title('Oscillator Omega (Natural Frequency)')
        ax2.set_ylabel('Omega Value')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: K (Coupling) over time
    valid_k = data['oscillator_k'].dropna()
    if len(valid_k) > 0:
        ax3.plot(valid_k.index, valid_k.values, color='green', linewidth=2)
        ax3.set_title('Oscillator K (Coupling Strength)')
        ax3.set_ylabel('K Value')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Current oscillator state
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
    st.pyplot(fig)
    
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

def display_monte_carlo_results(data, results, driver_symbol, target_symbol, horizon):
    """Display Monte Carlo results"""
    
    price_paths = results['price_paths']
    predictions = results['predictions']
    current_price = results['current_price']
    
    # Generate future dates
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(horizon + 1)]
    
    # Key metrics
    pred = predictions[horizon]
    expected_change = ((pred['median_price'] - current_price) / current_price) * 100
    prob_up = np.mean(pred['prices'] > current_price) * 100
    
    st.subheader("📈 Monte Carlo Price Prediction")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${current_price:.2f}")
    col2.metric("Median Prediction", f"${pred['median_price']:.2f}", f"{expected_change:+.1f}%")
    col3.metric("Probability Up", f"{prob_up:.1f}%")
    col4.metric("Confidence Interval", f"${pred['lower_bound']:.2f} - ${pred['upper_bound']:.2f}")
    
    # Monte Carlo visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical data
    historical_data = data['target_close'].tail(60)
    ax.plot(historical_data.index, historical_data.values, 'b-', linewidth=2, label='Historical Price')
    
    # Plot Monte Carlo paths
    if show_paths:
        for i in range(min(100, num_simulations)):
            ax.plot(future_dates, price_paths[:, i], 'gray', alpha=0.1, linewidth=0.5)
    
    # Plot confidence intervals
    percentiles = np.percentile(price_paths, [5, 25, 50, 75, 95], axis=1)
    ax.fill_between(future_dates, percentiles[0], percentiles[4], alpha=0.3, color='red', label='90% CI')
    ax.fill_between(future_dates, percentiles[1], percentiles[3], alpha=0.4, color='orange', label='50% CI')
    ax.plot(future_dates, percentiles[2], 'r-', linewidth=3, label='Median Path')
    
    # Current price and prediction markers
    ax.axhline(y=current_price, color='black', linestyle='--', alpha=0.7, label='Current Price')
    ax.axvline(x=future_dates[0], color='green', linestyle=':', alpha=0.7, label='Today')
    ax.axvline(x=future_dates[horizon], color='purple', linestyle='--', alpha=0.7, label=f'{horizon}-Day Horizon')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title(f'{target_symbol} Monte Carlo Price Simulation\n{driver_symbol} as Driver | {num_simulations} simulations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)

def display_ml_results(data, results, method, driver_symbol, target_symbol, horizon):
    """Display machine learning results"""
    
    current_prediction = results['current_prediction']
    y_test = results['y_test']
    y_pred = results['y_pred']
    y_pred_proba = results['y_pred_proba']
    dates_test = results['dates_test']
    
    # Performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    prob_up = current_prediction * 100
    direction = "UP" if current_prediction >= 0.5 else "DOWN"
    
    st.subheader(f"📊 {method} Direction Prediction")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model Accuracy", f"{accuracy:.3f}")
    col2.metric("Current Prediction", direction)
    col3.metric("Probability", f"{current_prediction:.3f}")
    col4.metric("Confidence", f"{prob_up:.1f}%")
    
    # Prediction timeline
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(dates_test, y_pred_proba, 'b-', linewidth=2, label='Prediction Probability', alpha=0.7)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Boundary')
    
    # Highlight correct/incorrect predictions
    correct_predictions = (y_pred == y_test)
    incorrect_predictions = ~correct_predictions
    
    if any(correct_predictions):
        ax.scatter(dates_test[correct_predictions], y_pred_proba[correct_predictions], 
                  color='green', alpha=0.6, label='Correct Prediction', s=30)
    
    if any(incorrect_predictions):
        ax.scatter(dates_test[incorrect_predictions], y_pred_proba[incorrect_predictions], 
                  color='red', alpha=0.6, label='Incorrect Prediction', s=30)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Probability of Price Increase')
    ax.set_title(f'{target_symbol} {method} Predictions ({horizon}-day horizon)\n{driver_symbol} as Driver | Accuracy: {accuracy:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Feature importance
    st.subheader("🔍 Feature Importance")
    
    feature_importance = results['feature_importance']
    feature_names = results['feature_names']
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(importance_df))
    
    ax2.barh(y_pos, importance_df['Importance'], color='skyblue', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(importance_df['Feature'])
    ax2.set_xlabel('Importance')
    ax2.set_title(f'Feature Importance - {method}\n(Oscillator features in red)')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Highlight oscillator features
    for i, feature in enumerate(importance_df['Feature']):
        if 'oscillator' in feature:
            ax2.get_children()[i].set_color('red')
    
    plt.tight_layout()
    st.pyplot(fig2)

# Run the app
if __name__ == "__main__":
    main()

# Add method comparison
st.sidebar.markdown("---")
st.sidebar.info("""
**🤖 Method Comparison:**
- **Monte Carlo**: Price range predictions with confidence intervals
- **Logistic Regression**: Direction (Up/Down) with probability scores  
- **Random Forest**: Robust pattern recognition with feature importance
- **Oscillators**: Driver-Target relationship dynamics
""")

# # # V2 Sistem
# # import streamlit as st
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from datetime import datetime, timedelta

# # try:
# #     import yfinance as yf
# #     from sklearn.linear_model import LogisticRegression
# #     from sklearn.ensemble import RandomForestClassifier
# #     from sklearn.preprocessing import StandardScaler
# #     from sklearn.pipeline import Pipeline
# #     from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# # except ImportError as e:
# #     st.error(f"Missing required packages: {e}")
# #     st.stop()

# # # Streamlit app configuration
# # st.set_page_config(
# #     page_title="Crypto Predictor with Monte Carlo",
# #     page_icon="🚀",
# #     layout="wide"
# # )

# # # App title and description
# # st.title("🚀 Advanced Crypto Predictor with Monte Carlo Simulation")
# # st.markdown("""
# # **Enhanced Features:**
# # - **Monte Carlo Simulation** for price prediction intervals
# # - **Prediction Horizon Positioning** with future dates
# # - **Confidence Bounds** (Lower/Upper bounds)
# # - **Probability Distribution** of future prices
# # """)

# # # Sidebar configuration
# # st.sidebar.header("Configuration")

# # # Date range input
# # st.sidebar.subheader("📅 Date Range")
# # col1, col2 = st.sidebar.columns(2)
# # with col1:
# #     start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
# # with col2:
# #     end_date = st.date_input("End Date", value=pd.to_datetime("today"))

# # # Cryptocurrency selection
# # st.sidebar.subheader("💰 Cryptocurrencies")
# # crypto_options = {
# #     "ETH-USD": "Ethereum",
# #     "BTC-USD": "Bitcoin", 
# #     "LINK-USD": "Chainlink",
# #     "INJ-USD": "Injective",
# #     "ADA-USD": "Cardano",
# #     "DOT-USD": "Polkadot",
# #     "SOL-USD": "Solana"
# # }

# # driver_symbol = st.sidebar.selectbox(
# #     "Driver Cryptocurrency",
# #     options=list(crypto_options.keys()),
# #     format_func=lambda x: f"{x} ({crypto_options[x]})",
# #     index=0
# # )

# # target_symbol = st.sidebar.selectbox(
# #     "Target Cryptocurrency", 
# #     options=list(crypto_options.keys()),
# #     format_func=lambda x: f"{x} ({crypto_options[x]})",
# #     index=3
# # )

# # # Rolling Window Configuration
# # st.sidebar.subheader("🔄 Rolling Window Settings")

# # window_size = st.sidebar.slider(
# #     "Rolling Window Size (days)",
# #     min_value=7,
# #     max_value=90,
# #     value=30
# # )

# # short_window = st.sidebar.slider("Short Window (days)", 5, 20, 7)
# # medium_window = st.sidebar.slider("Medium Window (days)", 15, 45, 20)
# # long_window = st.sidebar.slider("Long Window (days)", 30, 90, 50)

# # # Model parameters
# # st.sidebar.subheader("⚙️ Model Parameters")
# # prediction_horizon = st.sidebar.slider(
# #     "Prediction Horizon (days)",
# #     min_value=1,
# #     max_value=30,
# #     value=7,
# #     help="Number of days ahead to predict"
# # )

# # model_type = st.sidebar.selectbox(
# #     "Model Type",
# #     ["Logistic Regression", "Random Forest"],
# #     index=0
# # )

# # # Monte Carlo Simulation Parameters
# # st.sidebar.subheader("🎲 Monte Carlo Simulation")
# # num_simulations = st.sidebar.slider(
# #     "Number of Simulations",
# #     min_value=100,
# #     max_value=5000,
# #     value=1000,
# #     help="More simulations = more accurate confidence intervals"
# # )

# # confidence_level = st.sidebar.slider(
# #     "Confidence Level (%)",
# #     min_value=80,
# #     max_value=99,
# #     value=95,
# #     help="Confidence interval for prediction bounds"
# # )

# # # Monte Carlo Simulation Functions
# # def monte_carlo_simulation(current_price, volatility, prediction_horizon, num_simulations):
# #     """
# #     Run Monte Carlo simulation for price prediction
# #     """
# #     # Calculate daily returns and volatility
# #     simulations = np.zeros((prediction_horizon, num_simulations))
# #     simulations[0] = current_price
    
# #     for day in range(1, prediction_horizon):
# #         # Geometric Brownian Motion
# #         random_returns = np.random.normal(0, volatility, num_simulations)
# #         simulations[day] = simulations[day-1] * np.exp(random_returns)
    
# #     return simulations

# # def calculate_prediction_intervals(simulations, confidence_level):
# #     """
# #     Calculate prediction intervals from Monte Carlo simulations
# #     """
# #     final_prices = simulations[-1, :]
    
# #     # Calculate percentiles based on confidence level
# #     alpha = (100 - confidence_level) / 2
# #     lower_percentile = alpha
# #     upper_percentile = 100 - alpha
    
# #     lower_bound = np.percentile(final_prices, lower_percentile)
# #     upper_bound = np.percentile(final_prices, upper_percentile)
# #     median_price = np.median(final_prices)
# #     mean_price = np.mean(final_prices)
    
# #     return {
# #         'lower_bound': lower_bound,
# #         'upper_bound': upper_bound,
# #         'median_price': median_price,
# #         'mean_price': mean_price,
# #         'final_prices': final_prices
# #     }

# # # Main function
# # def main():
# #     if st.sidebar.button("🎯 Run Advanced Prediction", type="primary"):
# #         run_advanced_prediction()

# # def run_advanced_prediction():
# #     """Main prediction function with Monte Carlo"""
    
# #     # Convert dates to string format
# #     start_str = start_date.strftime("%Y-%m-%d")
# #     end_str = end_date.strftime("%Y-%m-%d")
    
# #     # Progress tracking
# #     progress_bar = st.progress(0)
# #     status_text = st.empty()
    
# #     try:
# #         # Step 1: Download data
# #         status_text.text("📥 Downloading cryptocurrency data...")
# #         progress_bar.progress(10)
        
# #         df_driver = yf.download(driver_symbol, start=start_str, end=end_str, progress=False)
# #         df_target = yf.download(target_symbol, start=start_str, end=end_str, progress=False)
        
# #         if df_driver.empty or df_target.empty:
# #             st.error("❌ Failed to download data. Please check your internet connection and try again.")
# #             return
        
# #         # Step 2: Process data with rolling windows
# #         status_text.text("🔄 Processing data with rolling windows...")
# #         progress_bar.progress(30)
        
# #         data = process_data_with_rolling_windows(df_driver, df_target, window_size, short_window, medium_window, long_window)
        
# #         if data is None:
# #             st.error("❌ Not enough data for analysis. Try a longer date range.")
# #             return
        
# #         # Step 3: Prepare features and target
# #         status_text.text("🔧 Preparing features...")
# #         progress_bar.progress(50)
        
# #         feature_cols = prepare_rolling_features(data)
# #         X, y, features_for_forecast, dates_clean = create_features_target(data, feature_cols, prediction_horizon)
        
# #         if X.shape[0] == 0:
# #             st.error("❌ No valid data for training after preprocessing.")
# #             return
        
# #         # Step 4: Train model and predict
# #         status_text.text("🤖 Training model...")
# #         progress_bar.progress(60)
        
# #         classification_results = train_and_predict(X, y, features_for_forecast, model_type, dates_clean)
        
# #         # Step 5: Monte Carlo Simulation
# #         status_text.text("🎲 Running Monte Carlo simulations...")
# #         progress_bar.progress(80)
        
# #         # Get current price and volatility for Monte Carlo
# #         current_price = data['target_close'].iloc[-1]
# #         volatility = data['target_vol_medium'].iloc[-1] if not np.isnan(data['target_vol_medium'].iloc[-1]) else data['target_ret_1'].std()
        
# #         # Run Monte Carlo simulation
# #         simulations = monte_carlo_simulation(current_price, volatility, prediction_horizon, num_simulations)
# #         prediction_intervals = calculate_prediction_intervals(simulations, confidence_level)
        
# #         # Combine results
# #         results = {
# #             **classification_results,
# #             'monte_carlo': {
# #                 'simulations': simulations,
# #                 'intervals': prediction_intervals,
# #                 'current_price': current_price,
# #                 'volatility': volatility
# #             }
# #         }
        
# #         # Step 6: Display results
# #         status_text.text("📊 Generating advanced results...")
# #         progress_bar.progress(90)
        
# #         display_advanced_results(data, results, driver_symbol, target_symbol, prediction_horizon)
        
# #         progress_bar.progress(100)
# #         status_text.text("✅ Advanced analysis complete!")
        
# #     except Exception as e:
# #         st.error(f"❌ An error occurred: {str(e)}")
# #         import traceback
# #         st.code(traceback.format_exc())

# # def process_data_with_rolling_windows(df_driver, df_target, main_window, short_window, medium_window, long_window):
# #     """Process data with multiple rolling windows"""
    
# #     # Align data
# #     data = pd.DataFrame(index=df_driver.index)
# #     data['driver_close'] = df_driver['Close']
# #     data['target_close'] = df_target['Close']
# #     data = data.dropna()
    
# #     if len(data) < main_window + 10:
# #         return None
    
# #     # Returns with different windows
# #     data['driver_ret_1'] = data['driver_close'].pct_change(1)
# #     data['driver_ret_short'] = data['driver_close'].pct_change(short_window)
# #     data['driver_ret_medium'] = data['driver_close'].pct_change(medium_window)
    
# #     # Rolling volatility
# #     data['driver_vol_short'] = data['driver_ret_1'].rolling(short_window).std()
# #     data['driver_vol_medium'] = data['driver_ret_1'].rolling(medium_window).std()
# #     data['driver_vol_long'] = data['driver_ret_1'].rolling(long_window).std()
    
# #     # Rolling averages
# #     data['driver_ma_short'] = data['driver_close'].rolling(short_window).mean()
# #     data['driver_ma_medium'] = data['driver_close'].rolling(medium_window).mean()
# #     data['driver_ma_long'] = data['driver_close'].rolling(long_window).mean()
    
# #     # Target features
# #     data['target_ret_1'] = data['target_close'].pct_change(1)
# #     data['target_ret_short'] = data['target_close'].pct_change(short_window)
# #     data['target_ret_medium'] = data['target_close'].pct_change(medium_window)
    
# #     data['target_vol_short'] = data['target_ret_1'].rolling(short_window).std()
# #     data['target_vol_medium'] = data['target_ret_1'].rolling(medium_window).std()
# #     data['target_vol_long'] = data['target_ret_1'].rolling(long_window).std()
    
# #     data['target_ma_short'] = data['target_close'].rolling(short_window).mean()
# #     data['target_ma_medium'] = data['target_close'].rolling(medium_window).mean()
# #     data['target_ma_long'] = data['target_close'].rolling(long_window).mean()
    
# #     # Cross-asset features
# #     data['rolling_corr_short'] = data['driver_ret_1'].rolling(short_window).corr(data['target_ret_1'])
# #     data['rolling_corr_medium'] = data['driver_ret_1'].rolling(medium_window).corr(data['target_ret_1'])
# #     data['rolling_beta'] = data['rolling_corr_medium'] * (data['target_vol_medium'] / data['driver_vol_medium'])
    
# #     data['price_ratio'] = data['target_close'] / data['driver_close']
# #     data['price_ratio_ma'] = data['price_ratio'].rolling(medium_window).mean()
# #     data['price_ratio_std'] = data['price_ratio'].rolling(medium_window).std()
    
# #     data['ret_spread'] = data['target_ret_1'] - data['driver_ret_1']
# #     data['ret_spread_ma'] = data['ret_spread'].rolling(medium_window).mean()
    
# #     # Momentum features
# #     data['driver_momentum'] = (data['driver_close'] - data['driver_ma_short']) / data['driver_vol_short']
# #     data['target_momentum'] = (data['target_close'] - data['target_ma_short']) / data['target_vol_short']
    
# #     data['driver_trend'] = (data['driver_ma_short'] - data['driver_ma_long']) / data['driver_vol_medium']
# #     data['target_trend'] = (data['target_ma_short'] - data['target_ma_long']) / data['target_vol_medium']
    
# #     # Use only recent data for performance
# #     data = data.tail(150).copy()
    
# #     return data

# # def prepare_rolling_features(data):
# #     """Prepare rolling feature column list"""
    
# #     return_features = [
# #         'driver_ret_1', 'driver_ret_short', 'driver_ret_medium',
# #         'target_ret_1', 'target_ret_short', 'target_ret_medium'
# #     ]
    
# #     volatility_features = [
# #         'driver_vol_short', 'driver_vol_medium', 'driver_vol_long',
# #         'target_vol_short', 'target_vol_medium', 'target_vol_long'
# #     ]
    
# #     moving_average_features = [
# #         'driver_ma_short', 'driver_ma_medium', 'driver_ma_long',
# #         'target_ma_short', 'target_ma_medium', 'target_ma_long'
# #     ]
    
# #     cross_asset_features = [
# #         'rolling_corr_short', 'rolling_corr_medium', 'rolling_beta',
# #         'price_ratio', 'price_ratio_ma', 'price_ratio_std',
# #         'ret_spread', 'ret_spread_ma'
# #     ]
    
# #     momentum_features = [
# #         'driver_momentum', 'target_momentum',
# #         'driver_trend', 'target_trend'
# #     ]
    
# #     all_features = (return_features + volatility_features + moving_average_features + 
# #                    cross_asset_features + momentum_features)
    
# #     feature_cols = [col for col in all_features if col in data.columns and not data[col].isna().all()]
    
# #     return feature_cols

# # def create_features_target(data, feature_cols, prediction_horizon):
# #     """Create features and target variable"""
    
# #     data['future_price'] = data['target_close'].shift(-prediction_horizon)
# #     data['target_up'] = (data['future_price'] > data['target_close']).astype(int)
    
# #     data_clean = data.dropna().copy()
    
# #     if len(data_clean) == 0:
# #         raise ValueError("No valid data after preprocessing")
    
# #     X = data_clean[feature_cols].values
# #     y = data_clean['target_up'].values
# #     dates_clean = data_clean.index
    
# #     features_for_forecast = data_clean.iloc[-1][feature_cols].values.reshape(1, -1)
    
# #     return X, y, features_for_forecast, dates_clean

# # def train_and_predict(X, y, features_for_forecast, model_type, dates):
# #     """Train model and make predictions"""
    
# #     min_samples = 20
# #     if len(X) < min_samples:
# #         raise ValueError(f"Not enough samples for training. Need at least {min_samples}, got {len(X)}")
    
# #     split_idx = max(1, int(len(X) * 0.7))
    
# #     if len(X) - split_idx < 5:
# #         split_idx = max(1, len(X) - 5)
    
# #     X_train, X_test = X[:split_idx], X[split_idx:]
# #     y_train, y_test = y[:split_idx], y[split_idx:]
# #     dates_test = dates[split_idx:]
    
# #     if len(X_test) == 0:
# #         raise ValueError("No test samples available.")
    
# #     if model_type == "Logistic Regression":
# #         classifier = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
# #     else:
# #         classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
# #     pipeline = Pipeline([
# #         ('scaler', StandardScaler()),
# #         ('classifier', classifier)
# #     ])
    
# #     pipeline.fit(X_train, y_train)
    
# #     y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
# #     y_pred = (y_pred_proba >= 0.5).astype(int)
    
# #     forecast_proba = pipeline.predict_proba(features_for_forecast)[0, 1]
    
# #     return {
# #         'pipeline': pipeline,
# #         'y_test': y_test,
# #         'y_pred': y_pred,
# #         'y_pred_proba': y_pred_proba,
# #         'forecast_proba': forecast_proba,
# #         'X_test': X_test,
# #         'dates_test': dates_test,
# #         'split_idx': split_idx
# #     }

# # def display_advanced_results(data, results, driver_symbol, target_symbol, prediction_horizon):
# #     """Display all results with Monte Carlo and prediction intervals"""
    
# #     current_price = data['target_close'].iloc[-1]
# #     mc_results = results['monte_carlo']
# #     intervals = mc_results['intervals']
    
# #     # Generate future dates for prediction horizon
# #     last_date = data.index[-1]
# #     future_dates = [last_date + timedelta(days=i) for i in range(1, prediction_horizon + 1)]
# #     prediction_date = future_dates[-1]
    
# #     # Display main prediction cards
# #     st.header("🎯 Advanced Prediction Results")
    
# #     col1, col2, col3, col4 = st.columns(4)
    
# #     with col1:
# #         st.metric(f"{target_symbol} Current", f"${current_price:.2f}")
    
# #     with col2:
# #         direction = "BULLISH" if results['forecast_proba'] >= 0.5 else "BEARISH"
# #         st.metric("Direction", f"{direction}")
    
# #     with col3:
# #         st.metric("Probability", f"{results['forecast_proba']:.3f}")
    
# #     with col4:
# #         change_pct = ((intervals['median_price'] - current_price) / current_price) * 100
# #         st.metric("Expected Change", f"{change_pct:+.1f}%")
    
# #     # Monte Carlo Prediction Intervals
# #     st.subheader("📊 Monte Carlo Prediction Intervals")
    
# #     col1, col2, col3, col4 = st.columns(4)
    
# #     with col1:
# #         st.metric(
# #             "Lower Bound", 
# #             f"${intervals['lower_bound']:.2f}",
# #             f"{((intervals['lower_bound'] - current_price) / current_price * 100):+.1f}%"
# #         )
    
# #     with col2:
# #         st.metric(
# #             "Median Price", 
# #             f"${intervals['median_price']:.2f}",
# #             f"{((intervals['median_price'] - current_price) / current_price * 100):+.1f}%"
# #         )
    
# #     with col3:
# #         st.metric(
# #             "Upper Bound", 
# #             f"${intervals['upper_bound']:.2f}",
# #             f"{((intervals['upper_bound'] - current_price) / current_price * 100):+.1f}%"
# #         )
    
# #     with col4:
# #         st.metric(
# #             "Prediction Date", 
# #             prediction_date.strftime('%Y-%m-%d')
# #         )
    
# #     # Enhanced Prediction Timeline with Future Horizon
# #     st.subheader("📈 Advanced Prediction Timeline")
    
# #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
# #     # Plot 1: Classification probabilities with future prediction
# #     dates_test = results['dates_test']
# #     y_pred_proba = results['y_pred_proba']
    
# #     # Extend timeline to include future prediction
# #     all_dates = list(dates_test) + [prediction_date]
# #     all_probabilities = list(y_pred_proba) + [results['forecast_proba']]
    
# #     ax1.plot(all_dates, all_probabilities, label='Prediction Probability', color='blue', linewidth=2, marker='o')
# #     ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Boundary')
    
# #     # Highlight future prediction
# #     ax1.plot(prediction_date, results['forecast_proba'], 'ro', markersize=10, label=f'Future Prediction')
# #     ax1.axvline(x=prediction_date, color='green', linestyle=':', alpha=0.7, label='Prediction Date')
    
# #     # Add confidence interval for future prediction
# #     ax1.fill_between([prediction_date, prediction_date], 
# #                      results['forecast_proba'] - 0.1, 
# #                      results['forecast_proba'] + 0.1, 
# #                      alpha=0.2, color='red', label='Prediction Uncertainty')
    
# #     ax1.set_xlabel('Date')
# #     ax1.set_ylabel('Probability of Price Increase')
# #     ax1.set_title(f'{target_symbol} Direction Prediction Timeline (Including Future Horizon)')
# #     ax1.legend()
# #     ax1.grid(True, alpha=0.3)
# #     plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
# #     # Plot 2: Monte Carlo Simulation Results
# #     final_prices = intervals['final_prices']
    
# #     # Plot histogram of final prices
# #     ax2.hist(final_prices, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
# #     ax2.axvline(intervals['lower_bound'], color='red', linestyle='--', linewidth=2, label=f'Lower Bound (${intervals["lower_bound"]:.2f})')
# #     ax2.axvline(intervals['median_price'], color='green', linestyle='-', linewidth=2, label=f'Median (${intervals["median_price"]:.2f})')
# #     ax2.axvline(intervals['upper_bound'], color='red', linestyle='--', linewidth=2, label=f'Upper Bound (${intervals["upper_bound"]:.2f})')
# #     ax2.axvline(current_price, color='blue', linestyle='-', linewidth=2, label=f'Current Price (${current_price:.2f})')
    
# #     ax2.set_xlabel('Predicted Price')
# #     ax2.set_ylabel('Probability Density')
# #     ax2.set_title(f'Monte Carlo Simulation: {target_symbol} Price Distribution after {prediction_horizon} days\n'
# #                  f'{confidence_level}% Confidence Interval: [${intervals["lower_bound"]:.2f}, ${intervals["upper_bound"]:.2f}]')
# #     ax2.legend()
# #     ax2.grid(True, alpha=0.3)
    
# #     plt.tight_layout()
# #     st.pyplot(fig)
    
# #     # Monte Carlo Path Visualization
# #     st.subheader("🎲 Monte Carlo Simulation Paths")
    
# #     fig2, ax3 = plt.subplots(figsize=(12, 6))
    
# #     # Plot a subset of simulation paths for clarity
# #     num_paths_to_plot = min(100, num_simulations)
# #     simulation_dates = [last_date + timedelta(days=i) for i in range(prediction_horizon)]
    
# #     for i in range(num_paths_to_plot):
# #         ax3.plot(simulation_dates, mc_results['simulations'][:, i], alpha=0.1, color='blue')
    
# #     # Plot confidence intervals
# #     percentiles = np.percentile(mc_results['simulations'], [5, 25, 50, 75, 95], axis=1)
# #     ax3.plot(simulation_dates, percentiles[2], 'g-', linewidth=3, label='Median Path')
# #     ax3.fill_between(simulation_dates, percentiles[0], percentiles[4], alpha=0.3, color='red', label='90% Confidence Interval')
# #     ax3.fill_between(simulation_dates, percentiles[1], percentiles[3], alpha=0.3, color='orange', label='50% Confidence Interval')
    
# #     ax3.axhline(y=current_price, color='black', linestyle='--', linewidth=2, label=f'Current Price (${current_price:.2f})')
# #     ax3.set_xlabel('Date')
# #     ax3.set_ylabel('Price')
# #     ax3.set_title(f'Monte Carlo Simulation Paths for {target_symbol} ({num_simulations} simulations)')
# #     ax3.legend()
# #     ax3.grid(True, alpha=0.3)
# #     plt.xticks(rotation=45)
# #     plt.tight_layout()
    
# #     st.pyplot(fig2)
    
# #     # Risk Analysis
# #     st.subheader("📉 Risk Analysis")
    
# #     col1, col2, col3, col4 = st.columns(4)
    
# #     with col1:
# #         probability_down = np.mean(final_prices < current_price) * 100
# #         st.metric("Probability of Decline", f"{probability_down:.1f}%")
    
# #     with col2:
# #         var_95 = np.percentile(final_prices, 5)
# #         var_loss = ((var_95 - current_price) / current_price) * 100
# #         st.metric("VaR (95%)", f"{var_loss:.1f}%")
    
# #     with col3:
# #         expected_shortfall = np.mean(final_prices[final_prices <= var_95])
# #         es_loss = ((expected_shortfall - current_price) / current_price) * 100
# #         st.metric("Expected Shortfall", f"{es_loss:.1f}%")
    
# #     with col4:
# #         best_case = np.max(final_prices)
# #         best_gain = ((best_case - current_price) / current_price) * 100
# #         st.metric("Best Case", f"{best_gain:+.1f}%")
    
# #     # Detailed Report
# #     with st.expander("📋 Detailed Analysis Report"):
# #         st.write("### Classification Performance")
# #         accuracy = accuracy_score(results['y_test'], results['y_pred'])
# #         st.metric("Model Accuracy", f"{accuracy:.3f}")
        
# #         st.write("### Monte Carlo Statistics")
# #         stats_data = {
# #             'Statistic': ['Current Price', 'Mean Prediction', 'Median Prediction', 'Std Deviation', 
# #                          'Skewness', 'Kurtosis', 'Confidence Interval'],
# #             'Value': [
# #                 f"${current_price:.2f}",
# #                 f"${intervals['mean_price']:.2f}",
# #                 f"${intervals['median_price']:.2f}",
# #                 f"${np.std(final_prices):.2f}",
# #                 f"{pd.Series(final_prices).skew():.3f}",
# #                 f"{pd.Series(final_prices).kurtosis():.3f}",
# #                 f"[${intervals['lower_bound']:.2f}, ${intervals['upper_bound']:.2f}]"
# #             ]
# #         }
# #         st.table(pd.DataFrame(stats_data))

# # # Run the app
# # if __name__ == "__main__":
# #     main()




