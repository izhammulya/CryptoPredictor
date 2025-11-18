import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    page_title="Crypto Predictor",
    page_icon="🚀",
    layout="wide"
)

# App title and description
st.title("🚀 Cryptocurrency Direction Predictor")
st.markdown("Predict cryptocurrency price movements using machine learning and oscillator features")

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

# Model parameters
st.sidebar.subheader("⚙️ Model Parameters")
prediction_horizon = st.sidebar.slider(
    "Prediction Horizon (days)",
    min_value=1,
    max_value=7,
    value=3,
    help="Number of days ahead to predict"
)

window = st.sidebar.slider(
    "Rolling Window Size",
    min_value=15,
    max_value=45,
    value=30,
    help="Window size for calculating rolling features"
)

use_oscillator = st.sidebar.checkbox(
    "Use Oscillator Features",
    value=True,
    help="Include physics-inspired oscillator features"
)

model_type = st.sidebar.selectbox(
    "Model Type",
    ["Logistic Regression", "Random Forest"],
    index=0
)

# Main function
def main():
    if st.sidebar.button("🎯 Run Prediction", type="primary"):
        run_prediction()

def run_prediction():
    """Main prediction function"""
    
    # Convert dates to string format
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Download data
        status_text.text("📥 Downloading cryptocurrency data...")
        progress_bar.progress(10)
        
        df_driver = yf.download(driver_symbol, start=start_str, end=end_str, progress=False)
        df_target = yf.download(target_symbol, start=start_str, end=end_str, progress=False)
        
        if df_driver.empty or df_target.empty:
            st.error("❌ Failed to download data. Please check your internet connection and try again.")
            return
        
        # Step 2: Process data
        status_text.text("🔄 Processing data...")
        progress_bar.progress(30)
        
        data = process_data(df_driver, df_target, window, use_oscillator)
        
        if data is None:
            st.error("❌ Not enough data for analysis. Try a longer date range.")
            return
        
        # Step 3: Prepare features and target
        status_text.text("🔧 Preparing features...")
        progress_bar.progress(50)
        
        feature_cols = prepare_features(data, use_oscillator)
        X, y, features_for_forecast, dates_clean = create_features_target(data, feature_cols, prediction_horizon)
        
        if X.shape[0] == 0:
            st.error("❌ No valid data for training after preprocessing.")
            return
        
        # Step 4: Train model and predict
        status_text.text("🤖 Training model...")
        progress_bar.progress(70)
        
        results = train_and_predict(X, y, features_for_forecast, model_type, dates_clean)
        
        # Step 5: Display results
        status_text.text("📊 Generating results...")
        progress_bar.progress(90)
        
        display_results(data, results, driver_symbol, target_symbol, prediction_horizon)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("💡 Try adjusting the parameters or using a different date range.")

def process_data(df_driver, df_target, window, use_oscillator):
    """Process and align cryptocurrency data"""
    
    # Align data
    data = pd.DataFrame(index=df_driver.index)
    data['driver_close'] = df_driver['Close']
    data['target_close'] = df_target['Close']
    data = data.dropna()
    
    if len(data) < window + 10:  # Need minimum data
        return None
    
    # Basic price features
    data['driver_ret_1'] = data['driver_close'].pct_change(1)
    data['driver_ret_3'] = data['driver_close'].pct_change(3)
    data['driver_vol_14'] = data['driver_close'].pct_change().rolling(14).std()
    
    data['target_ret_1'] = data['target_close'].pct_change(1)
    data['target_ret_3'] = data['target_close'].pct_change(3)
    data['target_vol_14'] = data['target_close'].pct_change().rolling(14).std()
    
    # Cross-asset features
    data['price_ratio'] = data['target_close'] / data['driver_close']
    data['ret_spread'] = data['target_ret_1'] - data['driver_ret_1']
    
    # Use only recent data for performance
    data = data.tail(100).copy()  # Increased from 60 to 100 for more training data
    
    # Oscillator features (simplified)
    if use_oscillator and len(data) >= window:
        data = add_oscillator_features(data, window)
    
    return data

def add_oscillator_features(data, window):
    """Add simplified oscillator features"""
    try:
        # Simplified oscillator-like features
        data['momentum'] = data['target_close'] / data['target_close'].rolling(window).mean() - 1
        data['driver_influence'] = data['driver_ret_1'].rolling(window).corr(data['target_ret_1'])
        data['volatility_ratio'] = data['target_vol_14'] / data['driver_vol_14']
        
    except Exception as e:
        st.warning(f"Oscillator features simplified due to: {e}")
    
    return data

def prepare_features(data, use_oscillator):
    """Prepare feature column list"""
    feature_cols = [
        'driver_ret_1', 'driver_ret_3', 'driver_vol_14',
        'target_ret_1', 'target_ret_3', 'target_vol_14',
        'price_ratio', 'ret_spread'
    ]
    
    if use_oscillator:
        oscillator_features = ['momentum', 'driver_influence', 'volatility_ratio']
        # Only add oscillator features that exist in data
        for feature in oscillator_features:
            if feature in data.columns and not data[feature].isna().all():
                feature_cols.append(feature)
    
    return feature_cols

def create_features_target(data, feature_cols, prediction_horizon):
    """Create features and target variable"""
    
    # Create target (price direction)
    data['future_price'] = data['target_close'].shift(-prediction_horizon)
    data['target_up'] = (data['future_price'] > data['target_close']).astype(int)
    
    # Remove rows with missing values
    data_clean = data.dropna().copy()
    
    if len(data_clean) == 0:
        raise ValueError("No valid data after preprocessing")
    
    # Prepare features
    X = data_clean[feature_cols].values
    y = data_clean['target_up'].values
    dates_clean = data_clean.index
    
    # Features for latest prediction
    features_for_forecast = data_clean.iloc[-1][feature_cols].values.reshape(1, -1)
    
    return X, y, features_for_forecast, dates_clean

def train_and_predict(X, y, features_for_forecast, model_type, dates):
    """Train model and make predictions"""
    
    # Ensure we have enough data for splitting
    min_samples = 10
    if len(X) < min_samples:
        raise ValueError(f"Not enough samples for training. Need at least {min_samples}, got {len(X)}")
    
    # Train-test split (time-based)
    split_idx = max(1, int(len(X) * 0.7))  # Ensure at least 1 sample in train set
    
    # Adjust split if test set would be too small
    if len(X) - split_idx < 5:
        split_idx = max(1, len(X) - 5)  # Ensure at least 5 test samples
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates[split_idx:]
    
    # Check if we have enough test samples
    if len(X_test) == 0:
        raise ValueError("No test samples available. Try using more data.")
    
    # Create pipeline
    if model_type == "Logistic Regression":
        classifier = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=50,  # Reduced for faster training
            random_state=42,
            class_weight='balanced'
        )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Forecast
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

def display_results(data, results, driver_symbol, target_symbol, prediction_horizon):
    """Display all results"""
    
    # Current prices
    current_driver_price = data['driver_close'].iloc[-1]
    current_target_price = data['target_close'].iloc[-1]
    
    # Prediction result
    forecast_proba = results['forecast_proba']
    prediction_label = "BULLISH 📈" if forecast_proba >= 0.5 else "BEARISH 📉"
    confidence = "High" if abs(forecast_proba - 0.5) > 0.2 else "Medium" if abs(forecast_proba - 0.5) > 0.1 else "Low"
    
    # Display main prediction
    st.header("🎯 Prediction Result")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(f"{target_symbol} Current Price", f"${current_target_price:.2f}")
    
    with col2:
        st.metric("Prediction", prediction_label)
    
    with col3:
        st.metric("Confidence", confidence)
    
    # Prediction details
    st.subheader("Prediction Details")
    st.write(f"**Probability of price increase in {prediction_horizon} days:** {forecast_proba:.3f}")
    
    if forecast_proba >= 0.5:
        st.success(f"🟢 **BULLISH SIGNAL** - {target_symbol} is predicted to increase with {forecast_proba:.1%} probability")
    else:
        st.error(f"🔴 **BEARISH SIGNAL** - {target_symbol} is predicted to decrease with {1-forecast_proba:.1%} probability")
    
    # Model performance
    st.subheader("📊 Model Performance")
    accuracy = accuracy_score(results['y_test'], results['y_pred'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    with col2:
        st.metric("Training Samples", results['split_idx'])
    with col3:
        st.metric("Test Samples", len(results['y_test']))
    
    # Plot results - FIXED DATA ALIGNMENT
    st.subheader("📈 Prediction Timeline")
    
    # Ensure dates and probabilities have the same length
    dates_test = results['dates_test']
    y_pred_proba = results['y_pred_proba']
    y_test = results['y_test']
    
    # Debug information
    st.write(f"Debug: Dates length: {len(dates_test)}, Probabilities length: {len(y_pred_proba)}")
    
    if len(dates_test) != len(y_pred_proba):
        st.warning(f"⚠️ Data alignment issue: {len(dates_test)} dates vs {len(y_pred_proba)} probabilities")
        # Use indices for plotting if dates don't match
        fig, ax = plt.subplots(figsize=(12, 6))
        x_values = range(len(y_pred_proba))
        ax.plot(x_values, y_pred_proba, label='Predicted Probability', color='blue', linewidth=2)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Boundary')
        
        # Highlight actual outcomes
        actual_up = y_test == 1
        actual_down = y_test == 0
        
        if any(actual_up):
            ax.scatter(np.array(x_values)[actual_up], y_pred_proba[actual_up], 
                      color='green', alpha=0.6, label='Actual Up', s=50)
        if any(actual_down):
            ax.scatter(np.array(x_values)[actual_down], y_pred_proba[actual_down], 
                      color='red', alpha=0.6, label='Actual Down', s=50)
        
        ax.set_xlabel('Test Sample Index')
        ax.set_ylabel('Probability of Price Increase')
        ax.set_title(f'{target_symbol} Price Direction Prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    else:
        # Normal plotting when lengths match
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates_test, y_pred_proba, label='Predicted Probability', color='blue', linewidth=2)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Boundary')
        
        # Highlight actual outcomes
        actual_up = y_test == 1
        actual_down = y_test == 0
        
        if any(actual_up):
            ax.scatter(dates_test[actual_up], y_pred_proba[actual_up], 
                      color='green', alpha=0.6, label='Actual Up', s=50)
        if any(actual_down):
            ax.scatter(dates_test[actual_down], y_pred_proba[actual_down], 
                      color='red', alpha=0.6, label='Actual Down', s=50)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Probability of Price Increase')
        ax.set_title(f'{target_symbol} Price Direction Prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Additional information
    with st.expander("📋 Detailed Report"):
        st.write("### Classification Report")
        st.text(classification_report(results['y_test'], results['y_pred']))
        
        st.write("### Confusion Matrix")
        cm = confusion_matrix(results['y_test'], results['y_pred'])
        cm_df = pd.DataFrame(cm, 
                           index=['Actual Down', 'Actual Up'],
                           columns=['Predicted Down', 'Predicted Up'])
        st.dataframe(cm_df)
        
        # Data summary
        st.write("### Data Summary")
        st.write(f"Total samples: {len(data)}")
        st.write(f"Training samples: {results['split_idx']}")
        st.write(f"Test samples: {len(results['y_test'])}")
        st.write(f"Features used: {len(results['pipeline'].named_steps['classifier'].coef_[0]) if hasattr(results['pipeline'].named_steps['classifier'], 'coef_') else 'N/A'}")

# Run the app
if __name__ == "__main__":
    main()

