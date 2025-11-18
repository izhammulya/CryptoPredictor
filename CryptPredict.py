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
except ImportError as e:
    st.error(f"Missing required packages: {e}")
    st.stop()

# Streamlit app configuration
st.set_page_config(
    page_title="Crypto Horizon Predictor",
    page_icon="🚀",
    layout="wide"
)

# App title and description
st.title("🚀 Crypto Horizon Predictor with Monte Carlo Paths")
st.markdown("""
**Clear Visualizations:**
- **Price Prediction Horizon** with future dates
- **Monte Carlo Simulation Paths** visible on graph
- **Confidence Intervals** for price targets
- **Multiple Timeframes** for predictions
""")

# Sidebar configuration
st.sidebar.header("🎯 Prediction Configuration")

# Date range input
st.sidebar.subheader("📅 Date Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
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
    "SOL-USD": "Solana"
}

target_symbol = st.sidebar.selectbox(
    "Target Cryptocurrency", 
    options=list(crypto_options.keys()),
    index=3
)

# Prediction Settings
st.sidebar.subheader("🎯 Prediction Settings")
prediction_horizon = st.sidebar.slider("Prediction Horizon (days)", 1, 30, 7)
show_multiple_horizons = st.sidebar.checkbox("Show Multiple Horizons", value=True)

if show_multiple_horizons:
    horizons = st.sidebar.multiselect(
        "Additional Horizons to Show",
        [1, 3, 5, 10, 14, 21, 30],
        default=[1, 3, 10]
    )
else:
    horizons = [prediction_horizon]

# Monte Carlo Settings
st.sidebar.subheader("🎲 Monte Carlo Simulation")
num_simulations = st.sidebar.slider("Number of Paths", 50, 1000, 200)
num_paths_to_show = st.sidebar.slider("Paths to Visualize", 10, 100, 50)
confidence_level = st.sidebar.slider("Confidence Level", 80, 99, 90)

show_paths = st.sidebar.checkbox("Show Monte Carlo Paths", value=True)
show_confidence_bands = st.sidebar.checkbox("Show Confidence Bands", value=True)

# Enhanced Monte Carlo Simulation
def enhanced_monte_carlo(current_price, historical_returns, horizon, n_simulations):
    """Enhanced Monte Carlo with better price simulation"""
    
    # Use historical return characteristics
    mean_return = np.mean(historical_returns)
    std_return = np.std(historical_returns)
    
    # Generate random returns based on historical characteristics
    random_returns = np.random.normal(mean_return, std_return, (horizon, n_simulations))
    
    # Calculate price paths
    price_paths = np.zeros((horizon + 1, n_simulations))
    price_paths[0] = current_price
    
    for day in range(1, horizon + 1):
        price_paths[day] = price_paths[day-1] * (1 + random_returns[day-1])
    
    return price_paths

def calculate_horizon_predictions(price_paths, confidence_level):
    """Calculate predictions for each horizon"""
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

# Main function
def main():
    if st.sidebar.button("🎯 Generate Horizon Predictions", type="primary"):
        run_horizon_prediction()

def run_horizon_prediction():
    """Main prediction function with clear horizon visualization"""
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Download data
        status_text.text("📥 Downloading price data...")
        progress_bar.progress(20)
        
        df = yf.download(target_symbol, start=start_str, end=end_str, progress=False)
        
        if df.empty:
            st.error("❌ Failed to download data.")
            return
        
        # Step 2: Prepare data
        status_text.text("🔄 Preparing data for simulation...")
        progress_bar.progress(40)
        
        data = df[['Close']].copy()
        data['Returns'] = data['Close'].pct_change()
        data = data.dropna()
        
        current_price = data['Close'].iloc[-1]
        historical_returns = data['Returns'].tail(100).values  # Use recent returns
        
        # Step 3: Run Monte Carlo for maximum horizon
        status_text.text("🎲 Running Monte Carlo simulations...")
        progress_bar.progress(60)
        
        max_horizon = max(horizons) if horizons else prediction_horizon
        price_paths = enhanced_monte_carlo(current_price, historical_returns, max_horizon, num_simulations)
        
        # Calculate predictions for all horizons
        horizon_predictions = calculate_horizon_predictions(price_paths, confidence_level)
        
        # Step 4: Generate future dates
        last_date = data.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(max_horizon + 1)]
        
        status_text.text("📊 Creating visualizations...")
        progress_bar.progress(80)
        
        # Display results
        display_horizon_predictions(data, price_paths, horizon_predictions, future_dates, 
                                  target_symbol, current_price, horizons)
        
        progress_bar.progress(100)
        status_text.text("✅ Horizon predictions complete!")
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

def display_horizon_predictions(data, price_paths, horizon_predictions, future_dates, 
                              symbol, current_price, horizons):
    """Display horizon predictions with clear Monte Carlo visualization"""
    
    # Main results header
    st.header(f"🎯 {symbol} Price Horizon Predictions")
    
    # Current price and key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    with col2:
        main_horizon = horizons[0] if horizons else 7
        main_pred = horizon_predictions[main_horizon]
        expected_change = ((main_pred['median_price'] - current_price) / current_price) * 100
        st.metric(f"{main_horizon}-Day Expected", f"${main_pred['median_price']:.2f}", 
                 f"{expected_change:+.1f}%")
    with col3:
        prob_up = np.mean(main_pred['prices'] > current_price) * 100
        st.metric("Probability Up", f"{prob_up:.1f}%")
    with col4:
        st.metric("Confidence Level", f"{confidence_level}%")
    
    # MAIN VISUALIZATION: Price History + Monte Carlo Future
    st.subheader("📈 Price History with Monte Carlo Future Paths")
    
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot historical prices (last 60 days for clarity)
    historical_to_show = data.tail(60)
    ax1.plot(historical_to_show.index, historical_to_show['Close'], 
             'b-', linewidth=3, label='Historical Price', alpha=0.8)
    
    # Plot current price as reference
    ax1.axhline(y=current_price, color='black', linestyle='--', alpha=0.7, 
                label=f'Current Price: ${current_price:.2f}')
    
    # Plot Monte Carlo paths
    if show_paths:
        paths_to_show = min(num_paths_to_show, num_simulations)
        for i in range(paths_to_show):
            ax1.plot(future_dates, price_paths[:, i], 'gray', alpha=0.1, linewidth=0.5)
    
    # Plot confidence bands
    if show_confidence_bands:
        percentiles = np.percentile(price_paths, [5, 25, 50, 75, 95], axis=1)
        
        # 90% confidence interval
        ax1.fill_between(future_dates, percentiles[0], percentiles[4], 
                        alpha=0.2, color='red', label='90% Confidence Interval')
        
        # 50% confidence interval
        ax1.fill_between(future_dates, percentiles[1], percentiles[3], 
                        alpha=0.3, color='orange', label='50% Confidence Interval')
        
        # Median path
        ax1.plot(future_dates, percentiles[2], 'r-', linewidth=3, 
                label='Median Prediction Path', alpha=0.8)
    
    # Add vertical line separating history from future
    ax1.axvline(x=future_dates[0], color='green', linestyle=':', alpha=0.7, 
                label='Prediction Start')
    
    # Add horizon markers
    colors = ['purple', 'brown', 'teal', 'magenta', 'navy']
    for i, horizon in enumerate(horizons):
        if horizon <= len(future_dates) - 1:
            horizon_date = future_dates[horizon]
            pred = horizon_predictions[horizon]
            
            # Vertical line at horizon
            ax1.axvline(x=horizon_date, color=colors[i % len(colors)], linestyle='--', 
                       alpha=0.6, label=f'{horizon}-Day Horizon')
            
            # Price target markers
            ax1.plot(horizon_date, pred['median_price'], 'o', color=colors[i % len(colors)], 
                    markersize=8, markeredgecolor='white', markeredgewidth=1)
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'{symbol} Price Prediction with Monte Carlo Simulation\n'
                 f'Showing {num_simulations} simulations with {confidence_level}% confidence intervals')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig1)
    
    # HORIZON PREDICTION TABLE
    st.subheader("📊 Horizon Prediction Summary")
    
    prediction_data = []
    for horizon in sorted(horizons):
        if horizon in horizon_predictions:
            pred = horizon_predictions[horizon]
            horizon_date = future_dates[horizon]
            
            price_change = ((pred['median_price'] - current_price) / current_price) * 100
            lower_change = ((pred['lower_bound'] - current_price) / current_price) * 100
            upper_change = ((pred['upper_bound'] - current_price) / current_price) * 100
            
            prob_up = np.mean(pred['prices'] > current_price) * 100
            
            prediction_data.append({
                'Horizon (Days)': horizon,
                'Prediction Date': horizon_date.strftime('%Y-%m-%d'),
                'Median Price': f"${pred['median_price']:.2f}",
                'Price Change': f"{price_change:+.1f}%",
                'Confidence Interval': f"${pred['lower_bound']:.2f} - ${pred['upper_bound']:.2f}",
                'Range Change': f"{lower_change:+.1f}% to {upper_change:+.1f}%",
                'Probability Up': f"{prob_up:.1f}%"
            })
    
    prediction_df = pd.DataFrame(prediction_data)
    st.dataframe(prediction_df, use_container_width=True)
    
    # MONTE CARLO DISTRIBUTION BY HORIZON
    st.subheader("📈 Price Distribution by Horizon")
    
    fig2, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    horizons_to_plot = horizons[:4]  # Plot first 4 horizons
    
    for idx, horizon in enumerate(horizons_to_plot):
        if idx < len(axes) and horizon in horizon_predictions:
            pred = horizon_predictions[horizon]
            prices = pred['prices']
            
            axes[idx].hist(prices, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
            axes[idx].axvline(pred['lower_bound'], color='red', linestyle='--', 
                            label=f'Lower: ${pred["lower_bound"]:.2f}')
            axes[idx].axvline(pred['median_price'], color='green', linestyle='-', 
                            label=f'Median: ${pred["median_price"]:.2f}')
            axes[idx].axvline(pred['upper_bound'], color='red', linestyle='--', 
                            label=f'Upper: ${pred["upper_bound"]:.2f}')
            axes[idx].axvline(current_price, color='blue', linestyle='-', 
                            label=f'Current: ${current_price:.2f}')
            
            axes[idx].set_xlabel('Price ($)')
            axes[idx].set_ylabel('Probability Density')
            axes[idx].set_title(f'{horizon}-Day Horizon Distribution\n'
                              f'90% CI: [${pred["lower_bound"]:.2f}, ${pred["upper_bound"]:.2f}]')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(horizons_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig2)
    
    # RISK ANALYSIS
    st.subheader("⚡ Risk Analysis by Horizon")
    
    risk_data = []
    for horizon in sorted(horizons):
        if horizon in horizon_predictions:
            pred = horizon_predictions[horizon]
            prices = pred['prices']
            
            # Risk metrics
            prob_decline = np.mean(prices < current_price) * 100
            var_95 = np.percentile(prices, 5)  # Value at Risk 95%
            var_loss = ((var_95 - current_price) / current_price) * 100
            best_case = np.max(prices)
            best_gain = ((best_case - current_price) / current_price) * 100
            worst_case = np.min(prices)
            worst_loss = ((worst_case - current_price) / current_price) * 100
            
            risk_data.append({
                'Horizon': f'{horizon} days',
                'Prob. Decline': f'{prob_decline:.1f}%',
                'VaR (95%)': f'{var_loss:.1f}%',
                'Worst Case': f'{worst_loss:.1f}%',
                'Best Case': f'{best_gain:.1f}%',
                'Expected Return': f"{((pred['median_price'] - current_price) / current_price * 100):+.1f}%"
            })
    
    risk_df = pd.DataFrame(risk_data)
    st.dataframe(risk_df, use_container_width=True)
    
    # DOWNLOAD PREDICTION DATA
    st.subheader("💾 Download Prediction Data")
    
    # Create downloadable dataframe
    download_data = []
    for horizon in sorted(horizons):
        if horizon in horizon_predictions:
            pred = horizon_predictions[horizon]
            download_data.append({
                'horizon_days': horizon,
                'prediction_date': future_dates[horizon].strftime('%Y-%m-%d'),
                'median_price': pred['median_price'],
                'lower_bound': pred['lower_bound'],
                'upper_bound': pred['upper_bound'],
                'probability_up': np.mean(pred['prices'] > current_price) * 100
            })
    
    download_df = pd.DataFrame(download_data)
    csv = download_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Horizon Predictions CSV",
        data=csv,
        file_name=f"{symbol}_horizon_predictions.csv",
        mime="text/csv"
    )

# Run the app
if __name__ == "__main__":
    main()

# Add explanation
st.sidebar.markdown("---")
st.sidebar.info("""
**🎯 What You'll See:**
- **Monte Carlo Paths**: Gray lines showing possible price trajectories
- **Confidence Bands**: Colored areas showing prediction uncertainty
- **Horizon Markers**: Vertical lines at prediction dates
- **Price Targets**: Dots showing median predictions at each horizon
- **Distribution Charts**: Probability distributions for each timeframe
""")


# # V2 Sistem
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


