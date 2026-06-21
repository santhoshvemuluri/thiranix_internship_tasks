import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score

def main():
    print("--- Predictive Analytics Using Historical Data ---\n")

    # ==========================================
    # 1. Generate Historical Dataset
    # ==========================================
    # Simulating 365 days of historical data with an upward trend and some noise
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', periods=365)
    trend = np.linspace(100, 250, 365) # Simulating a growth trend
    noise = np.random.normal(0, 15, 365)
    values = trend + noise

    df = pd.DataFrame({'Date': dates, 'Value': values})

    # Introducing synthetic missing data to demonstrate the cleaning requirement
    df.loc[20:25, 'Value'] = np.nan

    # ==========================================
    # 2. Clean and Preprocess Historical Datasets
    # ==========================================
    print("Cleaning and preprocessing data...")
    # Fill missing values using forward fill (common in time-series)
    df['Value'] = df['Value'].ffill()

    # Feature Engineering: Convert Date to an integer feature for the regression model
    df['Days_Since_Start'] = (df['Date'] - df['Date'].min()).dt.days

    # Define features (X) and target variable (y)
    X = df[['Days_Since_Start']]
    y = df['Value']

    # Chronological train-test split (80% train, 20% test for forecasting)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # ==========================================
    # 3. Use Regression Models for Prediction
    # ==========================================
    print("Training the Huber Regression model for greater robustness...")
    model = HuberRegressor()
    model.fit(X_train, y_train)

    # Forecast the trend on the test set
    y_pred = model.predict(X_test)

    # ==========================================
    # 4. Evaluate Model Accuracy
    # ==========================================
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2 Score): {r2:.2f}")

    # ==========================================
    # 5. Visualize Predictions
    # ==========================================
    print("\nGenerating visualization...")
    plt.figure(figsize=(12, 6))

    # Plot actual historical training data
    plt.plot(df['Date'].iloc[X_train.index], y_train, label='Training Data (Historical)', color='blue', alpha=0.6)
    
    # Plot actual historical testing data
    plt.plot(df['Date'].iloc[X_test.index], y_test, label='Actual Future Data (Testing)', color='green', alpha=0.6)

    # Plot the model's predictions
    plt.plot(df['Date'].iloc[X_test.index], y_pred, label='Predicted Trend', color='red', linewidth=2.5, linestyle='--')

    plt.title('Predictive Analytics: Forecasting Future Trends')
    plt.xlabel('Date')
    plt.ylabel('Value / Metric')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot as an image file and display it
    plt.savefig('prediction_chart.png')
    plt.show()

if __name__ == "__main__":
    main()
