import yfinance as yf

# Fetch historical data for Reliance
ticker = 'RELIANCE.NS'  # Use the correct ticker for Indian stocks on NSE
data = yf.Ticker(ticker)

# Get historical data from 2015 onwards
historical_data = data.history(start='2015-01-01', end='2024-12-27')  # Adjust 'end' date if required

# Extract relevant columns
reliance_data = historical_data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Display the data
print(reliance_data)

# Optionally save to a CSV file
reliance_data.to_csv('Reliance_Historical_Data_2015.csv'