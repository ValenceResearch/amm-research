import pandas as pd
import numpy as np
import os

# Use the current directory where the Python file is located
csv_directory = './datasets'

# Initialize a list to store the results
results = []

# Loop through each CSV file in the directory
for filename in os.listdir(csv_directory):
    if filename.endswith('.csv'):
        # Construct the full path to the file
        filepath = os.path.join(csv_directory, filename)
        
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # Calculate the spread for each row
        df['spread'] = df['ask_price'] - df['bid_price']
        
        # Calculate the mid-price for each row
        df['mid_price'] = (df['ask_price'] + df['bid_price']) / 2
        
        # Calculate the spread as a percentage of the mid-price (basis point) and round to 4 decimal places
        df['basis_point'] = ((df['spread'] / df['mid_price']) * 10000).round(4)
        
        # Calculate ask volume and bid volume
        df['ask_volume'] = df['ask_amount'] * df['ask_price']
        df['bid_volume'] = df['bid_amount'] * df['bid_price']
        
        # Calculate the average bid_volume and ask_volume
        average_ask_volume = df['ask_volume'].mean()
        average_bid_volume = df['bid_volume'].mean()
        
        # Calculate the standard deviation of basis points and round to 4 decimal places
        basis_point_std_dev = df['basis_point'].std().round(4)
        
        # Percentiles for basis points, rounded to 4 decimal places
        percentiles = [1, 10, 25, 50, 75, 90, 99]
        basis_point_percentiles = np.percentile(df['basis_point'], percentiles).round(8)
        
        # Extract the exchange name, date, and symbol from the filename and DataFrame
        filename_parts = filename.split('_')
        exchange_name = filename_parts[0]
        date = filename_parts[2].split('.')[0]  # Assuming the date part is immediately before the ".csv"
        symbol = df['symbol'].iloc[0]  # Assuming the 'symbol' column exists and is uniform across the file
        
        # Append the results, including percentiles, symbol, average volumes, and standard deviation of basis points
        results.append((exchange_name, date, symbol, average_bid_volume, average_ask_volume, basis_point_std_dev) + tuple(basis_point_percentiles))

# Define column names, including for percentiles, average volumes, and standard deviation of basis points
columns = ['Exchange', 'Date', 'Symbol', 'Average Bid Volume', 'Average Ask Volume', 'Std Dev of Basis Points'] + [f'{p}th P' for p in percentiles]

# Convert the results into a DataFrame for nicer display
results_df = pd.DataFrame(results, columns=columns)

results_df = results_df.sort_values(by='Exchange')

# Store the DataFrame in a CSV file
output_filepath = os.path.join(csv_directory, 'aggregated_results_with_volumes_and_std_dev.csv')
results_df.to_csv(output_filepath, index=False)

# The modified script now includes rounding for the basis point calculations to ensure maximum 4 decimal places.
