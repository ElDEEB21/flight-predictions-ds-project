import pandas as pd  # Importing the pandas library for data manipulation

# Reading the CSV file and storing it in a DataFrame called 'df'
df = pd.read_csv('../data/raw/flights dataset.csv')

# Dropping the 'Unnamed: 0' column from the DataFrame
df = df.drop('Unnamed: 0', axis=1)

# Replacing the values in the 'stops' column with numeric equivalents
df['stops'] = df['stops'].replace({'one': 1, 'zero': 0, 'two_or_more': 2})

# Converting the 'stops' column to numeric data type
df['stops'] = pd.to_numeric(df['stops'])

# Dropping the 'flight' column from the DataFrame
df = df.drop('flight', axis=1)

# Setting the conversion rate for the 'price' column
dollar_conversation_rate = 0.012

# Multiplying the values in the 'price' column by the conversion rate
df['price'] = df['price'] * dollar_conversation_rate

# Creating a dictionary to map the old column names to new column names
new_column_names = {
    'airline': 'Airline',
    'source_city': 'Departure City',
    'departure_time': 'Departure Time',
    'stops': 'Number of Stops',
    'arrival_time': 'Arrival Time',
    'destination_city': 'Arrival City',
    'class': 'Flight Class',
    'duration': 'Flight Duration (hours)',
    'days_left': 'Days Until Departure',
    'price': 'Price (USD)'
}

# Renaming the columns of the DataFrame using the dictionary
df.rename(columns=new_column_names, inplace=True)

# Replacing underscores with spaces in the 'Departure Time' column
df['Departure Time'] = df['Departure Time'].str.replace('_', ' ')

# Replacing underscores with spaces in the 'Arrival Time' column
df['Arrival Time'] = df['Arrival Time'].str.replace('_', ' ')

# Saving the cleaned DataFrame to a new CSV file
df.to_csv('../data/processed/flights cleaned.csv', index=False)