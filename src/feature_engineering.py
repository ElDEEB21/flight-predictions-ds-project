import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/processed/flights cleaned.csv')
df = pd.get_dummies(df, columns=['Airline', 'Departure City', 'Arrival City', 
                                 'Departure Time', 'Arrival Time', 'Flight Class'], drop_first=True)

scaler = StandardScaler()
df[['Flight Duration (hours)', 'Days Until Departure']] = scaler.fit_transform(df[['Flight Duration (hours)', 'Days Until Departure']])

corr = df.corr()
print("Correlation matrix:\n", corr)

x = df.drop(columns=['Price (USD)'])
y = df['Price (USD)']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

x_train.to_csv('../data/processed/x_train_processed.csv', index=False)
x_test.to_csv('../data/processed/x_test_processed.csv', index=False)
y_train.to_csv('../data/processed/y_train.csv', index=False)
y_test.to_csv('../data/processed/y_test.csv', index=False)
