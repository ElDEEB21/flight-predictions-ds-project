import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('..\\data\\processed\\flights cleaned.csv')

sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
airline_palette = sns.color_palette("Set2")
sns.countplot(x='Airline', data=df, palette=airline_palette)
plt.title('Airline Distribution', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('..\\reports\\figures\\airline_distribution.png')

plt.figure(figsize=(10, 6))
city_palette = sns.color_palette("Spectral")
sns.countplot(x='Departure City', data=df, palette=city_palette)
plt.title('Departure City Distribution', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('..\\reports\\figures\\departure_city_distribution.png')

plt.figure(figsize=(10, 6))
sns.histplot(df['Price (USD)'], bins=30, kde=True, color='purple', edgecolor='black')
plt.title('Price Distribution', fontsize=16)
plt.xlabel('Price (USD)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.savefig('..\\reports\\figures\\price_distribution.png')

plt.figure(figsize=(10, 6))
sns.histplot(df['Flight Duration (hours)'], bins=30, kde=True, color='blue', edgecolor='black')
plt.title('Flight Duration Distribution', fontsize=16)
plt.xlabel('Flight Duration (hours)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.savefig('..\\reports\\figures\\flight_duration_distribution.png')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Flight Duration (hours)', y='Price (USD)', data=df, marker='o', color='green', edgecolor='black', s=100)
plt.title('Flight Duration vs Price', fontsize=16)
plt.xlabel('Flight Duration (hours)', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.grid(True)
plt.savefig('..\\reports\\figures\\flight_duration_vs_price.png')

plt.figure(figsize=(12, 6))
sns.boxplot(x='Airline', y='Price (USD)', data=df, palette='Set3')
plt.title('Price by Airline', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.savefig('..\\reports\\figures\\price_by_airline.png')

plt.figure(figsize=(12, 6))
sns.boxplot(x='Flight Class', y='Price (USD)', data=df, palette='coolwarm')
plt.title('Price by Flight Class', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.savefig('..\\reports\\figures\\price_by_flight_class.png')

plt.figure(figsize=(10, 6))
sns.violinplot(x='Flight Class', y='Price (USD)', data=df, palette='pastel')
plt.title('Price Distribution by Flight Class', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.savefig('..\\reports\\figures\\price_distribution_by_flight_class.png')

plt.figure(figsize=(12, 6))
sns.countplot(x='Airline', hue='Flight Class', data=df, palette='bright')
plt.title('Airline Distribution by Flight Class', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.savefig('..\\reports\\figures\\airline_distribution_by_flight_class.png')

plt.figure(figsize=(12, 6))
sns.countplot(x='Departure City', hue='Flight Class', data=df, palette='Set1')
plt.title('Departure City Distribution by Flight Class', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.savefig('..\\reports\\figures\\departure_city_by_flight_class.png')

sns.pairplot(df, hue='Flight Class', palette='husl', markers=["o", "s"])
plt.suptitle('Pair Plot of Features', y=1.02, fontsize=16)
plt.savefig('..\\reports\\figures\\pair_plot_features.png')

pivot_table = pd.pivot_table(df, values='Price (USD)', index='Airline', columns='Flight Class', aggfunc='mean')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap='Blues', linewidths=1)
plt.title('Heatmap of Airlines vs Flight Class by Price', fontsize=16)
plt.savefig('..\\reports\\figures\\heatmap_airlines_flight_class_price.png')

plt.figure(figsize=(12, 6))
sns.boxplot(x='Departure Time', y='Price (USD)', data=df, palette='coolwarm')
plt.title('Price by Departure Time', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.savefig('..\\reports\\figures\\price_by_departure_time.png')

plt.figure(figsize=(10, 6))
sns.boxplot(x='Number of Stops', y='Flight Duration (hours)', data=df, palette='Spectral')
plt.title('Flight Duration by Number of Stops', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.savefig('..\\reports\\figures\\flight_duration_by_stops.png')

sns.jointplot(x='Days Until Departure', y='Price (USD)', data=df, kind='kde', fill=True, cmap='Purples')
plt.suptitle('Price vs Days Until Departure (KDE)', y=1.02, fontsize=16)
plt.savefig('..\\reports\\figures\\jointplot_price_vs_days_until_departure.png')

sns.jointplot(x='Flight Duration (hours)', y='Price (USD)', data=df, kind='scatter', color='blue', edgecolor='black')
plt.suptitle('Flight Duration vs Price', y=1.02, fontsize=16)
plt.savefig('..\\reports\\figures\\jointplot_flight_duration_vs_price.png')

g = sns.FacetGrid(df, col="Airline", hue="Flight Class", col_wrap=3, palette='cool')
g.map(sns.scatterplot, "Departure Time", "Price (USD)", edgecolor='black')
g.add_legend()
g.fig.suptitle('Facet Grid: Price by Airline and Departure Time', y=1.02, fontsize=16)
plt.savefig('..\\reports\\figures\\facetgrid_price_by_airline_departure_time.png')
