# Flight Price Prediction Project

## Project Overview
This project aims to predict flight prices using machine learning models. I analyze various flight-related factors to develop a robust model that provides accurate price predictions. The project is part of the **Digital Egyptian Pioneer Initiative (DEPI)** under the IBM Data Science Track.

### Dataset Description
The dataset contains the following information about flights:
- **Airline**: The airline operating the flight.
- **Source City**: The city from which the flight departs.
- **Departure Time**: The scheduled departure time (morning, evening, etc.).
- **Stops**: Number of stops during the flight (non-stop, 1 stop, 2+ stops).
- **Arrival Time**: The scheduled arrival time (morning, evening, etc.).
- **Destination City**: The city where the flight lands.
- **Class**: The class of travel (economy, business).
- **Duration**: The duration of the flight in hours.
- **Days Left**: The number of days left until the flight's departure.
- **Price**: The price of the flight in INR.

## Project Goal
The goal of this project is to build a predictive model that helps in estimating flight prices based on the above features, providing insights for travelers and airlines alike. This will be achieved by following a full data science pipeline, from data cleaning to model deployment.

## Stage 1: Data Cleaning
In the first stage, the dataset underwent cleaning to prepare it for analysis and modeling. The following key actions were taken:

1. **Column Renaming**: I renamed columns to make them more descriptive and user-friendly, such as renaming `source_city` to `Departure City` and `class` to `Flight Class`.
   
2. **Numeric Conversion**: Certain categorical columns, such as `stops`, were converted to numeric equivalents (e.g., 'one' to 1, 'zero' to 0).
   
3. **Currency Conversion**: The `price` column, originally in INR, was converted to USD using a fixed conversion rate.
   
4. **Dropping Unnecessary Columns**: Columns that were not relevant to the analysis, such as `flight` and `Unnamed: 0`, were removed.

This stage ensures the dataset is well-structured, clean, and ready for the next phases, such as Exploratory Data Analysis (EDA) and modeling.

## Stage 2: Exploratory Data Analysis (EDA)

In this stage, we conducted **Exploratory Data Analysis (EDA)** to gain insights from the dataset. Visualizations helped us understand patterns and relationships among key features, such as airlines, departure cities, flight duration, and prices.

### Key Insights:
- **Airline & City Distribution**: Most flights originate from a few key cities, and certain airlines dominate the data.
- **Price & Flight Duration**: There's a clear relationship between flight duration, number of stops, and price.
- **Flight Class & Price**: Business class flights tend to have significantly higher prices compared to economy.
- **Time of Departure**: Departure times influence price, with some periods being more expensive.

These insights will guide us in selecting features for our prediction model in the next stage.

## Next Steps:
Proceed with feature selection and model building to predict flight prices.
