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

## Next Steps
The next step involves performing **Exploratory Data Analysis (EDA)** to understand trends and patterns within the dataset.
