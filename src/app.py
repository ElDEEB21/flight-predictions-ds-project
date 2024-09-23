import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import joblib
import numpy as np
# Load the data
df = pd.read_csv('E:\\Abdo ElDeeb\\Courses\\DEPI\\flight-predictions-ds-project\\data\\processed\\flights cleaned.csv')

# Function for EDA Dashboard with interactive Plotly charts
def EDA_Dashboard():
    # Sidebar for figure selection
    st.title("📊 Exploratory Data Analysis")

    # Select figure to display
    st.subheader("Select Figure to Display")
    options = [
        "Airline Distribution", "Departure City Distribution", "Price Distribution", 
        "Flight Duration Distribution", "Flight Duration vs Price", 
        "Price by Airline", "Price by Flight Class", 
        "Price Distribution by Flight Class", "Airline Distribution by Flight Class", 
        "Departure City Distribution by Flight Class", "Heatmap of Airlines vs Flight Class by Price", 
        "Price by Departure Time", "Flight Duration by Number of Stops", 
        "Price vs Days Until Departure"
    ]
    choice = st.selectbox("Choose a Plot:", options, index=0)
    
    # Dynamic plot based on the user's choice using Plotly
    if choice == "Airline Distribution":
        fig = px.histogram(df, x='Airline', color_discrete_sequence=['#636EFA'], title='Airline Distribution')
        fig.update_layout(xaxis={'categoryorder': 'total descending'}, template='plotly_dark')
        st.plotly_chart(fig)

    elif choice == "Departure City Distribution":
        fig = px.histogram(df, x='Departure City', color_discrete_sequence=['#EF553B'], title='Departure City Distribution')
        fig.update_layout(xaxis={'categoryorder': 'total descending'}, template='plotly_dark')
        st.plotly_chart(fig)

    elif choice == "Price Distribution":
        fig = px.histogram(df, x='Price (USD)', nbins=50, color_discrete_sequence=['#00CC96'], title='Price Distribution')
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig)

    elif choice == "Flight Duration Distribution":
        fig = px.histogram(df, x='Flight Duration (hours)', nbins=50, color_discrete_sequence=['#AB63FA'], title='Flight Duration Distribution')
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig)

    elif choice == "Flight Duration vs Price":
        fig = px.scatter(df, x='Flight Duration (hours)', y='Price (USD)', color='Airline', title='Flight Duration vs Price')
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig)

    elif choice == "Price by Airline":
        fig = px.box(df, x='Airline', y='Price (USD)', color='Airline', title='Price by Airline')
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig)

    elif choice == "Price by Flight Class":
        fig = px.pie(df, names='Flight Class', values='Price (USD)', title='Price by Flight Class')
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig)

    elif choice == "Price Distribution by Flight Class":
        fig = px.violin(df, x='Flight Class', y='Price (USD)', color='Flight Class', box=True, points='all', title='Price Distribution by Flight Class')
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig)

    elif choice == "Airline Distribution by Flight Class":
        fig = px.histogram(df, x='Airline', color='Flight Class', title='Airline Distribution by Flight Class')
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig)

    elif choice == "Departure City Distribution by Flight Class":
        fig = px.histogram(df, x='Departure City', color='Flight Class', title='Departure City Distribution by Flight Class')
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig)

    elif choice == "Heatmap of Airlines vs Flight Class by Price":
        pivot_table = pd.pivot_table(df, values='Price (USD)', index='Airline', columns='Flight Class', aggfunc='mean')
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='Viridis'))
        fig.update_layout(title='Airlines vs Flight Class by Price', template='plotly_dark')
        st.plotly_chart(fig)

    elif choice == "Price by Departure Time":
        fig = px.box(df, x='Departure Time', y='Price (USD)', color='Departure Time', title='Price by Departure Time')
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig)

    elif choice == "Flight Duration by Number of Stops":
        fig = px.box(df, x='Number of Stops', y='Flight Duration (hours)', color='Number of Stops', title='Flight Duration by Number of Stops')
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig)

    elif choice == "Price vs Days Until Departure":
        fig = px.scatter(df, x='Days Until Departure', y='Price (USD)', color='Flight Class', title='Price vs Days Until Departure', trendline="ols")
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig)

# Empty function for model usage (to be developed later)
def load_model():
    model_path = 'E:\\Abdo ElDeeb\\Courses\\DEPI\\flight-predictions-ds-project\\RandomForest_model.pkl'
    with open(model_path, 'rb') as file:
        model = joblib.load(model_path)
    return model

def Model_Usage():
    st.title("✈️ Flight Price Prediction")
    st.markdown("### Predict the price of your flight based on various factors")

    # Load the model
    model = load_model()

    # Create columns for inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        num_stops = st.selectbox('Number of Stops', [0, 1, 2], key="num_stops", format_func=lambda x: f"{x} stop(s)")
        flight_duration = st.number_input('Flight Duration (hours)', min_value=1, max_value=50, key="flight_duration")
        airline = st.selectbox('Airline', 
                                ['Air India', 'GO FIRST', 'Indigo', 
                                 'SpiceJet', 'Vistara', 'AirAsia'], key="airline")

    with col2:
        days_until_departure = st.number_input('Days Until Departure', min_value=1, max_value=49, key="days_until_departure")
        departure_city = st.selectbox('Departure City', 
                                       ['Chennai', 'Delhi', 
                                        'Hyderabad', 'Kolkata', 
                                        'Mumbai', 'Bangalore'], key="departure_city")
        departure_time = st.selectbox('Departure Time', 
                                       ['Early Morning', 'Evening', 
                                        'Late Night', 'Morning', 
                                        'Night', 'Afternoon'], key="departure_time")

    with col3:
        arrival_city = st.selectbox('Arrival City', 
                                     ['Chennai', 'Delhi', 
                                      'Hyderabad', 'Kolkata', 
                                      'Mumbai', 'Bangalore'], key="arrival_city")
        arrival_time = st.selectbox('Arrival Time', 
                                     ['Early Morning', 'Evening', 
                                      'Late Night', 'Morning', 
                                      'Night', 'Afternoon'], key="arrival_time")
        flight_class = st.selectbox('Flight Class', 
                                     ['Economy', 'Business'], key="flight_class")

    # Button to make prediction
    if st.button("Predict Price"):
        # Prepare input for the model (one-hot encoding)
        input_data = np.array([[num_stops, flight_duration, days_until_departure,
                                 int(airline == 'Air India'),
                                 int(airline == 'GO FIRST'),
                                 int(airline == 'Indigo'),
                                 int(airline == 'SpiceJet'),
                                 int(airline == 'Vistara'),
                                 int(departure_city == 'Chennai'),
                                 int(departure_city == 'Delhi'),
                                 int(departure_city == 'Hyderabad'),
                                 int(departure_city == 'Kolkata'),
                                 int(departure_city == 'Mumbai'),
                                 int(arrival_city == 'Chennai'),
                                 int(arrival_city == 'Delhi'),
                                 int(arrival_city == 'Hyderabad'),
                                 int(arrival_city == 'Kolkata'),
                                 int(arrival_city == 'Mumbai'),
                                 int(departure_time == 'Early Morning'),
                                 int(departure_time == 'Evening'),
                                 int(departure_time == 'Late Night'),
                                 int(departure_time == 'Morning'),
                                 int(departure_time == 'Night'),
                                 int(arrival_time == 'Early Morning'),
                                 int(arrival_time == 'Evening'),
                                 int(arrival_time == 'Late Night'),
                                 int(arrival_time == 'Morning'),
                                 int(arrival_time == 'Night'),
                                 int(flight_class == 'Economy')]])

        # Make the prediction
        prediction = model.predict(input_data)
        # Displaying the predicted price with HTML styling
        st.markdown(
            f"<h2 style='color: #4CAF50;'>Predicted Price: <strong>${prediction[0]:.2f}</strong></h2>",
            unsafe_allow_html=True
        )


# Main function to create the two-page app
def main():
    st.title("🌍 Flight Analysis and Prediction Dashboard")

    # Custom CSS to enhance appearance
    st.markdown("""
        <style>
        .stRadio > div {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .stRadio label {
            margin-right: 20px;
            font-size: 18px;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    # Navigation between pages using radio buttons
    page = st.radio("Choose a Page", ["EDA Dashboard", "Model Usage"], index=0, horizontal=True)

    if page == "EDA Dashboard":
        EDA_Dashboard()
    elif page == "Model Usage":
        Model_Usage()

if __name__ == "__main__":
    main()
