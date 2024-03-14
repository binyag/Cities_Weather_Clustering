# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:04:10 2024

@author: USER
"""
import requests
import pandas as pd
#from datetime import datetime, timedelta
import io


def get_historical_weather_data(api_key, city_list, start_date, end_date):
    """
    Get historical weather data for a list of cities from Visual Crossing Weather API.

    Args:
        api_key: The API key for accessing the Visual Crossing Weather API.
        city_list: A list of city names for which historical weather data is requested.
        start_date: The start date for the historical data (format: 'YYYY-MM-DD').
        end_date: The end date for the historical data (format: 'YYYY-MM-DD').

    Returns:
        None: Saves a single CSV file containing weather data for all cities.
    """

    combined_weather_df = pd.DataFrame()  # Initialize empty DataFrame to store combined data

    for city in city_list:
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/weatherdata/history?aggregateHours=24&startDateTime={start_date}&endDateTime={end_date}&unitGroup=metric&location={city}&key={api_key}"

        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            historical_weather_data = response.text

            # Convert the weather data to a DataFrame
            weather_df = pd.read_csv(io.StringIO(historical_weather_data))

            # Add city name as a new column
            weather_df["City"] = city

            # Append city's weather data to combined DataFrame
            combined_weather_df = combined_weather_df.append(weather_df)

        except requests.exceptions.RequestException as e:
            print(f"Error retrieving data for {city}:", e)

    # Save combined weather data to CSV file
    combined_weather_df.to_csv("All_Cities_Weather.csv", index=False)


# Example usage:
api_key = "F94FD6MRZXVTHC8U737S5RHJ6"  # Your Visual Crossing Weather API key
city_list = [ 'Jakarta', 'Manila',  'Hyderabad', 'Pune', 'Kolkata', 'Ahmedabad', 'Jaipur', 'Lucknow'] # Example list of cities
end_date = "2024-03-10"  # End date is today
start_date = "2023-07-05"  # Start date is one year ago

get_historical_weather_data(api_key, city_list, start_date, end_date)

apis = ['ZLYH4N5W66JUEGT3FG77HV67G' , 'BTWS9XR57EUEYEVFVRRDWFFG4', ] 
past_city = ['Ho Chi Minh City',  'Chandigarh', 'Bengaluru', 'Surat', 'Kanpur', 'Bangalore', 'Chennai','Wellington', 'Christchurch', 'Perth', 'Adelaide', 'Brisbane', 'Gold Coast', 'Hobart', 'Canberra', 'Darwin', 'Birmingham', 'Liverpool', 'Glasgow', 'Edmonton', 'Calgary', 'Montreal', 'Quebec City', 'Halifax', 'Victoria','Marseille', 'Osaka', 'Moscow', 'Copenhagen', 'Dublin', 'Stockholm', 'Helsinki', 'Reykjavik', 'Warsaw', 'Lisbon', 'Edinburgh', 'Manchester', 'Bristol', 'Ottawa','Abuja','Accra','Algiers','Cape Town','Casablanca','Dar es Salaam',   'Johannesburg',         'Khartoum',         'Libreville',         'Windhoek', 'Auckland',         'Beijing',         'Cairo',             'Delhi',         'Hong Kong','Kuala Lumpur',         'Melbourne',         'Mumbai','Nairobi',         'Seattle',         'Seoul',         'Shanghai',         'Sydney',  'Taipei',         'Washington D.C.'"London", "Paris", "New York", "Tokyo", "Rome","Singapore", "Bangkok", "Dubai", "Istanbul", "Berlin", "Madrid","Vienna", "Prague","Budapest", "Amsterdam", "Barcelona", "Milan","Athens", "Los Angeles", "Chicago", "Toronto", "Vancouver", "Las Vegas","Rio de Janeiro", "Buenos Aires"]
cities = []


x = pd.read_csv("All_Cities_Weather.csv")
pd.unique(x["Address"])
