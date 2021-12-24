
### RENTAL_ANALYSIS 

# imports
import panel as pn
pn.extension('plotly')
import plotly.express as px
import pandas as pd
import hvplot.pandas
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Read the Mapbox API key
load_dotenv()
map_box_api = os.getenv("mapbox")

# Read the census data into a Pandas DataFrame
file_path = Path("Data/toronto_neighbourhoods_census_data.csv")
to_data = pd.read_csv(file_path, index_col="year")
to_data.head()

# Calculate the sum number of dwelling types units per year (hint: use groupby)
# YOUR CODE HERE!

to_dwelling_types = to_data.groupby("year").sum()
to_dwelling_types.drop(["average_house_value","shelter_costs_owned","shelter_costs_rented"], axis=1, inplace=True)
to_dwelling_types


# Save the dataframe as a csv file

to_dwelling_types.to_csv("./Data/To Dwelling Types.csv")

# Helper create_bar_chart function
def create_bar_chart(data, title, xlabel, ylabel, color):
    """
    Create a barplot based in the data argument.
    """
    fig = plt.figure,
    fig = data.plot.bar(color=color)
    plt.title(title,fontsize=17)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel,fontsize=12)
    plt.show()
    
    return fig
   
   
# Create a bar chart per year to show the number of dwelling types

# Bar chart for 2001                      
create_bar_chart(to_dwelling_types.loc[2001],"Dwelling Types in Toronto in 2001", "2001", "Dwelling Type Units","green");

# Bar chart for 2006
create_bar_chart(to_dwelling_types.loc[2006],"Dwelling Types in Toronto in 2006", "2006", "Dwelling Type Units","red");

# Bar chart for 2011
create_bar_chart(to_dwelling_types.loc[2011],"Dwelling Types in Toronto in 2011", "2011", "Dwelling Type Units","orange");

# Bar chart for 2016
create_bar_chart(to_dwelling_types.loc[2016],"Dwelling Types in Toronto in 2016", "2016", "Dwelling Type Units","gold");


# Calculate the average monthly shelter costs for owned and rented dwellings
# YOUR CODE HERE!
df_avg_monthly_costs = to_data[["shelter_costs_owned","shelter_costs_rented"]].groupby("year").mean()
df_avg_monthly_costs 


# Helper create_line_chart function
def create_line_chart(data, title, xlabel, ylabel, color):
    fig=plt.figure()
    line_chart = data.plot(title=title,xlabel=xlabel,ylabel=ylabel,color=color)
    return fig
    
    
# Create two line charts, one to plot the monthly shelter costs for owned dwelleing and other for rented dwellings per year

# Line chart for owned dwellings
create_line_chart(df_avg_monthly_costs.iloc[:,0],"Average Monthly Shelter Cost for Owned Dwellings in Toronto", "year", "shelter_costs_owned", "red");

# Line chart for rented dwellings
shelter_rented = create_line_chart(df_avg_monthly_costs.iloc[:,1],"Average Monthly Shelter Cost for Rented Dwellings in Toronto", "year", "shelter_costs_rented", "green");


# Calculate the average house value per year
# YOUR CODE HERE!
df_avg_house_value = to_data[["average_house_value"]].groupby("year").mean()
df_avg_house_value


# Plot the average house value per year as a line chart
# YOUR CODE HERE!
df_avg_house_value.plot.line(title="Avgerage house value per year");


# Create a new DataFrame with the mean house values by neighbourhood per year
# YOUR CODE HERE!
df_neighbourhood_index = to_data.reset_index()
df_neighbourhood_index.head()



# Use hvplot to create an interactive line chart of the average house value per neighbourhood
# The plot should have a dropdown selector for the neighbourhood
# YOUR CODE HERE!
df_avg_value_neighbourhood = df_neighbourhood_index[['year','neighbourhood','average_house_value']]
df_avg_value_neighbourhood.head(10)



# Use hvplot to create an interactive bar chart of the number of dwelling types per neighbourhood
# The plot should have a dropdown selector for the neighbourhood
# YOUR CODE HERE!

df_avg_value_neighbourhood.hvplot(
    x="year", 
    y="average_house_value", 
    groupby= "neighbourhood", 
    yformatter='%.0f'

)

# Getting the data from the top 10 expensive neighbourhoods
# YOUR CODE HERE!

top_10_neighbourhoods = to_data.sort_values("average_house_value", ascending=False).head(10)
top_10_neighbourhoods = to_data.groupby(by='neighbourhood').mean()
top_10_neighbourhoods = top_10_neighbourhoods.nlargest(10, columns='average_house_value')
top_10_neighbourhoods


# Plotting the data from the top 10 expensive neighbourhoods
# YOUR CODE HERE!

top_10_neighbourhoods.hvplot.bar(
    x="neighbourhood", 
    y="average_house_value", 
    rot=90, 
    height=600, 
    width=600,
    xlabel="Neighbourhood", 
    ylabel="Avg. House value", 
    yformatter="$%.0f", 
    title="Top 10 most expensive neighbourhoods in Toronto"

)

# Load neighbourhoods coordinates data
file_path = Path("Data/toronto_neighbourhoods_coordinates.csv")
df_neighbourhood_locations = pd.read_csv(file_path)
df_neighbourhood_locations.head()

# Calculate the mean values for each neighborhood
# YOUR CODE HERE!
avg_neighbourhood_value = to_data.groupby("neighbourhood").mean().reset_index()
avg_neighbourhood_value.head()


# Join the average values with the neighbourhood locations
# YOUR CODE HERE!
to_neighbourhood_loc_data = pd.concat(
    [df_neighbourhood_locations.set_index("neighbourhood"), avg_neighbourhood_value.set_index("neighbourhood")], axis="columns", join="inner"
).reset_index()
to_neighbourhood_loc_data.head()


# Create a scatter mapbox to analyze neighbourhood info
# YOUR CODE HERE!

px.set_mapbox_access_token(map_box_api)     
population_plot = px.scatter_mapbox(
    to_neighbourhood_loc_data,
    hover_name="neighbourhood",
    lat="lat",
    lon="lon",
    size="average_house_value",
    color="average_house_value",
    color_continuous_scale = px.colors.sequential.Turbo,
    title="Average House Values in Toronto",
    zoom=10,
    width=1200,
    height=600                  
)

map_plot.show()


#### DASHBOARD

# imports
import panel as pn
pn.extension('plotly')
import plotly.express as px
import pandas as pd
import hvplot.pandas
import matplotlib.pyplot as plt
import os
from pathlib import Path
from dotenv import load_dotenv

# Initialize the Panel Extensions (for Plotly)
import panel as pn
pn.extension("plotly")

# Read the Mapbox API key
load_dotenv()
map_box_api = os.getenv("mapbox")
px.set_mapbox_access_token(map_box_api)

# Import the CSVs to Pandas DataFrames
file_path = Path("Data/toronto_neighbourhoods_census_data.csv")
to_data = pd.read_csv(file_path, index_col="year")

file_path = Path("Data/toronto_neighbourhoods_coordinates.csv")
df_neighbourhood_locations = pd.read_csv(file_path)

# Getting the data from the top 10 expensive neighbourhoods
# YOUR CODE HERE!
top_10_neighbourhoods = to_data.groupby(by='neighbourhood').mean()
top_10_neighbourhoods = top_10_neighbourhoods.nlargest(10, columns='average_house_value')
top_10_neighbourhoods

avg_dwelling_types = to_data.groupby(['year','neighbourhood']).mean()
avg_dwelling_types.drop(['average_house_value', 'shelter_costs_owned', 'shelter_costs_rented'], axis=1, inplace= True)
avg_dwelling_types = avg_dwelling_types.reset_index()
avg_dwelling_types

avg_shelter_costs = to_data[['shelter_costs_owned', 'shelter_costs_rented']].mean()
avg_shelter_costs

# Define Panel visualization functions
def neighbourhood_map():
    to_neighbourhood_loc_data = pd.merge(
    df_neighbourhood_locations, avg_neighbourhood_value, on="neighbourhood"
).reset_index()
    
    population_plot = px.scatter_mapbox(
    to_neighbourhood_loc_data,
    hover_name="neighbourhood",
    lat="lat",
    lon="lon",
    size="average_house_value",
    color="average_house_value",
    color_continuous_scale=px.colors.cyclical.IceFire,
    title="Average House Values in Toronto",
    zoom=10,
    width=1200,
    height=600
)

    return population_plot

def create_bar_chart(data, title, xlabel, ylabel, color):
    """
    Create a barplot based in the data argument.
    """
    
    fig=plt.figure,
    bar=data.plot.bar(color=color)
    bar.set_xlabel(xlabel)
    bar.set_ylabel(ylabel)
    bar.set_title(title)
    plt.close()

    return fig

def create_line_chart(data, title, xlabel, ylabel, color):
    """
    Create a line chart based in the data argument.
    """
    
    fig=plt.figure()
    line=data.plot.line(color=color)
    line.set_xlabel(xlabel)
    line.set_ylabel(ylabel)
    line.set_title(title)
    plt.close()
    
    return fig

def average_house_value():
    """Average house values per year."""
    
    avg_house_value = to_data[["average_house_value"]].groupby("year").mean()
    create_line_chart(avg_house_value, 
    color = "orange",
    xlabel= "Year",
    ylabel= "Avg House Value",
    title= "Average House Value In Toronto")

    return avg_house_value

def average_value_by_neighbourhood():
    """Average house values by neighbourhood."""
     
    avg_neighbourhood_price = avg_neighbourhood_value.hvplot.line(
    x='year',
    y='average_house_value',
    groupby="neighbourhood")
    
    return avg_neighbourhood_price

def number_dwelling_types():
    """Number of dwelling types per year"""
              
    list_dwellings = to_dwelling_types.hvplot.bar(
    x='year', 
    groupby='neighbourhood', 
    rot=90, 
    height=500, 
    xlabel='Year', 
    ylabel='Dwelling Type Units')
    
        
    return list_dwellings 

def average_house_value_snapshot():
    """Average house value for all Toronto's neighbourhoods per year."""

     
    avg_value = avg_neighbourhood_value.hvplot.bar(
    x='year', 
    y="average_house_value",
    groupby='neighbourhood', 
    rot=90, 
    height=500, 
    xlabel='Toronto Neighbourhoods', 
    ylabel='Average House Value')
    
    return avg_value

def top_most_expensive_neighbourhoods():
    """Top 10 most expensive neighbourhoods."""
    plot_avg_house_value_top10 = df_avg_house_value_top10.hvplot.bar(x="neighbourhood", y="average_house_value",yformatter="%.0f", xlabel="Neighbourhood", ylabel="Avg. House Value", rot=90, height=500, title="Top 10 Expensive Neighbourhoods in Toronto", color="purple")
    return plot_avg_house_value_top10

def sunburts_cost_analysis():
    """Sunburst chart to conduct a costs analysis of most expensive neighbourhoods in Toronto per year."""
    
    dwelling_types_2001 = create_bar_chart(df_dwelling_units.loc[2001], "Dwelling Types in 2001", '2001', 'Units', 'yellow')
    dwelling_types_2006 = create_bar_chart(df_dwelling_units.loc[2006],"Dwelling Types in 2006","2006", "Units", "red")
    dwelling_types_2011 = create_bar_chart(df_dwelling_units.loc[2011],"Dwelling Types in 2011","2011", "Units", "orange")
    dwelling_types_2016 = create_bar_chart(df_dwelling_units.loc[2016],"Dwelling Types in 2016","2016", "Units", "black")
