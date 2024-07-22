# Binary Classification Web App

This Streamlit web application performs binary classification on mushroom data to determine if mushrooms are edible or poisonous.

## Features

- Loads and preprocesses mushroom dataset
- Allows users to choose from multiple classifiers
- Provides options to adjust model hyperparameters
- Displays raw data (optional)
- Calculates and displays model performance metrics
- Generates various plots for model evaluation

## Requirements

- Python 3.x
- Streamlit
- Pandas
- Scikit-learn

## Installation

1. Clone this repository
2. Install the required packages

## Usage

1. Ensure you have a file named `mushrooms.csv` in the same directory as the script
2. Run the Streamlit app: streamlit run app.py

3. Use the sidebar to:
- View raw data
- Choose a classifier
- Adjust model hyperparameters
- Select metrics to plot
4. Click "Classify" to train the model and view results

## Code Structure

- `load_data()`: Loads and preprocesses the mushroom dataset
- `split()`: Splits the data into training and testing sets
- `plot_metrics()`: Generates plots for selected metrics
- `main()`: Contains the primary logic for the Streamlit app


## Note

Ensure that the `settings.py` file is present and properly configured with the necessary classifier settings and utility functions.
