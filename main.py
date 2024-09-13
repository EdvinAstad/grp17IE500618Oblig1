import pandas as pd

try:
    housing = pd.read_csv('AmesHousing.csv')
    print(housing.head())
except FileNotFoundError:
    print("The file 'AmesHousing.csv' was not found. Please check the file path.")
except pd.errors.ParserError:
    print("The file could not be parsed. Please ensure it is in the correct CSV format.")
except Exception as e:
    print(f"An error occurred: {e}")

