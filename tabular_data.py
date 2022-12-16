import pandas as pd
import numpy as np

def remove_rows_with_missing_data():
     df.dropna(subset=['Description', 'Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating'], inplace=True)

def combine_description_strings():    
    df['Description'] = df['Description'].str.replace('About this space', '')
    df['Description'] = df['Description'].apply(lambda x: [item for item in eval(x) if item != ''])
    df['Description'] = df['Description'].apply(lambda x: ''.join(x))

def set_default_feature_values():
    df.update(df[['guests', 'bedrooms', 'beds', 'bathrooms']].fillna(1))

def clean_tabular_data():
    remove_rows_with_missing_data()
    combine_description_strings()
    set_default_feature_values()

if __name__=='__main__':
    df = pd.read_csv('tabular_data/AirBnbData.csv')
    df.drop(df.columns[19], axis=1, inplace=True) # removes unecessary final column
    clean_tabular_data()
    df.to_csv('clean_tabular_data.csv')


def load_airbnb(label):
    df = pd.read_csv('tabular_data/clean_tabular_data.csv')
    features = df.drop(label, axis=1).values
    labels = df.pop(label).values
    return features, labels
