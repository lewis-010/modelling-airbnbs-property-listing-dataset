import pandas as pd

def remove_rows_with_missing_data():
    '''Removes specific rows with missing data.'''
    df.dropna(subset=['Description', 'Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating'], inplace=True)

def combine_description_strings():  
    '''Combines indivdual strings in description column into single string and removes whitespace.'''
    df['Description'] = df['Description'].str.replace('About this space', '')
    df['Description'] = df['Description'].apply(lambda x: [item for item in eval(x) if item != ''])
    df['Description'] = df['Description'].apply(lambda x: ''.join(x))

def set_default_feature_values():
    '''Sets missing values in specific columns to default value (1).'''
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


def load_airbnb(df, label):
    ''' 
    Sets the features and labels of a specified dataset.

    Parameters 
    ----------
    df: pandas Dataframe
        The dataset to select the features and label from.
    label: str
        The column header of the selected label.  
    '''
    df.drop(df.columns[0], axis=1, inplace=True)
    features = df.drop([label], axis=1).select_dtypes(include=['int64', 'float64']).values 
    label = df[label].values
    return features, label