# The Machine Learning Model Project
Datasets are crucial tools for training machine learning (ML) models, effectively allowing them to learn and therefore predict future outcomes based on trends and patterns that are identified in the original dataset. 

This project will focus on using [AirBnB's](https://www.airbnb.co.uk/) property listing dataset to train an ML model that can predict the price per night for future listings. 

## Milestone 1
- The dataset (*AirBnBData.csv*) consists of 829 property listings and a variety of characteristics for each property. 
- The characteristics are a mixture of text and numerical values.
    - E.g., location, bedrooms, guests, ratings, price per night etc
- The first task was to clean the dataset with the following steps:
    - Remove rows with missing ratings.
    - Combine description strings into singular string and remove whitespace.
    - Apply default feature values (1) for guests, beds, bathrooms and bedrooms for missing values.
```Python
def remove_rows_with_missing_data():
     df.dropna(subset=['Description', 'Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating'], inplace=True)

def combine_description_strings():    
    df['Description'] = df['Description'].str.replace('About this space', '')
    df['Description'] = df['Description'].apply(lambda x: [item for item in eval(x) if item != ''])
    df['Description'] = df['Description'].apply(lambda x: ''.join(x))

def set_default_feature_values():
    df.update(df[['guests', 'bedrooms', 'beds', 'bathrooms']].fillna(1))
```
- Alongside the dataset, there are images corresponding to each property listing.
- These were uploaded to an [AWS](https://aws.amazon.com/) S3 bucket and subsequently downloaded and cleaned.
    - Discarded any non-rgb images.
    - Resized them all to the same height of the smallest image whilst maintaining aspect ratio.
```Python
def resize_images():
    base_dir = r'C:\Users\lcox1\Documents\VSCode\AiCore\Data science\images'

    rgb_file_paths = []

    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            for f in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, f)
                if os.path.isfile(file_path):
                    with Image.open(file_path) as img:
                        if img.mode == 'RGB':
                            rgb_file_paths.append(file_path)
    
    min_height = float('inf')
    for checked_file in rgb_file_paths:
        with Image.open(checked_file) as im:
            min_height = min(min_height, im.height)

    for file_path in rgb_file_paths:
        with Image.open(file_path) as im:
            width, height = im.size
            new_height = min_height
            new_width = int(width * new_height / height)

            resized_im = im.resize((new_width, new_height))

            resized_im.save(os.path.join('processed_images', os.path.basename(file_path)))
```
## Milestone 2
- With the data now cleaned, the next step was to begin building the ML model.
- Firstly, the features and labels were chosen (price per night is the target).
    - Only numerical columns were used as features.
```Python
def load_airbnb(label):
    df = pd.read_csv('tabular_data/clean_tabular_data.csv')
    df.drop(columns=['ID', 'Category', 'Title', 'Description', 'Amenities', 'Location', 'url'])
    features = df.drop('Price_Night', axis=1).values
    labels = df.pop('Price_Night').values
    return features, labels

features, labels = load_airbnb('Price_Night')
```