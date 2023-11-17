#EXP 3 IMPLEMENT DIFFERENT ENCODING SCHEMES (LABEL ENCODING AND ONE HOT ENCODING)

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Sample dataset with categorical features
data = {
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue', 'Green', 'Red'],
    'Size': ['Small', 'Medium', 'Large', 'Medium', 'Small','Small', 'Large'],
    'Shape': ['Circle', 'Rectangle', 'Circle', 'Rectangle', 'Circle','square', 'square']
}

df = pd.DataFrame(data)

# Display the original dataset
print("Original Dataset:")
print(df)

# Label Encoding
label_encoder = LabelEncoder()
df_encoded = df.copy()
for column in df.columns:
    if df[column].dtype == 'object':
        df_encoded[column] = label_encoder.fit_transform(df[column])

# Display the dataset after label encoding
print("\nDataset after Label Encoding:")
print(df_encoded)

# One-Hot Encoding
#one_hot_encoder = OneHotEncoder(drop='first', sparse=False)
one_hot_encoder = OneHotEncoder(drop=None, sparse=False)

df_encoded = df.copy()
for column in df.columns:
    if df[column].dtype == 'object':
        encoded_values = one_hot_encoder.fit_transform(df[[column]])
        encoded_df = pd.DataFrame(encoded_values, columns=[f"{column}_{val}" for val in one_hot_encoder.get_feature_names_out([column])])
        df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
        df_encoded = df_encoded.drop(column, axis=1)

# Display the dataset after one-hot encoding
print("\nDataset after One-Hot Encoding:")
pd.set_option('display.max_columns', None)
print(df_encoded)