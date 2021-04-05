import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# A_F = pd.read_csv('feature_ABCDEF.csv')
# G_I = pd.read_csv('feature_GHI.csv')
# J_L = pd.read_csv('feature_JKL.csv')
# M_O = pd.read_csv('feature_MNO.csv')
# P_R = pd.read_csv('feature_PQR.csv')
# label = pd.read_csv('data_geocoded/data_with_coor.csv', usecols=[10])

# df = pd.concat([A_F, G_I, J_L, M_O, P_R, label], axis=1)
# df.drop_duplicates(inplace=True)
# df.dropna(inplace=True)
# df.to_csv('data_combined_cleaned.csv', index=False)

# df = pd.read_csv('data_combined_cleaned.csv')
# df_shuffled = df.sample(frac=1)
# df_shuffled.to_csv('data_combined_cleaned_shuffled.csv', index=False)

df = pd.read_csv('data_intermediate/data_combined_cleaned_shuffled.csv')
df['price_per_sqm'] = df['resale_price'] / df['floor_area_sqm']

l_training = round(len(df) * 0.8)
training = df.iloc[:l_training]
test = df.iloc[l_training:]

training.to_csv('training_data.csv', index=False)
test.to_csv('test_data.csv', index=False)