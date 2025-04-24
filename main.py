# import pandas as pd
# df_cleaned = pd.read_csv('Cleaned_dataset.csv')
# # print first 5 rows of the dataframe
# print(df_cleaned.head())
# # print types of the columns in the dataframe
# print(df_cleaned.dtypes)
import modin.pandas as mpd
df = mpd.read_csv(r"2022_Yellow_Taxi_Trip_Data_20250414.csv")
print(df.head())
print(df.dtypes)
# check how many values are missing in each column
print(df.isnull().sum())
# make a copy of the dataframe, remove 2, 3 columns and rename the columns
df_cleaned = df.copy()
df_cleaned = df_cleaned.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
