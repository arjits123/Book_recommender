import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from dataclasses import dataclass
from utils import save_obj

# ML libraries
import pandas as pd # type: ignore
import numpy as np # type: ignore

# Create the Data transformation config class 
@dataclass
class DataTranformationConfig:
    """Data Transformation Config Class"""
    cleaned_data_path : str = os.path.join('artifacts', 'final_data_set.csv')
    transformed_data_path: str = os.path.join('artifacts', 'pivot_table.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTranformationConfig()
    
    def clean_data(self, data):
        try:
            df = pd.read_csv(data, low_memory=False)
            year_corrections = {
            'DK Publishing Inc': 2000,
            'Gallimard': 2003
            }

            #replaceing object values and converting to int
            df['Year-Of-Publication'] = df['Year-Of-Publication'].replace(year_corrections).astype(int)

            #converting year = 0 with median
            df['Year-Of-Publication'] = df['Year-Of-Publication'].replace(0, df['Year-Of-Publication'].median())

            #Removing outliers
            Q1 = df['Year-Of-Publication'].quantile(0.25)
            Q3 = df['Year-Of-Publication'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df['Year-Of-Publication'] >= lower_bound) & (df['Year-Of-Publication'] <= upper_bound)]

            df.to_csv(self.data_transformation_config.cleaned_data_path, index=False, header= True)
            logging.info('Data cleaning completed ')

            return df

        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self, new_df):
        try:
            # users indexes who have rated more than 200 books
            rating_200_above = new_df.groupby('User-ID').count()['Book-Rating'] > 200
            y = rating_200_above[rating_200_above == True].index
            filtered_rating_df = new_df[new_df['User-ID'].isin(y)]

            # gives the books which has more than 50 ratings
            book_rating_above_50 = filtered_rating_df.groupby('Book-Title').count()['Book-Rating'] >= 50
            famous_books = book_rating_above_50[book_rating_above_50 == True].index
            final_ratings_df = filtered_rating_df[filtered_rating_df['Book-Title'].isin(famous_books)]

            #Convert it into pivot table
            pt = final_ratings_df.pivot_table(index='Book-Title', columns='User-ID', values = 'Book-Rating')
            pt.fillna(0, inplace=True)

            #saving the pivot table object
            save_obj(file_path=self.data_transformation_config.transformed_data_path, data_frame=pt)
            logging.info('Data Transformation completed ')

            return pt

        except Exception as e:
            raise CustomException(e,sys)

