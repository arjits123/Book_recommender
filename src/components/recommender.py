import os
import sys
from dataclasses import dataclass
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging

#Importing important libraries
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

@dataclass
class ModelTrainerConfig:
    recommender_file_path : str = os.path.join('artifacts', 'recommender.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_recommendation(self, book_name:str, pivot):
        try:
            book_data = pd.read_csv('Notebook/dataset/Books.csv', low_memory=False)
            #Calculate the similarity score
            similarity_score = cosine_similarity(pivot)

            # get the book_index using np.where
            book_index = np.where(pivot.index == book_name)[0][0]

            #distances from that book to other books
            distances = similarity_score[book_index]

            #get the indices of the 10 most similar books
            index_with_score = list(enumerate(distances))
            sorted_list = sorted(index_with_score, key=lambda x: x[1], reverse = True)

            #Get the top 5 recommendations
            top_5_recommendation = sorted_list[1:5]
            recommendations = []
            for i in top_5_recommendation:
                items = []
                temp_df = book_data[book_data['Book-Title'] == pivot.index[i[0]]]
                items.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
                items.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
                items.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
            
                recommendations.append(items)
            logging.info('Recommnedations completed')

            return recommendations
        except Exception as e:
            raise CustomException(e,sys)
