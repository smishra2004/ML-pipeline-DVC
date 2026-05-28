import os
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_path = os.path.join(log_dir,'model_building.log')
file_handler = logging.FileHandler(file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(path:str)->pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train:np.ndarray,y_train:np.ndarray):
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be the same.")
        
        logger.debug('Initializing RandomForest model : %s')
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        
        logger.debug('Model training started with %d samples', X_train.shape[0])
        clf.fit(X_train, y_train)
        logger.debug('Model training completed')
        
        return clf
    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise
        
def save_model(model,model_path:str)->None:
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path,'wb') as file:
            pickle.dump(model,file)
            logger.debug('Model saved to %s', file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        
        clf = train_model(X_train, y_train)
        
        model_save_path = 'models/model.pkl'
        save_model(clf,model_save_path)
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()