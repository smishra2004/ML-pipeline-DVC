import os
import logging
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

#create a logs folder
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

#create a logger object and set debug level
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

#create a terminal handler and set level
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

#create a logs/data_ingestion.log file and save logs there in file format with debug level
log_file_path = os.path.join(log_dir,'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

#set fromat of the logging
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

#finally add the loggers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path:str)->dict:
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
            logger.debug('Parameters retrieved from %s', params_path)
            return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise
            

def load_data(data_path:str)->pd.DataFrame :
    try:
        df = pd.read_csv(data_path)
        logger.debug('Data loaded successfully from %s',data_path)
        return df
    except pd.errors.ParserError as e:
        logger.debug('Failed to parse the csv file %s',e)
        raise
    except Exception as e:
        logger.debug('Failed to load the data %s',e)
        raise 

def preprocessing(df:pd.DataFrame)->pd.DataFrame:
    try:
        df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
        df.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)
        logger.debug('Data preprocessing completed')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise
    
def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    try:
        raw_data_path = os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(
            os.path.join(raw_data_path,'train.csv'),
            index=False
        )
        test_data.to_csv(
            os.path.join(raw_data_path,'test.csv'),
            index=False
        )
        logger.debug('Train data and test data saved to %s',raw_data_path)
    except Exception as e:
        logger.debug('Failed to save train and test data to %s',raw_data_path)
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        data_path = "https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv"
        df = load_data(data_path)
        preprocessed_df = preprocessing(df)
        train_data,test_data = train_test_split(preprocessed_df,test_size=test_size,random_state=42)
        save_data(train_data,test_data,data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
    