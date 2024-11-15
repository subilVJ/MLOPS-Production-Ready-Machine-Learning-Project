import os
from datetime import date

DATABASE_NAME="US_VISA"

COLLECTION_NAME= "visa_da"

MONGODB_URL_KEY="mongodb+srv://subiljacob:subiljacob@cluster0.atusd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

PIPELINE_NAME:str="usvisa"

ARTIFACT_DIR:str="artifacts"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

FILE_NAME: str = "usvisa.csv"

MODEL_FILE_NAME: str="model.pkl"

TARGET_COLUMNS="case_status"
CURRENT_YEAR=date.today().year
PREPROCESSING_OBJECT_FILE_NAME="preprocessing.pkl"
SCHEMA_FILE_PATH=os.path.join("config","schema.yaml")

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "visa_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

'''
Data Validation related constants starting with DATA VALIDATION VAR NAME
'''
DATA_VALIDATION_DIR_NAME:str="data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR:str="drift_report"
DATA_VALIDATION_DRIFT_REOPORT_FILE_NAME:str="report.yamls"

'''
Data Transformation related constants starting with DATA_TRANSFORMATION_RELATED VAR NAME
'''

DATA_TRANSFORMATION_DIR_NAME:str="data_transformation"
DATA_TRANSFORMED_DATA_DIR:str="transformed"
DATA_TRANSFROMATION_TRANSFORMED_OBJECT_DIR:str="transfromed_object"

"""
Model Trainer related constants starts with MODEL_TRAINER VAR name
"""

MODEL_TRAINER_DIR_NAME="model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR:str="trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME:str="model.pkl"
MODEL_TRAINER_EXPECTED_SCORE:float=0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH:str=os.path.join('config','model.yaml')

