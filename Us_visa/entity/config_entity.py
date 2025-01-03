import os
from Us_visa.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()



@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name:str = DATA_INGESTION_COLLECTION_NAME

@dataclass
class DataValidationConfig:
    data_validation_dir:str=os.path.join(training_pipeline_config.artifact_dir,DATA_VALIDATION_DIR_NAME)
    drift_report_file_path:str=os.path.join(data_validation_dir,DATA_VALIDATION_DRIFT_REPORT_DIR,
                                            DATA_VALIDATION_DRIFT_REOPORT_FILE_NAME)
    
@dataclass
class DataTranssformationConfig:
    data_transformation_dir:str=os.path.join(training_pipeline_config.artifact_dir,DATA_TRANSFORMATION_DIR_NAME)
    transfromed_trained_file_path:str=os.path.join(data_transformation_dir,DATA_TRANSFORMED_DATA_DIR,
                                               TEST_FILE_NAME.replace("csv","npy"))
    transformed_test_file_path:str=os.path.join(data_transformation_dir,DATA_INGESTION_DIR_NAME,
                                            TEST_FILE_NAME.replace("csv","npy"))
    transfromed_object_file_path:str=os.path.join(data_transformation_dir,DATA_TRANSFORMATION_DIR_NAME,
                                                  PREPROCESSING_OBJECT_FILE_NAME)
    
@dataclass
class ModelTrainerConfig:
    model_trainer_dir:str=os.path.join(TrainingPipelineConfig.artifact_dir,MODEL_TRAINER_DIR_NAME)
    trained_model_file_path=os.path.join(model_trainer_dir,MODEL_TRAINER_TRAINED_MODEL_DIR,MODEL_TRAINER_TRAINED_MODEL_NAME)
    expected_accuracy_:float=MODEL_TRAINER_EXPECTED_SCORE
    model_config_file_path=MODEL_TRAINER_MODEL_CONFIG_FILE_PATH