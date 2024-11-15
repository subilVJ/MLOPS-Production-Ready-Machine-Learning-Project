import sys
from Us_visa.exception import USVisaException
from Us_visa.logger import logging
from Us_visa.components.data_ingestion import DataIngestion
from Us_visa.components.data_transformation import DataTransfromation
from Us_visa.components.model_trainer import ModelTrainer

from Us_visa.entity.config_entity import DataIngestionConfig,DataTranssformationConfig,ModelTrainerConfig
from Us_visa.entity.artifact_entity import DataIngestionArtifact,DataTransformationartifact,ModelTrainerArtifact

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config=DataTranssformationConfig()
        self.model_trainer_config= ModelTrainerConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
            """
            This method of TrainPipeline class is responsible for starting data ingestion component
            """
            try:
                logging.info("Entered the start_data_ingestion method of TrainPipeline class")
                logging.info("Getting the data from mongodb")
                data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
                data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
                logging.info("Got the train_set and test_set from mongodb")
                logging.info(
                    "Exited the start_data_ingestion method of TrainPipeline class"
                )
                return data_ingestion_artifact
            except Exception as e:
                raise USVisaException(e, sys) from e
            
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationartifact:
        """
        This method of TrainPipeline class is responsible for starting data transformation component
        """
        try:
            data_transformation = DataTransfromation(data_ingestion_artifact=data_ingestion_artifact,
                                                     data_transformation_config=self.data_transformation_config,
                                                     )
            data_transformation_artifact = data_transformation.intiate_data_transfromation()
            return data_transformation_artifact
        except Exception as e:
            raise USVisaException(e, sys)
    
    def start_model_trainer(self, data_transformation_artifact: DataTransformationartifact) -> ModelTrainerArtifact:
        """
        This method of TrainPipeline class is responsible for starting model training
        """
        try:
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                         model_trainer_config=self.model_trainer_config
                                         )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact

        except Exception as e:
            raise USVisaException(e, sys)
    
    def run_pipeline(self, ) -> None:
        """
        This method of TrainPipeline class is responsible for running complete pipeline
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact)
            model_trainer_artifact=self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
        except Exception as e:
            raise USVisaException(e, sys)