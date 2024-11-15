import sys
import os
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
from neuro_mf  import ModelFactory

from Us_visa.exception import USVisaException
from Us_visa.logger import logging
from Us_visa.utils.main_utils import load_numpy_array_data,read_yaml_file,load_object,save_object
from Us_visa.entity.config_entity import ModelTrainerConfig
from Us_visa.entity.artifact_entity import ModelTrainerArtifact,DataTransformationartifact,ClassificationMetricArtifact
from Us_visa.entity.estimator import USvisamodel

class ModelTrainer:
    def __init__(self,data_transformation_artifact:DataTransformationartifact,
                 model_trainer_config:ModelTrainerConfig):
        self.data_transformation_artifacts=data_transformation_artifact
        self.model_trainer_config=model_trainer_config
    
    def get_model_object_and_report(self,train:np.array,test:np.array)-> Tuple[object,object]:
        '''
        Method name : get model object and report 
        Description : This method uses neuro_mf to get best mode and report the model

        return metrics artifacts object and best model object
        on failue write an exception log and then raise Exception

        '''
        try:
            logging.info("Using neuro_mf to get best model object and report")
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
            
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

            best_model_detail = model_factory.get_best_model(
                X=x_train,y=y_train,base_accuracy=self.model_trainer_config.expected_accuracy_
            )
            model_obj = best_model_detail.best_model

            y_pred = model_obj.predict(x_test)
            
            accuracy = accuracy_score(y_test, y_pred) 
            f1 = f1_score(y_test, y_pred)  
            precision = precision_score(y_test, y_pred)  
            recall = recall_score(y_test, y_pred)
            metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)
            
            return best_model_detail, metric_artifact

        except Exception as e:
            raise USVisaException(e,sys)
    
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method inside ModelTrainer class")
        '''
        Methode Name: Initiate model trainer
        Description: This function will intiate a model trainer step

        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        '''
        try:
            train_arr=load_numpy_array_data(file_path=self.data_transformation_artifacts.transformed_train_file_path)
            test_arr=load_numpy_array_data(file_path=self.data_transformation_artifacts.transformed_test_file_path)

            best_model_detail,metric_artifact=self.get_model_object_and_report(train=train_arr,test=test_arr)
            preprocessing_obj=load_object(file_path=self.data_transformation_artifacts.transformed_object_file_path)

            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy_:
                logging.info("No best bodel found with score more than base score")
                raise Exception("No best model found with score more than base score")
            usvisamodel=USvisamodel(preprocessing_object=preprocessing_obj,trained_model_object=best_model_detail.best_model)
            logging.info("Created usvisa model object with preprocessor and model")
            logging.info("Created best model file path.")
            save_object(self.model_trainer_config.trained_model_file_path, usvisamodel)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise USVisaException(e, sys) 

        

    
