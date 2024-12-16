import json
import sys

import pandas as pd


from pandas import DataFrame

from anoma_data.exception import AnomaDataException
from anoma_data.logger import logging
from anoma_data.utils.main_utils import read_yaml_file
from anoma_data.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from anoma_data.entity.config_entity import DataValidationConfig
from anoma_data.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_config: configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config =read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise AnomaDataException(e,sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            
            #logging.info(f"Is required number of column present okay?: [{status}]")
            return status
        except Exception as e:
            raise AnomaDataException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Method Name :   is_column_exist
        Description :   This method validates the existence of a numerical and categorical columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            dataframe_feature_columns = df.drop(['y','y.1'],axis=1).columns
           

            missing_columns = []
            #missing_categorical_columns = []
            for column in self._schema_config["feature_columns"]:
                if column not in dataframe_feature_columns:
                    missing_columns.append(column)

            if len(missing_columns)>0:
                logging.info(f"Missing column: {missing_columns}")

            return False if len(missing_columns)>0 else True
        except Exception as e:
            raise AnomaDataException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise AnomaDataException(e, sys)
        
   

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            validation_error_msg = ""
            logging.info("Starting data validation")
            train_df, test_df = (DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
                                 DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))

            status = self.validate_number_of_columns(dataframe=train_df)
            logging.info(f"Number of columns present in training dataframe okay?: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."
            status = self.validate_number_of_columns(dataframe=test_df)

            logging.info(f"Number of columns present in test dataframe okay?: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."

            status = self.is_column_exist(df=train_df)
            logging.info(f"All required columns present in training dataframe status: {status}")

            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."

            status = self.is_column_exist(df=test_df)
            logging.info(f"All required columns present in test dataframe: {status}")

            if not status:
                validation_error_msg += f"columns are missing in test dataframe."

            validation_status = len(validation_error_msg) == 0
            logging.info(f'Validation status: {validation_status}')
            

            if not validation_status: 
                logging.info(f"Validation_error: {validation_error_msg}")
                

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg
                
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise AnomaDataException(e, sys) from e