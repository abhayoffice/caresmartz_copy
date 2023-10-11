import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    """
            The Data Transformation will perform simple imputation on the data and also I have added a custom method that
                    will replace the null values in the data with the median.
    """
    def __init__(self, config):
        self.data_transformation_config = config

    def get_data_transformer_object(self):
        try:
            num_pipeline = Pipeline(
                steps=[
                    ("Invoice Amount", SimpleImputer(strategy="median")),
                ]
            )

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, ['Invoice Amount'])
            ])
            return preprocessor

        except Exception as e:
            raise CustomException(e)

    # def initiate_data_transformation(self, train_data, test_data):
    def initiate_data_transformation(self):
        try:
            train_df = pd.read_csv(self.data_transformation_config["train_data_path"])
            test_df = pd.read_csv(self.data_transformation_config["test_data_path"])
            df = pd.read_csv(self.data_transformation_config["raw_data_path"])

            logging.info("Read train and test data completed")

            # print(f"Invoice date for train {train_df[['Invoice Date']]}")

            logging.info("Obtaining time series data transformation parameters")

            preprocessor = self.get_data_transformer_object()

            input_feature_train_arr = preprocessor.fit_transform(train_df[['Invoice Amount']])
            input_feature_test_arr = preprocessor.fit_transform(test_df[['Invoice Amount']])
            df_arr = preprocessor.fit_transform(df[['Invoice Amount']])

            print(f"The test array is {len(input_feature_test_arr)}")
            print(f"The train array is {len(input_feature_train_arr)}")

            train_df_transformed = pd.DataFrame(input_feature_train_arr, columns=['Invoice Amount'])
            test_df_transformed = pd.DataFrame(input_feature_test_arr, columns=['Invoice Amount'])
            df_transformed = pd.DataFrame(df_arr, columns=['Invoice Amount'])

            # Add the 'Invoice Date' column back to the DataFrames
            train_df_transformed['Invoice Date'] = train_df['Invoice Date']
            test_df_transformed['Invoice Date'] = test_df['Invoice Date']
            df_transformed['Invoice Date'] = df['Invoice Date']

            # Convert "Invoice Date" column to datetime type and set it as the index
            train_df_transformed['Invoice Date'] = pd.to_datetime(train_df_transformed['Invoice Date'])
            test_df_transformed['Invoice Date'] = pd.to_datetime(test_df_transformed['Invoice Date'])
            df_transformed['Invoice Date'] = pd.to_datetime(df_transformed['Invoice Date'])

            # Set the index as datetime for time series analysis.
            train_df_transformed.set_index('Invoice Date', inplace=True)
            test_df_transformed.set_index('Invoice Date', inplace=True)
            df_transformed.set_index('Invoice Date', inplace=True)

            # To check the total size of dataframe.
            if train_df_transformed.shape[0] + test_df_transformed.shape[0] <= 36:
                self.skip_file()

            train_df_transformed = self.replace_zero_with_median(train_df_transformed)
            test_df_transformed = self.replace_zero_with_median(test_df_transformed)
            df_transformed = self.replace_zero_with_median(df_transformed)

            save_object(
                file_path=self.data_transformation_config["preprocessor_file"], obj=preprocessor
            )
            return train_df_transformed, test_df_transformed, df_transformed
            # return train_monthly, test_monthly

        except Exception as e:
            raise CustomException(e)

    def skip_file(self):
        print("Skip this file and use os.write() here")
        return

    def replace_zero_with_median(self, df):
        median_value = df[df['Invoice Amount'] != 0.0]['Invoice Amount'].median()
        df.loc[df['Invoice Amount'] == 0.0, 'Invoice Amount'] = median_value
        return df
