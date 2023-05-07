import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from geopy.distance import geodesic
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object,filter_int_float_values,remove_outliers

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self,X):
        try:
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = X.select_dtypes(include='object').columns
            numerical_cols = X.select_dtypes(exclude='object').columns
            
            # Define the custom ranking for each ordinal variable
            weather_categories = ['Sunny','Stormy','Sandstorms','Windy','Cloudy','Fog']
            traffic_categories = ['Low','Medium','High','Jam']
            vehicleType_categories = ['electric_scooter','scooter','motorcycle']
            festival_categories = ['No','Yes']
            city_categories= ['Urban','Metropolitian','Semi-Urban']
            
            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[weather_categories,traffic_categories,vehicleType_categories,festival_categories,city_categories])),
                    ('scaler',StandardScaler())
                ]

            )

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            
            return preprocessor

            #logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
    def initaite_data_transformation(self,raw_data_path):
        try:
            # Reading train and test data
            data = pd.read_csv(raw_data_path)

            logging.info('Read raw data completed')
            logging.info(f'Raw Dataframe Head : \n{data.head().to_string()}')

            logging.info("EDA and Feature Engineering Started")

            # dropping nan values from dataframe
            data = data.dropna()

            # Now find the distance between restaurant location and delivery location
            restaurant_coordinates = data[['Restaurant_latitude','Restaurant_longitude']].to_numpy()
            delivery_coordinates = data[['Delivery_location_latitude','Delivery_location_longitude']].to_numpy()

            data['distance(km)'] = ''
            for i in range(len(data)):
                data['distance(km)'].iloc[i]=geodesic(restaurant_coordinates[i],delivery_coordinates[i]).km
                
            data['Restaurant_Delivery_distance'] = data['distance(km)'].astype('float')

            # removing outliers from Restaurant_Delivery_distance column
            data = remove_outliers(data)

            # After outliers removal data shape is 
            # it is found that there are some float and integer values in Time_Orderd and Time_Order_picked column, filter those and also get the data with upto 5 digits
            data['Time_Orderd'] = filter_int_float_values(data['Time_Orderd']).str.slice(0,5)
            data['Time_Order_picked'] = filter_int_float_values(data['Time_Order_picked']).str.slice(0,5)

            ## Concatenating :00 atlast to make it date format and also we can calculate the preparation time using these values
            data['Time_Orderd'] = data['Time_Orderd'] + ':00'
            data['Time_Order_picked'] = data['Time_Order_picked'] + ':00'
            data = data.reset_index()
            data.drop(columns=['index'],axis=1,inplace=True)

            ## Let's calculate preparation time
            data['Time_Orderd'] = pd.to_timedelta(data['Time_Orderd'])
            data['Time_Order_picked'] = pd.to_timedelta(data['Time_Order_picked'])
            td = pd.Timedelta(1, "d") # to indicate 1 day

            data.loc[(data['Time_Order_picked'] < data['Time_Orderd']), 'preparation1'] = data['Time_Order_picked'] - data['Time_Orderd'] + td
            data.loc[(data['Time_Order_picked'] > data['Time_Orderd']), 'preparation2'] = data['Time_Order_picked'] - data['Time_Orderd'] 

            data['preparation1'].fillna(data['preparation2'], inplace=True)
            data['preparation_time'] = pd.to_timedelta(data['preparation1'], "minute")
            for i in range(len(data['preparation_time'])):
                data['preparation_time'][i] = data['preparation_time'][i].total_seconds()/60 # converting into minutes
            data['preparation_time'] = data['preparation_time'].astype(float)

            # Round restaurant delivery distance distance to 2 decimals
            data['Restaurant_Delivery_distance'] = round(data['Restaurant_Delivery_distance'],2)
            data['Festival'] = data['Festival'].fillna('No')

            # Drop unwanted columns
            data.drop(columns=['ID','Delivery_person_ID','Time_Orderd','Time_Order_picked','Order_Date','preparation1','preparation2','Type_of_order','Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude','distance(km)'],axis=1,inplace=True)

            # Rearrange columns
            columns_data = ['Delivery_person_Age','Delivery_person_Ratings','Restaurant_Delivery_distance','preparation_time','Weather_conditions','Road_traffic_density','Vehicle_condition','Type_of_vehicle','multiple_deliveries','Festival','City','Time_taken (min)']
            data = data[columns_data]            

            logging.info("Splitting train and test split")
            ## Independent and dependent features
            logging.info(data.columns)
            X = data.drop(labels=['Time_taken (min)'],axis=1)
            Y = data[['Time_taken (min)']]
            
            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object(X)

            logging.info(f"Train Input features: \n {X.head().to_string()}")
            logging.info(f"Train Target features: \n {Y.head().to_string()}")

            input_feature_train_df,input_feature_test_df,target_feature_train_df,target_feature_test_df=train_test_split(X,Y,test_size=0.20,random_state=36)
            logging.info('Train and test split completed')
            
            # Saving train and test split data inside artifacts folder
            train_set = pd.concat([input_feature_train_df,target_feature_train_df],axis=1)
            test_set = pd.concat([input_feature_test_df,target_feature_test_df],axis=1)
            train_set.to_csv(self.data_transformation_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_transformation_config.test_data_path,index=False,header=True)

            # Transformating using preprocessor obj
            input_feature_train_arr=pd.DataFrame(preprocessing_obj.fit_transform(input_feature_train_df),columns=preprocessing_obj.get_feature_names_out())
            input_feature_test_arr=pd.DataFrame(preprocessing_obj.transform(input_feature_test_df),columns=preprocessing_obj.get_feature_names_out())
            

            logging.info("Applying preprocessing object on training and testing datasets.")            

            train_arr = np.c_[input_feature_train_arr.to_numpy(), np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr.to_numpy(), np.array(target_feature_test_df)]            

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)