import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 Delivery_person_Age:float,
                 Delivery_person_Ratings:float,
                 Restaurant_Delivery_distance:float,
                 preparation_time:float,
                 Weather_conditions:str,
                 Road_traffic_density:str,
                 Vehicle_condition:int,
                 Type_of_vehicle:str,
                 multiple_deliveries:float,
                 Festival:str,
                 City:str):
        
        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings = Delivery_person_Ratings
        self.Restaurant_Delivery_distance = Restaurant_Delivery_distance
        self.preparation_time = preparation_time
        self.Weather_conditions = Weather_conditions
        self.Road_traffic_density = Road_traffic_density
        self.Vehicle_condition = Vehicle_condition
        self.Type_of_vehicle = Type_of_vehicle
        self.multiple_deliveries = multiple_deliveries
        self.Festival = Festival
        self.City = City

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Delivery_person_Age':[self.Delivery_person_Age],
                'Delivery_person_Ratings':[self.Delivery_person_Ratings],
                'Restaurant_Delivery_distance':[self.Restaurant_Delivery_distance],
                'preparation_time':[self.preparation_time],
                'Weather_conditions':[self.Weather_conditions],
                'Road_traffic_density':[self.Road_traffic_density],
                'Vehicle_condition':[self.Vehicle_condition],
                'Type_of_vehicle':[self.Type_of_vehicle],
                'multiple_deliveries':[self.multiple_deliveries],
                'Festival':[self.Festival],
                'City':[self.City]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
        
'''if __name__=='__main__':
    data=CustomData(
                 Delivery_person_Age=36.0,
                 Delivery_person_Ratings=4.2,
                 Restaurant_Delivery_distance=10.27,
                 preparation_time=15.0,
                 Weather_conditions='Sunny',
                 Road_traffic_density='Jam',
                 Vehicle_condition=2,
                 Type_of_vehicle='motorcycle',
                 multiple_deliveries=1.0,
                 Festival='No',
                 City='Metropolitian'
            )
    final_new_data=data.get_data_as_dataframe()
    predict_pipeline=PredictPipeline()
    pred=predict_pipeline.predict(final_new_data)
    results=round(pred[0],2)
    print(results)'''