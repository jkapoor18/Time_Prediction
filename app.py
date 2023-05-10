from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline
from src.pipeline.training_pipeline import train_model


application=Flask(__name__)

app=application


@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/train',methods=['GET'])
def training_page():
    results = train_model()
    return render_template('trainingcomplete.html',final_result=results)

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            Delivery_person_Age=float(str(request.form.get('Delivery_person_Age')).strip()),
            Delivery_person_Ratings = float(str(request.form.get('Delivery_person_Ratings')).strip()),
            Restaurant_Delivery_distance = float(str(request.form.get('Restaurant_Delivery_distance')).strip()),
            preparation_time = float(str(request.form.get('preparation_time')).strip()),
            Weather_conditions = str(request.form.get('Weather_conditions')).strip(),
            Road_traffic_density = str(request.form.get('Road_traffic_density')).strip(),
            Vehicle_condition = int(str(request.form.get('Vehicle_condition')).strip()),
            Type_of_vehicle = str(request.form.get('Type_of_vehicle')).strip(),
            multiple_deliveries = float(str(request.form.get('multiple_deliveries')).strip()),
            Festival = str(request.form.get('Festival')).strip(),
            City = str(request.form.get('City')).strip()
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('results.html',final_result=results)

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)