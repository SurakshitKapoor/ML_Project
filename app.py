
from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

app = Flask(__name__)

# home route
@app.route('/')
def home():
    return render_template('index.html')

# prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict_data():
    if request.method == 'GET':
        return render_template("form.html")
    
    # fetching data from form
    gender = request.form['gender']
    race = request.form['race_ethnicity']
    parent_edu = request.form['parental_level_of_education']
    lunch = request.form['lunch']
    test_prep = request.form['test_preparation_course']
    reading = int(request.form['reading_score'])
    writing = int(request.form['writing_score'])

    print(f"Got: {gender}, {race}, {parent_edu}, {lunch}, {test_prep}, {reading}, {writing}")

    # create CustomData object
    data = CustomData(
        gender=gender,
        race_ethnicity=race,
        parental_level_of_education=parent_edu,
        lunch=lunch,
        test_preparation_course=test_prep,
        reading_score=reading,
        writing_score=writing
    )

    final_data_frame = data.get_data_as_dict()
    print("Final customised data as dataframe:\n", final_data_frame)

    # prediction
    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict(final_data_frame)    
    print("The Prediction is:", pred)

    logging.info("Got the predicted result into Flask application")

    return render_template("results.html", prediction=pred)


# main entry
if __name__ == "__main__":
    print("Running the Flask Application from app.py")
    app.run(debug=True, port=8080)
