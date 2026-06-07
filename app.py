from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline

app = Flask(__name__) # initializing a flask app

@app.route('/', methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train', methods=['GET'])  # route to train the pipeline
def training():
    try:
        os.system("python main.py")
        return "Training Successful!" 
    except Exception as e:
        return f"Training Failed: {str(e)}"


@app.route('/predict', methods=['POST', 'GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            # 1. Read the inputs given by the user and structure them as a dictionary
            data_dict = {
                'fixed_acidity': [float(request.form['fixed_acidity'])],
                'volatile_acidity': [float(request.form['volatile_acidity'])],
                'citric_acid': [float(request.form['citric_acid'])],
                'residual_sugar': [float(request.form['residual_sugar'])],
                'chlorides': [float(request.form['chlorides'])],
                'free_sulfur_dioxide': [float(request.form['free_sulfur_dioxide'])],
                'total_sulfur_dioxide': [float(request.form['total_sulfur_dioxide'])],
                'density': [float(request.form['density'])],
                'pH': [float(request.form['pH'])],
                'sulphates': [float(request.form['sulphates'])],
                'alcohol': [float(request.form['alcohol'])]
            }
            
            # 2. Convert dictionary directly to a 2D Numpy Array to bypass ALL feature name/order conflicts
            # Scikit-Learn will now strictly look at the values, completely ignoring column name mismatches!
            data_array = np.array([list(data_dict.values())]).reshape(1, -1)
            
            obj = PredictionPipeline()
            predict = obj.predict(data_array)

            # 3. Format output cleanly for display
            output_score = round(predict[0], 2)

            return render_template('results.html', prediction=str(output_score))

        except Exception as e:
            print('The Exception message is: ', e)
            return f"Something went wrong during model evaluation: {str(e)}"

    else:
        return render_template('index.html')


if __name__ == "__main__":
    # CRITICAL AZURE FIX: Pull the dynamic port allocated by Azure Router, fallback to 8080 locally
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)