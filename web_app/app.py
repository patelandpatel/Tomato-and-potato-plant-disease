"""
Web application for the Student Performance Predictor.
Provides a user interface for making predictions.
"""

import os
import sys
import yaml
from flask import Flask, request, render_template, redirect, url_for, flash

from src.utils.logger import Log
from src.utils.exception_handler import CustomException
from src.models.prediction import PredictionPipeline, CustomData
from config.config import ConfigurationManager


# Initialize Flask app
app = Flask(__name__)
app.secret_key = "student_performance_predictor"


# Load web app configuration
def get_web_app_config():
    """
    Load web app configuration from params.yaml.
    
    Returns:
        Dictionary of web app configuration
    """
    try:
        with open("config/params.yaml", "r") as f:
            params = yaml.safe_load(f)
        return params["web_app"]
    except Exception as e:
        Log.error(f"Exception occurred while loading web app config: {str(e)}")
        return {
            "host": "0.0.0.0",
            "port": 5000,
            "debug": True
        }


@app.route('/')
def index():
    """
    Home page route.
    
    Returns:
        Rendered index.html template
    """
    try:
        return render_template('index.html')
    except Exception as e:
        Log.error(f"Exception occurred in index route: {str(e)}")
        return render_template('error.html', error=str(e))


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """
    Route for making predictions.
    
    Returns:
        Rendered home.html template with prediction results
    """
    try:
        if request.method == 'GET':
            return render_template('home.html')
        else:
            # Get form data
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )
            
            # Convert to DataFrame
            pred_df = data.get_data_as_dataframe()
            Log.info(f"Input data: {pred_df}")
            
            # Make prediction
            prediction_pipeline_config = ConfigurationManager().get_prediction_pipeline_config()
            prediction_pipeline = PredictionPipeline(config=prediction_pipeline_config)
            results = prediction_pipeline.predict(pred_df)
            
            # Round the prediction to 2 decimal places
            predicted_score = round(float(results[0]), 2)
            
            Log.info(f"Prediction: {predicted_score}")
            
            return render_template('result.html', prediction=predicted_score)
            
    except Exception as e:
        Log.error(f"Exception occurred in predict_datapoint route: {str(e)}")
        return render_template('error.html', error=str(e))


def run_web_app():
    """
    Run the Flask web application.
    
    Raises:
        CustomException: If the app fails to start
    """
    try:
        # Get web app configuration
        web_app_config = get_web_app_config()
        
        # Run the app
        Log.info(f"Starting web application on {web_app_config['host']}:{web_app_config['port']}")
        app.run(
            host=web_app_config['host'],
            port=web_app_config['port'],
            debug=web_app_config['debug']
        )
        
    except Exception as e:
        Log.error(f"Exception occurred while running web app: {str(e)}")
        raise CustomException(str(e), sys) from e


if __name__ == "__main__":
    run_web_app()