"""
Forms for the Student Performance Predictor web application.
"""

from flask_wtf import FlaskForm
from wtforms import (
    StringField, SelectField, FloatField, SubmitField
)
from wtforms.validators import (
    DataRequired, Length, NumberRange
)


class PredictionForm(FlaskForm):
    """
    Form for student performance prediction.
    """
    
    gender = SelectField(
        'Gender',
        choices=[
            ('', 'Select Gender'),
            ('male', 'Male'),
            ('female', 'Female')
        ],
        validators=[DataRequired()]
    )
    
    race_ethnicity = SelectField(
        'Race/Ethnicity',
        choices=[
            ('', 'Select Race/Ethnicity'),
            ('group A', 'Group A'),
            ('group B', 'Group B'),
            ('group C', 'Group C'),
            ('group D', 'Group D'),
            ('group E', 'Group E')
        ],
        validators=[DataRequired()]
    )
    
    parental_level_of_education = SelectField(
        'Parental Level of Education',
        choices=[
            ('', 'Select Parental Education'),
            ("associate's degree", "Associate's Degree"),
            ("bachelor's degree", "Bachelor's Degree"),
            ('high school', 'High School'),
            ("master's degree", "Master's Degree"),
            ('some college', 'Some College'),
            ('some high school', 'Some High School')
        ],
        validators=[DataRequired()]
    )
    
    lunch = SelectField(
        'Lunch Type',
        choices=[
            ('', 'Select Lunch Type'),
            ('standard', 'Standard'),
            ('free/reduced', 'Free/Reduced')
        ],
        validators=[DataRequired()]
    )
    
    test_preparation_course = SelectField(
        'Test Preparation Course',
        choices=[
            ('', 'Select Test Preparation'),
            ('none', 'None'),
            ('completed', 'Completed')
        ],
        validators=[DataRequired()]
    )
    
    reading_score = FloatField(
        'Reading Score',
        validators=[
            DataRequired(),
            NumberRange(min=0, max=100, message='Score must be between 0 and 100')
        ]
    )
    
    writing_score = FloatField(
        'Writing Score',
        validators=[
            DataRequired(),
            NumberRange(min=0, max=100, message='Score must be between 0 and 100')
        ]
    )
    
    submit = SubmitField('Predict Math Score')