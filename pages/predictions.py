import dash
import dash_daq as daq
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from joblib import load
pipeline = load('assets/pipeline.joblib')

from app import app


# Import our dataframe
csv_content = 'https://raw.githubusercontent.com/Nburkhal/US-Mortality-Project/master/cleaned_df.csv'
df = pd.read_csv(csv_content, encoding = 'ISO-8859-1')

# Split into train & test sets first
# Seed for reproduceability
train, test = train_test_split(df, train_size=0.8, test_size=0.2, 
                                  stratify=df['sex'], random_state=42)

# Set X feature matrix and y target vector
target = 'sex'
X_train = train.drop(columns=target)
y_train = train[target]


column1 = dbc.Col(
    [
        dcc.Markdown(
            """
        
            ## Predictions


            """
        ),

        html.Br(),

    # Slider for Age
    dcc.Markdown('#### Age'),
    dcc.Slider(
        id='age',
        min=0,
        max=150,
        step=1,
        marks={i: '{}'.format(i) for i in range(0, 150, 10)},
        value=50
    ),

    html.Br(),
    html.Br(),

    # Drop down for marital status
    dcc.Markdown('#### Marital Status'),
    dcc.Dropdown(
        id='marital_status',
        options=[
            {'label': 'Married', 'value': 'Married'},
            {'label': 'Divorced', 'value': 'Divorced'},
            {'label': 'Widowed', 'value': 'Widowed'},
            {'label': 'Single', 'value': 'Never married, single'},
            {'label': 'Unknown', 'value': 'Marital Status unknown'}
        ],
        value='Married'
    ),

    html.Br(),

    # Drop down for education level
    dcc.Markdown('#### Education Level'),
    dcc.Dropdown(
        id='education_level',
        options=[
            {'label': '8th grade or less', 'value': '8th grade or less'},
            {'label': 'High School', 'value': 'high school'},
            {'label': 'College', 'value': 'college'},
            {'label': 'Graduate Degree', 'value': 'graduate'},
            {'label': 'Postgraduate Degree', 'value': 'postgraduate'}
        ],
        value='College'
    ),

    html.Br(),

    # Drop down for race
    dcc.Markdown('#### Race'),
    dcc.Dropdown(
        id='race',
        options=[
            {'label': 'White', 'value': 'White'},
            {'label': 'Black', 'value': 'Black'},
            {'label': 'Vietnamese', 'value': 'Vietnamese'},
            {'label': 'American Indian', 'value': 'American Indian (includes Aleuts and Eskimos)'},
            {'label': 'Filipino', 'value': 'Filipino'},
            {'label': 'Chinese', 'value': 'Chinese'},
            {'label': 'Japanese', 'value': 'Japanese'},
            {'label': 'Combined Asian/Pacific Islander', 'value': 'Combined other Asian or Pacific Islander, includes codes 18-68'},
            {'label': 'Asian Indian', 'value': 'Asian Indian'},
            {'label': 'Korean', 'value': 'Korean'},
            {'label': 'Samoan', 'value': 'Samoan'},
            {'label': 'Guamanian', 'value': 'Guamanian'},
            {'label': 'Hawaiian', 'value': 'Hawaiian (includes Part-Hawaiian)'},
            {'label': 'Asian (Other)', 'value': 'Other Asian or Pacific Islander in areas reporting codes 18-58'}
        ],
        value='White'
    ),

    html.Br(),

    # Drop down for cause of death
    dcc.Markdown('#### Cause of Death'),
    dcc.Dropdown(
        id='cause_of_death',
        options=[
            {'label': 'Heart Disease', 'value': 'heart_disease'},
            {'label': 'Cancer', 'value': 'cancer'},
            {'label': 'HIV', 'value': 'hiv'},
            {'label': 'Suicide', 'value': 'suicide'},
            {'label': 'Homicide', 'value': 'homicide'},
            {'label': 'Accident', 'value': 'accident'},
            {'label': 'Alzheimers', 'value': 'alzheimers'},
            {'label': 'Respiratory Failure', 'value': 'respiratory'},
            {'label': 'Diabetes', 'value': 'diabetes'},
            {'label': 'Natural', 'value': 'none'}
        ],
        value='Natural'
    ),

    
    ],
    md=4,
)

column2 = dbc.Col(
    [

    html.Div(id='prediction-gauge', style={'marginTop': '2.2em'}),

    html.Br(),

    html.Div(id='prediction-label', className='lead', style={'fontWeight': 'bold', 'fontSize': '25px', 'position': 'relative', 'left': '180px', 'bottom': '82px'})
        
    ]
)

layout = dbc.Row([column1, column2])

@app.callback(
    [Output('prediction-label', 'children'),
     Output('prediction-gauge', 'children')],
    [Input('age', 'value'),
     Input('marital_status', 'value'),
     Input('education_level', 'value'),
     Input('race', 'value'),
     Input('cause_of_death', 'value')]

)

def predict(age, marital_status, education_level, race, cause_of_death):
    
    # Create dataframe
    df = pd.DataFrame(     # Set generic entries based on most frequent values
        data=[[X_train['resident_status'].mode()[0], 
              X_train['month_of_death'].mode()[0],
              0,
              X_train['place_of_death_and_decedents_status'].mode()[0], 
              0,
              X_train['injury_at_work'].mode()[0],
              X_train['manner_of_death'].mode()[0], 
              X_train['method_of_disposition'].mode()[0],
              X_train['autopsy'].mode()[0],
              X_train['activity_code'].mode()[0],
              0,
              X_train['hispanic_origin'].mode()[0],
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        columns=['resident_status', 'month_of_death', 'detail_age', 'place_of_death_and_decedents_status', 
                 'marital_status', 'injury_at_work', 'manner_of_death', 'method_of_disposition', 'autopsy', 
                 'activity_code', 'race', 'hispanic_origin', 'education_level', 'heart_disease', 'cancer', 
                 'hiv', 'suicide', 'homicide', 'accident', 'alzheimers', 'respiratory', 'diabetes']
    )
    
    # Store input data in dataframe
    df['detail_age'] = age
    df['marital_status'] = marital_status
    df['education_level'] = education_level
    df['race'] = race
    
    # Assign 1 to column for cause_of_death, else 0
    if cause_of_death in df.columns:
        df[cause_of_death] = 1

    # Assign positive class and index number
    positive_class = 'Male'
    positive_class_index = 1 
    
    # Call model for prediction
    pred = pipeline.predict(df)
    predict = pred[0]
    
    # Get predicted probability
    pred_proba = pipeline.predict_proba(df)[0,positive_class_index]
    
    probability = pred_proba * 100
    if pred != positive_class:
        probability = 100 - probability

    probability = round(probability)
    
    # Return prediction and probability
    output1 = f'Gender is {probability:.0f}% likely to be {predict}\n'
    output2 = daq.Gauge(id='my-gauge',
                        value=probability,
                        label='      ',
                        max=100,
                        min=0,
                        size=400,
                        showCurrentValue=True,
                        units='%')
    return output1, output2