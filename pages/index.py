import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


from app import app

"""
https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout

Layout in Bootstrap is controlled using the grid system. The Bootstrap grid has 
twelve columns.

There are three main layout components in dash-bootstrap-components: Container, 
Row, and Col.

The layout of your app should be built as a series of rows of columns.

We set md=4 indicating that on a 'medium' sized or larger screen each column 
should take up a third of the width. Since we don't specify behaviour on 
smaller size screens Bootstrap will allow the rows to wrap so as not to squash 
the content.
"""

csv_content = 'https://raw.githubusercontent.com/Nburkhal/US-Mortality-Project/master/cleaned_df.csv'
df = pd.read_csv(csv_content, encoding = 'ISO-8859-1')
ct = pd.crosstab(df['sex'], df['marital_status'], normalize='index') * 100

column1 = dbc.Col(
    [
        dcc.Markdown(
            """
        
            ## Does Getting Married _Really_ Kill You?

            Of course not! But if a person's gender is unknown at their time of death, marital status _could_ indicate that they're probably male.

            What if we could predict the gender of the deceased based on demographic and generic medical information? What if males are more likely to die from heart disease, or are more likely to commit suicide? What if females are more prone to developing Alzheimer's?

            With this information, health providers, governments, and even individuals can take more targeted approaches toward promoting healthy lifestyles.

            This app predicts the probability of a person being male or female at their time of death based their age, marital status, education level, race, and specific disease contracted.

            """
        ),
        dcc.Link(dbc.Button('Try It Out', color='primary'), href='/predictions')
    ],
    md=4,
)

sex = ['Male', 'Female']

fig = go.Figure(data=[
    go.Bar(name='Married', x=sex, y=[ct.loc['Male','Married'], ct.loc['Female','Married']]),
    go.Bar(name='Divorced', x=sex, y=[ct.loc['Male','Divorced'], ct.loc['Female','Divorced']]),
    go.Bar(name='Widowed', x=sex, y=[ct.loc['Male','Widowed'], ct.loc['Female','Widowed']]),
    go.Bar(name='Never Married, single', x=sex, y=[ct.loc['Male','Never married, single'], 
                                                   ct.loc['Female','Never married, single']]),
    go.Bar(name='Marital Status unknown', x=sex, y=[ct.loc['Male','Marital Status unknown'], 
                                                    ct.loc['Female','Marital Status unknown']])
])
# Change the bar mode
fig.update_layout(barmode='stack', 
                  title_text='Proportion of Decedent Marital Statuses by Gender (%)',
                  width=700, 
                  height=500)

column2 = dbc.Col(
    [
        dcc.Graph(figure=fig),
    ]
)

layout = dbc.Row([column1, column2])