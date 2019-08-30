import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app

column1 = dbc.Col(
    [
        dcc.Markdown(
            """
        
            ## Process


            """
        ),

        html.Br(),

        dcc.Markdown(
            """

            #### __Gathering the Data__


            """
        ),

        html.Br(),

        dcc.Markdown(
            """
            Data was originally gathered from the CDC and converted into CSV files on [Kaggle](https://www.kaggle.com/cdc/mortality). Due to the 
            combined size of all the files, this project focused solely on data from 2015.

            """
        ),

        html.Br(),

        dcc.Markdown(
            """
            #### __Processing the Data__
            """
        ),

        html.Br(),

        dcc.Markdown(
            """
            Even with our focus on just one year's worth of the total data, the file size exceeded 1.5 GB, so a smaller sample was drawn randomly to create 
            an even smaller subset of the 2015 data:

            ```python
            # Select sample from data to help with memory issues
            df =  df.sample(700000, random_state=42)
            ```

            Once we had our smaller subset, the next step was to wrangle the data. After a lot of trial and error, the following function was created to clean 
            the dataset in one fell swoop:

            ```python
            # Create function to wrangle new features
            def wrangle(X):
    
                # Make copy of X so as not to contaminate original
                X = X.copy()
    
                # Create heart_disease feature (did person die of heart disease?)
                X['heart_disease'] = (X['39_cause_recode'].str.contains('heart')) | (X['39_cause_recode'].str.contains('heart'))
                
                # Create feature for cancer (keyword 'malignant neoplasm')
                X['cancer'] = ((X['39_cause_recode'].str.contains('Malignant neoplasm')) | 
                               (X['39_cause_recode'].str.contains('malignant neoplasm')) | 
                               (X['39_cause_recode'].str.contains('Leukemia')))
                
                # Create feature for HIV
                X['hiv'] = (X['39_cause_recode'].str.contains('HIV'))
                
                # Create feature for suicide
                X['suicide'] = (X['39_cause_recode'].str.contains('suicide'))
                
                # Create feaure for homicide
                X['homicide'] = (X['39_cause_recode'].str.contains('homicide'))
                
                # Create feature for accidents
                X['accident'] = (X['39_cause_recode'].str.contains('accident'))
                
                # Create feature for Alzheimer's
                X['alzheimers'] = (X['39_cause_recode'].str.contains("Alzheimer's"))
                
                # Create feature for respiratory
                X['respiratory'] = ((X['39_cause_recode'].str.contains('respiratory')) | 
                                    (X['39_cause_recode'].str.contains('Respiratory')))
                
                # Create feature for diabetes
                X['diabetes'] = (X['39_cause_recode'].str.contains('Diabetes'))
    
                # Replace 999 value for age with np.nan
                X['detail_age'] = X['detail_age'].replace(999, np.nan)
                
                # Fill na with mean age
                X['detail_age'] = X['detail_age'].fillna(round(X['detail_age'].mean()))
    
                # Convert our engineered features to int
                eng_feats = ['heart_disease', 'cancer', 'hiv', 'suicide', 'homicide', 
                             'accident', 'alzheimers', 'respiratory', 'diabetes']
                X[eng_feats] = X[eng_feats].astype(int)
    
                # Fill education_level NaNs with Unknown label
                X['education_level'] = X['education_level'].fillna('Unknown')
    
                # Fill activity_code NaNs with Not applicable
                X['activity_code'] = X['activity_code'].fillna('Not applicable')
    
                # Drop columns with supermajority NaNs, duplicates, and others that had < 0
                # permutation importance scores from earlier trials
                X = X.drop(columns=['place_of_injury_for_causes_w00_y34_except_y06_and_y07_', 
                                    '130_infant_cause_recode', 'race_recode_3', 
                                    'race_recode_5', 'age_bin', 'hispanic_originrace_recode', 
                                    'day_of_week_of_death', 'detail_age_type', 'infant_age_recode_22'])
  
                # Drop 39_cause_recode column to avoid leakage
                 X = X.drop(columns='39_cause_recode')
    
                return X
            ```

            For this function, special attention was paid to the column titled `39_cause_recode`. Since the goal of this project was to predict gender of the deceased, 
            understanding _how_ the person died would be beneficial to the model -- especially if the model could predict gender-based patterns in certain diseases. 
            However, `39_cause_recode` provided _too_ much information, and therefore presented a risk of leakage. For example, let's assume a person in our sample died of cancer. 
            `39_cause_recode` would not only tell us this fact, but also the _type_ of cancer the individual succumbed to. This quickly became an issue because certain
            types of cancer are dependent on a person's gender (a person dying of ovarian cancer will always be female). To avoid our model picking up on this nuance, broad, 
            boolean features were engineered, where the value would be `True` if the person died, for example, of cancer, and `False` otherwise. Once the features were engineered, 
            column `39_cause_recode` was dropped.
            """
        ),

        dcc.Markdown(
            """
            Additionally, many columns were dropped. These columns were dropped either because their values were too highly correlated with other columns, or as a result of 
            running permutation importance tests. Also important to note is how the function replaced `detail_age` values of 999 with the mean age of all samples in the 
            dataset. Choosing how to fill these values was difficult, since the samples with 999 for age seemed to be either infants or foreign nationals that happened to 
            pass away in the United States. However, since there were roughly 260 of these instances (out of 700,000), and the infant instances were a minority of these outliers, 
            we deemed that replacing these values with the mean of the total sample would not impact the model in a negative manner.
            """
        ),

        html.Br(),

        dcc.Markdown(
            """
            #### __Generating the Model__
            """
        ), 
        html.Br(),

        dcc.Markdown(
            """
            The data was split using the hold-out method (a train, validation, test split). This method was determined due to the fact that our sample size was quite large. The 
            split among the train, validation, and test sets was 60-20-20%.
            """
        ),

        dcc.Markdown(
            """
            The model of choice for this task was the `XGBClassifier`. We wrapped the classifier in a pipeline so that the categorical variables would be encoded prior to being fit to the 
            model. After performing a `RandomizedSearchCV` to find our optimal hyper parameters, the final pipeline looked like:

            ```python
            pipeline = make_pipeline(
                ce.OrdinalEncoder(),
                XGBClassifier(
                    boost='gbtree',
                    n_estimators=469,
                    random_state=42,
                    n_jobs=-1,
                    learning_rate=0.3,
                    max_depth=4
                )
            )
            ```


            """
        ),

        html.Br(),

        dcc.Markdown(
            """
            #### __Scoring the Model__
            """
        ), 

        html.Br(),

        dcc.Markdown(
            """
            For this model, scikit-learn's `roc_auc_score` was the metric we used to determine how well it performed. A simple accuracy score was avoided because the data was evenly split, 
            across almost all features in the dataset (there was a lot of overlap). Therefore, looking at the trade-off between our True Positive and False Positive rates would prove more beneficial. 
            Additionally, because we used the `roc_auc_score` as our metric, we had to redefine the aim of our model. Instead of asking "Based on the data we have, what is the gender of the person 
            in question?", we asked "Based on the data we have, _what is the probability that the gender is male_?"
            """
        ),

        dcc.Markdown(
            """
            We chose 'Male' as the positive class simply because our binary classifiers' `predict_proba` module assigns indexes alphabetically (if not already specified). Since the values at index `[1]` 
            are the positive class, and `M` comes after `F` in the alphabet, it's how things worked out. Once we calculated the predicted probability of our samples being male, we then got our ROC 
            AUC Score:

            ```python
            # Get ROC AUC score for the model
            y_pred_proba = model.predict_proba(X_val_encoded)[:,1]
            roc_auc_score(y_val, y_pred_proba)

            >>>0.737818456968959
            ```

            Our classification report for the model was as follows:

            ```python
                          precision    recall  f1-score   support

                  Female       0.70      0.62      0.66     55166
                    Male       0.67      0.74      0.70     56834

                accuracy                           0.68    112000
               macro avg       0.69      0.68      0.68    112000
            weighted avg       0.69      0.68      0.68    112000
            ```
            And our confusion matrix was as such:
            """
        ),

        html.Img(src='assets/Model-confusion-matrix.png', className='img-fluid', style={'height':'300px','width':'500px'}),

        html.Br(),
        html.Br(),

        dcc.Markdown(
            """
            #### __Conclusion__
            """
        ),

        html.Br(),

        dcc.Markdown(
            """
            While this model significantly outperforms a baseline assumption of simply guessing, there is still a lot to be desired. However, one big takeaway from this experiment is 
            that life is unpredictable. Though there are features that may [sway a prediction in a given direction](https://gendermortality.herokuapp.com/insights), ultimately death 
            and/or disease do not generally target a specific gender. _Specific_ diseases target specific genders, and death targets everyone. Based on the constraints presented by the 
            available data, the model performed reasonably well. One interesting way to possibly imporve model performance, aside from running the entire file set provided in Kaggle would 
            be to get more specific demographic data (by state, county, city) and see how those values relate to the other features in the model.

            """


        )



    ],
)

layout = dbc.Row([column1])