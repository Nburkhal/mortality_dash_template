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
        
            ## Insights


            """
        ),

        html.Br(),

        dcc.Markdown(
            """
            #### __How the Model Works__
            """
        ),

        html.Br(),

        dcc.Markdown(
            """
            While documenting our [process](https://gendermortality.herokuapp.com/process), we saw how well the model performed, but what exactly is going on 
            under the hood? What exactly _causes_ the model to make a certain prediction?
            """
        ),

        dcc.Markdown(
            """
            Taking a quick look at the model's permutation importances, we can see which features carry the most weight in the model's determination of gender:
            """
        ),

        html.Img(src='assets/permutation-importance.png', className='img-fluid'),

        dcc.Markdown(
            """
            Here we see that marital status holds the most weight when determining whether a person is male or female. And intuitively, this makes sense. On average, 
            women live longer than men, so if a person is still married at their time of death (assuming a natural death, of course), then chances are good that they're 
            male. Another interesting point is that a person's education level plays a significant role in the model's determination of gender. What's fascinating about 
            this information is that it is, in a sense, a historical snapshot of how gender roles were shaped in the past. An interactive partial dependency plot of 
            education level and marital status will help make this point clear:
            """
        ),

        html.Img(src='assets/ms-edu-pdp-interact.png', className='img-fluid'),

        dcc.Markdown(
            """
            In this plot, you can see that as education level increases (from top to bottom on the visualization), the probability that the person is male increases, despite 
            their marital status. This is also intuitive, as "back in the old days" men were far more likely to pursue higher education than women. But aside from the historical 
            snapshot, this graph also does a fantastic job of illustrating the wieght our model puts on marital status. A married person is overwhelmingly likely to be male, while 
            a widowed person is overwhelmingly likely to be female.
            """
        ),

        dcc.Markdown(
            """
            Let's take a look at a couple more partial dependency plots to see how marital status interacts with other features.
            """

        ),

        html.Img(src='assets/ms-disp-pdp-interact.png', className='img-fluid'),

        html.Img(src='assets/ms-place-pdp-interact.png', className='img-fluid'),

        dcc.Markdown(
            """
            Here again, we see a familiar pattern. Marital status far outweighs all other features in determining a person's gender, according to our model.
            """
        ),

        html.Br(),

        dcc.Markdown(
            """
            #### __Individual Predictions__
            """
        ),

        html.Br(),

        dcc.Markdown(
            """
            Now let's take a look at how the model scored individual predictions, using shapley force plots. We'll start with a correct prediction of 'Male':
            """
        ),

        html.Img(src='assets/male-right.png', className='img-fluid'),

        dcc.Markdown(
            """
            These kind of graphs may look tricky, but they're really simple to understand. The red shows "scores" for our positive class (Male), while blue indicate 
            scores for Female. A score above the base value of 0.03177 means the model predicts male, while a score below the base value means a prediction of female.
            """
        ),

        dcc.Markdown(
            """
            In this example, we see that marital status provided the biggest boost in favor of Male, though this person happened to be single. The most interesting 
            part of this graphic, though, is that the model tends to weight in favor of female if the person _did not_ die of heart disease. It's a very small score 
            in this example, but if you play around with the [prediction app](https://gendermortality.herokuapp.com/predictions), you'll notice that you'll be able to 
            tip the scale in favor of male if you select heart disease as the cause of death.
            """
        ),

        dcc.Markdown(
            """
            Now let's look at an example where the model was _wildly_ wrong:
            """
        ),

        html.Img(src='assets/female-wrong.png', className='img-fluid'),

        dcc.Markdown(
            """
            For this prediction, the model was overwhelmingly certain that the person was male. On the contrary, she was female. Here we see that the model weighed heavily in 
            favor of the positive class if the manner of death was a homicide. And again, we see that the model weighs being single in favor of males.
            """
        )




    ],
    md=4,
)


layout = dbc.Row([column1])