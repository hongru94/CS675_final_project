# CS675_final_project

##Project name: Covid-19 Incident Projection <br>

<br>
Team member: Ensheng Dong, Hongru Du, Huaizhong Zhang, Stefano Tusa <br>
<br>

##Introduction: 

This project is aimed to predict the daily and weekly COVID-19 incident cases in the US at state level with Long Short-Term Memory (LSTM) Recurrent Neural Networks method. We set up two LSTM-RNN models, one to predict the daily new cases another for predicting future features. Models for cases prediction and feature prediction have the same netwrok frameworks, but different training set (cases prediction fit to cases data, while feature prediction fit to all other dependent variables). The main_cases.py script in the [Model directory](https://github.com/hongru94/CS675_final_project/tree/main/Model) contains the data sampling methods and a LSTM RNN model implemented with PyTorch library. The [Prediction directory](https://github.com/hongru94/CS675_final_project/tree/main/Predictions) contains two trained models and also a notebook, which takes those two models as input and expand prediction to future 4 weeks. The [Data directory](https://github.com/hongru94/CS675_final_project/tree/main/Data) contains some of the raw data we used for training this model. However, due to the unshareable datasets we adopted from the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University (JHU), if you want to test out the model yourself, please contact xxx at xxx@xxx to get access to the entire datasets. Some of the results and evaluation of the model are shown in the figures below. <br>
<br>

![Image text](https://github.com/arthurzhang434/CS675_final_project/blob/main/weekly.png)   
Figure 1: The predictions made to several states's weekly new cases based on our model.<br>
<br>
![Image text](https://github.com/arthurzhang434/CS675_final_project/blob/main/model_comparison_week46.png)   
Figure 2: The comparison of the mean absolute error on weekly new cases prediction from the groundtruth between our model and CDC ensemble model. <br>

##Reference
