# CS675_final_project
Project name: Covid-19 Incident Projection <br>
<br>
Team member: Ensheng Dong, Hongru Du, Huaizhong Zhang, Stefano Tusa <br>
<br>
Introduction: This project is aimed to predict the daily and weekly COVID-19 incident cases in the US on state level with Long Short-Term Memory (LSTM) Recurrent Neural Networks method. The main_cases.py script in the Model directory contains the data pre-processing methods and a LSTM RNN model implemented with PyTorch library. The Data directory contains some of the raw data we used for training this model. However, due to the unshareable datasets we adopted from the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University (JHU), if you want to test out the model yourself, please contact xxx at xxx@xxx to get access to the entire datasets. Some of the results and evaluation of the model are shown in the figures below. <br>
<br>
![Image text](https://github.com/arthurzhang434/CS675_final_project/blob/main/weekly.png)   
Figure 1: The predictions made to several states's weekly new cases based on our model.<br>
<br>
![Image text](https://github.com/arthurzhang434/CS675_final_project/blob/main/model_comparison_week46.png)   
Figure 2: The comparison of the mean absolute error on weekly new cases prediction from the groundtruth between our model and CDC ensemble model. <br>
