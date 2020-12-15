# CS675 Final Project: Covid-19 Incident Projection <br>

Team member: Ensheng Dong, Hongru Du, Huaizhong Zhang, Stefano Tusa <br>

Presentation slides: https://arcg.is/nLuaG


## Introduction:

A local outbreak of “pneumonia of unknown cause” detected in Wuhan, Hubei Province, China in late December 2019, has since spread to more than 190 countries, with more than 72 million cases as of December 14, 2020. The situation became more serious in places outside of China, such as the US, Europe, India and Brazil. In response to this ongoing public health emergency, we are planning to predict the incidence rate for each US state by considering factors including historical cases, county or state demographics, local public health facilities, community mobility, temperature and humidity for each county. Recent studies show that the purpose of trips and density of destination also play important roles to quantify transmission risk. We want to adopt the information from those generated variable to make predictions on daily incident in 1 to 4 weeks. While, we realize that there are a lot of models available in the CDC COVID projection, most of them are good at trend fitting. Thus some trend changes may not well projected. Our baseline for this project is make reasonable state level projections (output within the interval of CDC ensemble model) and optimal outcome is to capture those trend changing points. Unlike traditional SEIR transmission model (such as IHME) and time series models, we adopt LSTM networks, a special kind of RNN, to make more accurate incident projection. This project is aimed to predict the daily and weekly COVID-19 incident cases in the US at state level with Long Short-Term Memory (LSTM) Recurrent Neural Networks method. We set up two LSTM-RNN models, one to predict the daily new cases another for predicting future features. Models for cases prediction and feature prediction have the same netwrok frameworks, but different training set (cases prediction fit to cases data, while feature prediction fit to all other dependent variables). 


## How to run the code:

- Jupyter Notebook shares the whole story, data sets and sample outputs of this project.
- The `main_cases.py` script in the [Model directory](https://github.com/hongru94/CS675_final_project/tree/main/Model) contains the data sampling methods and a LSTM RNN model implemented with PyTorch library. 
- The [Prediction directory](https://github.com/hongru94/CS675_final_project/tree/main/Predictions) contains two trained models and also a notebook, which takes those two models as input and expand prediction to future 4 weeks. 
- The [Data directory](https://github.com/hongru94/CS675_final_project/tree/main/Data) contains some of the raw data we used for training this model. However, due to the unshareable datasets we adopted from the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University (JHU), if you want to test out the model yourself, please contact [Hongru Du](mailto:hd9@jhu.edu) to get access to the entire datasets. Some of the results and evaluation of the model are shown in the figures below.

# Method

![Image text](https://github.com/enshengdong/CS675_final_project/blob/main/method.PNG)   
Figure 1: LSTM RNN.<br>

# Split of train and test data sets:

![Image text](https://github.com/enshengdong/CS675_final_project/blob/main/split.png)   
Figure 2: Split of train and test data sets.<br>

## Sample outputs:

![Image text](https://github.com/arthurzhang434/CS675_final_project/blob/main/weekly.png)   
Figure 3: The predictions made to several states's weekly new cases based on our model.<br>
<br>
![Image text](https://github.com/arthurzhang434/CS675_final_project/blob/main/model_comparison_week46.png)   
Figure 4: The comparison of the mean absolute error on weekly new cases prediction from the groundtruth between our model and CDC ensemble model. <br>

## References:

- Chang, S., Pierson, E., Koh, P.W., Gerardin, J., Redbird, B., Grusky, D. and Leskovec, J., 2020. Mobility network models of COVID-19 explaininequities and inform reopening. Nature, pp.1-8.
- Rader, B., Scarpino, S.V., Nande, A., Hill, A.L., Adlam, B., Reiner, R.C., Pigott, D.M., Gutierrez, B., Zarebski, A.E., Shrestha, M. and Brownstein,J.S., 2020. Crowding and the shape of COVID-19 epidemics. Nature medicine, pp.1-6.
- The COVID-19 Forecast Hub. https://covid19forecasthub.org/.
- JHU CSSE GitHub. https://github.com/CSSEGISandData/COVID-19
- Badr, H.S., Du, H., Marshall, M., Dong, E., Squire, M.M. and Gardner, L.M., 2020.  Association between mobility patterns and COVID-19transmission in the USA: a mathematical modelling study. The Lancet Infectious Diseases, 20(11), pp.1247-1254.
- Ünlü, R. and Namlı, E., 2020.  Machine Learning and Classical Forecasting Methods Based Decision Support Systems for COVID-19.  CMC-COMPUTERS MATERIALS & CONTINUA, 64(3), pp.1383-1399.
- Kefayati, S., Huang, H., Chakraborty, P., Roberts, F., Gopalakrishnan, V., Srinivasan, R., Pethe, S., Madan, P., Deshpande, A., Liu, X. and Hu, J.,2020. On machine learning-based short-term adjustment of epidemiological projections of covid-19 in us. medRxiv.[8] https://www.cdc.gov/coronavirus/2019-ncov/cases-updates/forecasts-cases.html[9] The Institute for Health Metrics and Evaluation (IHME): https://covid19.healthdata.org/global
