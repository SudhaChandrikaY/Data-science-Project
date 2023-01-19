“Ensemble Based Approach for Effective Heart Disease Prediction”

Heart Disease is one of the major concerns for society for a long time. The term 
heart disease is often used interchangeably with the term cardiovascular disease 
(CVD). It generally refers to the conditions that involve narrowed or blocked blood 
vessels that can lead to heart attack, chest pain or stroke. Prediction of CVD is 
regarded as one of the most important subjects in clinical data analysis. The 
amount of data in the healthcare industry is huge. It is difficult to identify heart 
disease because of several contributory risk factors such as diabetes, high blood 
pressure, high cholesterol, abnormal pulse rate, and many other factors. It is 
difficult to manually determine the odds of getting heart disease based on risk 
factors. Due to such constraints, many have turned towards modern approaches 
like Data Mining and Machine Learning for predicting the disease. Machine 
learning techniques are useful to predict the output from existing data and that 
helps in effectively predicting the large quantities of data produced by the 
healthcare industry. In this work, we have used multiple Machine learning 
approaches to predict whether a person is suffering from a heart disease or not 
and then constructed an ensemble method for better accuracy, along with some 
visualization techniques to visualize the trends.
Dataset used: Cleveland Dataset (from UCI repository)
https://archive.ics.uci.edu/ml/datasets/heart+disease

DataSet Description :

The heart disease dataset folder comprises four different databases which include 
Cleveland, Hungary, Switzerland and VA Long Beach databases. Among these, 
our project is implemented based on the Cleveland database which is the only one 
that is mostly used by ML researchers. The Cleveland dataset is made of 303 
individuals' data. There are 14 columns in this data consisting of 13 features 
(relevant patient information) and 1 target attribute (presence or absence of heart 
disease in the patient). The data was collected by Robert Detrano, M.D., Ph.D. of 
the Cleveland Clinic Foundation. Below are the 13 features (patient information) 
taken into consideration from the dataset:
● age: The person's age in years
● sex: The person's sex (1 = male, 0= female)
● cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical 
angina, Value 3: non-anginal pain, Value 4: asymptomatic)
● trestbps: The person's resting blood pressure (mm Hg on admission to the 
hospital)
● chol: The person's cholesterol measurement in mg/dl
● fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
● restecg: Resting electrocardiographic measurement (0 = normal, 
1 = having ST-T wave abnormality, 2 = showing probable or definite left 
ventricular hypertrophy by Estes' criteria)
● thalach: The person's maximum heart rate achieved
● exang: Exercise induced angina (1 = yes; 0 = no)
● oldpeak: ST depression induced by exercise relative to rest ('ST' relates to 
positions on the ECG plot. See more here)
● slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 
2: flat, Value 3: downsloping) 
● ca: The number of major vessels (0-3)
● thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = 
reversible defect)


Implementation :

Visualization-

For better understanding of the dataset, Exploratory Data Analysis is an 
essential step. Data visualization can help understanding the relation among the 
features. To identify any missing values in columns we used the Missingno library. 
Another important preprocessing step achieved through visualization is to identify 
outliers. We identified outliers using box plots and later dropped them. Outliers are 
those data points in the box plot which lies outside of the whiskers. Later, we 
plotted a bar graph for each feature to understand their distribution. We found that 
age, trestbps and chol are normally distributed. For understanding the correlation 
between the features we generated a heatmap, from which we found that cp, 
thalach, slope had good positive correlation with target. Oldpeak, exang, ca, thal, 
sex, age had good negative correlation with target. Whereas, fbs chol, trestbps, 
restecg had low correlation with our target. To understand if there are any features 
that could be separated linearly, we plotted pair plots using seaborn. Only oldpeak 
had a clear linear separation relation between disease and non-disease, and 
thalach had mild separation. We also generated an automatic EDA report using 
pandas profiling. 

Machine Learning-

A single algorithm may not make the perfect prediction for a given data set. 
Machine learning algorithms have their limitations and producing a model with high 
accuracy is challenging. Therefore, we have used multiple machine learning 
algorithms to check and find the model which gives high accuracy for the heart 
disease data-set. If we build and combine multiple models, we have the chance to 
boost the overall accuracy. We then implement the combination of models by 
aggregating the output from each model using a voting classifier (Ensemble).

RESULTS:

Data Visualization -
Missing values identification, outliers’ identification and dropping, linearly 
separable features identification, finding correlation between features and feature 
distribution are implemented as a part of data visualization and shown as results 
in the google collab link.

ML models Used -
The following are the Machine Learning classification models used and their final 
accuracies in percentage
Logistic Regression 87.913
K Nearest Neighbor Classifier 90.109
Naïve Bayes Classifier 85.71
Support Vector Machine Classifier 87.91
Neural Network 89.0109
Random Forest 90.109

Ensemble Model -
We cannot always rely on the prediction from a single Machine Learning Algorithm 
when we are dealing with health-related datasets. The more precise the algorithm 
is, the better it fits the real time and hence we constructed an ensemble using the 
predictions from multiple machine learning algorithms for better accuracy. We have 
used a voting classifier as our ensemble method and the accuracy of the ensemble 
method is 91.21%.

CONCLUSION :

From our work throughout the project, we have learned the importance of data 
preprocessing and how it impacts the model training and its accuracy. We have 
also figured out that using an ensemble model boosts the overall accuracy. By 
combining multiple Machine Learning models using a voting classifier as our 
ensemble model, we have successfully shown that there is an improvement in the 
accuracy.


Project Execution Collab link:
https://colab.research.google.com/drive/1I9HGe2xRTN7SDygZvpt_lC23Xy5-92Ed?usp=sharing

