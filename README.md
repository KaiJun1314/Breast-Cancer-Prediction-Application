# Breast Cancer Prediction Application
Breast cancer is a type of disease in which cells in the breast grow and divide abnormally, creating a mass of tissue called a tumor.  
- If the cells are normal cells, the tumor is called benign (non-cancerous).
- If the cells are abnormal and do not function like the body's normal cells, the tumor is called malignant (cancerous). 

The accurate classification between benign and malignant tumors has become the trending topic in many studies as it could prevent the patients from receiving unnecessary treatment. However, the laboratory test to determine the type of tumor are time consuming and cost consuming. In this project, we are building a machine learning model which adapt the logistics regression algorithm to predict the type of tumor

## File
1. model_training.py : trains and saves the model to the disk. 
2. model.pkb : the pickle model
3. app.py : contains all the requiered for flask and to manage APIs.
4. wdbc.data: Breast Cancer Dataset
5. wdbc.names: Metadata of Breast Cancer Dataset

## Dataset Source
UCI Machine Learning Repository \
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

## Performance Measurement
- Sensitivity : 0.971 
- Specificity : 0.978 
- Precision : 0.985
- Recall : 0.971 
- Accuracy : 0.974 

**ROC Curve**
<p align="center">
  <img src=assests/ROC.png alt="ROC Curve" width="80%" height="30%">
</p>

## User Interface
<p align="center">
  <img src=assests/UserInterface.png alt="ROC Curve" width="80%" height="30%">
</p>

## Procedure
1. Install the required library
	```
   pip install -r requirements.txt 
   ```
2. Run python app.py
3. Visit 127.0.0.1:5000 address using web browser
