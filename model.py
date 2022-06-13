# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Importing the dataset
dataset = pd.read_csv('wdbc.data', header = None)
dataset.columns = ['ID number','Diagnosis','mean radius','mean texture','mean perimeter','mean area','mean smoothness','mean compactness','mean concavity',
                   'mean concave points','mean symmetry','mean fractal dimension','radius error','texture error','perimeter error','area error',
                   'smoothness error','compactness error','concavity error','concave points error','symmetry error','fractal dimension error',
                   'worst radius','worst texture','worst perimeter','worst area','worst smoothness','worst compactness','worst concavity',
                   'worst concave points','worst symmetry','worst fractal dimension']
dataset = dataset.drop(columns= "ID number")
X = dataset.drop(columns='Diagnosis', axis=1)
Y = dataset['Diagnosis']

#Encoding categorical data values
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Simple Linear Regression to the Training set
model = LogisticRegression()
model.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)

# Saving model using pickle
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load( open('model.pkl','rb'))
print(model.predict([[13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]]))
