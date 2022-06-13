import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = request.form.getlist('data')
    final_features = np.array(int_features).astype(float)
    num_cell = int(final_features.size/10)
    
    worst_list = []
    mean_list = []
    error_list = []

    for index in range(1,11):
        if index == 1:
            start = index - 1
        else: 
            start = (num_cell * index) - num_cell
        
        end = int(index * num_cell)
        data = final_features[start:end]

        worst_list.append(np.max(data))
        error_list.append(np.std(data, ddof=1)/np.sqrt(len(data)))
        mean_list.append(np.mean(data))
    
    parameter = np.concatenate((np.array(mean_list), np.array(error_list), np.array(worst_list)))
    parameter = parameter.reshape(1, -1)
    prediction = model.predict(parameter)
    if(prediction == 1):
        output = 'Malignant'
    else :
        output = 'Benign'

    return render_template('index.html', prediction_text='The Breast Cancer is {}'.format(output))
   
if __name__ == "__main__":
    app.run(debug=True)