from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app = application

##Route for the home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods = ['GET','POST'])
def predict_datapoints():
    if request.method =='GET':
        return render_template('home.html')
    else:
        #Code for post method.
        data = CustomData