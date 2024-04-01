from flask import Flask, request, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'

model = None
std_scalar = None
encoder = None

def initialize():

    global model
    global std_scalar
    global encoder

    model = tf.keras.models.load_model("model.keras")
    encoder = joblib.load('encoder.bin')
    std_scalar =  joblib.load('std_scaler.bin')

def preprocess_numerical_cols(df):
    numerical_cols = df.select_dtypes(exclude=["object"]).columns
    df[numerical_cols] = df[numerical_cols].fillna(0)
    return df

def preprocess_categorical_cols(df):
    categorical_cols = df.select_dtypes(include=["object"]).columns
    df[categorical_cols] = df[categorical_cols].replace('-', "None")
    df[categorical_cols] = df[categorical_cols].fillna("None")
    return df

def drop_columns(df, to_drop = ['id','attack_cat','label']):
    df.drop(to_drop,axis=1,inplace=True)
    return df

def standardize(df):
    scaler =  joblib.load('std_scaler.bin')
    df = scaler.fit_transform(df)
    return df

def preprocess_data(filename):

    sample_df = pd.read_csv(filename)
    sample_df = preprocess_numerical_cols(sample_df)
    sample_df = preprocess_categorical_cols(sample_df)
    sample_df = drop_columns(sample_df)

    categorical_columns = sample_df.select_dtypes(include=['object']).columns

    onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore', categories=encoder.categories_)
    sample_df_encoded = onehot_encoder.fit_transform(sample_df[categorical_columns])

    sample_df_numeric = sample_df.drop(categorical_columns, axis=1)
    sample_df_combined = np.hstack((sample_df_numeric, sample_df_encoded))
    sample_df_combined = standardize(sample_df_combined)

    return sample_df_combined

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)

            df = pd.read_csv(file_path)
            df = df[['proto','service','dur','state','sbytes', 'dbytes','sttl','dttl']]
            packets = df.to_dict(orient='records')
            session['filepath'] = file_path
            session['packets'] = packets  # Save the packets in session
            return redirect(url_for('index'))
    
    return render_template('index.html', packets=session.get('packets'))

@app.route('/predict', methods=['POST'])
def predict():
    
    filename = session.get('filepath')
    packets = session.get('packets')

    if filename and packets:
        preprocessed_data = preprocess_data(filename)  # You need to implement this
        preds = model.predict(preprocessed_data)
        predictions = []
        for p in preds:
           
            prediction = "Attack Packet" if p > 0.5 else "Normal Traffic"
            predictions.append(prediction)
        return render_template('index.html', packets=packets, predictions=predictions)
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    initialize()
    app.run(debug=True)