from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler



model = joblib.load('dtree_model.pkl')
uploaded_file = None
processed_data = None

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    global uploaded_file
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            # df.to_csv('temp_file.csv', index=False)
            uploaded_file = df

            return redirect(url_for('preprocessing'))
    return render_template('index.html')


@app.route('/preprocessing')
def preprocessing():
    global uploaded_file, processed_data
    df = uploaded_file
    # df = pd.read_csv('temp_file.csv')
    preprocessing_steps = []

    # Cleaning input data
    preprocessing_steps.append("Cleaning Input Data.")
    time.sleep(3)

    # Replace infinite values with NaN
    preprocessing_steps.append("Replacing +ve and -ve Infinity with NaN")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop missing values
    preprocessing_steps.append("Dropping Missing Values")
    df.dropna(inplace=True)

    duplicate = df.duplicated().sum()
    preprocessing_steps.append(f"Number of Duplicate Values :{duplicate}")
    # print(f'\nThe number of duplicate values :{duplicate}')
    df.drop_duplicates(inplace=True)
    duplicate_left = df.duplicated().sum()
    preprocessing_steps.append(f"Duplicate Values Left :{duplicate_left}")

    # Drop unnecessary columns
    preprocessing_steps.append("Dropping Unnecessary Columns")
    time.sleep(2)
    columns_to_drop = ['dest_ip', 'src_ip', 'time_start', 'time_end']
    dff = df.drop(columns=columns_to_drop)
    df = df[df['label'] != 'outlier']


    # Feature scaling
    time.sleep(2)
    preprocessing_steps.append("Feature Scaling.")
    min_max_scaler = MinMaxScaler().fit(df[['avg_ipt', 'bytes_in', 'bytes_out', 'dest_port', 'entropy',
                                            'num_pkts_out', 'num_pkts_in', 'src_port',
                                            'total_entropy', 'duration']])

    scaled_columns = ['avg_ipt', 'bytes_in', 'bytes_out', 'dest_port', 'entropy',
                      'num_pkts_out', 'num_pkts_in', 'src_port',
                      'total_entropy', 'duration']
    df.loc[:, scaled_columns] = min_max_scaler.transform(df[scaled_columns])
    preprocessing_steps.append("Everything Done")

    # df.to_csv('new-file.csv', index=False, sep=',')
    new_file = df
    processed_data = new_file
    return render_template('preprocessing.html', preprocessing_steps=preprocessing_steps)


@app.route('/result')
def result():
    global  processed_data
    df = processed_data

    # df = pd.read_csv('new-file.csv')
    X_new = df[df.columns[:-2]].values
    y_new = df[df.columns[-2]].values
    predictions = model.predict(X_new)
    predictions = predictions.astype(int)

    percentage_malicious = (predictions.sum() / len(predictions)) * 100
    percentage_benign = 100 - percentage_malicious
    return render_template('result.html', benign_percentage=percentage_benign, malicious_percentage=percentage_malicious)





if __name__ == '__main__':
    app.run(debug=True)
