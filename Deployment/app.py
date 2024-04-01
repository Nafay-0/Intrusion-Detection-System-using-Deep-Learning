from flask import Flask, request, jsonify
import pandas as pd
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
# Load the TensorFlow model
model = load_model('model.keras')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a file is present in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        uploaded_file = request.files['file']
        print(uploaded_file)
        if uploaded_file:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(uploaded_file)
            # Perform prediction using the loaded model
            predictions = model.predict(df)
            predictions = predictions > 0.5

            return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
