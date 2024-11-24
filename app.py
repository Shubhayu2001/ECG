from flask import Flask, render_template, request, jsonify
import joblib
import os

app = Flask(__name__)

# Correct model path
MODEL_PATH = os.path.join('model', 'model.pkl')  # Ensure 'model.pkl' is in the 'model' folder
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Expect JSON with "features"
        features = data['features']
        prediction = model.predict([features])
        probability = model.predict_proba([features])
        return jsonify({
            'status': 'success',
            'prediction': prediction.tolist(),
            'probability': probability.tolist()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
