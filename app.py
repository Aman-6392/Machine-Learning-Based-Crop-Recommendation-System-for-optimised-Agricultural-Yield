from flask import Flask, request, render_template, abort
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def load_artifact(name):
    p = BASE_DIR / name
    if not p.exists():
        abort(500, f"Missing artifact: {name}. Train your notebook and save it here.")
    try:
        return joblib.load(p)
    except Exception as e:
        abort(500, f"Failed to load {name}: {e}")

# Load artifacts
model = load_artifact('model.pkl')
ms = load_artifact('minmaxscaler.pkl')      # MinMaxScaler (fit during training)
sc = load_artifact('standscaler.pkl')       # StandardScaler (fit during training)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])
    except Exception as e:
        abort(400, f"Invalid input: {e}")

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single = np.array(feature_list, dtype=float).reshape(1, -1)

    # The transform order must match training: MinMax -> Standardize
    single = ms.transform(single)
    single = sc.transform(single)
    pred = model.predict(single)

    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    result = "Sorry, we could not determine the best crop."
    key = int(pred[0]) if hasattr(pred, '__iter__') else int(pred)
    if key in crop_dict:
        result = f"{crop_dict[key]} is the best crop to be cultivated right there"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
