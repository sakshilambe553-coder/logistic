import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained scikit-learn model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Features expected by the model in exact order
FEATURES = [
    "age", "gender", "city", "bmi", "family_history_diabetes",
    "physical_activity_level", "diet_type", "smoking_status",
    "alcohol_consumption", "hours_sleep_per_night", "stress_level",
    "fasting_blood_sugar", "hba1c_level", "blood_pressure_systolic",
    "blood_pressure_diastolic", "waist_circumference_cm", "income_bracket"
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Health Risk Assessment</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --input-bg: #334155;
            --accent: #38bdf8;
            --accent-hover: #0284c7;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #475569;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 850px;
            background: var(--card-bg);
            border-radius: 16px;
            padding: 2.5rem;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
            border: 1px solid var(--border-color);
        }

        header {
            margin-bottom: 2rem;
            text-align: center;
        }

        header h1 {
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent);
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: capitalize;
        }

        input {
            background: var(--input-bg);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.2s ease;
        }

        input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
        }

        .button-group {
            grid-column: 1 / -1;
            margin-top: 1rem;
        }

        button {
            width: 100%;
            background: var(--accent);
            color: #0f172a;
            border: none;
            padding: 0.9rem;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s ease;
        }

        button:hover {
            background: var(--accent-hover);
            color: #ffffff;
        }

        .result-box {
            grid-column: 1 / -1;
            margin-top: 1.5rem;
            padding: 1.25rem;
            border-radius: 10px;
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid var(--accent);
            text-align: center;
        }

        .result-box h2 {
            font-size: 1.2rem;
            color: var(--text-muted);
            margin-bottom: 0.25rem;
        }

        .result-box .prediction {
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent);
        }
    </style>
</head>
<body>

<div class="container">
    <header>
        <h1>Health Risk Assessment</h1>
        <p>Enter patient parameters to predict risk level (High, Moderate, Low)</p>
    </header>

    <form method="POST" action="/predict">
        {% for feature in features %}
        <div class="form-group">
            <label for="{{ feature }}">{{ feature.replace('_', ' ') }}</label>
            <input 
                type="number" 
                step="any" 
                id="{{ feature }}" 
                name="{{ feature }}" 
                value="{{ request.form.get(feature, '') }}" 
                required>
        </div>
        {% endfor %}

        <div class="button-group">
            <button type="submit">Predict Risk Category</button>
        </div>
    </form>

    {% if prediction %}
    <div class="result-box">
        <h2>Predicted Risk Level</h2>
        <div class="prediction">{{ prediction }}</div>
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, features=FEATURES, prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features from form in expected order
        input_data = [float(request.form[feature]) for feature in FEATURES]
        
        # Reshape for prediction
        array_data = np.array(input_data).reshape(1, -1)
        
        # Predict using LogisticRegression model
        prediction = model.predict(array_data)[0]
        
        return render_template_string(
            HTML_TEMPLATE, 
            features=FEATURES, 
            prediction=prediction
        )
    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE, 
            features=FEATURES, 
            prediction=f"Error processing input: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)
