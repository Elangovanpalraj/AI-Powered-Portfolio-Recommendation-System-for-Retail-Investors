from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("portfolio_model.pkl")
le = joblib.load("risk_encoder.pkl")
risk_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        salary = float(request.form['salary'])
        sip = float(request.form['sip'])
        risk = request.form['risk']

        # Encode risk
        risk_encoded = risk_mapping[risk]

        input_data = np.array([[age, salary, sip, risk_encoded]])
        prediction = model.predict(input_data)[0]

        return render_template('index.html',
                               prediction_text=f"Diversification: "
                                               f"Equity={prediction[0]:.2f}, "
                                               f"Bonds={prediction[1]:.2f}, "
                                               f"FD={prediction[2]:.2f}, "
                                               f"Real Estate={prediction[3]:.2f}")
    except Exception as e:
        return render_template('index.html', prediction_text="Error in input! Please check again.")

if __name__ == '__main__':
    app.run(debug=True)
