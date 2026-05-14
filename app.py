from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model once when server starts
model = joblib.load("artifacts/titanic_model.pkl")

@app.route("/")
def home():
    return jsonify({"message": "Titanic Model API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    # Get JSON data sent by user
    data = request.get_json()

    age  = data["age"]
    fare = data["fare"]

    # Make prediction
    input_df = pd.DataFrame({"Age": [age], "Fare": [fare]})
    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return jsonify({
        "age":             age,
        "fare":            fare,
        "prediction":      int(prediction),
        "result":          "Survived" if prediction == 1 else "Did not survive",
        "survival_chance": f"{probability * 100:.1f}%"
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)