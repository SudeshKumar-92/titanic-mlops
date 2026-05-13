import pandas as pd
import joblib

# Load saved model — no retraining needed
model = joblib.load('artifacts/titanic_model.pkl')
print("Model loaded!")

# Take input
age  = float(input("Enter Age:  "))
fare = float(input("Enter Fare: "))

# Predict
my_input = pd.DataFrame({'Age': [age], 'Fare': [fare]})
prediction = model.predict(my_input)
probability = model.predict_proba(my_input)

print(f"\nSurvival chance: {probability[0][1]*100:.1f}%")
print("Result: ✅ Survived" if prediction[0] == 1 else "Result: ❌ Did not survive")