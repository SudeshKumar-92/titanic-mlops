# titanic-mlops

<img width="870" height="594" alt="image" src="https://github.com/user-attachments/assets/ff3eab3b-c50a-49c7-ba66-4d13f29a0eb0" />

Yes — your understanding is perfectly correct. Let me explain exactly what happens:

Phase 1 — Training (run only once)
Dataset → train.py → model object in RAM → joblib.dump → model.pkl on disk

You download the dataset (Titanic CSV)
Your code cleans it and trains the model
The trained model exists only in RAM (temporary memory)
joblib.dump() snapshots that model object and writes it to disk as model.pkl
Script ends → RAM is wiped → but model.pkl stays on disk forever


Phase 2 — Prediction (run anytime, forever)
model.pkl → joblib.load → model object restored in RAM → .predict() → result

joblib.load() reads model.pkl from disk
The model object is rebuilt in RAM — exactly as it was after training
You give it new input (Age=22, Fare=50)
It returns a prediction — no dataset, no retraining needed


What is actually stored inside .pkl?
The .pkl file is not the dataset. It stores the learned patterns — specifically the mathematical weights the algorithm found during training:
model.pkl contains:
├── coefficients (weights for Age, Fare)
├── intercept value
├── feature names it expects
└── all settings of LogisticRegression
Think of it like this — the dataset is the school. The .pkl file is the degree certificate. Once you have the degree, you don't need to go back to school every time someone asks you a question.

The golden rule
train.pypredict.pyNeeds dataset?✅ Yes❌ NoNeeds internet?✅ Yes❌ NoSlow?Yes (training takes time)No (instant)Run how often?Once (or when retraining)Every predictionProducesmodel.pklprediction result
This is the core idea behind every ML deployment in the world — train once, predict millions of times.















Great observation! You're questioning the logic of the model — and this is a really important ML concept.

Short answer — Not exactly
The model doesn't follow a simple rule like:

"young age + more money = survive"

It learned statistical patterns from 891 real Titanic passengers. What it actually found was:









run app.py having all the code to run application
To run the application :
use : Curl command :---- $body = '{\"age\": 27, \"fare\": 800}'
curl.exe -X POST "http://127.0.0.1:5000/predict" -H "Content-Type: application/json" -d $body

<img width="1098" height="180" alt="image" src="https://github.com/user-attachments/assets/4d871eeb-1442-4ee1-935c-b75e2605716a" />



Higher Fare → more likely to survive (1st class passengers had better access to lifeboats)
Younger Age → slightly more likely to survive (children were prioritized)
