from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model 
import joblib
import numpy as np

# Creates the Flask application object.
app = Flask(__name__)

# Loads your neural network model (titanic_nn.h5).
# Loads your scaler (probably a StandardScaler or MinMaxScaler used to normalize age/pclass/sex).
# Scaling is needed because neural networks usually expect normalized input.
model = load_model("titanic_nn.h5")
scaler = joblib.load("scaler.pkl") 

@app.route("/") #Defines the root route ("/"). When user visits your site, it sends them the index.html page.
def index():
    return render_template("index.html")  # Load HTML frontend


# This is the API endpoint for predictions.
@app.route("/predict", methods=["POST"]) # Defines another route ("/predict") that only accepts POST requests.
def predict():
    try:
        # Get data from form
        pclass = int(request.form["pclass"]) #Casts pclass to int and age to float so they match the model’s expected input.
        age = float(request.form["age"])
        sex = request.form["sex"]

        # Encode sex (since model was trained with 0=male, 1=female)
        sex_code = 1 if sex.lower() == "female" else 0

        # Shape input for model
                                                                # features = np.array([[pclass, age, sex_code]]) just ignore, this is for older logistic regression model (.pkl).
                                                                # prediction = model.predict(features)[0]

        # result = "Survived" if prediction == 1 else "Did Not Survive"
        features = np.array([[pclass, age, sex_code]])  #Makes a 2D numpy array (needed by the model).
        features = scaler.transform(features)  # scale input same as training data
        prediction = model.predict(features)[0][0] 
        # Runs the neural network on the input.
        # model.predict(features) outputs a nested array like [[0.8479]].
        # [0][0] extracts the single float value 0.8479.
        # This is the probability of survival (0 → 100%).
        result = "Survived" if prediction >= 0.5 else "Did Not Survive"
        percentage = round(float(prediction) * 100, 2)  # ensure it's a Python float

        return jsonify({ #Sends back JSON response to frontend
            "result": result,
            "probability": percentage 
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__": #Runs the app when you start this script directly.
    app.run(debug=True) #Flask restarts automatically if you change the code. You get detailed error messages in your browser/terminal.


