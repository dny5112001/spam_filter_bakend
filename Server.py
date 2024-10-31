from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
model = load("spam_model.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["text"]
    prediction = model.predict([text])[0]
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Make sure to bind to 0.0.0.0
