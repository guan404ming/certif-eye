from model import Model
from flask import Flask, request

app = Flask(__name__)


@app.route("/")
def home():
    # Render the HTML form
    return ""


@app.route("/api/infer", methods=["POST"])
def infer():
    # Get the text from the form
    user_input = request.get_json()

    # Create a model instance and perform inference
    model = Model()
    result = model.infer(user_input.get("data"))

    # Return the inference result
    return f"{result}"
