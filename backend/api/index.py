from model.main import Model
from flask import Flask, request

app = Flask(__name__)
model = Model()


@app.route("/api/infer", methods=["POST"])
def infer():
    # Get the text from the form
    user_input = request.get_json()
    result = model.infer(user_input.get("data"))

    # Return the inference result
    return f"{result}"

@app.route("/api/get-place-info", methods=["POST"])
def get_place_info():
    # Get the text from the form
    user_input = request.get_json()

    # Return the inference result
    return model.get_place_info(user_input.get("place_id"))
