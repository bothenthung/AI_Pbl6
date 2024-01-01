"""
    Example Controllers
"""

from project import app, corrector
import sys
from flask import render_template, redirect, url_for, request,jsonify
from utils.api_utils import correctFunction, postprocessing_result

#route index
@app.route('/', methods = ['GET'])
def index():
    data = {
        "title": "Hello World",
        "body": "Flask simple MVC"
    }
    return render_template('index.html', data = data)

@app.route('/spelling', methods = ['GET'])
def correct():
    text = request.args.get("text")
    if not text or text == "" or len(text) < 10:
        print("Received nothing!", file=sys.stderr)
        data = {"error": f"Received short text '{text}'"}
        return render_template('index.html', data = data)
    out = correctFunction(text, corrector)

    result = postprocessing_result(out)
    response_data = {"correction": result}
    return jsonify(out)
