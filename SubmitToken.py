from flask import Flask, g, render_template, Blueprint, redirect, request

app = Blueprint("SubmitToken", __name__)

@app.route('/SubmitToken/', methods=['POST'])
def RefreshToken():
    token = request.values.get("token")
    print(token)

    with open('token.txt', 'w') as f:
         f.write(token)

    return 'OK'