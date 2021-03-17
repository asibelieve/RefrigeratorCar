from flask import Flask, g, render_template, Blueprint, redirect, request

app = Blueprint("RefreshToken", __name__)

@app.route('/RefreshToken/', methods=['GET'])
def RefreshToken():
    return render_template('RefreshToken.htm')
