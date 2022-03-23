from flask import Flask, render_template, request, jsonify
import re
import model

app = Flask(__name__, static_url_path="/static")


@app.route('/message', methods = ['POST'])
def reply():
    question = request.form["msg"]      
    domain, domain_name = model.getDomainPrediction(question)
    answer = model.getIntent(domain, question)
    answer = domain_name + " "+ str(answer)
    return jsonify( { 'text': [answer], 'reload' : False }) 
     

@app.route('/postmethod', methods = ['POST']) 
def post_javascript_data():       
   return ''
    
@app.route("/")
def index():
    return render_template("index.html")

if (__name__ == "__main__"):         
    app.run(host="localhost", port = 5000)

