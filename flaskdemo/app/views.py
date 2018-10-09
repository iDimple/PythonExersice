# -*- coding: utf-8 -*-
from flask import  Flask,jsonify,make_response,abort,request
app = Flask(__name__)

tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web',
        'done': False
    }
]

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route("/",methods=["GET"])
def index():
    return jsonify({"task":tasks})

@app.route("/<int:task_id>",methods=["GET"])
def getTask(task_id):
    if task_id=="1234":
        abort(404)
    return jsonify({"task":tasks})

@app.route("/",methods=["POST"])
def createTask():
    if  not request.json or not "title" in request.json:
        print request.json
        abort(400)
    task = {
        'id': tasks[-1]['id'] + 1,
        'title': request.json['title'],
        'description': request.json.get('description', ""),
        'done': False
    }
    tasks.append(task)
    return jsonify({"task":tasks}),201

if __name__=='__main__':
        app.run(debug=False,host="0.0.0.0",port=9000)