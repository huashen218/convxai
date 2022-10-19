from flask_socketio import SocketIO, emit
from flask import (
    Flask, render_template, send_file,
    request, abort, Response, jsonify,
)
import os
import json
from pymongo import MongoClient
from bson.objectid import ObjectId
from threading import Thread, Event
import time
from datetime import datetime 
import pytz
import logging
from convxai.services.utils import create_folder

app = Flask(__name__)
app.config["SECRET_KEY"] = "liaopi6u123sdfakjb23sd"
# app.config["SECRET_KEY"] = "leampi6u123sdfakjb23sd"


socketio = SocketIO(app, logger=True, engineio_logger=True, async_mode="threading")
mongo = MongoClient("localhost")["convxai"]

thread = Thread()
thread_stop_event = Event()
task_mapping = {} # {task_id : sid}


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
with open(os.path.join(root_dir, 'runners/sysconfig.json')) as json_file:
    logfilePath = json.load(json_file)['logfilePath']
logFileName = "log_" + datetime.now().astimezone(pytz.timezone('US/Eastern')).strftime("%m%d%Y_%H%M%S") + ".txt"
create_folder([logfilePath])
# logfile = open(os.path.join(logfilePath, logFileName), "a")
logFile = open(os.path.join(logfilePath, logFileName), "a")


########################################
# Threading and MongoDB
########################################

def init_threading():
    global thread
    if not thread.is_alive():
        thread = Thread(
            name='mongo-monitoring', 
            target=mongo_monitoring, 
        )
        thread.start()
        print("Starting Mongo Monitoring Thread")


def mongo_monitoring():
    global task_mapping

    pipeline = [{
        "$match": {
            "$and": [
                {"operationType": "insert"},
                {"fullDocument.role": "agent"},
                {"fullDocument.reply_to": {"$exists": True}}
            ]
        }
    }]

    with mongo.message.watch(pipeline) as stream:
        for insert_change in stream:
            source_message_id = insert_change["fullDocument"]["reply_to"]
            response_text = insert_change["fullDocument"]["text"]
            message_type = insert_change["fullDocument"]["payload"]["message_type"]
            if message_type == "task":
                socketio.emit(
                    "task_response",
                    {"text": response_text},
                    to=task_mapping[source_message_id],
                    namespace="/connection"
                )
            elif message_type == "conv":
                writingIndex = insert_change["fullDocument"]["payload"]["writingIndex"]
                socketio.emit(
                    "conv_response",
                    {"text": response_text, "writingIndex": writingIndex},
                    to=task_mapping[source_message_id],
                    namespace="/connection"
                )
            del task_mapping[source_message_id],


########################################
# Flask Implementation
########################################

def get_data():
    data = json.loads(str(request.data, encoding='utf-8'))
    return data


@app.route("/")
def index():
    return render_template("user_interface.html")


@app.route("/init_conversation", methods=["POST"])
def init_conversation():
    body = get_data()
    res = mongo.message.insert_one({
        "text": body["text"],
        "role": "user",
        "init": True,
        "time": datetime.now(),
        "conversation_id": body.get("conversation_id"),
    })
    return jsonify({"text": "hello world", "conversation_id": str(res.inserted_id)})


@app.route("/reset")
def reset():
    # send text using ws
    data = json.dumps({"text": "[RESET]"})
    #ws.send(data)
    return jsonify({"text": "RESET"})





########################################
# SocketIO Implementation
########################################
"""
SocketIO Implementation.
"""

@socketio.on('connect', namespace="/connection")
def test_connect():
    emit('connected', {'data': 'Connected'})
    init_threading()


@socketio.on('disconnect', namespace="/connection")
def test_disconnect():
    print('Client disconnected')


@socketio.on("interact", namespace="/connection")
def interact_socket(body):
    result = mongo.message.insert_one({
        "text": body["text"],
        "role": "user",
        "init": False,
        "time": datetime.now(),
        "conversation_id": body["conversation_id"],
        "payload": {
            "message_type": body["message_type"],
            "writing_model": body.get("writing_model", None),
        },
        "note": "socket",
    })


    ### Write logs into files ###
    task_mapping[result.inserted_id] = request.sid
    
    responseTime = datetime.now().astimezone(pytz.timezone('US/Eastern')).strftime("%m-%d-%Y %H:%M:%S")
    responseText = body["text"]
    responseMessageType = body["message_type"]
    responseWritingModel = body.get("writing_model", None)

    logEntry = responseTime + "  Text: " + responseText + "  \n\t\tMessageType:" + responseMessageType + "  \n\t\tWritingModel: " + responseWritingModel + "\n"
    # logFile = open(os.path.join(logfilePath, logFileName), "a")
    logFile.write(logEntry+"\n")
    logFile.close()
    print("Written to file" +logEntry)




#########################################
"""
Run Flask and Websocket.
"""
def run_flask_socketio():
    socketio.run(app, host="0.0.0.0", port=8080, debug=False)


# def run_flask():
#     #app.run(host="0.0.0.0", port=5000, debug=True)
#     app.run(host="0.0.0.0", port=8080, debug=True)


# def run_web_socket():
#     ws.run_forever()


if __name__ == "__main__":
    init_threading()
    run_flask_socketio()



    #threading.Thread(target=run_web_socket).start()
    #run_flask()