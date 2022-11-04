#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This source code supports the web server of the ConvXAI system.
# Copyright (c) Hua Shen, Chieh-Yang Huang, 2022.
#


import os
import json
import pytz
import logging
from flask_socketio import SocketIO, emit
from flask import (Flask, render_template, request, jsonify)
from pymongo import MongoClient
from threading import Thread, Event
from datetime import datetime
from convxai.utils import *


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


app = Flask(__name__)
app.config["SECRET_KEY"] = "liaopi6u123sdfakjb23sd"
socketio = SocketIO(app, logger=True, engineio_logger=True,
                    async_mode="threading")
mongo = get_mongo_connection()
thread = Thread()
thread_stop_event = Event()
task_mapping = {}


# ########################################
# # Set up paths to save log files
# ########################################
# system_config = parse_system_config_file()
# logFileName = "log_" + datetime.now().astimezone(pytz.timezone('US/Eastern')
#                                                  ).strftime("%m%d%Y_%H%M%S") + ".txt"
# logFile = open(os.path.join(
#     system_config['system']['logfilePath'], logFileName), "a")


########################################
# Threading and MongoDB
########################################

def init_threading() -> None:
    """
    Initiate the thread to monitor interaction and save data into mongoDB database.
    """
    global thread
    if not thread.is_alive():
        thread = Thread(
            name='mongo-monitoring',
            target=mongo_monitoring,
        )
        thread.start()
        logging.info("Starting Mongo Monitoring Thread")


def mongo_monitoring():
    """
    Set up mongoDB monitoring.
    """
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
    """
    Get the request data.
    """
    data = json.loads(str(request.data, encoding='utf-8'))
    return data


@app.route("/")
def index():
    return render_template("user_interface.html")


@app.route("/init_conversation", methods=["POST"])
def init_conversation():
    """
    Initiate the conversation.
    """
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
    """
    Reset the interface.
    """
    data = json.dumps({"text": "[RESET]"})
    return jsonify({"text": "RESET"})


########################################
# SocketIO Implementation
########################################

@socketio.on('connect', namespace="/connection")
def test_connect():
    emit('connected', {'data': 'Connected'})
    init_threading()


@socketio.on('disconnect', namespace="/connection")
def test_disconnect():
    logging.info('Client disconnected')


@socketio.on("interact", namespace="/connection")
def interact_socket(body):
    """
    Interaction between the web server and interface via socketIO.
    """
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

    responseTime = datetime.now().astimezone(
        pytz.timezone('US/Eastern')).strftime("%m-%d-%Y %H:%M:%S")
    responseText = body["text"]
    responseMessageType = body["message_type"]
    responseWritingModel = body.get("writing_model", None)

    # logEntry = responseTime + "  Text: " + responseText + "  \n\t\tMessageType:" + \
    #     responseMessageType + "  \n\t\tWritingModel: " + responseWritingModel + "\n"
    # logFile.write(logEntry+"\n")
    # logFile.close()
    # logging.info("Written to file" + logEntry)


########################################
# Run Flask and SocketIO.
########################################
def run_flask_socketio():
    """
    Run Flask and Websocket.
    """
    socketio.run(app, host="0.0.0.0", port=8080, debug=False)


if __name__ == "__main__":
    init_threading()
    run_flask_socketio()
