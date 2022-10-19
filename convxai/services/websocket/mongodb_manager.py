#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Websocket Manager Module Contains implementation of the WebsocketManager which helps run
ParlAI via websockets.
"""
import json
import asyncio
import logging
from parlai.core.agents import create_agent
from parlai.chat_service.core.chat_service_manager import ChatServiceManager

import parlai.chat_service.utils.logging as log_utils
import parlai.chat_service.utils.misc as utils
from .mongodb_agents import MongoAgent
from pymongo import MongoClient
from datetime import datetime
from threading import Thread, Event
import time
from bson.objectid import ObjectId

class MongoManager(ChatServiceManager):
    """
    Manages interactions between agents on a websocket as well as direct interactions
    between agents and an overworld.
    """

    class MessageSender(ChatServiceManager.ChatServiceMessageSender):
        def send_read(self, receiver_id):
            pass

        def typing_on(self, receiver_id, persona_id=None):
            pass

    def __init__(self, opt):
        """
        Create a WebsocketManager using the given setup options.
        """
        super().__init__(opt)
        self.opt = opt
        #self.port = opt.get('port')
        # self.mongo_host = "157.230.188.155"
        # self.mongo_host = "130.203.139.47"
        self.mongo_host = "localhost"
        self.db_name = "convxai_test" #"convxai" 
        self.mongo = MongoClient(self.mongo_host)[self.db_name]
        
        self.mongo_lock = Event()
        self.mongo_thread = Thread()

        self.worker_lock = Event()
        self.worker_thread = Thread()

        self.debug = opt.get('is_debug', False)

        self.message_sender = MongoManager.MessageSender()
        self.service_reference_id = None

        self._parse_config(opt)
        self._complete_setup()

    def parse_additional_args(self, opt):
        self.should_load_model = self.config['additional_args'].get('load_model', True)

    def _complete_setup(self):
        """
        Complete necessary setup items.
        """
        self.agent_pool = {}
        self.messenger_agent_states = {}
        self.agent_id_to_overworld_future = {}
        self.task_to_agent_ids = {}
        self._load_model()

    def _load_model(self):
        """
        Load model if necessary.
        """
        if 'models' in self.opt and self.should_load_model:
            model_params = {}
            model_info = {}
            for model in self.opt['models']:
                model_opt = self.opt['models'][model]
                override = model_opt.get('override', {})
                if type(override) is list:
                    model_opt['override'] = override[0]
                model_params[model] = create_agent(model_opt).share()
                model_info[model] = {'override': override}
            self.runner_opt['model_info'] = model_info
            self.runner_opt['shared_bot_params'] = model_params

    def _handle_message_read(self, event):
        """
        Send read receipt back to user who sent message This function is left empty as
        it is not applicable to websockets since read receipts are not supported.
        """
        pass

    def _manager_loop_fn(self):
        """
        An iteration of the manager's main loop to launch worlds.
        """
        while True:
            time.sleep(10)

            with self.agent_pool_change_condition:
                valid_pools = self._get_unique_pool()
                for world_type, agent_pool in valid_pools.items():
                    # check if agent has exceeded max time in pool
                    world_config = self.task_configs[world_type]
                    if world_config.max_time_in_pool is not None:
                        self.check_timeout_in_pool(
                            world_type,
                            agent_pool,
                            world_config.max_time_in_pool,
                            world_config.backup_task,
                        )

                    needed_agents = self.max_agents_for[world_type]
                    if len(agent_pool) >= needed_agents:
                        log_utils.print_and_log(
                            logging.INFO, 'starting pool', should_print=True
                        )
                        # enough agents in pool to start new conversation
                        self.conversation_index += 1
                        task_id = 't_{}'.format(self.conversation_index)

                        # Add the required number of valid agents to the conv
                        agent_states = [w for w in agent_pool[:needed_agents]]
                        agents = []
                        for state in agent_states:
                            agent = self._create_agent(task_id, state.get_id())
                            agent.onboard_data = state.onboard_data
                            agent.data = state.data
                            state.assign_agent_to_task(agent, task_id)
                            state.set_active_agent(agent)
                            agents.append(agent)
                            # reset wait message state
                            state.stored_data['seen_wait_message'] = False
                        assign_role_function = utils.get_assign_roles_fn(
                            self.world_module, self.taskworld_map[world_type]
                        )
                        if assign_role_function is None:
                            assign_role_function = utils.default_assign_roles_fn
                        assign_role_function(agents)
                        # Allow task creator to filter out workers and run
                        # versions of the task that require fewer agents
                        for a in agents:
                            # Remove selected workers from the agent pool
                            self.remove_agent_from_pool(
                                self.get_agent_state(a.id),
                                world_type=world_type,
                                mark_removed=False,
                            )
                        for a in agents:
                            partner_list = agents.copy()
                            partner_list.remove(a)
                            a.message_partners = partner_list

                        done_callback = self._get_done_callback_for_agents(
                            task_id, world_type, agents
                        )

                        # launch task world.
                        future = self.world_runner.launch_task_world(
                            task_id, self.taskworld_map[world_type], agents
                        )
                        future.add_done_callback(done_callback)
                        self.active_worlds[task_id] = future

    def start_task(self):
        """
        Begin handling task.
        """
        self.running = True

        # run mongo watch
        if not self.mongo_thread.is_alive():
            self.mongo_thread = Thread(
                name='mongo-monitoring',
                target=self.mongo_watch,
            )
            self.mongo_thread.start()

        # run manager while loop
        # TODO: Probably send this args (callback_time = utils.THREAD_MEDIUM_SLEEP * 1000)
        """
        if not self.worker_thread.is_alive():
            self.worker_thread = Thread(
                name="worker-pool-monitoring",
                target=self._manager_loop_fn,
                args=(self,)
            )
            self.worker_thread.start()
        """
        self._manager_loop_fn()

    def shutdown(self):
        """
        Defined to shutown the tornado application.
        """
        try:
            self.world_runner.shutdown()
            self._expire_all_conversations()
        finally:
            pass

        # TODO: how to terminate both of the thread elegantly?
        quit()

    def _create_agent(self, task_id, socketID):
        """
        Initialize an agent and return it.

        Called each time an agent is placed into a new task.

        :param task_id:
            string task identifier
        :param agent_id:
            int agent id
        """
        return MongoAgent(self.opt, self, socketID, task_id)

    def observe_message(self, socket_id, message, quick_replies=None):
        """
        Send a message through the message manager.

        :param socket_id:
            int identifier for agent socket to send message to
        :param message:
            (str) message to send through the socket.
        :param quick_replies:
            (list) list of strings to send as quick replies.

        Returns a tornado future for tracking the `write_message` action.
        """
        if quick_replies is not None:
            quick_replies = list(quick_replies)

        self.mongo.message.insert_one({
            "text": message,
            "quick_replies": quick_replies,
            "conversation_id": socket_id,
            "init": False,
            "time": datetime.now(),
            "role": "agent",
        })

    def observe_payload(self, socket_id, text, payload, quick_replies=None):
        """
        Send a message through the message manager.

        :param socket_id:
            int identifier for agent socket to send message to
        :param payload:
            (dict) payload to send through the socket. The mandatory keys are:
                    'type': (str) Type of the payload (e.g. 'image')
                    'data': str. base64 encoded content
                    If 'type' is 'image', the 'mime_type' (str) key can be provided
                    to specify the Mime type of the image

        Returns a tornado future for tracking the `write_message` action.
        """
        if quick_replies is not None:
            quick_replies = list(quick_replies)

        self.mongo.message.insert_one({
            "text": text,
            "quick_replies": quick_replies,
            "payload": payload,
            "conversation_id": socket_id,
            "init": False,
            "time": datetime.now(),
            "role": "agent",
            "reply_to": ObjectId(payload["reply_to"]),
        })

    def restructure_message(self, message):
        """
        This is to restructure a new message to conform to the message structure defined
        in the `chat_service` README.
        >> {'text': 'start', 'payload': None, 'sender': {'id': 'd6897316-d16f-4aa7-b27a-16626e3ca2ee'}, 'recipient': {'id': 0}}
        """
        print(message)
        return message

    def _handle_bot_read(self, agent_id):
        pass

    def _confirm_message_delivery(self, event):
        pass

    def setup_server(self):
        pass

    def setup_socket(self):
        pass

    def mongo_watch(self):
        pipeline = [{
            "$match": {
                "$and": [
                    {"operationType": "insert"}, # incoming message
                    {"fullDocument.role": "user"}, # role has to be "user"
                ]
            }
        }]

        with self.mongo.message.watch(pipeline) as stream:
            for insert_change in stream:
                self.handle_new_message(insert_change["fullDocument"])

    def handle_new_message(self, message):
        """
        message = {
            "_id": ObjectId("_id"),
            "text": text-string,
            "payload": other-json-data,
            "conversation_id": _id for the initial conversation,
            "init": True/False,
            "time": timestamp,
            "role": "user" / "agent"
        }
        """

        logging.info('websocket message from client: {}'.format(str(message)))

        if message["init"]:
            sender_id = str(message["_id"])
        else:
            sender_id = str(message["conversation_id"])

            # add message_id to payload
            payload = message.get("payload")
            payload["message_id"] = message["_id"]

        message_info = {
            'text': message.get('text', ''),
            'payload': message.get('payload'),
            'sender': {'id': sender_id},
            'recipient': {'id': 0},
        }
        self._on_new_message(message_info)


