#!/usr/bin/env python3

# This source code supports the deep learning server of the ConvXAI system.
# Copyright (c) Hua Shen 2022.

import os
import json
import logging
import numpy as np
from parlai.core.worlds import World
from parlai.core.agents import create_agent_from_shared
from convxai.writing_models.models import *
from convxai.utils import *



os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger("my-logger")
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format=FORMAT)
logger.setLevel(logging.INFO)


class MessengerOverworld(World):
    """
    World to handle moving agents to their proper places.
    """
    MODEL_KEY = 'blender_90M'
    def __init__(self, opt, agent, bot):
        self.agent = agent
        self.opt = opt
        self.first_time = True
        self.episodeDone = False
        self.convai_model = bot
        self.predict_results = {}
        self.configs = parse_system_config_file()
        self.diversity_model = DiversityModel(self.configs['scientific_writing']["diversity_model_dir"])
        self.quality_model = QualityModel(self.configs['scientific_writing']["quality_model_dir"])

    @staticmethod
    def generate_world(opt, agents):
        if opt['models'] is None:
            raise RuntimeError("Model must be specified")
        return MessengerOverworld(
            opt,
            agents[0],
            create_agent_from_shared(
                opt['shared_bot_params'][MessengerOverworld.MODEL_KEY]
            ),
        )

    @staticmethod
    def assign_roles(agents):
        for a in agents:
            a.disp_id = 'Agent'

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        self.agent.shutdown()

    def parley(self):
        if self.first_time:
            self.agent.observe(
                {
                    'id': 'Overworld',
                    'text': 'Welcome to the overworld for the ParlAI messenger '
                    'chatbot demo. Please type "begin" to start, or "exit" to exit',
                    'quick_replies': ['begin', 'exit'],
                }
            )
            self.first_time = False
        a = self.agent.act()
        if a is None:
            return

        message_info = a["payload"]
        message_info["text"] = a["text"]

        ##############################################
        # PANEL 1 - Scientific Writing Support Tasks
        ##############################################
        """message_info Example
        message_info = {
            "message_type": 'task',
            "text": 'string of the user input writing',
            "writing_model": 'model-writing-1',
            "message_id": 'ObjectId('622ab03276bdb0ec3f36cb18')',
        }
        """
        if message_info["message_type"] == "task":
            logging.info(
                f"\nReceiving Writing Task Inputs from Users:{message_info}")
            input_paragraph = message_info["text"]

            # ****** Return both models' predictions ****** #
            # ******  Writing Model 1 -- Diversity Model ****** #
            logging.info(
                "\nGenerating writing prediction from the Diversity_Model....")
            diversity_model_outputs = self.diversity_model.inference(
                input_paragraph)
            (predict, probability) = diversity_model_outputs['outputs']
            diversity_model_x_text = diversity_model_outputs['inputs']
            diversity_model_post_text = [{"text": diversity_model_x_text[n], "classname": diversity_model_label_mapping[predict[n]], "score": str(
                predict[n])} for n in range(len(diversity_model_x_text))]

            # ******  Writing Model 2 -- Quality Model ****** #
            logging.info(
                "\nGenerating writing prediction from the Quality_Model....")
            quality_model_outputs = self.quality_model.inference(
                input_paragraph)
            (perplexities) = quality_model_outputs['outputs']
            quality_model_x_text = quality_model_outputs['inputs']
            quality_model_post_text = [{"text": quality_model_x_text[n], "classname": quality_model_label_mapping(
                float(perplexities[n])), "score": str(perplexities[n])} for n in range(len(quality_model_x_text))]

            # ****** Send Results to Website ****** #
            post_text = diversity_model_post_text + quality_model_post_text
            self.agent.observe(
                {
                    'text': post_text,
                    'payload': {
                        "message_type": message_info["message_type"],
                        "reply_to": message_info["message_id"]
                    }
                }
            )
            self.predict_results = {
                "diversity_model_outputs": {
                    "inputs": diversity_model_outputs['inputs'],
                    "outputs_predict": ','.join(str(v) for v in diversity_model_outputs["outputs"][0]),
                    "outputs_probability": ','.join(str(v) for v in diversity_model_outputs["outputs"][1])
                },
                "quality_model_outputs": {
                    "inputs": quality_model_outputs['inputs'],
                    "outputs_perplexity": ','.join(str(v) for v in quality_model_outputs["outputs"])
                }
            }
            del input_paragraph

        ##############################################
        # PANEL 2 - Conversational XAI Tasks
        ##############################################
            """
            Example of the message_info format: 
            message_info = {
                "message_type": 'conv',
                "text": '{"writingInput": 'List[str]', "explainInput": 'str', "writingIndex": 'List[str]'}',
                "writing_model": 'model-writing-1',
                "message_id": 'ObjectId('622ab03276bdb0ec3f36cb18')',
            }
            Example of the response format: 
            response = {
                "text": response_text,
                "writingIndex": writingIndex,
            }
            """
        elif message_info["message_type"] == "conv":
            logger.info(f"Receiving ConvXAI Inputs from Users:{message_info}")
            if '[DONE]' in message_info['text']:
                self.episodeDone = True
            elif '[RESET]' in message_info['text']:
                self.convai_model.reset()
                self.agent.observe(
                    {"text": "[History Cleared]", "episode_done": False})
            else:
                message_info['predict_results'] = self.predict_results
                self.convai_model.observe(message_info)
                response = self.convai_model.act()
                self.agent.observe(
                    {
                        'text': response['text'],
                        'payload': {
                            "message_type": message_info["message_type"],
                            "reply_to": message_info["message_id"],
                            "writingIndex": response["writingIndex"]
                        }
                    }
                )
