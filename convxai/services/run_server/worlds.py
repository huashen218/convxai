#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import json
import logging
import numpy as np
from parlai.core.worlds import World
from parlai.core.agents import create_agent_from_shared
from convxai.writing_models.models import *


os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger("my-logger")
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format=FORMAT)
logger.setLevel(logging.INFO)


###### Diversity Model Output Mapping ######
diversity_model_label_mapping = {
    0: "background",
    1: "purpose",
    2: "method",
    3: "finding",
    4: "other",
}


###### Quality Model Output Mapping ######
levels = [35, 52, 74, 116]


def quality_model_label_mapping(x): return \
    "quality1" if x > levels[3] else (
    "quality2" if x >= levels[2] and x < levels[3] else (
        "quality3" if x >= levels[1] and x < levels[2] else (
            "quality4" if x >= levels[0] and x < levels[1] else
            "quality5"))
)


# ---------- Overworld -------- #
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

        ###### Define writing_models' configurations ######

        ### Diversity_Model ###
        root_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "../../"))

        with open(os.path.join(root_dir, "writing_models/configs/diversity_model_config.json"), 'r') as fp:
            self.diversity_model_config = json.load(fp)
        self.diversity_model = DiversityModel(
            self.diversity_model_config["save_dirs"]["output_dir"])

        ### Quality_Model ###
        with open(os.path.join(root_dir, "writing_models/configs/quality_model_config.json"), 'r') as fp:
            self.quality_model_config = json.load(fp)
        self.quality_model = QualityModel(
            self.quality_model_config["save_configs"]["output_dir"])

        ###### define conversational XAI model ######
        self.convai_model = bot
        self.predict_results = {}

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

        ###### Convert user inputs into 'message_info' dict ######
        message_info = a["payload"]
        message_info["text"] = a["text"]

        ###### PANEL ONE - Writing Support Tasks ######
        if message_info["message_type"] == "task":
            """message_info Example
            message_info = {
                "message_type": 'task',
                "text": 'string of the user input writing',
                "writing_model": 'model-writing-1',
                "message_id": 'ObjectId('622ab03276bdb0ec3f36cb18')',
            }
            """
            print(f"\nReceiving Writing Task Inputs from Users:{message_info}")

            input_paragraph = message_info["text"]
            writing_task = message_info["writing_model"]

            ################## Return only one model output ##################
            """
            ### Writing Model 1 -- Diversity Model ### 
            if writing_task == "model-writing-1":
                print("\nGenerating writing prediction from the Diversity_Model....")
                model_outputs = self.diversity_model.inference(input_paragraph)
                (predict, probability) = model_outputs['outputs']
                x_text = model_outputs['inputs']
                post_text = [{"text": x_text[n], "classname": diversity_model_label_mapping[predict[n]], "score": str(predict[n])} for n in range(len(x_text))]


            ### Writing Model 2 -- Quality Model ### 
            elif writing_task == "model-writing-2":
                print("\nGenerating writing prediction from the Quality_Model....")
                model_outputs = self.quality_model.inference(input_paragraph)
                (perplexities) = model_outputs['outputs']

                ###### Quality Model Output Curve ######
                perplexities = 1 / np.log(perplexities)
                x_text = model_outputs['inputs']
                post_text = [{"text": x_text[n], "classname": quality_model_label_mapping(float(perplexities[n])), "score": str(perplexities[n])} for n in range(len(x_text))]
            
            
            ### Send Results to Website ### 
            self.agent.observe(
                {
                    'text': post_text,
                    'payload': {
                        "message_type": message_info["message_type"],
                        "reply_to": message_info["message_id"]
                    }
                }
            )
            
            
            """
            ########################################################################

            ################## Return both models' outputs ##################

            ### Writing Model 1 -- Diversity Model ###
            print("\nGenerating writing prediction from the Diversity_Model....")
            diversity_model_outputs = self.diversity_model.inference(
                input_paragraph)

            (predict, probability) = diversity_model_outputs['outputs']
            diversity_model_x_text = diversity_model_outputs['inputs']
            diversity_model_post_text = [{"text": diversity_model_x_text[n], "classname": diversity_model_label_mapping[predict[n]], "score": str(
                predict[n])} for n in range(len(diversity_model_x_text))]
            # diversity_model_post_text = [{"diversity_model_text": diversity_model_x_text[n], "diversity_model_classname": diversity_model_label_mapping[predict[n]], "diversity_model_score": str(predict[n])} for n in range(len(diversity_model_x_text))]

            ### Writing Model 2 -- Quality Model ###
            print("\nGenerating writing prediction from the Quality_Model....")
            quality_model_outputs = self.quality_model.inference(
                input_paragraph)
            (perplexities) = quality_model_outputs['outputs']

            ###### Quality Model Output Curve ######
            # perplexities = 1 / np.log(perplexities)
            quality_model_x_text = quality_model_outputs['inputs']
            quality_model_post_text = [{"text": quality_model_x_text[n], "classname": quality_model_label_mapping(
                float(perplexities[n])), "score": str(perplexities[n])} for n in range(len(quality_model_x_text))]

            post_text = diversity_model_post_text + quality_model_post_text

            ### Send Results to Website ###
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

        ########################################################################

        ###### PANEL TWO - ConvXAI Tasks ######
        elif message_info["message_type"] == "conv":
            logger.info(f"Receiving ConvXAI Inputs from Users:{message_info}")
            """message_info Example
            message_info = {
                "message_type": 'conv',
                "text": '{"writingInput": 'List[str]', "explainInput": 'str', "writingIndex": 'List[str]'}',
                "writing_model": 'model-writing-1',
                "message_id": 'ObjectId('622ab03276bdb0ec3f36cb18')',
            }
            """

            if '[DONE]' in message_info['text']:
                self.episodeDone = True
            elif '[RESET]' in message_info['text']:
                self.convai_model.reset()
                self.agent.observe(
                    {"text": "[History Cleared]", "episode_done": False})
            else:

                # self.predict_results['text'] = response['text']
                message_info['predict_results'] = self.predict_results

                self.convai_model.observe(message_info)
                response = self.convai_model.act()
                """
                response = {
                    "text": response_text,
                    "writingIndex": writingIndex,
                }

                """

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
