#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This source code supports the web server of the ConvXAI system.
# Copyright (c) Hua Shen 2023.
#


import os
import json
import openai

class ChatGPT():
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.parameters = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.95,
            "max_tokens": 1000,
            "top_p": 0.95,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
        }

    def generate(self, input_messages):
        response = openai.ChatCompletion.create(
            messages=input_messages,
            **self.parameters
        )
        return response



# Quick test
def main():

    chatgpt = ChatGPT()
    messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                {"role": "user", "content": "Where was it played?"}
            ]
    response = chatgpt.generate(messages)
    
    content = response["choices"][0]["message"]["content"]
    role = response["choices"][0]["message"]["role"]

    with open("chat-gpt.json", 'w', encoding='utf-8') as infile:
        json.dump(response, infile, indent=2)


if __name__ == "__main__":
    main()
