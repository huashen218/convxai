import os
import json
import openai
import argparse
import pandas as pd
import pdb
from tqdm import tqdm




class ChatGPT():
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.parameters = {
            "model": "gpt-3.5-turbo",
            # "temperature": 0.95,
            # "top_p": 0.95,
            # "frequency_penalty": 0.5,
            # "presence_penalty": 0.5,
        }

    def generate(self, input_messages):
        response = openai.ChatCompletion.create(
            messages=input_messages,
            **self.parameters
        )
        return response




### Define the prompt for generating the XAI question samples.
PROMPT = """Here are eleven intents:
{PROMPT_EXAMPLE}

Generate {N} samples for {LABEL} intent. Put the generated samples into a list without index. Please try to vary the text (both length, style and formal or not) and sort them from short to long.\n

Here are previous samples. Do not repeat.

{QUESTION_SAMPLES}
"""



def main(args):

    DATA_FOLDER = args.data_folder
    data = pd.read_csv(os.path.join(DATA_FOLDER, "xai_intent_seed.csv")) 

    N = 20
    PROMPT_EXAMPLE = ""
    chatgpt = ChatGPT()

    classes = list(set(data['label']))
    print(f"======= The XAI question seeds include {len(classes)} classes = {classes} ======")
    print(f"======= Each class includes {len(data['label'] == classes[1])} questions. ")

    for cls in classes:
        PROMPT_EXAMPLE += f" - {cls} {list(data[data['label'] == cls]['question'])[0]}\n"

    classes = list(set(data['label']))
    for cls in tqdm(classes):
        question_list = data[data['label'] == cls]['question']
        QUESTION_SAMPLES = "\n".join(question_list)
        prompt = PROMPT.format(PROMPT_EXAMPLE=PROMPT_EXAMPLE, LABEL=cls, N=N, QUESTION_SAMPLES=QUESTION_SAMPLES)
        print("The prompt is:", prompt)

        messages = [
                    {"role": "system", "content": "You are a model to generate user AI explanation questions based on provided examples."},
                    {"role": "user", "content": prompt},
                ]
        response = chatgpt.generate(messages)
        content = response["choices"][0]["message"]["content"]
        print("The generated XAI questions are:",content)

        question_list = list(map(str.strip, content.replace('-','').split('\n')))

        add_pd = pd.DataFrame({
            "label": len(question_list) * [cls],
            "question": question_list
        })

        data = pd.concat([data, add_pd])
    
    data.to_csv(os.path.join(DATA_FOLDER, "xai_intent_all.csv"), index=False)  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", dest="data_folder", type=str, default="../data")
    args = parser.parse_args()
    main(args)