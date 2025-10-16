import os 
import csv
import ipdb
import pandas as pd
import json
from openai import OpenAI
from argparse import ArgumentParser
from copy import deepcopy
import random


with open('../openai.txt', 'r') as f:
    api_key = f.read().strip()
    
client = OpenAI(api_key=api_key)
    
task_template = {
        "custom_id": "",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": "gpt-5-2025-08-07",
            "temperature": 1,
            "input": "",
            "reasoning" :{ "effort": "low" },
        }
    }
    
def create_batch():
    with open('../prompts/user-prompt-single.txt', 'r') as f:
        bare_prompt = f.read()
    
    prompt_genres = sum([['Adventure',  'Science Fiction']*100,
              ['Purple Prose', 'Romance', 'Crime & Detective', 'Vile Puns']*150, 
              ['Western', 'Historical Fiction', "Children's Literature", 'Fantasy & Horror']*50], [])
    random.seed(1)
    random.shuffle(prompt_genres)
    
    prompts = []
    for ind, genre in enumerate(prompt_genres):
        user_prompt = eval(f'f"""{bare_prompt}"""')
        task = deepcopy(task_template)
        task['body']['input'] = user_prompt
        task['custom_id'] = f"bl-{ind}"
        prompts.append(task)
        
    file_name = "outputs/gpt5-batch/input.jsonl"
    
    with open(file_name, 'w') as file:
        for obj in prompts:
            file.write(json.dumps(obj) + '\n')

def submit_job(batch_file):
    '''
    Submits batch job to OpenAI client
    '''
    batch_input_file = client.files.create(
        file=open(batch_file, "rb"),
        purpose="batch"
    )
    print(batch_input_file)
    batch_input_file_id = batch_input_file.id
    
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "description": "Bulwer Lytton one-shot prompt"
        }
    )
        
def main():
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default='prep', choices=['prep', 'submit', 'check', 'retreive'])
    args = parser.parse_args()
    
    if args.mode=='prep':
        create_batch()
    elif args.mode=='submit':
        submit_job('outputs/gpt5-batch/input.jsonl')
    elif args.mode=='check':
        output = client.batches.list(limit=100)
        for batch in output.model_dump()['data']:
            print(batch['status'], batch['created_at'], batch['output_file_id'])
    elif args.mode=='retreive':
        result_file_id = "file-FrCnXgVZcAertW7AexmEsk"
        result = client.files.content(result_file_id).content
        with open("outputs/gpt5-batch/outputs.jsonl", 'wb') as file:
            file.write(result)
    
if __name__ == "__main__":
    main()