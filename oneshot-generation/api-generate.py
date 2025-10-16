import ipdb
from argparse import ArgumentParser
from tqdm import tqdm
import logging
import sys
import random
import asyncio
from together import AsyncTogether
from time import sleep

with open('../../together.txt', 'r') as f:
    api_key = f.read().strip()
    
client = AsyncTogether(
    api_key = api_key
)

def setup_logging(log_file: str) -> None:
    """
    Configure logging to both file and console.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            # logging.StreamHandler()  # This will print to console as well
        ]
    )

def create_prompts():
    with open('../prompts/user-prompt-single.txt', 'r') as f:
        bare_prompt = f.read()
    
    prompt_genres = sum([['Adventure',  'Science Fiction']*100,
              ['Purple Prose', 'Romance', 'Crime & Detective', 'Vile Puns']*150, 
              ['Western', 'Historical Fiction', "Children's Literature", 'Fantasy & Horror']*50], [])
    random.seed(1)
    random.shuffle(prompt_genres)
    
    prompts = []
    for genre in prompt_genres:
        user_prompt = eval(f'f"""{bare_prompt}"""')
        prompts.append(user_prompt)
        
    return prompts
    
def create_prompts_test():
    with open('../prompts/user-prompt-single.txt', 'r') as f:
        bare_prompt = f.read()
    
    prompt_genres = ['Adventure',  'Science Fiction', 'Purple Prose', 'Romance', 'Crime & Detective', 'Vile Puns', 'Western', 'Historical Fiction', "Children's Literature", 'Fantasy & Horror']
    random.seed(1)
    random.shuffle(prompt_genres)
    
    prompts = []
    for genre in prompt_genres:
        user_prompt = eval(f'f"""{bare_prompt}"""')
        prompts.append(user_prompt)
        
    return prompts
    
async def async_chat_completion(model_name, messages, thinking, max_new_tokens):
    tasks = [client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": message}],
        temperature=1,
        seed=1,
        max_tokens=max_new_tokens,
        chat_template_kwargs={"thinking": thinking},
        )
        for message in messages
    ]
    
    responses = await asyncio.gather(*tasks)
    
    return responses
    
def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-V3.1")
    parser.add_argument("--thinking", action='store_true')
    parser.add_argument("--batch_size", type=int, default=10)
    args = parser.parse_args()
    
    prompts = create_prompts()
    
    # Set token limit based on thinking. Give 100 extra tokens for medium and high
    if args.thinking:
        max_new_tokens=400
    else:
        max_new_tokens=200
    
    num_prompts = len(prompts)
    
    log_file = f"outputs/{args.model.split('/')[-1]}_thinking:{args.thinking}_inference.log"
    setup_logging(log_file)
    
    total_prompts = len(prompts)
    logging.info(f"Starting inference on {total_prompts} prompts")
    
    output_file = f"outputs/{args.model.split('/')[-1]}_thinking:{args.thinking}_raw.txt"
    with open(output_file, 'a', encoding='utf-8') as file:
        processed_prompts = 0
        
        for i in range(processed_prompts, total_prompts, args.batch_size):
            batch = prompts[i:i + args.batch_size]
            
            responses = asyncio.run(async_chat_completion(args.model, batch, args.thinking, max_new_tokens))
            for ind, response in enumerate(responses):
                file.write(response.choices[0].message.content + "\n")
            
            processed_prompts += args.batch_size
            logging.info(f"Progress: {processed_prompts}/{total_prompts} prompts processed")
                
            if processed_prompts%(args.batch_size*10)==0:
                logging.info(f"Sleeping for 30 seconds")
                sleep(30)
    
        
if __name__ == "__main__":
    main()