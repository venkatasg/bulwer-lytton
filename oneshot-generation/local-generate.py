from transformers import AutoModelForCausalLM, AutoTokenizer
import ipdb
from argparse import ArgumentParser
from tqdm import tqdm
import logging
import sys
import random

# 1. Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_prompts(tokenizer):
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
        message = [
            {"role": "user", "content": user_prompt},
        ]
        prompts.append(message)
        
    return prompts
    
def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--reasoning", type=str, default='low')
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    # Create handlers
    # Info handler for training_eval.log
    info_handler = logging.FileHandler(f"./outputs/logs/{args.model.split('/')[-1]}-info.log")
    info_handler.setLevel(logging.INFO)
    
    # Warning handler for warnings.log
    warning_handler = logging.FileHandler(f"./outputs/logs/{args.model.split('/')[-1]}-warning.log")
    warning_handler.setLevel(logging.WARNING)
    
    # Stream handler for console output
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    
    # Create formatters and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    info_handler.setFormatter(formatter)
    warning_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(info_handler)
    logger.addHandler(warning_handler)
    logger.addHandler(stream_handler)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype="auto",
        device_map="auto"
    )
    
    prompts = create_prompts(tokenizer)
    
    # Set token limit based on reasoning. Give 100 extra tokens for medium and high
    if args.reasoning=='low':
        max_new_tokens=300
    elif args.reasoning=='medium':
        max_new_tokens=400
    else:
        max_new_tokens=500
    
    num_prompts = len(prompts)
    
    with open(f"outputs/{args.model.split('/')[-1]}_reasoning_{args.reasoning}_raw_temp_1.txt", 'w') as f:
        for batch_ind in tqdm(range(0, len(prompts), args.batch_size)):
            inputs = tokenizer.apply_chat_template(
                prompts[batch_ind:batch_ind+args.batch_size],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                reasoning_effort=args.reasoning,
                padding=True
            ).to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1,
                repetition_penalty=1.2
            )
            logger.info(f"Completed {batch_ind+args.batch_size}/{num_prompts} prompts")
            for output in tokenizer.batch_decode(outputs):
                f.write(output + "\n\n")
        
if __name__ == "__main__":
    main()