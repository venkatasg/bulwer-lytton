import argparse
import os
import torch
import yaml


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("model", choices=["meta-llama/Llama-3.3-70B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", "openai/gpt-oss-20b", "openai/gpt-oss-120b", "gpt-4.1-nano", "gpt-4.1"])
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--api", default="vllm", choices=["vllm", "openai"])
    return parser.parse_args()


def get_api_args(quantize, num_gpus, max_num_seqs=8, model=None):
    args = {"tensor_parallel_size": num_gpus, "max_num_seqs": max_num_seqs}
    if not quantize:
        return args
    quantization = "mxfp4" if model and "gpt" in model else "bitsandbytes"
    # args from https://docs.vllm.ai/en/latest/features/quantization/bnb.html
    args.update({"dtype": torch.bfloat16, "trust_remote_code": True, "quantization": quantization})
    return args


def process_config(config, model, api):
    for k, v in config.items():
        if isinstance(v, dict):
            process_config(v, model, api)
        else:
            if "output" in k:
                simplified_model = model.split('/')[1].lower() if api == "vllm" else model
                root, ext = os.path.splitext(v)
                config[k] = f"{root}.{simplified_model}{ext}"


def load_config(config_file, model, api=None):
    """
    Load a config file and modify output filenames to include model information
    """
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    process_config(config, model, api)
    return config
