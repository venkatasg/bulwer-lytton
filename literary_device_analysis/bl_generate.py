from topicgpt_python import *

from bl_util import parse_arguments, get_api_args, load_config

def build_examples(sentences, dataset):
    return [{"id": f"{dataset}-{i}", "text": sent} for i, sent in enumerate(sentences)]

def main():

    args = parse_arguments()

    # determine whether to use quantization
    api_args = get_api_args(args.quantize, args.num_gpus, model=args.model)

    # get data from config, customized with model in output file paths
    config = load_config(args.config_file, args.model, api=args.api)

    generate_topic_lvl1(
        args.api,
        args.model,
        config["data_sample"],
        config["generation"]["prompt"],
        config["generation"]["seed"],
        config["generation"]["output"],
        config["generation"]["topic_output"],
        verbose=config["verbose"],
        api_args=api_args
    )



if __name__ == "__main__":
    main()
