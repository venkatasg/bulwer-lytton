from topicgpt_python import *
import yaml

from bl_util import parse_arguments, get_api_args, load_config

def main():
    args = parse_arguments()

    # determine whether to use quantization
    api_args = get_api_args(args.quantize, args.num_gpus, model=args.model)

    # get data from config, customized with model in output file paths
    config = load_config(args.config_file, args.model, api=args.api)

    # Assignment
    assign_topics(
        args.api,
        args.model,
        config["data_sample"],
        config["assignment"]["prompt"],
        config["assignment"]["output"],
        config["refinement"][
            "topic_output"
        ],  # TODO: change to generation_2 if you have subtopics, or config['refinement']['topic_output'] if you refined topics
        verbose=config["verbose"],
        api_args=api_args
    )

if __name__ == "__main__":
    main()
