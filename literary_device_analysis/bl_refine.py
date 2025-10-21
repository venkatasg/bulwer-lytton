from topicgpt_python import *
import yaml

from bl_util import parse_arguments, get_api_args, load_config

def main():
    args = parse_arguments()

    # determine whether to use quantization
    api_args = get_api_args(args.quantize, args.num_gpus, model=args.model)

    # get data from config, customized with model in output file paths
    config = load_config(args.config_file, args.model, api=args.api)

    refine_topics(
        args.api,
        args.model,
        config["refinement"]["prompt"],
        config["generation"]["output"],
        config["generation"]["topic_output"],
        config["refinement"]["topic_output"],
        config["refinement"]["output"],
        verbose=config["verbose"],
        remove=config["refinement"]["remove"],
        mapping_file=config["refinement"]["mapping_file"],
        api_args=api_args
    )


if __name__ == "__main__":
    main()
