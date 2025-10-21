"""
Save humor predictions to use later
"""
import argparse

from hr_roberta import HumorResearchRobertaDetector
from util import DATASETS, HumorDetectionResults, load_sentences

MODEL_NAME_TO_CLASS = {
    "hr_roberta": HumorResearchRobertaDetector,
}
MODELS = list(MODEL_NAME_TO_CLASS.keys())


def main(args=None):
    """
    Stores humor predictions for a dataset to be analyzed later
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS, help="Model to initialize")
    parser.add_argument(
        "--datasets", choices=DATASETS, nargs="+", help="Datasets to run the model on"
    )
    parser.add_argument(
        "--model_args",
        nargs="*",
        help="Arguments that are passed to the model to be initialized",
    )
    args = parser.parse_args(args)

    model = MODEL_NAME_TO_CLASS[args.model](*args.model_args)
    for dataset in args.datasets:
        results = HumorDetectionResults(dataset, model.model_name())
        if not results.exist:
            sentences = load_sentences(dataset)
            predictions = model.predict(sentences)
            results.save(predictions)


if __name__ == "__main__":
    main()
