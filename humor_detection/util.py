"""
Shared utility for multiple humor detection models
"""
import glob
import os
import re
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd

DATASET_PATHS = {
    "bulwer": "data/Bulwer-Lytton.tsv",
    "bulwer-puns": "data/bulwer-lytton.tsv",
    "PotD-test": "data/humor_datasets/pun_of_the_day/files/test.csv",
    "combo-test": "data/humor_datasets/comb/files/test.csv",
    "crowd": "data/crowdsourced_sents_cleaned.txt",
    "fewshot-gpt5": "fewshot-generation/outputs/gpt5-sents.txt",
    "fewshot-gpt41": "fewshot-generation/outputs/gpt41-sents.txt",
    "fewshot-DeepSeek": "fewshot-generation/outputs/DeepSeek_sents.txt",
    "fewshot-gpt-120b_temp_1": "fewshot-generation/outputs/gpt-120b_temp_1.txt"

}
DATASETS = (
    list(DATASET_PATHS.keys())
    + list(dataset + "-pos" for dataset in DATASET_PATHS if dataset.endswith("test"))
    + list(dataset + "-neg" for dataset in DATASET_PATHS if dataset.endswith("test"))
)

OUTPUT_BASE_PATH = "../data/humor_detection"


class HumorDetector(ABC):
    """
    Abstract class for humor detection models
    """

    @abstractmethod
    def predict(self, sentences: List[str]) -> List[float]:
        """
        Predict funniness score for a set of sentences

        Args:
            sentences (List[str]): the sentences to score

        Returns:
            List[float]: the humor scores
        """

    @abstractmethod
    def model_name(self):
        """
        A unique identifier for the model, which may account for some
        specific args used when initializing

        Returns:
            str: the unique identifier
        """


class HumorDetectionResults:
    """
    Class for storing and loading results from all humor detection models
    Assumes results are lists of floats
    """

    def __init__(self, dataset_name: str, model_name: str, combine_seeds: bool = False):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.combine_seeds = combine_seeds

    def save(self, predictions: List[float]):
        """
        Save predictions to a txt file

        Args:
            predictions (List[float]): a humor model's predictions
        """
        os.makedirs(os.path.split(self._output_filename)[0], exist_ok=True)
        with open(self._output_filename, "w", encoding="utf-8") as file:
            for pred in predictions:
                file.write(str(pred))
                file.write("\n")

    def load(self) -> List[float]:
        """
        Load stored results from a txt file

        Returns:
            List[float]: the predictions from the model
        """
        if self.combine_seeds:
            all_files = glob.glob(
                os.path.join(
                    OUTPUT_BASE_PATH, f"{self.model_name}*", f"{self.dataset_name}.txt"
                )
            )
            all_seed_data = []
            for filename in all_files:
                seeded_model_name = re.match(
                    rf".*/.*/({self.model_name}-[\d]+)/.*", filename
                ).group(1)
                all_seed_data.append(
                    HumorDetectionResults(self.dataset_name, seeded_model_name).load()
                )
            return np.array(all_seed_data).mean(axis=0).tolist()
        else:
            with open(self._output_filename, "r", encoding="utf-8") as file:
                return [float(s) for s in file.readlines()]

    @property
    def _output_filename(self):
        return os.path.join(
            OUTPUT_BASE_PATH, self.model_name, f"{self.dataset_name}.txt"
        )

    @property
    def exist(self) -> bool:
        """
        Check for the existence of the output file for this model/dataset pair

        Returns:
            bool: whether the results have already been saved
        """
        return os.path.exists(self._output_filename)


def load_sentences(dataset: str) -> List[str]:
    """
    Load all sentences from a given dataset

    Args:
        dataset (str): the dataset's name

    Returns:
        List[str]: list of sentences in the dataset
    """
    quoting = 3 if "bulwer" in dataset else 0
    # filter positive/negative instances
    if dataset.endswith("-pos") or dataset.endswith("-neg"):
        dataset_path = DATASET_PATHS[dataset[:-4]]
        label = 1 if dataset.endswith("-pos") else 0
        df = pd.read_csv(dataset_path, encoding="utf-8", quoting=quoting)
        df = df[df["label"] == label]
        return df["text"].to_list()
    dataset_path = DATASET_PATHS[dataset]
    # filter puns
    if dataset.startswith("bulwer-puns"):
        bulwer_full = pd.read_csv(
            DATASET_PATHS["bulwer"], sep="\t", encoding="utf-8", quoting=quoting
        )
        pun_index = bulwer_full[bulwer_full["category"] == "Vile Puns"].index
        if dataset == "bulwer-puns":
            return bulwer_full.iloc[pun_index]["sentence"].tolist()
        elif dataset == "bulwer-puns-swap":
            with open(dataset_path, "r", encoding="utf-8") as file:
                data = [s for s in file.read().splitlines() if len(s.strip()) > 0]
            return np.array(data)[pun_index].tolist()
    # csv or tsv
    if dataset_path.endswith("csv") or dataset_path.endswith("tsv"):
        sep = "," if dataset_path.endswith("csv") else "\t"
        col = "text" if dataset == "claude" or dataset.endswith("test") else "sentence"
        return pd.read_csv(dataset_path, sep=sep, encoding="utf-8", quoting=quoting)[
            col
        ].to_list()
    # txt
    with open(dataset_path, "r", encoding="utf-8") as file:
        return [s for s in file.read().splitlines() if len(s.strip()) > 0]
