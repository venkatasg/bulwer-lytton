"""
Humor detection models from:
    Baranov, A., Kniazhevsky, V., & Braslavski, P. (2023, December).
    You told me that joke twice:
    A systematic investigation of transferability and robustness of humor detection models.
    In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing
    (pp. 13701-13715).
"""
from typing import List

from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    TextClassificationPipeline,
)
from util import HumorDetector

# seeds available: 23, 977, 693, 453, 47
SUPPORTED_HF_MODELS = {
    "humor-detection-comb",  # authors say this is their best model
    # each model was trained with multiple seeds, basically chose randomly between them
    "humor-detection-pun-of-the-day",
}


class HumorResearchRobertaDetector(HumorDetector):
    """
    Predict using pre-trained RoBERTa models on various humor datasets
    """

    def __init__(self, hf_model: str):
        assert (
            any(hf_model.startswith(supported_model) for supported_model in SUPPORTED_HF_MODELS)
        ), f"Must choose from supported models {SUPPORTED_HF_MODELS}"
        self.hf_model = hf_model
        self.model = f"Humor-Research/{hf_model}"

    def predict(self, sentences: List[str]) -> List[float]:
        model = RobertaForSequenceClassification.from_pretrained(self.model)
        tokenizer = RobertaTokenizerFast.from_pretrained(
            "roberta-base", max_length=512, truncation=True
        )
        pipe = TextClassificationPipeline(
            model=model, tokenizer=tokenizer, max_length=512, truncation=True
        )
        results = pipe(sentences)
        # convert results to 0-1
        return [
            result["score"] if result["label"] == "LABEL_1" else 1 - result["score"]
            for result in results
        ]

    def model_name(self):
        return self.hf_model
