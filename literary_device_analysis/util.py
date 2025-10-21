import json
import re

import pandas as pd


def load_assignments(assignment_path, verbose=False):
    with open(assignment_path) as f:
        data = [json.loads(line) for line in f]

    all_topics = set()
    for instance in data:
        instance["topics"] = []
        for response in instance["responses"].split("\n"):
            topic = re.match(r"^\s*\[1\] ([^:]+): ", response)
            if topic:
                instance["topics"].append(topic.group(1))
                all_topics.add(topic.group(1))
            elif verbose:
                print("Failed topic: ", response)
    all_topics = list(all_topics)
    df = pd.DataFrame.from_records(data, columns=["id", "topics"])
    # from https://stackoverflow.com/a/71937657
    df = pd.concat((df[["id"]], df["topics"].str.join('|').str.get_dummies()), axis=1)
    df["dataset"] = df["id"].apply(lambda s: s.split("-")[0])
    return df


def load_features(feature_path):
    feature_re = r"\[1\] ([\w ]+) \(Count: \d+\): (.*)\n"
    with open(feature_path, "r", encoding="utf-8") as f:
        features_text = f.readlines()
    features = {}
    for feature_text in features_text:
        match = re.match(feature_re, feature_text)
        features[match.group(1)] = match.group(2)
    assert len(features) == len(features_text)
    return features


def load_text(text_path):
    out = {}
    with open(text_path, "r") as f:
        for line in f:
            json_obj = json.loads(line)
            out[json_obj["id"]] = json_obj["text"]

    return out