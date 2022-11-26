from pathlib import Path
import joblib
import json
from dataclasses import dataclass

from typing import Dict, List
from convxai.utils import *


@dataclass
class AspectPattern:
    _id: str
    aspect_sequence: List[int]


class TfidfAspectModel:

    def __init__(self, conference):
        self.configs = parse_system_config_file()
        model_path = self.configs['conversational_xai']['checkpoints_root_dir'] + self.configs['conversational_xai']['xai_writing_aspect_prediction_dir']+f"/{conference}"
        self.vectorizer = joblib.load(Path(model_path, "vectorizer.joblib"))
        self.model = joblib.load(Path(model_path, "model.joblib"))
        with open(Path(model_path, "centers.json"), 'r', encoding='utf-8') as infile:
            self.centers = json.load(infile)

    def get_feature(self, aspect_sequence: List[int]):
        return " ".join([
            f"{aspect}"
            for i, aspect in enumerate(aspect_sequence)
        ])

    def predict(self, aspect_sequence: List[int]) -> AspectPattern:
        feature = self.get_feature(aspect_sequence)
        vectors = self.vectorizer.transform([feature])
        labels = self.model.predict(vectors)
        result = AspectPattern(*self.centers[labels[0]])
        return result

