import yaml
import os
from dotenv import load_dotenv

load_dotenv()

class LoadConfig:
    """Loads configuration settings from YAML file."""
    def __init__(self, config_path="configs/app_config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.data_directory = self.config["directories"]["data_directory"]
        self.persist_directory = self.config["directories"]["persist_directory"]
        self.embedding_model = self.config["embedding_model_config"]["model"]
        self.llm_model = self.config["llm_config"]["model"]
        self.temperature = self.config["llm_config"]["temperature"]
        self.chunk_size = self.config["splitter_config"]["chunk_size"]
        self.chunk_overlap = self.config["splitter_config"]["chunk_overlap"]
        self.k = self.config["retrieval_config"]["k"]
