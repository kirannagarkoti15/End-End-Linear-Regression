import yaml
import os

class configuration:
    def __init__(self):
        self.root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = self.root + "/config.yaml"

    def load_config(self):
        with open(self.config_path, "r") as f:
            configurations = yaml.safe_load(f)
        return configurations