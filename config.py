import yaml

def load_config(file_path):
    with open(file_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def save_config(config, filepath):
    with open(filepath, "w") as f:
        yaml.dump(config, f)