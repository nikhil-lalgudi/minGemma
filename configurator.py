import argparse
import importlib.util
from typing import Any, Dict

def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from a Python file."""
    spec = importlib.util.spec_from_file_location("config", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return {k: v for k, v in vars(config_module).items() if not k.startswith('__')}

def parse_cli_args() -> Dict[str, Any]:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Configuration parser")
    parser.add_argument('config_file', help="Path to the configuration file")
    parser.add_argument('--batch_size', type=int, help="Batch size for training")
    # Add more arguments as needed
    return vars(parser.parse_args())

def merge_configs(base_config: Dict[str, Any], cli_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge CLI arguments into the base configuration."""
    return {**base_config, **{k: v for k, v in cli_config.items() if v is not None}}

def get_config() -> Dict[str, Any]:
    """Load and merge configurations."""
    cli_args = parse_cli_args()
    base_config = load_config(cli_args['config_file'])
    return merge_configs(base_config, cli_args)
"""
Example Usage:
from config_loader import get_config

def main():
    config = get_config()
    batch_size = config['batch_size']
    ...

if __name__ == "__main__":
    main()
"""