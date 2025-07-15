import logging
import logging.config
from datetime import datetime

import yaml

from .._const import LOG_CONFIG_FILE, LOG_DIR


def logging_setup(exp_name: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(LOG_CONFIG_FILE, "rt") as f:
            config = yaml.safe_load(f.read())
    except FileNotFoundError:
        print(f"Error: Logging configuration file not found at {LOG_CONFIG_FILE}")
        return False
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        return False

    if "handlers" in config and "file" in config["handlers"]:
        timestamp = datetime.now().strftime("%Y%m%d_%H:%M:%S")
        # Construct the full log file path within the predefined LOG_DIR
        log_filename = LOG_DIR / f"{exp_name}_{timestamp}.log"
        config["handlers"]["file"]["filename"] = str(log_filename)
        print(f"Configuring logging to: {log_filename}")
    else:
        print(
            "Warning: 'file' handler not found in logging configuration. Dynamic filename not set."
        )
        return False

    logging.config.dictConfig(config)
    return True
