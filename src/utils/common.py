import os
import sys
import gzip
import yaml
import json
import logging
import joblib
import pandas as pd
from typing import Any
from pathlib import WindowsPath, Path
from box import ConfigBox
from ensure import ensure_annotations
from typing import Iterator, Dict, Any, Union, List
from box.exceptions import BoxValueError


# For Logging
logging_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

logging_directory = "logs"
os.makedirs(logging_directory, exist_ok=True)
log_filepath = os.path.join(logging_directory, "running_logs.log")

logging.basicConfig(
    level=logging.INFO,  # set minimum log level to respond
    format=logging_format,  # set the log output format
    handlers=[
        logging.FileHandler(log_filepath),  # set file to write the log messages
        logging.StreamHandler(sys.stdout),  # send log messages to the system output
    ],
)  # set logging configuration

logger = logging.getLogger("credit-score-modelling-logger")  # get logger


def interpret_information_value(score: float) -> str:
    """
    Interpret Information Score

    Args:
        score (float): Information Value.
    Returns:
        str: Interpretation of the score.
    """
    if score < 0.02:
        return "Not Predictive"
    elif 0.02 <= score and score < 0.1:
        return "Weak Predictive"
    elif 0.1 <= score and score < 0.3:
        return "Medium Predictive"
    elif 0.3 <= score and score < 0.5:
        return "Strong Predictive"
    else:
        return "Very Strong Predictive"
