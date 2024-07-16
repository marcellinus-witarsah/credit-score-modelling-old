from enum import Enum


class FileConstants(Enum):
    CONFIG_FILE_PATH = "config.yaml"
    SCHEMA_FILE_PATH = "schema.yaml"


class GlobalConstants(Enum):
    RANDOM_STATE = 42


class CreditScoreReportConstants(Enum):
    CREDIT_LEVELS_DESCRIPTIONS = {
        1: "Very Poor",
        2: "Poor",
        3: "Below Average",
        4: "Average",
        5: "Above Average",
        6: "Good",
        7: "Very Good",
        8: "Excellent",
        9: "Exceptional",
    }
    LEVEL_BOUNDS = [float("-inf"), 350, 400, 450, 500, 550, 600, 650, 700, float("inf")]
    RANDOM_STATE = 42
