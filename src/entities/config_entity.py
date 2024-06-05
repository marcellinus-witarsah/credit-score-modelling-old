from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Data class for storing data ingestion configuration.

    Attributes:
        root_dir (Path): Root directory for data ingestion.
        source_path (Path): Source path of the data.
        target_path (Path): Target path for the processed data.
    """
    root_dir: Path
    source_path: Path
    target_path: Path
