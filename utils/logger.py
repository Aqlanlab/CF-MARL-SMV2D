import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class TrainingLogger:
    def __init__(
        self,
        log_dir: str = "./logs",
        tensorboard_dir: Optional[str] = None,
        log_level: int = logging.INFO
    ):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self._log_dir / f"training_{timestamp}.log"

        self._logger = logging.getLogger("CF-MARL-SMV2D")
        self._logger.setLevel(log_level)

        if not self._logger.handlers:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)

            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self._logger.addHandler(file_handler)
            self._logger.addHandler(console_handler)

        self._tensorboard_writer = None
        if tensorboard_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tensorboard_writer = SummaryWriter(tensorboard_dir)
            except ImportError:
                self._logger.warning("TensorBoard not available")

        self._metrics_history = []

    def log_info(self, message: str):
        self._logger.info(message)

    def log_warning(self, message: str):
        self._logger.warning(message)

    def log_error(self, message: str):
        self._logger.error(message)

    def log_debug(self, message: str):
        self._logger.debug(message)

    def log_scalar(self, tag: str, value: float, step: int):
        if self._tensorboard_writer:
            self._tensorboard_writer.add_scalar(tag, value, step)

        self._metrics_history.append({
            "step": step,
            "tag": tag,
            "value": value,
            "timestamp": datetime.now().isoformat()
        })

    def log_dict(self, prefix: str, values: Dict[str, Any], step: int):
        for key, value in values.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f"{prefix}/{key}", value, step)

    def save_metrics(self):
        metrics_file = self._log_dir / "metrics_history.json"
        with open(metrics_file, "w") as f:
            json.dump(self._metrics_history, f, indent=2)

    def close(self):
        if self._tensorboard_writer:
            self._tensorboard_writer.close()
        self.save_metrics()