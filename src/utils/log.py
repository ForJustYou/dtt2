# src/utils/log.py
import csv
import os
from datetime import datetime
from typing import Dict, Iterable, Optional
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
Logger = None

DEFAULT_FIELDS = [
    'epoch', 'iter', 'train_loss', 'vali_loss', 'test_loss',
    'vali_mse', 'test_mse', 'speed', 'cost_time', 'epoch_time'
]


def init_csv_logger(arg_model,arg_data):
    global Logger
    if Logger is None: 
        Logger = CSVLogger(
            fieldnames=DEFAULT_FIELDS,
            model=arg_model,
            data=arg_data 
        )

def default_log_filename(model: str, data: str, log_dir: str = DEFAULT_LOG_DIR) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    return os.path.join(log_dir, f"{model}_{data}_{ts}.csv")

class CSVLogger:
    """
    统一写入一个 CSV 文件：
    - 初始化时写表头（若文件不存在）。
    - log(row) 追加一行；缺失字段自动填空。
    """

    def __init__(
        self,
        fieldnames: Iterable[str],
        filename: Optional[str] = None,
        model: str = "model",
        data: str = "data",
        log_dir: str = DEFAULT_LOG_DIR,
    ):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_path = filename or default_log_filename(model, data, self.log_dir)
        self.fieldnames = list(fieldnames)

        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: Dict, default: str = "") -> None:
        """追加一行；缺失字段填 default（默认空字符串）。"""
        to_write = {k: row.get(k, default) for k in self.fieldnames}
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(to_write)

    @property
    def path(self) -> str:
        return self.log_path
