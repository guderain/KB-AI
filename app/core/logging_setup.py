import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def setup_logging() -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler_exists = any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, TimedRotatingFileHandler)
        for handler in root_logger.handlers
    )
    if not console_handler_exists:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    project_root = Path(__file__).resolve().parents[2]
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "backend.log"

    file_handler_exists = any(
        isinstance(handler, TimedRotatingFileHandler) and Path(handler.baseFilename) == log_file
        for handler in root_logger.handlers
    )
    if not file_handler_exists:
        file_handler = TimedRotatingFileHandler(
            filename=str(log_file),
            when="midnight",
            interval=1,
            backupCount=14,
            encoding="utf-8",
        )
        file_handler.suffix = "%Y-%m-%d"
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
