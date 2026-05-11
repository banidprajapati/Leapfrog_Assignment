import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance with colored console output."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)

        parent_dir = Path.cwd()
        log_dir = parent_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        # Rotates logs every 10 days, keeping the active file as `app.log` and archiving old logs with date suffixes.
        timed_handler = TimedRotatingFileHandler(
            filename=log_dir / "app.log", when="D", interval=1, backupCount=10
        )
        timed_handler.setLevel(logging.DEBUG)
        timed_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(console_handler)
        logger.addHandler(timed_handler)

    return logger
