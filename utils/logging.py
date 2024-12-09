import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class LoggerSetup:
    """Handles the setup and configuration of logging for the application."""

    @staticmethod
    def setup_logger(
        logger_name: str, log_file: Optional[str] = None, level: int = logging.INFO
    ) -> logging.Logger:
        """
        Sets up a logger with both file and console handlers.

        Args:
            logger_name: Name of the logger
            log_file: Optional path to log file
            level: Logging level

        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)

        # Create formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")

        # Set up console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Set up file handler if log_file is provided
        if log_file:
            # Create logs directory if it doesn't exist
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

            # Add timestamp to log filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_handler = logging.FileHandler(log_dir / f"{log_file}_{timestamp}.log")
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)

        return logger
