# app_logger.py
import logging
import sys

class LoggerFactory:
    _configured = False

    @staticmethod
    def _configure():
        if LoggerFactory._configured:
            return

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout)
            ],
        )
        LoggerFactory._configured = True

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        LoggerFactory._configure()
        return logging.getLogger(name)
