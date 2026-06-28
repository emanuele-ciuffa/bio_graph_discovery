import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler


class Logging_handler:
    def __init__(self, config_handler):
        # Read the base path (e.g., 'logs/pipeline.log')
        self.base_path = config_handler.read_property("log.path")
        self.level = config_handler.read_property("log.level") or "INFO"

        # Generate dynamic filename with timestamp for each run
        name, ext = os.path.splitext(self.base_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"{name}_{timestamp}{ext}"

        # Automatically create the folder if it doesn't exist
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        self.logger = logging.getLogger("bio_graph_discovery")
        self.logger.setLevel(self.level)

        if not self.logger.handlers:
            # The handler will now find the directory ready for the file
            file_handler = RotatingFileHandler(
                self.log_file, maxBytes=5 * 1024 * 1024, backupCount=2
            )

            formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def get_logger(self, module_name):
        return logging.getLogger(f"bio_graph_discovery.{module_name}")

