import logging
import os

def setup_logger(name: str, log_file: str, level=logging.INFO):
    """"
        Creates and configure a timestamped log handler writing directly to disk.

        Arguments:
            name(str): name of the lgging module.
            log_file(str): file path of saved logs
            level: severity visibility reporting thresholds(Default - INFO)
    """

    # generate new folder if log folder is missing 
    logs_dir = os.path.dirname(log_file)
    if logs_dir:
        os.makedirs(logs_dir, exist_ok=True)

    # Initialising a FileHandler to put logs directly to disk file
    handler = logging.FileHandler(log_file)
    # format of the log entry
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # assigner the format to the FileHandler
    handler.setFormatter(formatter)

    # get named systm logger
    logger = logging.getLogger(name)
    # set log severity level
    logger.setLevel(level)

    # check for existing handlers and add file text handler if needed
    if not logger.handlers:
        logger.addHandler(handler)

    # returns the logger object
    return logger
