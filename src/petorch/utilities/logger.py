import inspect
import logging
import os
import sys

import loguru

logger = loguru.logger

class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get the corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

logger_info_set = ["lightning.pytorch"]
logger_debug_set = ["__main__", "petorch"]

def is_info_logger(name: str):
    for l in logger_info_set:
        if l in name:
            return True
    return False


def is_debug_logger(name: str):
    for l in logger_debug_set:
        if l in name:
            return True
    return False

DEFAULT_LOGGER_FORMAT = ("<green><b>[{time:YYYY-MM-DD HH:mm:ss.SSS}]</b></green> "
                         "<level>[PID {process.id} | {level: <8}]</level> "
                         "<cyan><i>{name}:{function}:{line}</i></cyan> "
                         "<level> :~ {message}</level>"
                         )
def setup_logger(level=None):
    level = level or os.getenv("LOG_LEVEL","INFO")
    def log_filter(record: "loguru.Record") -> bool:
        name = record.get("name")
        level_no = record["level"].no
        if name:
            if is_info_logger(name):
                return level_no >= logger.level("INFO").no

            if is_debug_logger(name):
                return level_no >= logger.level("DEBUG").no

        # Does not have a name, or name is not prespecified
        return level_no >= logger.level("WARNING").no

    logging.basicConfig(handlers=[InterceptHandler()], level=logging.NOTSET, force=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        filter=log_filter,
        format=DEFAULT_LOGGER_FORMAT
    )
    logger.info(f"Log level is set to `{level}`.")
    return logger

setup_logger()
