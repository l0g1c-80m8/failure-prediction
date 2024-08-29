import logging

# local imports
from src.constants import LOGGER_OPTIONS


LOGGER = logging.getLogger(LOGGER_OPTIONS.NAME)
logging.basicConfig(filename=LOGGER_OPTIONS.FILE, level=LOGGER_OPTIONS.LEVEL)
