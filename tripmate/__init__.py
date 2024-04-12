import logging

console_format = logging.Formatter('%(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_format)

logger = logging.getLogger('tripmate')
logger.propagate = False
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)