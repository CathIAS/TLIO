# Copyright 2004-present Facebook. All Rights Reserved.

# Put some color in you day!
import logging
import sys


try:
    import coloredlogs

    coloredlogs.install()
except BaseException:
    pass

logging.basicConfig(
    stream=sys.stdout,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    level=logging.INFO,
)
