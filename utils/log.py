import os
import sys
from colorama import init
import logging
import colorlog
import re

# LOG_COLOR_CONFIG = {
#     'DEBUG': 'cyan',
#     'WARNING': 'yellow',
#     'ERROR': 'red',
#     'CRITICAL': 'red',
# }

def set_color(log, color, highlight=True):
    color_set = ['black', 'red', 'green',
                 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = '\033['
    if highlight:
        prev_log += '1;3'
    else:
        prev_log += '0;3'
    prev_log += str(index) + 'm'
    return prev_log + log + '\033[0m'


class RemoveColorFilter(logging.Filter):
    def filter(self, record):
        if record:
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            record.msg = ansi_escape.sub('', str(record.msg))
        return True


class Logger(object):
    def __init__(self, logname, now):
        path = os.path.join('log-files', now.split('_')[0])

        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, now.split('_')[1] + '-' + '.txt')
        print('saving log to ', path)

        self.terminal = sys.stdout
        self.file = None

        self.open(path)

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message:
            is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def close(self):
        self.file.close()
