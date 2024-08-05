import logging
import os
import uuid
from types import SimpleNamespace


SIM_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SIM_ROOT_PATH)


RES = SimpleNamespace(
    UR5_MODEL=os.path.join(SIM_ROOT_PATH, 'model/universal_robots_ur5e/ur5e.xml')
)

LOG_LEVEL = logging.INFO  # [FATAL/CRITICAL, ERROR, WARNING/WARN, INFO, DEBUG, NOTSET]
LOG_FILE = os.path.join(SIM_ROOT_PATH, f'log/log-{uuid.uuid5()}.log')

if __name__ == '__main__':
    current_module = globals()
    constants = {name: value for name, value in current_module.items() 
                 if not name.startswith('__') and not callable(value)}
    
    print('Constants:')
    for name, value in constants.items():
        print(f'{name} = {value}')
