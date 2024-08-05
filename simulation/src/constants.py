import logging
import os
import uuid
from types import SimpleNamespace


# path definitions
SIM_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SIM_ROOT_PATH)


# resources
RES = SimpleNamespace(
    UR5_MODEL=os.path.join(SIM_ROOT_PATH, 'model/universal_robots_ur5e/ur5e.xml')
)

# logging options
LOGGER_OPTIONS = SimpleNamespace(
    # Options: [logging.FATAL, logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG, logging.NOTSET]
    LEVEL=logging.INFO,
    FILE=os.path.join(SIM_ROOT_PATH, f'log/sim-log-{uuid.uuid5(uuid.NAMESPACE_X500, "log.local")}.log'),
    NAME='default_logger'
)

# check the resolved values of the constants
if __name__ == '__main__':
    current_module = globals()
    constants = {name: value for name, value in current_module.items() 
                 if not name.startswith('__') and not callable(value)}
    
    print('Constants:')
    for name, value in constants.items():
        print(f'{name} = {value}')
