import os
from types import SimpleNamespace


SIM_ROOT_PATH = os.path.join(__file__, '../..')
REPO_ROOT = os.path.join(SIM_ROOT_PATH, '..')


RES = SimpleNamespace(
    UR5_MODEL=os.path.join(SIM_ROOT_PATH, 'model/ur5e.xml')
)


if __name__ == '__main__':
    current_module = globals()
    constants = {name: value for name, value in current_module.items() 
                 if not name.startswith('__') and not callable(value)}
    
    print('Constants:')
    for name, value in constants.items():
        print(f'{name} = {value}')
