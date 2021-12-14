from carla_utils import carla

import os, sys
from os.path import join


if __name__ == "__main__":
    current_path = os.getcwd()

    args = ''
    for arg in sys.argv[1:]:
        args += ' ' + arg

    server_path = os.environ['CARLAPATH']

    target_path = join(server_path, 'PythonAPI/util/')
    os.chdir(target_path)

    cmd = sys.executable + ' ' + 'config.py'
    cmd = cmd + args

    print('\nrunning:\n    '+cmd+'\n')
    os.system(cmd)

