import carla_utils as cu

import os

if __name__ == "__main__":
    config = cu.basic.YamlConfig()
    args = cu.utils.default_argparser().parse_args()
    config.update(args)

    carla_version = cu.system.get_carla_version()

    cmd1 = 'docker pull mjxu96/carlaviz:' + carla_version + ' && '
    cmd2 = 'docker run -it --network="host" -e CARLAVIZ_BACKEND_HOST=localhost -e CARLA_SERVER_HOST="{}" -e CARLA_SERVER_PORT={} mjxu96/carlaviz:{}'.format(
        config.host, str(config.port), carla_version
    )
    cmd = cmd1 + cmd2

    print('\nrunning:\n    '+cmd+'\n')
    print('carlaviz:{} at http://localhost:{}/ (Press CTRL+C to quit)'.format(carla_version, str(8080)))
    print('\n\n')

    os.system(cmd)

