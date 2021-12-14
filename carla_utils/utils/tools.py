

def default_argparser():
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--timeout',
        default=2.0,
        type=float,
        help='(default: 2.0)')
    argparser.add_argument(
        '--map',
        dest='map_name',
        help='load a new map, use --list to see available maps')
    
    argparser.add_argument('--real', action='store_true', help='real mode (default: False)')   ### ! warning to delete maybe
    argparser.add_argument('--render', action='store_true', help='render the env (default: False)')
    return argparser
