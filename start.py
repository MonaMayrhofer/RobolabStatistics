#!/usr/bin/env python
# coding=utf-8
import argparse
import importlib
import sys
import os
import numpy as np


ALIASES = {
    "pong": "facepong2.pong"
}


def main():
    parser = argparse.ArgumentParser(description='Execute apps')
    parser.add_argument('app', type=str, help='The app to be started.')
    parser.add_argument('arguments', type=str, nargs='*',
                        help='Supply arguments to specified app')
    args = parser.parse_args()

    app = args.app
    arguments = args.arguments

    if app in ALIASES:
        app = ALIASES[app]

    print(app)
    print(arguments)

    app_path = "apps.{}".format(app)

    print("====== [{}] ======".format(app_path))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    sys.argv = np.concatenate(([app], arguments))
    importlib.import_module(app_path).main()


if __name__ == '__main__':
    main()
