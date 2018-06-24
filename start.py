#!/usr/bin/env python
# coding=utf-8
import argparse
import importlib
import sys
import os
import numpy as np


ALIASES = {
    "pong": "facepong2.pong",
    "paparazzi": "mainloop_and_utils.paparazzi",
    "mainloop": "mainloop_and_utils.mainloop",
    "converter": "mainloop_and_utils.converter"
}


def main():
    keyargs = sys.argv
    otherind = 0
    for ind, val in enumerate(sys.argv):
        if not val[0] == "-":
            otherind = ind
        if otherind > 0:
            break

    parser = argparse.ArgumentParser(description='Execute apps')
    parser.add_argument('app', type=str, help='The app to be started.')
    parser.add_argument('arguments', type=str, nargs='*',
                        help='Supply arguments to specified app')
    args = parser.parse_args(keyargs[1:otherind+1])
    app = args.app

    for i in range(otherind):
        keyargs.remove(keyargs[0])
    arguments = keyargs

    if app in ALIASES:
        app = ALIASES[app]

    app_path = "apps.{}".format(app)

    print("====== [{}] ======".format(app_path))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    sys.argv = arguments
    importlib.import_module(app_path).main()


if __name__ == '__main__':
    main()
