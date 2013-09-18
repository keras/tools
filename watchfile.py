#!/usr/bin/env python

import os
import sys
import fnmatch
import time
import subprocess
import signal
import argparse

def sigint_handler(a,b):
    print
    exit(0)

signal.signal(signal.SIGINT, sigint_handler)


parser = argparse.ArgumentParser(description='Run process on file update')
parser.add_argument("command", metavar="command", type=str)
parser.add_argument("args", metavar="args", type=str, nargs="*")
parser.add_argument("--name", dest="name", nargs=1, default="*", help="file name filter")
parser.add_argument("--path", dest="path", nargs=1, default=".", help="search path for scanner")

args = parser.parse_args()

cmd = [args.command] + args.args

def check_file(modify_times, path):
    try:
        modified = os.stat(path).st_mtime
    except Exception:
        return False
    if path not in modify_times:
        modify_times[path] = modified
        return True
    if modify_times[path] != modified:
        modify_times[path] = modified
        return True


matches = set()
mtimes = dict()
name_filter = args.name
while True:
    for root, dirnames, filenames in os.walk("".join(args.path)):
        for filename in fnmatch.filter(filenames, "".join(args.name)):
            full_path = os.path.join(root, filename)
            if full_path not in matches:
                matches.add(full_path)

    do_check = False
    for f in matches:
        if check_file(mtimes, f):
            do_check = True
    if do_check:
	start = time.time()
        subprocess.call(cmd)
	end = time.time()
	print "Ran '%s' in %f seconds\n" % (" ".join(cmd), end-start)
    time.sleep(1)
