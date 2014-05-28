#!/usr/bin/env python

import argparse
import collections
import datetime
import itertools
import re
import time
import sys
import os
import math
import subprocess


parser = argparse.ArgumentParser(description='show diff')
parser.add_argument(
    '--cmd',
    help='Program output to monitor'
)
parser.add_argument(
    '--file',
    help='File to monitor'
)
parser.add_argument('-i', '--interval', dest='interval', type=float, default=2.0)
parser.add_argument(
    '-u', '--units', default='bin', metavar='U',
    help='units, bin or dec'
)


def read_file(file_name):
    with file(file_name) as fh:
        return fh.read()

def read_cmd(cmd):
    proc = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = proc.communicate()
    return stdout



class Node(object):

    def __init__(self, match=None, source=None, start=None, end=None, groups=None, tpe=None, text=None):
        self.source = source or match.string
        self.start  = start  or match.start()
        self.end    = end    or match.end()
        self.groups = groups or match.groupdict()
        self.text   = (text   or self.source[self.start : self.end]).strip()
        self.tpe    = tpe

    def __str__(self):
        return 'Node[num:%6s, %r]' % (
            self.is_numeric,
            self.text,
        )

    def __repr__(self):
        return '%s(%s)' % (
            self.__class__.__name__,
            ','.join(
                '%s=%s' % item
                for item in self.__dict__.iteritems()
            )
        )


    def __len__(self):
        return self.end - self.start

    @property
    def is_numeric(self):
        return self.tpe == 'num'

    def get_number(self):
        if self.tpe == 'num':
            return float(self.text)


def parse_blob(text):
    separator = r'[:=|]+'
    number = r'-?\d+(?:\.\d+)?'
    item_parser = re.compile(
        r'(?P<sep>\s*{sep})|(?P<num>\s*{num})(?=\s|$)|(?P<label>\s*[^\s:=|]+)'.format(
            num=number,
            sep=separator,
        )
    )
    types = ('sep', 'num', 'label')
    lines = text.split('\n')

    def parse_line(line):
        s = item_parser.scanner(line)
        m = s.match()
        while m:
            tpe, val = [(t,g) for g,t in zip(m.groups(), types) if g is not None][0]
            yield Node(m, text=val, tpe=tpe)
            m = s.match()

    for line in lines:
        yield list(parse_line(line))

class Formatter(object):
    symbols = ['u', 'm', '', 'k', 'M', 'G', 'T', 'E']
    def __init__(self, units):
        self.units = units

    def pretty_num(self, value):
        if self.units == 'bin':
            power = 1024
        elif self.units == 'dec':
            power = 1000
        else:
            raise RuntimeException('invalid unit %r', self.unit)

        if value != 0:
            e = int(math.log(abs(value))/math.log( power ))
            if e != 0:
                v = float(value) / power**e
            else:
                v = value
        else:
            e = 0
            v = value
        frac, _ = math.modf(abs(v))
        if frac < 0.0001:
            v = int(v)
        return v, self.symbols[e+2]

    def fmt(self, value, width):
        v, sym = self.pretty_num(value)
        if sym:
            if isinstance(v, float):
                decimals = 4-len(str(int(v)))
                tpl = '%%%s.%sf%s' % (width-1, decimals, sym)
            else:
                tpl = '%%%ss%s' % (width-1, sym)
        else:
            if isinstance(v, float):
                decimals = 4-len(str(int(v)))
                tpl = '%%%s.%sf' % (width, decimals)
            else:
                tpl = '%%%ss' % width

        return tpl % v

def main():
    args = parser.parse_args()
    formatter = Formatter(
        units = args.units
    )

    if args.cmd:
        reader = lambda: read_cmd(args.cmd)
    elif args.file:
        reader = lambda: read_file(args.file)
    else:
        parser.print_help()
        sys.exit(1)

    t0 = time.time()
    last = None
    while True:
        current = list(parse_blob(reader()))
        t1 = time.time()
        os.system('clear')
        print "Every %ss: file: %s" % (args.interval, args.file)
        if last:
            line_z = itertools.izip(last, current)
            for linepair in line_z:
                node_z = itertools.izip(*linepair)
                for last_n, current_n in node_z:
                    if last_n.is_numeric and current_n.is_numeric:
                        l = last_n.get_number()
                        c = current_n.get_number()
                        formatted = formatter.fmt(
                            (c-l)/(t1-t0),
                            len(current_n)
                        )
                        sys.stdout.write( formatted )
                    else:
                        pass
                        sys.stdout.write( current_n.text )
                    #print current_n
                print '\n',
        else:
            for line in current:
                for node in line:
                    sys.stdout.write( node.text )
                print '\n',
        last = current
        t0 = t1

        time.sleep(args.interval)


if __name__ == '__main__':
    main()
