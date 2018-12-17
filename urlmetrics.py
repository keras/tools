#!/usr/bin/env python

"""
test with `nosetests path_metrics.py`
TODO:
  - deeper level grandchildren heuristics on compaction pass1
  - parse also url parameters
"""

from __future__ import print_function
import fileinput
import unittest
import re
import collections


class TreeTest(unittest.TestCase):
    def test_create(self):
        tree = Tree()
        tree.append(' a b'.split(), 1)
        tree.append(' a b c'.split(), 1)
        tree.append(' a b c'.split(), 1)
        self.assertEqual(tree.count_nodes(), 4)

    def test_merge_trees(self):
        tree_a = Tree()
        tree_b = Tree()

        tree_a.append('a a'.split(), 1)
        tree_b.append('a b'.split(), 2)

        tree_a.merge_tree(tree_b)

        print(tree_a.debug_dict())
        self.assertEqual(
            tree_a.debug_dict(),
            {'a': {'a': 1, 'b': 2}}
        )

    def test_merge_children(self):
        tree = Tree()
        tree.append('a'.split(), 1)
        b_node = tree.append('a b'.split(), 2)
        tree.append('a b ba'.split(), 3)
        tree.append('a b bb'.split(), 4)

        self.assertEqual(len(b_node.children), 2)
        wc_node = b_node.merge_children()
        self.assertIs(wc_node, b_node.children[Tree.wildcard])
        self.assertEqual(len(b_node.children), 1)
        self.assertEqual(wc_node.values, Tree.Values(min=3, max=4, sum=7, count=2))

        tree.append('a b bb'.split(), 5)
        self.assertEqual(len(b_node.children), 1)

    def test_merge_children_deep(self):
        tree = Tree()
        tree.append('a'.split(), 1)
        b_node = tree.append('a b'.split(), 2)
        tree.append('a b ba bd'.split(), 3)
        tree.append('a b bb bd'.split(), 4)
        tree.append('a b bc bd'.split(), 5)

        """ tree.debug_yaml()
        a:
          b:
            ba:
              bd: 3
            bb:
              bd: 4
            bc:
              bd: 5
        """
        self.assertEqual(tree.count_nodes(), 9)
        b_node.merge_children()

        """ tree.debug_yaml()
        a:
          b:
            '*':
              bd: 12
        """
        self.assertEqual(tree.count_nodes(), 5)


class Tree(object):
    wildcard = '*'

    class NodeStatistic(object):
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def __str__(self):
            return 'Stats[{}]'.format(', '.join(
                '{}={:}'.format(k, v)
                for k, v in sorted(self.__dict__.items())
            ))

    class Values(object):
        def __init__(self, min, max, sum, count):
            (
                self.min,
                self.max,
                self.sum,
                self.count,
            ) = min, max, sum, count

        @classmethod
        def for_value(cls, value):
            return cls(value, value, value, 1)

        def __str__(self):
            return 'Tree.Values(min={min}, max={max}, sum={sum}, count={count}, avg={avg})' \
                .format(
                    avg = float(self.sum) / self.count,
                    **self.__dict__
                )

        def __repr__(self):
            return 'Tree.Values(min={min}, max={max}, sum={sum}, count={count})' \
                .format(
                    **vars(self)
                )

        def __eq__(self, other):
            return type(self) == type(other) and self.__dict__ == other.__dict__

        def copy(self):
            return Tree.Values(self.min, self.max, self.sum, self.count)

        def add(self, value):
            self.min = min(self.min, value)
            self.max = max(self.max, value)
            self.sum = self.sum + value
            self.count += 1

        def add_values(self, other):
            self.min = min(self.min, other.min)
            self.max = max(self.max, other.max)
            self.count += other.count
            self.sum += other.sum

    def __init__(self, name=''):
        self.values = None
        self.children = {}

    @classmethod
    def make(cls, name, values, children):
        obj = cls()
        obj.values = values
        obj.children = children
        return obj

    def debug(self):
        import json
        return json.dumps(self.debug_dict(), indent=2)

    def debug_dict(self):
        if self.children:
            return {
                k: v.debug_dict()
                for k, v in self.children.items()
            }
        else:
            return self.values.sum

    def debug_yaml(self):
        import yaml
        return yaml.dump(self.debug_dict(), default_flow_style=False)

    def __repr__(self):
        return 'Tree.make(%r, %r)' % (
            self.values,
            self.children,
        )

    def append(self, path, value):
        if path:
            head, tail = path[0], path[1:]
            if head in self.children:
                child = self.children[head]
            elif self.wildcard in self.children:
                child = self.children[self.wildcard]
            else:
                child = Tree(head)
                self.children[head] = child
            return child.append(tail, value)
        else:
            if self.values is None:
                self.values = self.Values.for_value(value)
            else:
                self.values.add(value)
            return self

    def merge_children(self):
        children = self.children
        wc_child = Tree(self.wildcard)
        self.children = {
            self.wildcard: wc_child
        }
        for name, child in children.items():
            wc_child.merge_tree(child)
        return wc_child

    def merge_tree(self, other):
        if self.values and other.values:
            self.values.add_values(other.values)
        else:
            self.values = other.values
        for name in other.children.keys():
            if name not in self.children:
                self.children[name] = other.children[name]
            else:
                self.children[name].merge_tree(other.children[name])

    def pretty_print(self, fields):
        #fields = 'count avg min max sum path'.split()
        fields_dict = dict(zip(fields, fields))
        Format = collections.namedtuple('Format', 'header body')
        field_formats = {
            'count': Format(':<7', ':<7'),
            'avg': Format(':<10', ':<10.2f'),
            'min': Format(':<9', ':<9'),
            'max': Format(':<9', ':<9'),
            'sum': Format(':<10', ':<10'),
            'path': Format('', ''),
        }
        header_template = ' '.join(
            '{%s%s}' % (f, field_formats[f].header)
            for f in fields
        )
        body_template = ' '.join(
            '{%s%s}' % (f, field_formats[f].body)
            for f in fields
        )

        print(header_template.format(**fields_dict))
        total = None
        for path, values in self._pretty_iter(''):
            if total is not None:
                total.add_values(values)
            else:
                total = values.copy()
            print(
                body_template.format(
                    path = '/'.join(path),
                    avg = float(values.sum) / float(values.count),
                    **values.__dict__
                )
            )
        print(body_template.format(
            path='[TOTAL]',
            avg = float(values.sum) / float(values.count),
            **total.__dict__
        ))

    def _pretty_iter(self, name):
        if self.values:
            yield [name], self.values
        for child_name, child in sorted(self.children.items()):
            for path, values in child._pretty_iter(child_name):
                yield [name] + path, values

    def count_nodes(self):
        return 1 + sum(child.count_nodes() for child in self.children.values())

    def hitcount(self):
        my_count = self.values.count if self.values else 0
        child_count = sum(child.hitcount() for child in self.children.values())
        return my_count + child_count

    def _node_stats(self, path=tuple()):
        if self.children:
            cn = len(self.children)
            nn = self.count_nodes()
            grand_child_counts = []
            child_hitcounts = []
            for key, child in self.children.items():
                grand_child_counts.append(len(child.children))
                child_hitcounts.append(child.hitcount())
                for item in child._node_stats(path + (key,)):
                    yield item

            # 'gc' = grandchildren
            # 'hc' = hitcount
            gc_mean = sum(grand_child_counts) / float(len(grand_child_counts))
            gc_var = sum((gc_mean - gcc) ** 2 for gcc in grand_child_counts) / len(grand_child_counts)
            hc_mean = sum(child_hitcounts) / float(len(child_hitcounts))
            hc_var = sum((hc_mean - chc) ** 2 for chc in child_hitcounts) / len(child_hitcounts)
            yield self.NodeStatistic(children=cn, all_children=nn, gc_mean=gc_mean, gc_variance=gc_var, hitcount=self.hitcount(), hc_hean=hc_mean, hc_variance=hc_var, path=path)

    def _compact_pass1(self, min_cardinality, child_ratio):
        """
        Try finding good candidates for compaction
        """
        def grandchildren(tree):
            for c in tree.children.values():
                for gc in c.children.keys():
                    yield gc

        def recursive(tree):
            child_count = len(tree.children)
            if child_count > min_cardinality:
                gc = list(grandchildren(tree))
                gc_set = set(gc)

                # if there are more children than grandchildren merge children
                if child_count > len(gc_set) * child_ratio:
                    tree.merge_children()
            for child in tree.children.values():
                recursive(child)
        recursive(self)

    def _compact_pass2(self, hard_count):
        """
        Try to keep the whole tree smaller than :hard_count: nodes
        Find the best candidate (according some scoring) and merge its children
        """
        node_count = self.count_nodes()
        if node_count > hard_count:
            self_stats = list(self._node_stats())
            stats_with_scores = [
                (
                    stat.children / (1 + stat.gc_variance),
                    # stat.children * (stat.hc_variance),
                    # 1.0 / float(stat.hitcount),
                    # stat.gc_variance,
                    stat
                )
                for stat in self_stats
            ]
            sorted_stats = sorted(stats_with_scores, reverse=True)
            score, best_candidate = sorted_stats[0]
            gc_variances = sorted((stat.gc_variance for _, stat in sorted_stats), reverse=True)

            tree = self
            for p in best_candidate.path:
                tree = tree.children[p]

            if best_candidate.gc_variance > gc_variances[0] / 10:
                # do partial merge
                # merge low hitcount nodes together leaving high hitcounts untouched
                children_by_hitcount = sorted(
                    (child.hitcount(), key)
                    for key, child in tree.children.items()
                )
                mean_hitcount = sum(hc for hc, _ in children_by_hitcount) / len(children_by_hitcount)
                wc_child = Tree()
                tree.children[self.wildcard] = wc_child
                noise_children = [key for hitcount, key, in children_by_hitcount if hitcount < mean_hitcount]
                for key in noise_children:
                    child = tree.children.pop(key)
                    wc_child.merge_tree(child)
            else:
                tree.merge_children()

    def compact(self, min_cardinality=10, child_ratio=10, hard_count=500, force=False):
        do_compact = True
        while do_compact:
            self._compact_pass1(min_cardinality, child_ratio)
            self._compact_pass2(hard_count)
            if force and self.count_nodes() > hard_count:
                do_compact = True
            else:
                do_compact = False


class TestReader(unittest.TestCase):
    def test_parse_path(self):
        reader = Reader('/', '^([^?]+)')
        self.assertEqual(reader.parse_path('/foo/bar'), ['foo', 'bar'])

    def test_parse_without_path_arguments(self):
        reader = Reader('/', '^([^?]+)')
        self.assertEqual(reader.parse_path('/foo/bar?hello=world'), ['foo', 'bar'])


class Reader(object):
    def __init__(self, path_separator, path_parser):
        self.path_separator = path_separator
        self.path_parser = path_parser
        self.parser_re = re.compile(self.path_parser)

    def parse_path(self, path):
        match = self.parser_re.match(path)
        path = match.group()
        # split with separator
        path_arr = path.split(self.path_separator)

        # filter empty path segments foo//bar
        path_arr = list(filter(bool, path_arr))
        return path_arr


class TestCompaction(unittest.TestCase):
    @staticmethod
    def dedent(data):
        import textwrap
        return textwrap.dedent(data).strip()

    def test_single_level_child_ratio(self):
        test_data = self.dedent("""
            /foo/1/ooo
            /foo/2/iii
            /foo/3/ooo
            /foo/4/iii
            /foo/5/ooo
        """)

        reader = Reader('/', '^([^?]+)')
        tree = Tree()

        for line in test_data.split('\n'):
            path_arr = reader.parse_path(line)
            tree.append(path_arr, 1)

        self.assertEqual(tree.count_nodes(), 12)

        tree.compact(min_cardinality=3, child_ratio=3)
        self.assertEqual(tree.count_nodes(), 12)

        tree.compact(min_cardinality=3, child_ratio=2)
        self.assertEqual(tree.count_nodes(), 5)

        self.assertEqual(
            tree.debug_dict(),
            {
                "foo": {
                    "*": {
                        "ooo": 3,
                        "iii": 2,
                    }
                }
            }
        )


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='asdasdasd')
    parser.add_argument('--key-separator', default='/')
    parser.add_argument('--key-selector', default='^([^?]+)')
    parser.add_argument('--no-metric', dest='metric', action='store_false')
    parser.add_argument('-q', '--quiet', dest='quiet', action='store_true')
    parser.add_argument('files', metavar='FILE', nargs='*')
    parser.set_defaults(metric=True)

    args = parser.parse_args()

    reader = Reader(
        path_parser = args.key_selector,
        path_separator = args.key_separator,
    )
    tree = Tree()

    n = 0
    try:
        for line in fileinput.input(args.files):
            line = line.strip()
            n += 1

            if args.metric:
                try:
                    value, path = line.split(' ', 1)
                except ValueError:
                    continue
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        continue
            else:
                value = 1
                path = line

            path_arr = reader.parse_path(path)
            tree.append(path_arr, value)
            if n % 1000 == 0:
                tree.compact()
                nodes_after = tree.count_nodes()
                if not args.quiet:
                    print(' [%s] nodes: %s\r' % (n, nodes_after), end='', file=sys.stderr)
    except KeyboardInterrupt:
        pass
    tree.compact(force=True)
    if args.metric:
        fields = ['count', 'avg', 'min', 'max', 'sum', 'path']
    else:
        fields = ['count', 'path']
    if not args.quiet:
        print((' ' * 30) + '\r', end='', file=sys.stderr)
    tree.pretty_print(fields)
