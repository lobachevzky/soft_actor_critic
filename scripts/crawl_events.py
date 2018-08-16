#! /usr/bin/env python

import argparse
from collections import namedtuple
from itertools import islice
from pathlib import Path
from typing import List, Optional

import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dirs', nargs='*', type=Path)
    parser.add_argument('--smoothing', type=int, default=2000)
    parser.add_argument('--tag', default='reward')
    parser.add_argument('--cache-file', default=None)
    parser.add_argument('--update-cache', action='store_true')
    args = parser.parse_args()
    data_points = crawl(
        dirs=args.dirs,
        tag=args.tag,
        smoothing=args.smoothing,
        cache_file=args.cache_file or args.tag,
        update_cache=args.update_cache)
    print('Sorted lowest to highest:')
    print('*************************')
    for data, event_file in sorted(data_points):
        print('{:10}: {}'.format(data, event_file))


DataPoint = namedtuple('DataPoint', 'data source')


def crawl(dirs: List[Path], tag: str, smoothing: int, cache_file: Path,
          update_cache: bool) -> List[DataPoint]:
    event_files = collect_events_files(dirs)
    data_points = []
    for event_file_path in event_files:
        data = collect_data(tag=tag, event_file_path=event_file_path, n=smoothing)
        if data:
            cache_path = Path(event_file_path.parent, cache_file)
            data_points.append((data, event_file_path))
            if update_cache or not cache_path.exists():
                print(f'Writing {cache_path}...')
                with cache_path.open('w') as f:
                    f.write(str(data))
        else:
            print('Tag not found')
    return data_points


def collect_events_files(dirs):
    pattern = '**/events*'
    return [path for d in dirs for path in d.glob(pattern)]


def collect_data(tag: str, event_file_path: Path, n: int) -> Optional[float]:
    """
    :param event_file_path: path to events file
    :param n: number of data points to average
    :return: average of last n data-points in events file or None if events file is empty
    """
    length = sum(1 for _ in tf.train.summary_iterator(str(event_file_path)))
    iterator = tf.train.summary_iterator(str(event_file_path))
    events = islice(iterator, max(length - n, 0), length)

    def get_tag(event):
        return next((v.simple_value for v in event.summary.value if v.tag == tag), None)

    data = (get_tag(e) for e in events)
    data = [d for d in data if d is not None]
    try:
        return sum(data) / len(data)
    except ZeroDivisionError:
        return None


if __name__ == '__main__':
    main()
