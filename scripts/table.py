#! /usr/bin/env python
import argparse
import json
from pathlib import Path, PurePath

from runs.database import DataBase, RunEntry
from runs.logger import Logger
from scripts.crawl_events import crawl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('patterns', nargs='*', type=PurePath)
    parser.add_argument('--tensorboard-dir', default=Path('.runs/tensorboard'), type=Path)
    parser.add_argument('--smoothing', type=int, default=2000)
    parser.add_argument('--tag', default='reward')
    parser.add_argument('--db-path', default='runs.db')
    parser.add_argument('--update-cache', action='store_true')
    args = parser.parse_args()

    header, rows = table(
        tag=args.tag,
        db_path=args.db_path,
        patterns=args.patterns,
        smoothing=args.smoothing,
        tensorboard_dir=args.tensorboard_dir,
        use_cache=not args.update_cache)
    print(*header, sep=',')
    for row in rows:
        print(*row, sep=',')


def table(tag, db_path, patterns, smoothing, tensorboard_dir, use_cache):
    logger = Logger(quiet=False)
    with DataBase(db_path, logger) as db:
        entries = {entry.path: entry for entry in db.get(patterns)}
    dirs = [Path(tensorboard_dir, path) for path in entries]
    data_points = crawl(dirs=dirs, tag=tag, smoothing=smoothing, use_cache=use_cache)
    len_tb_dir = len(tensorboard_dir.parts)
    rewards = {
        PurePath(*event_file.parts[len_tb_dir:-1]): data
        for data, event_file in data_points
    }

    header = list(RunEntry.fields()) + ['rewards']

    def get_run_attr(path, field):
        attr = getattr(entries[path], field)
        if ',' in str(attr):
            return json.dumps(attr)
        return attr

    def get_row(path, reward):
        row = [get_run_attr(path, field) for field in RunEntry.fields()]
        row += [reward]
        return row

    rows = [get_row(path, reward) for path, reward in rewards.items()]
    return header, rows


if __name__ == '__main__':
    main()
