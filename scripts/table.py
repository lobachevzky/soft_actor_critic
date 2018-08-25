#! /usr/bin/env python
import argparse
from pathlib import Path, PurePath

from runs.database import DataBase, RunEntry
from runs.logger import Logger
from scripts.crawl_events import crawl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('patterns', nargs='*', type=PurePath)
    parser.add_argument('--tensorboard-dir', default=Path('.runs/tensorboard'), type=Path)
    parser.add_argument('--smoothing', type=int, default=20000)
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
    rewards = {
        str(event_file).replace(tensorboard_dir, ''): data
        for data, event_file in data_points
    }
    assert set(rewards.values()) == set(entries.values())

    header = list(RunEntry.fields()) + ['rewards']
    rows = [[getattr(entry, field) for field in RunEntry.fields()] + [rewards[path]]
            for path, entry in entries.items()]
    return header, rows


if __name__ == '__main__':
    main()
