#! /usr/bin/env python
# stdlib
# stdlib
import argparse
from collections import namedtuple
from pathlib import Path, PurePath

# first party
from runs.commands.flags import parse_flags
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
    parser.add_argument('--delimiter', default='|')
    args = parser.parse_args()

    header, rows = get_table(
        tag=args.tag,
        db_path=args.db_path,
        patterns=args.patterns,
        smoothing=args.smoothing,
        tensorboard_dir=args.tensorboard_dir,
        use_cache=not args.update_cache)
    print(*header, sep=args.delimiter)
    for row in rows:
        assert len(header) == len(row)
        print(*row, sep=args.delimiter)


def get_table(tag, db_path, patterns, smoothing, tensorboard_dir, use_cache):
    logger = Logger(quiet=False)
    with DataBase(db_path, logger) as db:
        entries = {entry.path: entry for entry in db.get(patterns)}
    dirs = [Path(tensorboard_dir, path) for path in entries]
    data_points = crawl(
        dirs=dirs, tag=tag, smoothing=smoothing, use_cache=use_cache, quiet=True)
    rewards = {
        event_file.relative_to(tensorboard_dir).parent: data
        for data, event_file in data_points
    }

    def format_flag_name(name):
        return name.lstrip('-').replace('-', '_')

    commands = [e.command for e in entries.values()]
    flag_names = parse_flags(commands, delimiter='=').keys()
    flag_names = [format_flag_name(n) for n in flag_names]

    Row = namedtuple('Row', ['reward'] + list(RunEntry.fields()) + list(flag_names))

    def get_row(path):
        entry = entries[path]  # type: RunEntry
        flags = parse_flags([entry.command], delimiter='=')
        flags = {format_flag_name(k): v.pop() for k, v in flags.items()}
        entry_dict = {str(k): str(v) for k, v in entry.asdict().items()}
        return Row(reward=rewards[path], **entry_dict, **flags)

    return Row._fields, map(get_row, rewards)


if __name__ == '__main__':
    main()
