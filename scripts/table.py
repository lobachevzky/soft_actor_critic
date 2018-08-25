#! /usr/bin/env python
import argparse
import json
from collections import defaultdict
from pathlib import Path, PurePath

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

    table = defaultdict(list)
    flag_names = set(parse_flags([e.command for e in entries.values()],
                                  delimiter='=').values())

    for path, reward in rewards.items():
        table[path].append(reward)
        entry = entries[path]  # type: RunEntry
        flags = parse_flags([entry.command], delimiter='=')
        missing_flags = flag_names - set(flags.values())
        if missing_flags:
            raise RuntimeError(f"{path} with command {entry.command} "
                               f"is missing the following flags:"
                               + '\n'.join(missing_flags))

        for flag, value in flags.items():
            table[flag].append(value)
        for field, attr in entry.asdict():
            if ',' in str(attr):
                attr = json.dumps(attr)
            table[field].append(attr)

    for k, v in table.items():
        if len(v) != len(rewards):
            raise RuntimeError(f'Field {k} of table has only '
                               f'{len(v)} entries but should '
                               f'have {len(rewards)}.')

    return table




if __name__ == '__main__':
    main()
