#!/usr/bin/env python3
"""Generate frames.csv for existing XYZI bins from a rosbag2 SQLite bag."""

import argparse
import csv
import sqlite3
import struct
from pathlib import Path


def align4(value):
    return (value + 3) & ~3


def pointcloud_header(raw):
    if len(raw) < 24 or raw[:4] != b'\x00\x01\x00\x00':
        raise ValueError('expected little-endian CDR PointCloud2 data')
    sec, nanosec = struct.unpack_from('<iI', raw, 4)
    frame_id_length = struct.unpack_from('<I', raw, 12)[0]
    dimensions_offset = align4(16 + frame_id_length)
    height, width = struct.unpack_from('<II', raw, dimensions_offset)
    return sec * 1_000_000_000 + nanosec, int(height) * int(width)


def find_db3(path):
    path = Path(path)
    if path.is_file() and path.suffix == '.db3':
        return path
    files = sorted(path.glob('*.db3')) if path.is_dir() else []
    if len(files) != 1:
        raise RuntimeError(f'expected one .db3 file under {path}, found {len(files)}')
    return files[0]


def generate_manifest(bag_path, bins_path, topic):
    db_path = find_db3(bag_path)
    bins_path = Path(bins_path)
    bin_files = sorted(bins_path.glob('*.bin'))
    if not bin_files:
        raise RuntimeError(f'no .bin files under {bins_path}')

    with sqlite3.connect(db_path) as connection:
        topic_row = connection.execute(
            'select id from topics where name = ?', (topic,)).fetchone()
        if topic_row is None:
            raise RuntimeError(f'topic not found: {topic}')
        messages = connection.execute(
            'select data from messages where topic_id = ? order by timestamp',
            (topic_row[0],),
        ).fetchall()

    if len(messages) != len(bin_files):
        raise RuntimeError(
            f'frame count mismatch: bag={len(messages)} bins={len(bin_files)}')

    rows = []
    for index, (bin_file, (raw,)) in enumerate(zip(bin_files, messages)):
        expected_name = f'{index:06d}.bin'
        if bin_file.name != expected_name:
            raise RuntimeError(f'non-contiguous bin sequence at {bin_file.name}')
        stamp_ns, original_count = pointcloud_header(raw)
        if bin_file.stat().st_size % 16:
            raise RuntimeError(f'invalid XYZI file size: {bin_file}')
        rows.append({
            'filename': bin_file.name,
            'stamp_ns': stamp_ns,
            'point_count': bin_file.stat().st_size // 16,
            'original_point_count': original_count,
        })

    output = bins_path / 'frames.csv'
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=['filename', 'stamp_ns', 'point_count', 'original_point_count'],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output, rows


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'bag', nargs='?',
        default=root / 'datasets' / 'rooster_2020-07-10-09-19-26_2-001_ros2')
    parser.add_argument(
        'bins', nargs='?', default=root / 'datasets' / 'rooster_bin_001')
    parser.add_argument('--topic', default='/os1_cloud_node/points')
    args = parser.parse_args()
    output, rows = generate_manifest(args.bag, args.bins, args.topic)
    print(f'wrote {len(rows)} frames -> {output}')
    print(
        f"first stamp={rows[0]['stamp_ns']} points={rows[0]['point_count']}/"
        f"{rows[0]['original_point_count']}")


if __name__ == '__main__':
    main()
