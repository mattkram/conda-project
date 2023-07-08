# Copyright (C) 2022 Anaconda, Inc
# SPDX-License-Identifier: BSD-3-Clause

"""Perform a recursive hash of every file in a directory"""
import hashlib
import sys
from pathlib import Path


def get_total_hash_of_directory(directory: Path) -> str:
    def _file_hashes(env_path: Path) -> bytes:
        """Yield the md5 digest of every file inside a directory"""
        patterns = [
            "conda-meta/history",
            # "conda-meta/*.json",
            "lib/python*/site-packages/**/*.py",
        ]
        for p in patterns:
            for f in env_path.glob(p):
                # # Hash just the filename
                # yield hashlib.md5(f.name.encode("utf-8")).digest()

                # Hash the filename + size
                # yield hashlib.md5((f.name + str(f.lstat().st_size)).encode("utf-8")).digest()

                # Hash the file contents
                # with f.open("rb") as fp:
                #     yield hashlib.md5(fp.read()).digest()

                # Read bytes using pathlib
                yield hashlib.md5(f.read_bytes()).digest()

    sorted_hashes = sorted(_file_hashes(directory))

    total_hash = hashlib.md5()
    for h in sorted_hashes:
        total_hash.update(h)
    return total_hash.hexdigest()


if __name__ == "__main__":
    env_directory = Path(sys.argv[1])
    total_hash = get_total_hash_of_directory(directory=env_directory)
    print(total_hash)
