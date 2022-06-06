# -*- coding: utf-8 -*-
# Copyright (C) 2022 Anaconda, Inc
# SPDX-License-Identifier: BSD-3-Clause
import os

import pytest
import yaml

from conda_project.exceptions import CondaProjectError
from conda_project.project import CondaProject, DEFAULT_PLATFORMS


def test_conda_project_init_no_env_yml(tmpdir):
    with pytest.raises(CondaProjectError) as excinfo:
        CondaProject(tmpdir)
    assert "No Conda environment.yml or environment.yaml file was found" in str(
        excinfo.value
    )


def test_project_init_expands_cwd(monkeypatch, project_directory_factory):
    project_path = project_directory_factory()
    monkeypatch.chdir(project_path)

    project = CondaProject()
    assert project.directory.samefile(project_path)
    assert project.environment_file.exists()


def test_project_init_path(project_directory_factory):
    project_path = project_directory_factory()

    project = CondaProject(project_path)
    assert project.directory.samefile(project_path)
    assert project.environment_file


def test_prepare_no_dependencies(project_directory_factory):
    env_yaml = """name: test
dependencies: []
"""
    project_path = project_directory_factory(env_yaml=env_yaml)
    project = CondaProject(project_path)
    assert project.directory.samefile(project_path)

    env_dir = project.prepare()
    assert env_dir.samefile(project_path / "envs" / "default")

    conda_history = env_dir / "conda-meta" / "history"
    assert conda_history.exists()


@pytest.mark.slow
def test_prepare_and_clean(project_directory_factory):
    env_yaml = """name: test
dependencies:
  - python=3.8
"""
    project_path = project_directory_factory(env_yaml=env_yaml)

    project = CondaProject(project_path)
    env_dir = project.prepare()
    assert env_dir.samefile(project_path / "envs" / "default")

    conda_history = env_dir / "conda-meta" / "history"
    assert conda_history.exists()

    with conda_history.open() as f:
        assert "conda create -y --file" in f.read()
    conda_history_mtime = os.path.getmtime(conda_history)

    project.prepare()
    assert conda_history_mtime == os.path.getmtime(conda_history)

    project.prepare(force=True)
    assert conda_history_mtime < os.path.getmtime(conda_history)

    project.clean()
    assert not conda_history.exists()


@pytest.mark.slow
def test_lock(project_directory_factory):
    env_yaml = """name: test
dependencies:
  - python=3.8
"""
    project_path = project_directory_factory(env_yaml=env_yaml)

    project = CondaProject(project_path)
    project.lock()

    lockfile = project_path / 'conda-lock.yml'
    assert lockfile == project.lock_file
    assert lockfile.exists()


def test_lock_no_channels(project_directory_factory, capsys):
    env_yaml = """name: test
dependencies: []
"""
    project_path = project_directory_factory(env_yaml=env_yaml)

    project = CondaProject(project_path)
    project.lock(verbose=True)

    _, err = capsys.readouterr()
    assert "no 'channels:' key" in err

    with open(project.lock_file) as f:
        lock = yaml.safe_load(f)

    assert [c['url'] for c in lock['metadata']['channels']] == ['defaults']


def test_lock_with_channels(project_directory_factory):
    env_yaml = """name: test
channels: [defusco, conda-forge, defaults]
dependencies: []
"""
    project_path = project_directory_factory(env_yaml=env_yaml)

    project = CondaProject(project_path)
    project.lock()

    with open(project.lock_file) as f:
        lock = yaml.safe_load(f)

    assert [c['url'] for c in lock['metadata']['channels']] == ['defusco', 'conda-forge', 'defaults']


def test_lock_no_platforms(project_directory_factory):
    env_yaml = """name: test
dependencies: []
"""
    project_path = project_directory_factory(env_yaml=env_yaml)

    project = CondaProject(project_path)
    project.lock()

    with open(project.lock_file) as f:
        lock = yaml.safe_load(f)

    assert lock['metadata']['platforms'] == list(DEFAULT_PLATFORMS)


def test_lock_with_platforms(project_directory_factory):
    env_yaml = """name: test
dependencies: []
platforms: [linux-64, osx-64]
"""
    project_path = project_directory_factory(env_yaml=env_yaml)

    project = CondaProject(project_path)
    project.lock(verbose=True)

    with open(project.lock_file) as f:
        lock = yaml.safe_load(f)

    assert lock['metadata']['platforms'] == ['linux-64', 'osx-64']


def test_lock_wrong_platform(project_directory_factory):
    env_yaml = """name: test
dependencies: []
platforms: [dummy-platform]
"""

    project_path = project_directory_factory(env_yaml=env_yaml)

    project = CondaProject(project_path)
    project.lock()

    with pytest.raises(CondaProjectError) as e:
        project.prepare()
    assert "not in the supported locked platforms" in str(e.value)


def test_force_relock(project_directory_factory, capsys):
    env_yaml = """name: test
dependencies: []
"""
    project_path = project_directory_factory(env_yaml=env_yaml)

    project = CondaProject(project_path)
    project.lock(verbose=True)

    lockfile_mtime = os.path.getmtime(project.lock_file)
    project.lock()
    assert lockfile_mtime == os.path.getmtime(project.lock_file)

    project.lock(force=True)
    assert lockfile_mtime < os.path.getmtime(project.lock_file)


@pytest.mark.slow
def test_relock_add_packages(project_directory_factory):
    env_yaml = """name: test
dependencies:
  - python=3.8
"""
    project_path = project_directory_factory(env_yaml=env_yaml)

    project = CondaProject(project_path)
    project.lock()

    assert project.lock_file.exists()
    lockfile_mtime = os.path.getmtime(project.lock_file)
    with open(project.lock_file) as f:
        lock = f.read()
    assert 'requests' not in lock

    env_yaml = """name: test
dependencies:
  - python=3.8
  - requests
"""
    with open(project.environment_file, 'w') as f:
        f.write(env_yaml)

    project.lock()
    with open(project.lock_file) as f:
        lock = f.read()
    assert 'requests' in lock

    assert lockfile_mtime < os.path.getmtime(project.lock_file)
