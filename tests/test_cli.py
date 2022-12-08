# -*- coding: utf-8 -*-
# Copyright (C) 2022 Anaconda, Inc
# SPDX-License-Identifier: BSD-3-Clause
from functools import partial
from textwrap import dedent

import click.testing
import pytest
import yaml

from conda_project import __version__
from conda_project.cli.main import cli, main, new_cli, parse_and_run

PROJECT_COMMANDS = ("check",)
ENVIRONMENT_COMMANDS = ("clean", "prepare", "lock")
ALL_COMMANDS = PROJECT_COMMANDS + ENVIRONMENT_COMMANDS


@pytest.fixture()
def run_cli(tmp_path, monkeypatch):
    """A function to call the click CLI."""
    monkeypatch.chdir(tmp_path)
    runner = click.testing.CliRunner()

    def _run_cli(*args):
        return runner.invoke(new_cli, args)

    return _run_cli


def test_known_commands():
    parser = cli()
    assert parser._positionals._actions[2].choices.keys() == set(ALL_COMMANDS)


@pytest.mark.parametrize("flag", [None, "--help"])
def test_cli_display_help(run_cli, flag):
    """Help is displayed if no arguments provided, or explicit flag provided."""
    args = (flag,) if flag is not None else tuple()
    result = run_cli(*args)
    assert result.exit_code == 0
    assert result.output.startswith("Usage:")


@pytest.mark.parametrize("flag", ["-V", "--version"])
def test_cli_version(run_cli, flag: str):
    result = run_cli(flag)
    assert result.exit_code == 0
    assert f"conda-project {__version__}" in result.output


def test_cli_command(run_cli):
    # TODO: Remove once other commands are implemented
    result = run_cli("dummy")
    assert result.exit_code == 0
    assert "REPLACEME" in result.output


def test_cli_create(run_cli, tmp_path):
    """When we use `conda project create`, three files are generated."""
    filenames = [
        "conda-project.yml",
        "environment.yml",
        ".condarc",
        "default.conda-lock.yml",
    ]
    for f in filenames:
        assert not (tmp_path / f).exists()
    result = run_cli("create")
    assert result.exit_code == 0
    for f in filenames:
        assert (tmp_path / f).exists()


def test_cli_create_no_lock(run_cli, tmp_path):
    """When we use `conda project create --no-lock`, The conda-lock file is not generated."""
    lockfile_path = tmp_path / "default.conda-lock.yml"
    assert not lockfile_path.exists()
    result = run_cli("create", "--no-lock")
    assert result.exit_code == 0
    # Still doesn't exist
    assert not lockfile_path.exists()


def test_create_with_prepare(run_cli, tmp_path):
    result = run_cli("create", "--prepare")
    assert result.exit_code == 0
    assert (tmp_path / "envs" / "default" / "conda-meta" / "history").exists()


def test_create_with_channels(run_cli, tmp_path):
    """When we create with channels, those are embedded into the environment.yml in the order provided."""
    result = run_cli("create", "-c", "pyviz", "--channel", "main")
    assert result.exit_code == 0
    with (tmp_path / "environment.yml").open() as fp:
        data = yaml.safe_load(fp)
    assert data["channels"] == ["pyviz", "main"]


def test_create_with_platforms(run_cli, tmp_path):
    """When we create with platforms, those are embedded into the environment.yml."""
    # TODO: Should we accept multiple here instead of using a CSV string?
    result = run_cli("create", "--platforms", "osx-64,linux-64")
    assert result.exit_code == 0
    with (tmp_path / "environment.yml").open() as fp:
        data = yaml.safe_load(fp)
    assert data["platforms"] == ["osx-64", "linux-64"]


def test_create_with_conda_configs(run_cli, tmp_path):
    """When we create with platforms, those are embedded into the environment.yml."""
    # TODO: Should we accept multiple here instead of using a CSV string?
    result = run_cli(
        "create",
        "--conda-configs",
        "experimental_solver=libmamba,channel_priority=strict",
    )
    assert result.exit_code == 0
    with (tmp_path / ".condarc").open() as fp:
        data = yaml.safe_load(fp)
    assert data == {
        "experimental_solver": "libmamba",
        "channel_priority": "strict",
    }


def test_create_with_dependencies(run_cli, tmp_path):
    """When we create with package spec, those are embedded into the environment.yml."""
    result = run_cli(
        "create",
        "python=3.10",
        "numpy",
    )
    assert result.exit_code == 0
    with (tmp_path / "environment.yml").open() as fp:
        data = yaml.safe_load(fp)
    assert data["dependencies"] == ["python=3.10", "numpy"]


def test_no_env_yaml(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr("sys.argv", ["conda-project", "prepare"])
    assert main() == 1

    err = capsys.readouterr().err
    assert "No Conda environment.yml or environment.yaml file was found" in err


def test_unknown_command(capsys):
    with pytest.raises(SystemExit):
        assert parse_and_run(["__nope__"]) is None

    err = capsys.readouterr().err
    assert "invalid choice: '__nope__'" in err


@pytest.mark.parametrize("command", ALL_COMMANDS)
def test_command_args(command, monkeypatch, capsys):
    def mocked_command(command, args):
        print(f"I am {command}")
        assert args.directory == "project-dir"
        return 42

    monkeypatch.setattr(
        f"conda_project.cli.commands.{command}", partial(mocked_command, command)
    )

    ret = parse_and_run([command, "--directory", "project-dir"])
    assert ret == 42

    out, err = capsys.readouterr()
    assert f"I am {command}\n" == out
    assert "" == err


@pytest.mark.parametrize("command", ENVIRONMENT_COMMANDS)
@pytest.mark.parametrize("project_directory_factory", [".yml", ".yaml"], indirect=True)
def test_cli_verbose_env(command, monkeypatch, project_directory_factory):
    if command == "create":
        project_path = project_directory_factory()
    else:
        env_yaml = "dependencies: []\n"
        project_path = project_directory_factory(env_yaml=env_yaml)

    def mocked_action(*_, **kwargs):
        assert kwargs.get("verbose", False)

    monkeypatch.setattr(f"conda_project.project.Environment.{command}", mocked_action)

    ret = parse_and_run([command, "--directory", str(project_path)])
    assert ret == 0


@pytest.mark.parametrize("command", PROJECT_COMMANDS)
def test_cli_verbose_project(command, monkeypatch, project_directory_factory):
    if command == "create":
        project_path = project_directory_factory()
    else:
        env_yaml = "dependencies: []\n"
        project_path = project_directory_factory(env_yaml=env_yaml)

    def mocked_action(*_, **kwargs):
        assert kwargs.get("verbose", False)

    monkeypatch.setattr(f"conda_project.project.CondaProject.{command}", mocked_action)

    _ = parse_and_run([command, "--directory", str(project_path)])


@pytest.mark.parametrize("command", ENVIRONMENT_COMMANDS)
def test_command_with_environment_name(command, monkeypatch, project_directory_factory):
    env1 = env2 = "dependencies: []\n"
    project_yaml = dedent(
        f"""\
        name: multi-envs
        environments:
          env1: [env1{project_directory_factory._suffix}]
          env2: [env2{project_directory_factory._suffix}]
        """
    )
    project_path = project_directory_factory(
        project_yaml=project_yaml,
        files={
            f"env1{project_directory_factory._suffix}": env1,
            f"env2{project_directory_factory._suffix}": env2,
        },
    )

    def mocked_action(self, *args, **kwargs):
        assert self.name == "env1"

    monkeypatch.setattr(f"conda_project.project.Environment.{command}", mocked_action)

    ret = parse_and_run([command, "--directory", str(project_path), "env1"])
    assert ret == 0


def test_prepare_and_clean_all_environments(monkeypatch, project_directory_factory):
    env1 = env2 = "dependencies: []\n"
    project_yaml = dedent(
        f"""\
        name: multi-envs
        environments:
          env1: [env1{project_directory_factory._suffix}]
          env2: [env2{project_directory_factory._suffix}]
        """
    )
    project_path = project_directory_factory(
        project_yaml=project_yaml,
        files={
            f"env1{project_directory_factory._suffix}": env1,
            f"env2{project_directory_factory._suffix}": env2,
        },
    )

    def mocked_action(self, *args, **kwargs):
        assert self.name in ["env1", "env2"]

    monkeypatch.setattr("conda_project.project.Environment.prepare", mocked_action)
    monkeypatch.setattr("conda_project.project.Environment.clean", mocked_action)

    ret = parse_and_run(["prepare", "--directory", str(project_path), "--all"])
    assert ret == 0

    ret = parse_and_run(["clean", "--directory", str(project_path), "--all"])
    assert ret == 0


def test_lock_all_environments(monkeypatch, project_directory_factory):
    env1 = env2 = "dependencies: []\n"
    project_yaml = dedent(
        f"""\
        name: multi-envs
        environments:
          env1: [env1{project_directory_factory._suffix}]
          env2: [env2{project_directory_factory._suffix}]
        """
    )
    project_path = project_directory_factory(
        project_yaml=project_yaml,
        files={
            f"env1{project_directory_factory._suffix}": env1,
            f"env2{project_directory_factory._suffix}": env2,
        },
    )

    def mocked_action(self, *args, **kwargs):
        assert self.name in ["env1", "env2"]

    monkeypatch.setattr("conda_project.project.Environment.lock", mocked_action)

    ret = parse_and_run(["lock", "--directory", str(project_path)])
    assert ret == 0


@pytest.mark.slow
def test_check_multi_env(project_directory_factory, capsys):
    env1 = env2 = "dependencies: []\n"
    project_yaml = dedent(
        f"""\
        name: multi-envs
        environments:
          env1: [env1{project_directory_factory._suffix}]
          env2: [env2{project_directory_factory._suffix}]
        """
    )
    project_path = project_directory_factory(
        project_yaml=project_yaml,
        files={
            f"env1{project_directory_factory._suffix}": env1,
            f"env2{project_directory_factory._suffix}": env2,
        },
    )

    ret = parse_and_run(["check", "--directory", str(project_path)])
    assert ret == 1

    stderr = capsys.readouterr().err
    assert "The environment env1 is not locked" in stderr
    assert "The environment env2 is not locked" in stderr

    ret = parse_and_run(["lock", "--directory", str(project_path)])
    assert ret == 0

    ret = parse_and_run(["check", "--directory", str(project_path)])
    assert ret == 0

    env1 = "dependencies: [python=3.8]\n"
    with (project_path / f"env1{project_directory_factory._suffix}").open("w") as f:
        f.write(env1)

    ret = parse_and_run(["check", "--directory", str(project_path)])
    assert ret == 1

    stderr = capsys.readouterr().err
    assert stderr
    assert "The lockfile for environment env1 is out-of-date" in stderr
    assert "The lockfile for environment env2" not in stderr

    env2 = "dependencies: [python=3.8]\n"
    with (project_path / f"env2{project_directory_factory._suffix}").open("w") as f:
        f.write(env2)

    ret = parse_and_run(["check", "--directory", str(project_path)])
    assert ret == 1

    stderr = capsys.readouterr().err
    assert stderr
    assert "The lockfile for environment env1 is out-of-date" in stderr
    assert "The lockfile for environment env2 is out-of-date" in stderr
