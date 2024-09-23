from __future__ import annotations

from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from cleo.testers.command_tester import CommandTester
    from pytest_mock import MockerFixture

    from poetry.utils.env import VirtualEnv
    from tests.types import CommandTesterFactory


@pytest.fixture
def tester(command_tester_factory: CommandTesterFactory) -> CommandTester:
    return command_tester_factory("env activate")


@pytest.mark.parametrize(
    "shell, command, ext",
    (
        ("bash", "source", ""),
        ("zsh", "source", ""),
        ("fish", "source", ".fish"),
        ("nu", "overlay use", ".nu"),
        ("csh", "source", ".csh"),
        ("pwsh", ".", "Activate.ps1"),
        ("powershell", ".", "Activate.ps1"),
    ),
)
def test_env_activate_prints_correct_script(
    tmp_venv: VirtualEnv,
    mocker: MockerFixture,
    tester: CommandTester,
    shell: str,
    command: str,
    ext: str,
) -> None:
    mocker.patch("shellingham.detect_shell", return_value=(shell, None))
    mocker.patch("poetry.utils.env.EnvManager.get", return_value=tmp_venv)

    tester.execute()

    assert tester.io.fetch_output().startswith(command)
    assert tester.io.fetch_output().strip("\n").endswith(ext)
