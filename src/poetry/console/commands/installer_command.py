from __future__ import annotations

from functools import cached_property

from poetry.console.commands.env_command import EnvCommand
from poetry.console.commands.group_command import GroupCommand
from poetry.installation.installer import Installer


class InstallerCommand(GroupCommand, EnvCommand):
    def __init__(self) -> None:
        self._installer: Installer | None = None

        super().__init__()

    def reset_poetry(self) -> None:
        super().reset_poetry()

        self.installer.set_package(self.poetry.package)
        self.installer.set_locker(self.poetry.locker)

    @cached_property
    def installer(self) -> Installer:
        return Installer(
            self.io,
            self.env,
            self.poetry.package,
            self.poetry.locker,
            self.poetry.pool,
            self.poetry.config,
            disable_cache=self.poetry.disable_cache,
        )

    def set_installer(self, installer: Installer) -> None:
        self.installer = installer
