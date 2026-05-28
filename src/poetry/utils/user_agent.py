from __future__ import annotations

import platform

from typing import Self


def user_agent(name: str, version: str) -> str:
    """Return an internet-friendly user_agent string.

    :param name: The intended name of the user-agent, e.g. "python-requests".
    :param version: The version of the user-agent, e.g. "0.0.1".
    :returns: Formatted user-agent string
    """
    return (
        UserAgentBuilder(name, version)
        .include_implementation()
        .include_system()
        .build()
    )


class UserAgentBuilder:
    """Class to provide a greater level of control than :func:`user_agent`."""

    def __init__(self, name: str, version: str) -> None:
        """Initialize our builder with the name and version of our user agent.

        :param str name:
            Name of our user-agent.
        :param str version:
            The version string for user-agent.
        """
        self._pieces: list[tuple[str, str]] = [(name, version)]

    def build(self) -> str:
        """Finalize the User-Agent string.

        :returns:
            Formatted User-Agent string.
        :rtype:
            str
        """
        return " ".join([f"{piece[0]}/{piece[1]}" for piece in self._pieces])

    def include_implementation(self) -> Self:
        """Append the implementation string to the user-agent string."""
        implementation = platform.python_implementation()

        if implementation == "CPython":
            implementation_version = platform.python_version()
        elif implementation == "PyPy":
            import sys

            pypy_version = sys.pypy_version_info
            implementation_version = (
                f"{pypy_version.major}.{pypy_version.minor}.{pypy_version.micro}"
            )

            if release_level := pypy_version.releaselevel != "final":
                implementation_version = "".join(
                    [implementation_version, release_level]
                )
        else:
            implementation_version = "Unknown"

        self._pieces.append((implementation, implementation_version))
        return self

    def include_system(self) -> Self:
        """Append the information about the Operating System."""
        try:
            p_system = platform.system()
            p_release = platform.release()
        except OSError:
            p_system = "Unknown"
            p_release = "Unknown"
        self._pieces.append((p_system, p_release))
        return self
