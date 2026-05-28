from __future__ import annotations

from poetry.utils.user_agent import UserAgentBuilder
from poetry.utils.user_agent import user_agent


def test_only_user_agent_name():
    assert UserAgentBuilder("fake", "1.0.0").build() == "fake/1.0.0"


@patch("platform.python_implementation", return_value="CPython")
@patch("platform.python_version", return_value="2.7.13")
def test_include_implementation(*_):
    expected = "fake/1.0.0 CPython/2.7.13"
    actual = UserAgentBuilder("fake", "1.0.0").include_implementation().build()
    assert expected == actual


@patch("platform.system", return_value="Linux")
@patch("platform.release", return_value="4.9.5")
def test_include_system(*_):
    expected = "fake/1.0.0 Linux/4.9.5"
    actual = UserAgentBuilder("fake", "1.0.0").include_system().build()
    assert expected == actual


def test_user_agent_provides_package_name_and_version():
    assert "my-package/0.0.1" in user_agent("my-package", "0.0.1")
