"""
Copied from `pip`, src/pip/_internal/req/req_uninstall.py
Changes:
- replaced some pip internals with Poetry versions
- removed some `.egg` based code paths (debatable)
- inlined some util functions from pip internals
- `uninstall` function is an API that I aimed for. The rest should be implementation detail
- passing Env to `UninstallPathSet` to get sysconfig paths info for env
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import shutil
import sys
import sysconfig
import tempfile
import threading

from importlib.util import cache_from_source
from pathlib import Path
from pathlib import PurePath
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Generator
from typing import Iterable
from typing import Iterator
from typing import Protocol
from typing import Union

from packaging.utils import NormalizedName
from packaging.utils import canonicalize_name
from poetry.core.utils.helpers import robust_rmtree

from poetry.utils._compat import WINDOWS
from poetry.utils._compat import metadata
from poetry.utils.helpers import remove_directory


if TYPE_CHECKING:
    from poetry.utils.env import Env


logger = logging.getLogger(__name__)


class UninstallationError(Exception):
    """Uninstallation error."""


def uninstall(name: str, env: Env) -> None:
    """
    Dist uninstallation function
    :param name: name of the package to uninstall
    :param env: environment from which to uninstall
    """
    dist = env.site_packages.find_distribution(name)
    if not dist:
        logger.warning("Skipping %s as it is not installed.", name)
        return None
    logger.info("Found existing installation: %s", dist)

    uninstall_pathset = UninstallPathSet.from_dist(dist, env)
    uninstall_pathset.remove()

    if uninstall_pathset:
        uninstall_pathset.commit()


InfoPath = Union[str, PurePath]


class BasePath(Protocol):
    """A protocol that various path objects conform.

    This exists because importlib.metadata uses both ``pathlib.Path`` and
    ``zipfile.Path``, and we need a common base for type hints (Union does not
    work well since ``zipfile.Path`` is too new for our linter setup).

    This does not mean to be exhaustive, but only contains things that present
    in both classes *that we need*.
    """

    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def parent(self) -> BasePath:
        raise NotImplementedError()


class Distribution:
    def __init__(
        self,
        dist: metadata.Distribution,
        info_location: BasePath | None,
        installed_location: BasePath | None,
    ) -> None:
        self._dist = dist
        self._info_location = info_location
        self._installed_location = installed_location

    @property
    def location(self) -> str | None:
        if self._info_location is None:
            return None
        return str(self._info_location.parent)

    @property
    def info_location(self) -> str | None:
        if self._info_location is None:
            return None
        return str(self._info_location)

    @property
    def installed_location(self) -> str | None:
        if self._installed_location is None:
            return None
        return normalize_path(str(self._installed_location))

    def _get_dist_name_from_location(self) -> str | None:
        """Try to get the name from the metadata directory name.

        This is much faster than reading metadata.
        """
        if self._info_location is None:
            return None
        stem, suffix = os.path.splitext(self._info_location.name)
        if suffix not in (".dist-info", ".egg-info"):
            return None
        return stem.split("-", 1)[0]

    @property
    def canonical_name(self) -> NormalizedName:
        return canonicalize_name(self._dist.name)

    @property
    def version(self) -> str:
        return self._dist.version

    def is_file(self, path: InfoPath) -> bool:
        return self._dist.read_text(str(path)) is not None

    def iter_distutils_script_names(self) -> Iterator[str]:
        # A distutils installation is always "flat" (not in e.g. egg form), so
        # if this distribution's info location is NOT a pathlib.Path (but e.g.
        # zipfile.Path), it can never contain any distutils scripts.
        if not isinstance(self._info_location, Path):
            return
        for child in self._info_location.joinpath("scripts").iterdir():
            yield child.name

    def read_text(self, path: InfoPath) -> str:
        content = self._dist.read_text(str(path))
        if content is None:
            raise FileNotFoundError(path)
        return content

    def iter_entry_points(self) -> Iterable[metadata.EntryPoint]:
        # importlib.metadata's EntryPoint structure sasitfies BaseEntryPoint.
        return self._dist.entry_points

    @property
    def installed_with_dist_info(self) -> bool:
        """Whether this distribution is installed with the "modern format".

        This indicates a "modern" installation, e.g. storing metadata in the
        ``.dist-info`` directory. This applies to installations made by
        setuptools (but through pip, not directly), or anything using the
        standardized build backend interface (PEP 517).
        """
        info_location = self.info_location
        if not info_location:
            return False
        if not info_location.endswith(".dist-info"):
            return False
        return Path(info_location).is_dir()

    @property
    def raw_name(self) -> str:
        """Value of "Name:" in distribution metadata."""
        # The metadata should NEVER be missing the Name: key, but if it somehow
        # does, fall back to the known canonical name.
        return self._dist.metadata.get("Name", self.canonical_name)

    def _iter_declared_entries_from_record(self) -> Iterator[str] | None:
        import csv

        try:
            text = self.read_text("RECORD")
        except FileNotFoundError:
            return None
        # This extra Path-str cast normalizes entries.
        return (str(Path(row[0])) for row in csv.reader(text.splitlines()))

    def iter_declared_entries(self) -> Iterator[str] | None:
        """Iterate through file entries declared in this distribution.

        For modern .dist-info distributions, this is the files listed in the
        ``RECORD`` metadata file. For legacy setuptools distributions, this
        comes from ``installed-files.txt``, with entries normalized to be
        compatible with the format used by ``RECORD``.

        :return: An iterator for listed entries, or None if the distribution
            contains neither ``RECORD`` nor ``installed-files.txt``.
        """
        return self._iter_declared_entries_from_record()


def normalize_path(path: str, resolve_symlinks: bool = True) -> str:
    """
    Convert a path to its canonical, case-normalized, absolute version.

    """
    path = os.path.expanduser(path)
    path = os.path.realpath(path) if resolve_symlinks else os.path.abspath(path)
    return os.path.normcase(path)


def renames(old: str, new: str) -> None:
    """Like os.renames(), but handles renaming across devices."""
    # Implementation borrowed from os.renames().
    head, tail = os.path.split(new)
    if head and tail and not os.path.exists(head):
        os.makedirs(head)

    shutil.move(old, new)

    head, tail = os.path.split(old)
    if head and tail:
        with contextlib.suppress(OSError):
            os.removedirs(head)


_log_state = threading.local()


@contextlib.contextmanager
def indent_log(num: int = 2) -> Generator[None, None, None]:
    """
    A context manager which will cause the log output to be indented for any
    log messages emitted inside it.
    """
    # For thread-safety
    _log_state.indentation = get_indentation()
    _log_state.indentation += num
    try:
        yield
    finally:
        _log_state.indentation -= num


def get_indentation() -> int:
    return getattr(_log_state, "indentation", 0)


def _script_names(bin_dir: str, script_name: str, is_gui: bool) -> Iterator[str]:
    """Create the fully qualified name of the files created by
    {console,gui}_scripts for the given ``dist``.
    Returns the list of file names
    """
    exe_name = os.path.join(bin_dir, script_name)
    yield exe_name
    if not WINDOWS:
        return
    yield f"{exe_name}.exe"
    yield f"{exe_name}.exe.manifest"
    if is_gui:
        yield f"{exe_name}-script.pyw"
    else:
        yield f"{exe_name}-script.py"


def _unique(
    fn: Callable[..., Generator[Any, None, None]],
) -> Callable[..., Generator[Any, None, None]]:
    @functools.wraps(fn)
    def unique(*args: Any, **kw: Any) -> Generator[Any, None, None]:
        seen: set[Any] = set()
        for item in fn(*args, **kw):
            if item not in seen:
                seen.add(item)
                yield item

    return unique


@_unique
def uninstallation_paths(dist: Distribution) -> Iterator[str]:
    """
    Yield all the uninstallation paths for dist based on RECORD-without-.py[co]

    Yield paths to all the files in RECORD. For each .py file in RECORD, add
    the .pyc and .pyo in the same directory.

    UninstallPathSet.add() takes care of the __pycache__ .py[co].

    If RECORD is not found, raises UninstallationError,
    with possible information from the INSTALLER file.

    https://packaging.python.org/specifications/recording-installed-packages/
    """
    location = dist.location
    assert location is not None, "not installed"

    entries = dist.iter_declared_entries()
    if entries is None:
        msg = f"Cannot uninstall {dist}, RECORD file not found."
        raise UninstallationError(msg)

    for entry in entries:
        path = os.path.join(location, entry)
        yield path
        if path.endswith(".py"):
            dn, fn = os.path.split(path)
            base = fn[:-3]
            path = os.path.join(dn, base + ".pyc")
            yield path
            path = os.path.join(dn, base + ".pyo")
            yield path


def compact(paths: Iterable[str]) -> set[str]:
    """Compact a path set to contain the minimal number of paths
    necessary to contain all paths in the set. If /a/path/ and
    /a/path/to/a/file.txt are both in the set, leave only the
    shorter path."""

    sep = os.path.sep
    short_paths: set[str] = set()
    for path in sorted(paths, key=len):
        should_skip = any(
            path.startswith(shortpath.rstrip("*"))
            and path[len(shortpath.rstrip("*").rstrip(sep))] == sep
            for shortpath in short_paths
        )
        if not should_skip:
            short_paths.add(path)
    return short_paths


def compress_for_rename(paths: Iterable[str]) -> set[str]:
    """Returns a set containing the paths that need to be renamed.

    This set may include directories when the original sequence of paths
    included every file on disk.
    """
    case_map = {os.path.normcase(p): p for p in paths}
    remaining = set(case_map)
    unchecked = sorted({os.path.split(p)[0] for p in case_map.values()}, key=len)
    wildcards: set[str] = set()

    def norm_join(*a: str) -> str:
        return os.path.normcase(os.path.join(*a))

    for root in unchecked:
        if any(os.path.normcase(root).startswith(w) for w in wildcards):
            # This directory has already been handled.
            continue

        all_files: set[str] = set()
        all_subdirs: set[str] = set()
        for dirname, subdirs, files in os.walk(root):
            all_subdirs.update(norm_join(root, dirname, d) for d in subdirs)
            all_files.update(norm_join(root, dirname, f) for f in files)
        # If all the files we found are in our remaining set of files to
        # remove, then remove them from the latter set and add a wildcard
        # for the directory.
        if not (all_files - remaining):
            remaining.difference_update(all_files)
            wildcards.add(root + os.sep)

    return set(map(case_map.__getitem__, remaining)) | wildcards


def compress_for_output_listing(paths: Iterable[str]) -> tuple[set[str], set[str]]:
    """Returns a tuple of 2 sets of which paths to display to user

    The first set contains paths that would be deleted. Files of a package
    are not added and the top-level directory of the package has a '*' added
    at the end - to signify that all it's contents are removed.

    The second set contains files that would have been skipped in the above
    folders.
    """

    will_remove = set(paths)
    will_skip = set()

    # Determine folders and files
    folders = set()
    files = set()
    for path in will_remove:
        if path.endswith(".pyc"):
            continue
        if path.endswith("__init__.py") or ".dist-info" in path:
            folders.add(os.path.dirname(path))
        files.add(path)

    _normcased_files = set(map(os.path.normcase, files))

    folders = compact(folders)

    # This walks the tree using os.walk to not miss extra folders
    # that might get added.
    for folder in folders:
        for dirpath, _, dirfiles in os.walk(folder):
            for fname in dirfiles:
                if fname.endswith(".pyc"):
                    continue

                file_ = os.path.join(dirpath, fname)
                if (
                    os.path.isfile(file_)
                    and os.path.normcase(file_) not in _normcased_files
                ):
                    # We are skipping this file. Add it to the set.
                    will_skip.add(file_)

    will_remove = files | {os.path.join(folder, "*") for folder in folders}

    return will_remove, will_skip


class StashedUninstallPathSet:
    """A set of file rename operations to stash files while
    tentatively uninstalling them."""

    def __init__(self) -> None:
        # Mapping from source file root to [Adjacent]TempDirectory
        # for files under that directory.
        self._save_dirs: dict[str, str] = {}
        # (old path, new path) tuples for each move that may need
        # to be undone.
        self._moves: list[tuple[str, str]] = []

    def _get_directory_stash(self, path: str) -> str:
        """Stashes a directory.

        Directories are stashed adjacent to their original location if
        possible, or else moved/copied into the user's temp dir."""

        # TODO: consider reproducing with our code
        """
        try:
            save_dir: TempDirectory = AdjacentTempDirectory(path)
        except OSError:
            save_dir = TempDirectory(kind="uninstall")
        """
        save_dir = tempfile.mkdtemp(prefix="poetry-uninstall")
        self._save_dirs[os.path.normcase(path)] = save_dir

        return save_dir

    def _get_file_stash(self, path: str) -> str:
        """Stashes a file.

        If no root has been provided, one will be created for the directory
        in the user's temp directory."""
        path = os.path.normcase(path)
        head, old_head = os.path.dirname(path), None
        save_dir = None

        while head != old_head:
            with contextlib.suppress(KeyError):
                save_dir = self._save_dirs[head]
                break
            head, old_head = os.path.dirname(head), head
        else:
            # Did not find any suitable root
            head = os.path.dirname(path)
            save_dir = tempfile.mkdtemp(prefix="poetry-uninstall")
            self._save_dirs[head] = save_dir

        relpath = os.path.relpath(path, head)
        if relpath and relpath != os.path.curdir:
            return os.path.join(save_dir, relpath)
        return save_dir

    def stash(self, path: str) -> str:
        """Stashes the directory or file and returns its new location.
        Handle symlinks as files to avoid modifying the symlink targets.
        """
        path_is_dir = os.path.isdir(path) and not os.path.islink(path)
        if path_is_dir:
            new_path = self._get_directory_stash(path)
        else:
            new_path = self._get_file_stash(path)

        self._moves.append((path, new_path))
        if path_is_dir and os.path.isdir(new_path):
            # If we're moving a directory, we need to
            # remove the destination first or else it will be
            # moved to inside the existing directory.
            # We just created new_path ourselves, so it will
            # be removable.
            os.rmdir(new_path)
        renames(path, new_path)
        return new_path

    def commit(self) -> None:
        """Commits the uninstall by removing stashed files."""
        for save_dir in self._save_dirs.values():
            robust_rmtree(save_dir)
        self._moves = []
        self._save_dirs = {}

    def rollback(self) -> None:
        """Undoes the uninstall by moving stashed files back."""
        for p in self._moves:
            logger.info("Moving to %s\n from %s", *p)

        for new_path, path in self._moves:
            try:
                logger.debug("Replacing %s from %s", new_path, path)
                if os.path.isfile(new_path) or os.path.islink(new_path):
                    os.unlink(new_path)
                elif os.path.isdir(new_path):
                    remove_directory(Path(new_path))
                renames(path, new_path)
            except OSError as ex:
                logger.error("Failed to restore %s", new_path)
                logger.debug("Exception: %s", ex)

        self.commit()

    @property
    def can_rollback(self) -> bool:
        return bool(self._moves)


class UninstallPathSet:
    """A set of file paths to be removed in the uninstallation of a
    requirement."""

    def __init__(self, dist: Distribution) -> None:
        self._paths: set[str] = set()
        self._refuse: set[str] = set()
        self._pth: dict[str, UninstallPthEntries] = {}
        self._dist = dist
        self._moved_paths = StashedUninstallPathSet()
        # Create local cache of normalize_path results. Creating an UninstallPathSet
        # can result in hundreds/thousands of redundant calls to normalize_path with
        # the same args, which hurts performance.
        self._normalize_path_cached = functools.lru_cache(normalize_path)

    def _permitted(self, path: str) -> bool:
        """
        Return True if the given path is one we are permitted to
        remove/modify, False otherwise.

        """
        # aka is_local, but caching normalized sys.prefix
        # TODO: resolve
        """
        if not running_under_virtualenv():
            return True
        """
        return path.startswith(self._normalize_path_cached(sys.prefix))

    def add(self, path: str) -> None:
        head, tail = os.path.split(path)

        # we normalize the head to resolve parent directory symlinks, but not
        # the tail, since we only want to uninstall symlinks, not their targets
        path = os.path.join(self._normalize_path_cached(head), os.path.normcase(tail))

        if not os.path.exists(path):
            return
        if self._permitted(path):
            self._paths.add(path)
        else:
            self._refuse.add(path)

        # __pycache__ files can show up after 'installed-files.txt' is created,
        # due to imports
        if os.path.splitext(path)[1] == ".py":
            self.add(cache_from_source(path))

    def add_pth(self, pth_file: str, entry: str) -> None:
        pth_file = self._normalize_path_cached(pth_file)
        if self._permitted(pth_file):
            if pth_file not in self._pth:
                self._pth[pth_file] = UninstallPthEntries(pth_file)
            self._pth[pth_file].add(entry)
        else:
            self._refuse.add(pth_file)

    def remove(self) -> None:
        """Remove paths in ``self._paths`` with confirmation (unless
        ``auto_confirm`` is True)."""

        if not self._paths:
            logger.info(
                "Can't uninstall '%s'. No files were found to uninstall.",
                self._dist.raw_name,
            )
            return

        dist_name_version = f"{self._dist.raw_name}-{self._dist.version}"
        logger.info("Uninstalling %s:", dist_name_version)

        with indent_log():
            moved = self._moved_paths

            for_rename = compress_for_rename(self._paths)

            for path in sorted(compact(for_rename)):
                moved.stash(path)
                logger.debug("Removing file or directory %s", path)

            for pth in self._pth.values():
                pth.remove()

            logger.info("Successfully uninstalled %s", dist_name_version)

    def rollback(self) -> None:
        """Rollback the changes previously made by remove()."""
        if not self._moved_paths.can_rollback:
            logger.error(
                "Can't roll back %s; was not uninstalled",
                self._dist.raw_name,
            )
            return
        logger.info("Rolling back uninstall of %s", self._dist.raw_name)
        self._moved_paths.rollback()
        for pth in self._pth.values():
            pth.rollback()

    def commit(self) -> None:
        """Remove temporary save dir: rollback will no longer be possible."""
        self._moved_paths.commit()

    @classmethod
    def from_dist(cls, dist: Distribution, env: Env) -> UninstallPathSet:
        dist_location = dist.location
        if dist_location is None:
            logger.info(
                "Not uninstalling %s since it is not installed",
                dist.canonical_name,
            )
            return cls(dist)

        if normalized_dist_location := normalize_path(dist_location) in {
            p
            for p in {sysconfig.get_path("stdlib"), sysconfig.get_path("platstdlib")}
            if p
        }:
            logger.info(
                "Not uninstalling %s at %s, as it is in the standard library.",
                dist.canonical_name,
                normalized_dist_location,
            )
            return cls(dist)

        paths_to_remove = cls(dist)

        if dist.installed_with_dist_info:
            for path in uninstallation_paths(dist):
                paths_to_remove.add(path)

        else:
            logger.debug(
                "Not sure how to uninstall: %s - Check: %s",
                dist,
                dist_location,
            )

        bin_dir = env.paths.get("scripts")

        # find distutils scripts= scripts
        try:
            for script in dist.iter_distutils_script_names():
                paths_to_remove.add(os.path.join(bin_dir, script))
                if WINDOWS:
                    paths_to_remove.add(os.path.join(bin_dir, f"{script}.bat"))
        except (FileNotFoundError, NotADirectoryError):
            pass

        # find console_scripts and gui_scripts
        def iter_scripts_to_remove(
            dist: Distribution,
            bin_dir: str,
        ) -> Iterator[str]:
            for entry_point in dist.iter_entry_points():
                if entry_point.group == "console_scripts":
                    yield from _script_names(bin_dir, entry_point.name, False)
                elif entry_point.group == "gui_scripts":
                    yield from _script_names(bin_dir, entry_point.name, True)

        for s in iter_scripts_to_remove(dist, bin_dir):
            paths_to_remove.add(s)

        return paths_to_remove


class UninstallPthEntries:
    def __init__(self, pth_file: str) -> None:
        self.file = pth_file
        self.entries: set[str] = set()
        self._saved_lines: list[bytes] | None = None

    def add(self, entry: str) -> None:
        entry = os.path.normcase(entry)
        # On Windows, os.path.normcase converts the entry to use
        # backslashes.  This is correct for entries that describe absolute
        # paths outside of site-packages, but all the others use forward
        # slashes.
        # os.path.splitdrive is used instead of os.path.isabs because isabs
        # treats non-absolute paths with drive letter markings like c:foo\bar
        # as absolute paths. It also does not recognize UNC paths if they don't
        # have more than "\\sever\share". Valid examples: "\\server\share\" or
        # "\\server\share\folder".
        if WINDOWS and not os.path.splitdrive(entry)[0]:
            entry = entry.replace("\\", "/")
        self.entries.add(entry)

    def remove(self) -> None:
        logger.debug("Removing pth entries from %s:", self.file)

        # If the file doesn't exist, log a warning and return
        if not os.path.isfile(self.file):
            logger.warning("Cannot remove entries from nonexistent file %s", self.file)
            return
        with open(self.file, "rb") as fh:
            # windows uses '\r\n' with py3k, but uses '\n' with py2.x
            lines = fh.readlines()
            self._saved_lines = lines

        endline = "\r\n" if any(b"\r\n" in line for line in lines) else "\n"
        # handle missing trailing newline
        if lines and not lines[-1].endswith(endline.encode("utf-8")):
            lines[-1] = lines[-1] + endline.encode("utf-8")
        for entry in self.entries:
            try:
                logger.debug("Removing entry: %s", entry)
                lines.remove((entry + endline).encode("utf-8"))
            except ValueError:
                pass
        with open(self.file, "wb") as fh:
            fh.writelines(lines)

    def rollback(self) -> bool:
        if self._saved_lines is None:
            logger.error("Cannot roll back changes to %s, none were made", self.file)
            return False
        logger.debug("Rolling %s back to previous state", self.file)
        with open(self.file, "wb") as fh:
            fh.writelines(self._saved_lines)
        return True
