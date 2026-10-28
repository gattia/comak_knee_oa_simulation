"""Run configuration, and the rule that every path in one is relative.

A configuration says what to run and with which parameters. It does not say
where the data lives -- ``paths.py`` decides that, once. So every path in a
config file is relative to the data root and is resolved through here.

An absolute path in a config would point outside the versioned data root, at an
input that is undeclared, has no recorded provenance and can change underneath
the pipeline. Such a path is rejected, with a message saying what to write
instead.

Knee substitution
-----------------
A config is written once per gait and reused across ~750 knees, so path values
may contain ``{subject}``, ``{side}``, ``{timepoint}`` and ``{knee}``. They are
filled in by :func:`for_knee` before anything is resolved, and an unsubstituted
placeholder that reaches resolution is an error, so no directory literally
named ``{subject}`` is ever created.

Run substitution
----------------
``{run}`` is the run the data root declares (``paths.run_name``). It is filled
in when a value carrying it is read, so a caller that reads only other keys
works on any root. A root that declares no run cannot fill it and raises an
error.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .paths import STAGES, data_root, run_name

__all__ = [
    "Config",
    "Knee",
    "Label",
    "load_config",
    "ConfigError",
]

_PLACEHOLDER = re.compile(r"\{[a-z_]+\}")
_RUN = "{run}"

# A path value is a data path if it starts with a stage directory. Anything else
# (a filename, a format string, a solver keyword) is left alone.
_STAGE_DIRS = tuple(STAGES.values())


class ConfigError(ValueError):
    """A config is malformed, or asks for something that is not allowed."""


@dataclass(frozen=True)
class Knee:
    """One knee at one visit. The unit of work for stages 01 and 02."""

    subject: str
    side: str  # LEFT | RIGHT
    timepoint: str = "00m"

    @property
    def name(self) -> str:
        """``9003316_RIGHT`` -- how a knee is named in cohort lists and results."""
        return f"{self.subject}_{self.side}"

    @property
    def directory(self) -> str:
        """``9003316_00m_RIGHT`` -- how a knee is named as a directory."""
        return f"{self.subject}_{self.timepoint}_{self.side}"

    @classmethod
    def parse(cls, text: str, timepoint: str = "00m") -> "Knee":
        """Accept either naming form, since both appear in lists on disk."""
        parts = text.split("_")
        if len(parts) == 2:
            return cls(parts[0], parts[1], timepoint)
        if len(parts) == 3:
            return cls(parts[0], parts[2], parts[1])
        raise ConfigError(
            f"cannot read {text!r} as a knee; expected 'subject_SIDE' or "
            "'subject_timepoint_SIDE'"
        )

    def substitute(self, value: str) -> str:
        return (
            value.replace("{subject}", self.subject)
            .replace("{side}", self.side)
            .replace("{timepoint}", self.timepoint)
            .replace("{knee}", self.directory)
        )


@dataclass(frozen=True)
class Label:
    """A knee that is not a cohort knee -- a synthetic joint such as ``fig5_sigma_p0``.

    It stands in for :class:`Knee` wherever a stage only needs a name and a
    directory: ``{knee}`` becomes the label. It has no subject, side or
    timepoint, so a config that still asks for one is left with the placeholder
    and fails when resolved. (Parsing a
    label as a knee happens to succeed -- ``fig5_sigma_p0`` reads as subject
    ``fig5``, side ``p0`` -- which is why this type exists.)
    """

    label: str

    @property
    def name(self) -> str:
        return self.label

    @property
    def directory(self) -> str:
        return self.label

    def substitute(self, value: str) -> str:
        return value.replace("{knee}", self.label)


class Config:
    """A loaded run configuration.

    Values are read by section and key with a default: a config supplies
    overrides, and every default is visible in the code that asks for it.
    """

    def __init__(self, data: dict, origin: Path | None = None, knee: Knee | None = None):
        self._data = data
        self.origin = origin
        self.knee = knee

    # -- reading ---------------------------------------------------------

    def get(self, section: str, key: str, default=None):
        return _fill_run(self._data.get(section, {}).get(key, default))

    def section(self, name: str) -> dict:
        return _fill_run(self._data.get(name, {}))

    def require(self, section: str, key: str):
        try:
            value = self._data[section][key]
        except KeyError:
            where = f" in {self.origin}" if self.origin else ""
            raise ConfigError(f"{section}.{key} is required{where}") from None
        return _fill_run(value)

    @property
    def data(self) -> dict:
        """The raw dictionary, for recording into a run's provenance.

        Raw means ``{run}`` is left as written: this is the config's definition,
        and it reads the same on any root. Values read through ``get``,
        ``section``, ``require`` and ``path`` have it filled in.
        """
        return copy.deepcopy(self._data)

    # -- knee substitution -----------------------------------------------

    def for_knee(self, knee: "Knee | Label") -> "Config":
        """A copy with every ``{subject}``-style placeholder filled in."""

        def walk(node):
            if isinstance(node, dict):
                return {k: walk(v) for k, v in node.items()}
            if isinstance(node, list):
                return [walk(v) for v in node]
            if isinstance(node, str):
                return knee.substitute(node)
            return node

        return Config(walk(self._data), origin=self.origin, knee=knee)

    # -- path resolution --------------------------------------------------

    def path(self, section: str, key: str, default: str | None = None) -> Path:
        """Resolve a config path value against the data root.

        Raises
        ------
        ConfigError
            If the value is absolute, or still carries a placeholder, or does
            not begin with one of the stage directories. Each of those means
            the config is describing a location the pipeline does not own, so
            it is rejected.
        """
        value = self.get(section, key, default)
        if value is None:
            raise ConfigError(f"{section}.{key} is not set and has no default")
        return self.resolve(value, where=f"{section}.{key}")

    def resolve(self, value: str, where: str = "path") -> Path:
        if Path(value).is_absolute():
            raise ConfigError(
                f"{where} is an absolute path: {value}\n"
                "Config paths are relative to the data root, which "
                "comak_knee_oa.paths resolves. Write it as, for example, "
                f"'{_STAGE_DIRS[0]}/reference/...' instead."
            )
        leftover = _PLACEHOLDER.search(value)
        if leftover and leftover.group(0) == _RUN:
            raise ConfigError(
                f"{where} still contains {_RUN}: {value}\n"
                "It is filled from the data root's run declaration when a value "
                "is read through Config.get/section/require/path; a value taken "
                "from Config.data is the raw definition and still carries it."
            )
        if leftover:
            raise ConfigError(
                f"{where} still contains {leftover.group(0)}: {value}\n"
                "Call Config.for_knee() before resolving paths."
            )
        head = value.split("/", 1)[0]
        if head not in _STAGE_DIRS:
            raise ConfigError(
                f"{where} does not start with a pipeline stage: {value}\n"
                f"Expected one of: {', '.join(_STAGE_DIRS)}"
            )
        return data_root() / value

    def check_inputs(self, *specs: tuple[str, str]) -> list[str]:
        """Resolve several paths and report which do not exist.

        Called at the top of a stage, so a missing input is found by one
        directory listing instead of after the model has been built and an
        hour of solver time spent.
        """
        problems = []
        for section, key in specs:
            try:
                path = self.path(section, key)
            except ConfigError as exc:
                problems.append(f"{section}.{key}: {exc}")
                continue
            if not path.exists():
                problems.append(f"{section}.{key}: not found: {path}")
        return problems


def _walk(node, fn):
    if isinstance(node, dict):
        return {k: _walk(v, fn) for k, v in node.items()}
    if isinstance(node, list):
        return [_walk(v, fn) for v in node]
    if isinstance(node, str):
        return fn(node)
    return node


def _uses_run(node) -> bool:
    if isinstance(node, dict):
        return any(_uses_run(v) for v in node.values())
    if isinstance(node, list):
        return any(_uses_run(v) for v in node)
    return isinstance(node, str) and _RUN in node


def _fill_run(value):
    """``value`` with ``{run}`` replaced by the data root's run, if it carries it.

    The root is consulted only when the value names a run, so reading a value
    that does not -- on a machine with no data root at all -- still works.
    """
    if not _uses_run(value):
        return value
    name = run_name()
    return _walk(value, lambda text: text.replace(_RUN, name))


def load_config(path: str | Path, knee: Knee | None = None) -> Config:
    """Read a JSON config, optionally already substituted for one knee."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    config = Config(json.loads(path.read_text()), origin=path)
    return config.for_knee(knee) if knee else config
