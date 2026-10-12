"""Operator-supplied site dictionaries, loaded once and matched in one pass.

A site dictionary is explicit local vocabulary (facility names, departments,
provider names, identifier prefixes) supplied by the deploying site. It is
opt-in and off by default. Terms are indexed by folded text and a short
prefix, so a request costs one pass over the note plus a hash lookup per
candidate position, independent of how many terms are loaded.

Loading errors name the dictionary and the field, term index or file line,
never a term. Summaries report names, counts and content hashes only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import regex

MAX_DICTIONARIES = 64
MAX_TERMS = 1_000_000
MAX_TERM_LENGTH = 256
MAX_PATTERNS = 32
SOURCE_PREFIX = "site_dictionary:"
_NAME = regex.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_FIELDS = frozenset(
    (
        "name",
        "label",
        "terms",
        "terms_file",
        "case_sensitive",
        "word_boundaries",
        "patterns",
    )
)
# Terms are looked up by their first characters; shorter terms are checked
# directly. Longer prefixes mean fewer candidate lengths per hit.
_PREFIX = 4
_WORD = regex.compile(r"\w")
_ALNUM = regex.compile(r"[^\W_]")
# With word boundaries a match starts after a non-word character, and terms
# never begin with whitespace.
_START = regex.compile(r"(?<!\w)\S")
# Folding preserves offsets: every whitespace character becomes one space,
# and final sigma (lowercased by context) folds like any other sigma. All
# Unicode whitespace lies below U+3001.
_FOLD = {cp: " " for cp in range(0x3001) if chr(cp).isspace()} | {ord("ς"): "σ"}


class SiteDictionaryError(ValueError):
    """Invalid site dictionary configuration; the message contains no terms."""


@dataclass(frozen=True)
class SiteDictionary:
    name: str
    label: str
    terms: tuple[str, ...]
    patterns: tuple[str, ...] = ()
    case_sensitive: bool = False
    word_boundaries: bool = True

    @property
    def source(self):
        return SOURCE_PREFIX + self.name

    @property
    def digest(self):
        content = json.dumps(
            {
                "name": self.name,
                "label": self.label,
                "terms": sorted(self.terms),
                "patterns": list(self.patterns),
                "case_sensitive": self.case_sensitive,
                "word_boundaries": self.word_boundaries,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _fold(text, case_sensitive):
    if case_sensitive:
        return text.translate(_FOLD)
    lowered = text.lower()
    # Lowercasing never shortens a character, so equal lengths mean every
    # character mapped to exactly one; otherwise fold character by character.
    if len(lowered) != len(text):
        lowered = "".join(
            low if len(low := char.lower()) == 1 else char for char in text
        )
    return lowered.translate(_FOLD)


def _bounded(text, start, end):
    return (start == 0 or not _WORD.match(text, start - 1)) and (
        end == len(text) or not _WORD.match(text, end)
    )


class _Index:
    """Folded terms of the dictionaries sharing one case setting."""

    def __init__(self, case_sensitive, members, dictionaries):
        self.case_sensitive = case_sensitive
        self.members = frozenset(members)
        self.boundary_only = all(dictionaries[i].word_boundaries for i in members)
        owners = {}
        for index in members:
            for term in dictionaries[index].terms:
                found = owners.setdefault(_fold(term, case_sensitive), [])
                if index not in found:
                    found.append(index)
        prefixes, short = {}, set()
        for key in owners:
            if len(key) >= _PREFIX:
                prefixes.setdefault(key[:_PREFIX], set()).add(len(key))
            else:
                short.add(len(key))
        self.owners = {key: tuple(value) for key, value in owners.items()}
        self.prefixes = {
            key: tuple(sorted(value, reverse=True)) for key, value in prefixes.items()
        }
        self.short = tuple(sorted(short, reverse=True))

    def scan(self, text, selected, boundaries):
        """Longest valid term at each candidate start, as (start, end, index)."""
        folded = _fold(text, self.case_sensitive)
        size = len(text)
        owners, prefixes, short = self.owners, self.prefixes, self.short
        starts = (
            (match.start() for match in _START.finditer(text))
            if self.boundary_only
            else range(size)
        )
        found = []
        for start in starts:
            lengths = prefixes.get(folded[start : start + _PREFIX])
            if lengths is None and not short:
                continue
            for candidates in (lengths or (), short):
                matched = False
                for length in candidates:
                    end = start + length
                    if end > size:
                        continue
                    indices = owners.get(folded[start:end])
                    if indices is None:
                        continue
                    for index in indices:
                        if index in selected and (
                            not boundaries[index] or _bounded(text, start, end)
                        ):
                            found.append((start, end, index))
                            matched = True
                    if matched:
                        break
                if matched:
                    break
        return found


class SiteDictionaries:
    """An immutable set of validated dictionaries and their match indexes."""

    def __init__(self, dictionaries):
        from outis.policy import _compile_custom

        self.dictionaries = tuple(dictionaries)
        if not self.dictionaries:
            raise SiteDictionaryError("No site dictionaries were configured")
        self.names = tuple(item.name for item in self.dictionaries)
        self._positions = {name: index for index, name in enumerate(self.names)}
        self._boundaries = tuple(item.word_boundaries for item in self.dictionaries)
        self._indexes = []
        for case_sensitive in (True, False):
            members = [
                index
                for index, item in enumerate(self.dictionaries)
                if item.case_sensitive == case_sensitive and item.terms
            ]
            if members:
                self._indexes.append(_Index(case_sensitive, members, self.dictionaries))
        self._patterns = tuple(
            (index, _compile_custom(pattern))
            for index, item in enumerate(self.dictionaries)
            for pattern in item.patterns
        )
        digests = [item.digest for item in self.dictionaries]
        self.digest = hashlib.sha256("\n".join(digests).encode()).hexdigest()[:16]
        self._summary = {
            "count": len(self.dictionaries),
            "term_count": sum(len(item.terms) for item in self.dictionaries),
            "sha256": self.digest,
            "dictionaries": [
                {
                    "name": item.name,
                    "label": item.label,
                    "term_count": len(item.terms),
                    "pattern_count": len(item.patterns),
                    "case_sensitive": item.case_sensitive,
                    "word_boundaries": item.word_boundaries,
                    "sha256": digest,
                }
                for item, digest in zip(self.dictionaries, digests)
            ],
        }

    @property
    def labels(self):
        return frozenset(item.label for item in self.dictionaries)

    def summary(self):
        """Names, counts and content hashes; never terms or patterns."""
        return json.loads(json.dumps(self._summary))

    def select(self, names=None):
        """Indices of the named dictionaries (all when None), in configured order."""
        if names is None:
            return tuple(range(len(self.dictionaries)))
        if any(name not in self._positions for name in names):
            raise ValueError("Unknown site dictionary")
        wanted = set(names)
        return tuple(i for i, name in enumerate(self.names) if name in wanted)

    def find(self, text, selected):
        """Literal term matches as (start, end, index), dropping contained matches.

        Partially overlapping matches are all kept so that the policy merge
        covers their union. Identical spans keep the first configured
        dictionary.
        """
        selected = frozenset(selected)
        found = []
        for index in self._indexes:
            if index.members & selected:
                found.extend(index.scan(text, selected, self._boundaries))
        found.sort(key=lambda hit: (hit[0], -hit[1], hit[2]))
        result, edge = [], -1
        for start, end, index in found:
            if end > edge:
                result.append((start, end, index))
                edge = end
        return result

    def patterns(self, selected):
        """Compiled identifier-format expressions as (index, pattern)."""
        selected = frozenset(selected)
        return [(index, item) for index, item in self._patterns if index in selected]


def _where(name, position):
    return f"site dictionary {name!r}" if name else f"site dictionary #{position + 1}"


def _terms_from_file(path, where):
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError):
        raise SiteDictionaryError(f"{where}: terms_file cannot be read") from None
    return [
        (f"terms_file line {number}", line)
        for number, line in enumerate(lines, 1)
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _dictionary(spec, position, base, default_name=None):
    from outis.policy import _LABEL, MEDICAL_TYPES, CustomEntity

    if not isinstance(spec, dict):
        raise SiteDictionaryError(f"{_where(None, position)}: must be an object")
    name = spec.get("name", default_name)
    where = _where(name if isinstance(name, str) else None, position)
    unknown = set(spec) - _FIELDS
    if unknown:
        raise SiteDictionaryError(f"{where}: unknown fields {sorted(unknown)}")
    if not isinstance(name, str) or not _NAME.match(name):
        raise SiteDictionaryError(
            f"{where}: name must be 1-64 lowercase letters, digits, '_' or '-'"
        )
    label = spec.get("label")
    if not isinstance(label, str) or not _LABEL.match(label):
        raise SiteDictionaryError(f"{where}: invalid label")
    if label in MEDICAL_TYPES:
        raise SiteDictionaryError(f"{where}: medical concept labels are unsupported")
    flags = {}
    for flag, default in (("case_sensitive", False), ("word_boundaries", True)):
        flags[flag] = spec.get(flag, default)
        if type(flags[flag]) is not bool:
            raise SiteDictionaryError(f"{where}: {flag} must be true or false")
    raw = spec.get("terms", [])
    if not isinstance(raw, list):
        raise SiteDictionaryError(f"{where}: terms must be a list")
    entries = [(f"term {number}", term) for number, term in enumerate(raw, 1)]
    terms_file = spec.get("terms_file")
    if terms_file is not None:
        if not isinstance(terms_file, str) or not terms_file:
            raise SiteDictionaryError(f"{where}: terms_file must be a path")
        path = Path(terms_file)
        if base is not None and not path.is_absolute():
            path = base / path
        entries.extend(_terms_from_file(path, where))
    terms = {}
    for position_name, term in entries:
        if not isinstance(term, str):
            raise SiteDictionaryError(f"{where}: {position_name} must be a string")
        term = " ".join(term.split())
        if not 1 <= len(term) <= MAX_TERM_LENGTH:
            raise SiteDictionaryError(
                f"{where}: {position_name} must have 1 to {MAX_TERM_LENGTH} characters"
            )
        if not _ALNUM.search(term):
            raise SiteDictionaryError(
                f"{where}: {position_name} must contain a letter or digit"
            )
        terms.setdefault(term, None)
    patterns = spec.get("patterns", [])
    if not isinstance(patterns, list) or len(patterns) > MAX_PATTERNS:
        raise SiteDictionaryError(
            f"{where}: patterns must be a list of at most {MAX_PATTERNS} expressions"
        )
    for number, pattern in enumerate(patterns, 1):
        try:
            CustomEntity(label=label, pattern=pattern)
        except ValueError:
            raise SiteDictionaryError(f"{where}: pattern {number} is invalid") from None
    if not terms and not patterns:
        raise SiteDictionaryError(f"{where}: no terms or patterns")
    return SiteDictionary(
        name=name,
        label=label,
        terms=tuple(terms),
        patterns=tuple(patterns),
        **flags,
    )


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, ValueError, RecursionError):
        raise SiteDictionaryError(
            f"Site dictionary file {path.name!r} is not readable JSON"
        ) from None


def _manifest(value):
    if not isinstance(value, dict) or set(value) - {"version", "dictionaries"}:
        raise SiteDictionaryError(
            "A site dictionary manifest is an object with 'version' and 'dictionaries'"
        )
    if value.get("version", 1) != 1:
        raise SiteDictionaryError("Unsupported site dictionary manifest version")
    specs = value.get("dictionaries")
    if not isinstance(specs, list):
        raise SiteDictionaryError("Site dictionary manifest needs a dictionaries list")
    return specs


def load_site_dictionaries(source) -> SiteDictionaries | None:
    """Load dictionaries from a manifest path, a directory, or in-memory specs.

    ``source`` may be ``None`` (no dictionaries), a ``SiteDictionaries``, a path
    to a JSON manifest ``{"version": 1, "dictionaries": [...]}``, a directory
    whose ``*.json`` files each define one dictionary (named after the file by
    default), a manifest ``dict``, or a list of dictionary objects. Relative
    ``terms_file`` paths resolve against the defining file's directory.
    """
    if source is None or isinstance(source, SiteDictionaries):
        return source
    entries = []
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            files = sorted(path.glob("*.json"))
            entries = [(_read_json(file), file.parent, file.stem) for file in files]
        elif path.is_file():
            entries = [
                (spec, path.parent, None) for spec in _manifest(_read_json(path))
            ]
        else:
            raise SiteDictionaryError("Site dictionary path does not exist")
    elif isinstance(source, dict):
        entries = [(spec, None, None) for spec in _manifest(source)]
    elif isinstance(source, (list, tuple)):
        entries = [(spec, None, None) for spec in source]
    else:
        raise SiteDictionaryError("Unsupported site dictionary configuration")
    if len(entries) > MAX_DICTIONARIES:
        raise SiteDictionaryError(
            f"At most {MAX_DICTIONARIES} site dictionaries are supported"
        )
    dictionaries = [
        _dictionary(spec, position, base, default)
        for position, (spec, base, default) in enumerate(entries)
    ]
    names = [item.name for item in dictionaries]
    if len(set(names)) != len(names):
        raise SiteDictionaryError("Site dictionary names must be unique")
    if sum(len(item.terms) for item in dictionaries) > MAX_TERMS:
        raise SiteDictionaryError(f"At most {MAX_TERMS} site dictionary terms in total")
    return SiteDictionaries(dictionaries)
