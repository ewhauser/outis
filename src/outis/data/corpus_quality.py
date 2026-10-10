"""Read-only development corpus checks; no discovery or qualification access."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import unicodedata

from outis.data.annotation_policy import COHORTS, POLICY_VERSION, annotation_errors, identity_split
from outis.types import ENTITY_TYPES


class ProtectedCorpusError(ValueError):
    """Raised before opening a path outside development corpus scope."""


_PROTECTED = re.compile(r"(?:^|[_.-])(sealed|final|test|qualification|blind|v6|attempt|ledger)(?:$|[_.-])", re.I)
_PROTECTED_ANCESTOR = re.compile(r"(?:^|[_.-])(sealed|qualification|blind|v6|ledger)(?:$|[_.-])", re.I)
_EXPLANATIONS = (
    "age below 90 remains clinical information",
    "ages below 90 remain clinical information",
    "no identified individual",
    "this is not phi",
    "this is non-phi",
    "should not be tagged",
    "should not be labeled",
    "should be labeled",
    "not a patient identifier",
    "no patient dates were recorded",
    "not named people",
    "this was a disease term",
    "non-identifying",
)
_CUES = {
    "ordinary_pronoun": re.compile(r"\b(?:i|we|you|it|he|she|they|me|us|him|his|her|their|our|your)\b", re.I),
    "publication": re.compile(r"\b(?:publication|published|study|journal|article)\b", re.I),
    "duration": re.compile(r"\b\d+\s+(?:days?|weeks?|months?|years?|hours?|minutes?)\b", re.I),
    "age_wording": re.compile(r"\b(?:age|aged|years? old|year-old|y/o|\d+(?:yo|[FM])|\d+-year-old)\b", re.I),
    "teaching_wording": re.compile(r"\b(?:teaching|textbook|educational|example)\b", re.I),
    "name_field": re.compile(r"\b(?:patient name|named|preferred name)\b", re.I),
}


def _path_scope(path: Path, *, manifest=False) -> None:
    if not (path.name.endswith(".manifest.json") if manifest else path.suffix.lower() == ".jsonl"):
        raise ProtectedCorpusError("Only explicitly named development .jsonl files are accepted")
    if _PROTECTED.search(path.name):
        raise ProtectedCorpusError(f"Protected corpus path refused before reading: {path}")
    for part in path.parts[:-1]:
        if part.lower() in {"test", "tests", "final", "sealed"} or _PROTECTED_ANCESTOR.search(part):
            raise ProtectedCorpusError(f"Protected corpus directory refused before reading: {path}")


def _read_scoped_bytes(path: str | Path, *, manifest=False) -> tuple[Path, bytes]:
    """Check lexical/resolved scope, then open each component without symlinks.

    Holding directory descriptors avoids a symlink replacement between path
    validation and opening. All inputs are regular, explicit JSONL files.
    Manifests are accepted only through the explicitly scoped preflight below.
    """
    original = Path(path).expanduser()
    _path_scope(original, manifest=manifest)
    absolute = Path(os.path.abspath(original))
    _path_scope(absolute, manifest=manifest)
    resolved = absolute.resolve(strict=True)
    _path_scope(resolved, manifest=manifest)
    if absolute != resolved:
        raise ProtectedCorpusError(f"Symlink or noncanonical input path refused: {original}")
    directory_fd = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in absolute.parts[1:-1]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        fd = os.open(absolute.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ProtectedCorpusError("Corpus input must be a regular file")
            payload = handle.read()
            after = os.fstat(handle.fileno())
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
            ):
                raise ValueError("Corpus changed while being audited")
        current = os.stat(absolute.name, dir_fd=directory_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("Corpus path changed while being audited")
    finally:
        os.close(directory_fd)
    return resolved, payload


def _preflight_manifests(path: str | Path, role: str | None, *, required: bool) -> list[dict]:
    """Read only explicit adjacent declarations, before reading clinical text."""
    path = Path(path).expanduser()
    _path_scope(path)
    for marker in (Path(str(path) + ".sealed"), path.with_suffix(".sealed")):
        if marker.exists() or marker.is_symlink():
            raise ProtectedCorpusError(f"Sealed marker found; payload will not be read: {path}")
    manifests = []
    for candidate in (path.with_suffix(path.suffix + ".manifest.json"), path.with_suffix(".manifest.json")):
        if not candidate.exists() and not candidate.is_symlink():
            continue
        resolved, payload = _read_scoped_bytes(candidate, manifest=True)
        declaration = json.loads(payload.decode("utf-8"))
        if not isinstance(declaration, dict):
            raise ValueError("Corpus manifest must be an object")
        for key in ("split", "suite", "role", "corpus_role", "kind", "purpose"):
            value = declaration.get(key)
            if isinstance(value, str) and _PROTECTED.search(value):
                raise ProtectedCorpusError(f"Protected {key} declaration; payload will not be read: {path}")
        for key in ("sealed", "is_final", "independent_final", "qualification"):
            if declaration.get(key) not in (None, False):
                raise ProtectedCorpusError(f"Protected {key} declaration; payload will not be read: {path}")
        if required and declaration.get("split") != role:
            raise ValueError(f"Strict manifest must declare split={role!r}: {candidate}")
        if required and declaration.get("annotation_policy") != POLICY_VERSION:
            raise ValueError(f"Strict manifest must declare annotation_policy={POLICY_VERSION!r}: {candidate}")
        digest = declaration.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Corpus manifest must declare a SHA-256 digest: {candidate}")
        manifests.append({"path": str(resolved), "sha256": hashlib.sha256(payload).hexdigest(), "declared_sha256": digest})
    if required and not manifests:
        raise ValueError(f"Strict corpus audit requires an adjacent manifest before reading: {path}")
    return manifests


def read_development_bytes(path: str | Path) -> tuple[Path, bytes]:
    """Read one scoped input, honoring any adjacent protected declaration."""
    manifests = _preflight_manifests(path, None, required=False)
    resolved, payload = _read_scoped_bytes(path)
    if any(item["declared_sha256"] != hashlib.sha256(payload).hexdigest() for item in manifests):
        raise ValueError("Corpus SHA-256 differs from its manifest")
    return resolved, payload


def _summary(values: list[int]) -> dict:
    if not values:
        return {"count": 0}
    values = sorted(values)
    return {"count": len(values), "minimum": values[0], "maximum": values[-1],
            "mean": round(sum(values) / len(values), 3),
            "p50": values[math.ceil(len(values) * .5) - 1],
            "p95": values[math.ceil(len(values) * .95) - 1]}


def _perfect_wilson_lower(support: int) -> float:
    return support / (support + 1.959963984540054 ** 2) if support else 0.0


def _masked_template(text: str, spans: list[dict]) -> str:
    parts, offset = [], 0
    for span in sorted(spans, key=lambda item: item["start"]):
        parts.extend((text[offset:span["start"]], f"<{span['label']}>"))
        offset = span["end"]
    parts.append(text[offset:])
    return " ".join(unicodedata.normalize("NFC", "".join(parts)).casefold().split())


def _script(surface: str) -> str:
    scripts = set()
    for character in surface:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "UNKNOWN")
        scripts.add(next((item for item in ("LATIN", "CYRILLIC", "GREEK", "ARABIC", "HEBREW", "HIRAGANA", "KATAKANA", "HANGUL", "CJK", "DEVANAGARI") if item in name), "OTHER"))
    return "+".join(sorted(scripts)) or "NONE"


def audit_corpora(corpora: dict, *, strict_policy: bool = True, token_counter=None, tokenizer_name: str | None = None) -> dict:
    """Audit explicit ``{train: [paths], validation: [paths]}`` inputs.

    Structural defects and cross-split group contamination always fail. Missing
    v2 metadata is allowed only in legacy mode, which cannot certify v2 policy.
    Text is never rewritten, inferred labels are never added, and warnings do
    not count as annotation adjudication or qualification evidence.
    """
    if not corpora or any(role not in {"train", "validation"} for role in corpora):
        raise ValueError("Only explicit train and validation input roles are accepted")
    if token_counter is not None and not tokenizer_name:
        raise ValueError("A tokenizer_name is required with token_counter")
    inputs = []
    for role, paths in corpora.items():
        if isinstance(paths, (str, Path)):
            paths = [paths]
        if not isinstance(paths, (list, tuple)) or not paths:
            raise ValueError("Each corpus role requires one or more explicit paths")
        for path in paths:
            # Validate every lexical path before reading any input.
            _path_scope(Path(path).expanduser())
            inputs.append((role, path))
    # Preflight every sidecar before opening any corpus payload. A protected
    # declaration on a later input therefore cannot follow an earlier text read.
    declarations = [_preflight_manifests(path, role, required=strict_policy) for role, path in inputs]
    errors, warnings, artifacts = [], [], []
    all_rows, identifiers, text_hashes = [], {}, {}
    groups = {key: defaultdict(set) for key in ("identity_group", "pair_id", "template_family")}
    masked, masked_sentences = defaultdict(set), defaultdict(set)
    splits, label_support, cohort_support, phenomenon_support = Counter(), Counter(), defaultdict(Counter), defaultdict(Counter)
    lengths, name_lengths, name_tokens = defaultdict(list), defaultdict(list), defaultdict(list)
    token_lengths, scripts, name_shapes, cues, positive_negative = defaultdict(list), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter), Counter()
    policy_counts = Counter()

    def issue(code, message, path=None, row=None):
        item = {"code": code, "message": message}
        if path is not None:
            item["path"] = str(path)
        if row is not None:
            item["row"] = row
        errors.append(item)

    for (role, path), manifests in zip(inputs, declarations):
        resolved, payload = _read_scoped_bytes(path)
        digest = hashlib.sha256(payload).hexdigest()
        if any(item["declared_sha256"] != digest for item in manifests):
            raise ValueError(f"Corpus SHA-256 differs from its manifest: {path}")
        artifact = {"path": str(resolved), "role": role, "sha256": digest, "bytes": len(payload), "rows": 0, "manifests": manifests}
        artifacts.append(artifact)
        try:
            lines = payload.decode("utf-8").splitlines()
        except UnicodeDecodeError:
            issue("invalid_encoding", "Corpus must be UTF-8", resolved)
            continue
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            artifact["rows"] += 1
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                issue("invalid_json", f"Invalid JSON at line {line_number}", resolved)
                continue
            if not isinstance(row, dict):
                issue("invalid_row", f"Expected object at line {line_number}", resolved)
                continue
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier.strip():
                issue("invalid_id", f"Nonempty string ID required at line {line_number}", resolved)
                identifier = f"line:{line_number}"
            elif identifier in identifiers:
                issue("duplicate_id", f"ID also occurs in {identifiers[identifier]}", resolved, identifier)
            else:
                identifiers[identifier] = str(resolved)
            # Protected labels are rejected without processing row contents.
            if row.get("split") in ("test", "final", "sealed") or row.get("suite") in ("test", "final", "sealed", "qualification"):
                issue("protected_row", "Final/test/sealed row refused; no diagnostics computed", resolved, identifier)
                continue
            if row.get("split") != role:
                issue("split_mismatch", f"Row must declare split={role!r}", resolved, identifier)
            text, spans = row.get("text"), row.get("spans")
            invalid = annotation_errors(text, spans)
            for message in invalid:
                issue("invalid_annotation", message, resolved, identifier)
            if invalid:
                continue
            fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if fingerprint in text_hashes:
                prior = text_hashes[fingerprint]
                issue("duplicate_text", f"Exact text also occurs at {prior['path']} row {prior['id']} ({prior['role']})", resolved, identifier)
                if prior["spans"] != spans:
                    issue("conflicting_annotations", "Exact duplicated text has different annotations", resolved, identifier)
            else:
                text_hashes[fingerprint] = {"path": str(resolved), "id": identifier, "role": role, "spans": spans}
            policy = row.get("annotation_policy")
            policy_counts[str(policy)] += 1
            if strict_policy and policy != POLICY_VERSION:
                issue("policy_version", f"annotation_policy must be {POLICY_VERSION!r}", resolved, identifier)
            cohort = row.get("cohort", "legacy_unspecified")
            if not isinstance(cohort, str) or (strict_policy and cohort not in COHORTS):
                issue("cohort_metadata", f"cohort must be one of {COHORTS}", resolved, identifier)
                cohort = "invalid"
            phenomena = row.get("phenomena", [])
            if not isinstance(phenomena, list) or any(not isinstance(item, str) or not item.strip() for item in phenomena) or len(set(phenomena)) != len(phenomena):
                issue("phenomena_metadata", "phenomena must be a list of distinct nonempty strings", resolved, identifier)
                phenomena = []
            elif strict_policy and not phenomena:
                issue("phenomena_metadata", "Strict policy requires at least one phenomenon", resolved, identifier)
            for key, mapping in groups.items():
                group = row.get(key)
                if group is None and not strict_policy:
                    continue
                if not isinstance(group, str) or not group.strip():
                    issue("group_metadata", f"{key} must be a nonempty string", resolved, identifier)
                    continue
                mapping[group].add(role)
                if key == "identity_group":
                    is_digest = bool(re.fullmatch(r"[0-9a-f]{64}", group))
                    if is_digest and identity_split(group) != role:
                        issue("identity_partition", "SHA-256 identity group is assigned to the wrong shared hash split", resolved, identifier)
                    elif strict_policy and spans and not is_digest:
                        issue("identity_group_format", "PHI-bearing v2 rows require a canonical SHA-256 identity_group", resolved, identifier)
            if "identity_groups" in row:
                identities = row["identity_groups"]
                if (not isinstance(identities, list) or (not identities and spans)
                        or any(not isinstance(item, str) or not item.strip() for item in identities)
                        or len(set(identities)) != len(identities)
                        or (identities and row.get("identity_group") not in identities)):
                    issue("identity_groups_metadata", "identity_groups must contain unique nonempty strings including identity_group", resolved, identifier)
                else:
                    for group in identities:
                        groups["identity_group"][group].add(role)
                        if re.fullmatch(r"[0-9a-f]{64}", group):
                            if identity_split(group) != role:
                                issue("identity_partition", "An identity_groups member is assigned to the wrong shared hash split", resolved, identifier)
                        elif strict_policy and spans:
                            issue("identity_group_format", "PHI-bearing identity_groups members must be canonical SHA-256 digests", resolved, identifier)
            explanations = [phrase for phrase in _EXPLANATIONS if phrase in text.casefold()]
            if explanations:
                item = {"code": "annotation_explanation", "message": "Clinical text contains annotation-policy explanation cues", "row": identifier, "phrases": explanations}
                (errors if strict_policy else warnings).append(item)
            state = "phi_bearing" if spans else "phi_free"
            stratum = f"{role}/{state}"
            positive_negative[stratum] += 1
            for cue, pattern in _CUES.items():
                cues[stratum][cue] += bool(pattern.search(text))
            splits[role] += 1
            lengths[role].append(len(text))
            if token_counter is not None:
                count = token_counter(text)
                if type(count) is not int or count < 0:
                    raise ValueError("token_counter must return a nonnegative integer")
                token_lengths[role].append(count)
            masked_text = _masked_template(text, spans)
            masked[hashlib.sha256(masked_text.encode()).hexdigest()].add(role)
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", masked_text):
                # Normalized unannotated numeric vitals/durations make nearby
                # repeated context visible; this remains a review diagnostic.
                sentence = re.sub(r"\b\d+(?:[./:-]\d+)*\b", "<number>", sentence)
                if len(sentence.split()) >= 4:
                    masked_sentences[hashlib.sha256(sentence.encode()).hexdigest()].add(role)
            cohort_support[f"{role}/{cohort}"]["documents"] += 1
            cohort_support[f"{role}/{cohort}"][state] += 1
            for phenomenon in phenomena:
                phenomenon_support[f"{role}/{phenomenon}"]["documents"] += 1
            for span in spans:
                label = span["label"]
                label_support[(role, label)] += 1
                cohort_support[f"{role}/{cohort}"][label] += 1
                for phenomenon in phenomena:
                    phenomenon_support[f"{role}/{phenomenon}"][label] += 1
                if label == "NAME":
                    surface = text[span["start"]:span["end"]]
                    name_lengths[role].append(len(surface))
                    if token_counter is not None:
                        name_tokens[role].append(token_counter(surface))
                    scripts[role][_script(surface)] += 1
                    name_shapes[role]["single_word" if len(surface.split()) == 1 else "multiple_words"] += 1
                    name_shapes[role]["punctuated" if any(unicodedata.category(ch).startswith("P") for ch in surface) else "unpunctuated"] += 1
                    name_shapes[role]["ascii" if surface.isascii() else "non_ascii"] += 1
            all_rows.append((role, identifier))
        if not artifact["rows"]:
            issue("empty_corpus", "Corpus contains no rows", resolved)
    for key, mapping in groups.items():
        overlap = sorted(group for group, roles in mapping.items() if len(roles) > 1)
        if overlap:
            issue("cross_split_" + key, f"{len(overlap)} {key} groups occur in both train and validation")
            errors[-1]["examples"] = overlap[:10]
    overlap = sum(len(roles) > 1 for roles in masked.values())
    if overlap:
        warnings.append({"code": "masked_template_overlap", "message": f"{overlap} normalized PHI-masked document templates overlap across splits; review shared context", "count": overlap})
    sentence_overlap = sum(len(roles) > 1 for roles in masked_sentences.values())
    if sentence_overlap:
        warnings.append({"code": "masked_sentence_overlap", "message": f"{sentence_overlap} PHI-masked, number-normalized sentences overlap across splits; inspect context reuse", "count": sentence_overlap})
    for role in corpora:
        positive, negative = f"{role}/phi_bearing", f"{role}/phi_free"
        if min(positive_negative[positive], positive_negative[negative]) >= 20:
            for cue in _CUES:
                positive_rate = cues[positive][cue] / positive_negative[positive]
                negative_rate = cues[negative][cue] / positive_negative[negative]
                if abs(positive_rate - negative_rate) >= .25:
                    warnings.append({"code": "context_cue_imbalance", "message": f"{role} cue {cue!r} differs by at least 25 percentage points between document strata; review context diversity", "cue": cue, "role": role, "phi_bearing_rate": positive_rate, "phi_free_rate": negative_rate})
    support = {}
    for role in corpora:
        support[role] = {}
        for label in ENTITY_TYPES:
            count = label_support[(role, label)]
            support[role][label] = {"spans": count, "perfect_recall_wilson_lower": _perfect_wilson_lower(count), "could_demonstrate_99pct_wilson": count >= 381}
        if any(label_support[(role, label)] < 200 for label in ENTITY_TYPES):
            warnings.append({"code": "limited_label_support", "message": f"{role} has labels below 200 spans; see per-label support. This diagnostic bank does not establish qualification."})
    for role in corpora:
        for state in ("phi_bearing", "phi_free"):
            if not positive_negative[f"{role}/{state}"]:
                warnings.append({"code": "missing_document_stratum", "message": f"{role} has no {state} documents"})
    if token_counter is None:
        warnings.append({"code": "token_lengths_unavailable", "message": "No tokenizer supplied; character counts are not model token lengths"})
    if not strict_policy:
        warnings.append({"code": "legacy_mode", "message": "Legacy structural audit does not certify annotations against clinical-phi-v2"})
    return {
        "audit_version": 1, "annotation_policy": POLICY_VERSION,
        "strict_policy": strict_policy, "passed": not errors,
        "errors": errors, "warnings": warnings, "inputs": artifacts,
        "valid_structural_rows": len(all_rows), "documents_by_split": dict(splits),
        "policy_versions": dict(policy_counts),
        "label_support": support,
        "cohort_support": {key: dict(value) for key, value in sorted(cohort_support.items())},
        "phenomenon_support": {key: dict(value) for key, value in sorted(phenomenon_support.items())},
        "document_strata": dict(positive_negative),
        "masked_templates": {"unique": len(masked), "cross_split_unique": overlap,
                             "cross_split_fraction": overlap / len(masked) if masked else 0},
        "masked_sentences": {"unique": len(masked_sentences), "cross_split_unique": sentence_overlap,
                             "cross_split_fraction": sentence_overlap / len(masked_sentences) if masked_sentences else 0},
        "distributions": {
            "document_characters": {key: _summary(value) for key, value in lengths.items()},
            "document_tokens": {key: _summary(value) for key, value in token_lengths.items()},
            "name_characters": {key: _summary(value) for key, value in name_lengths.items()},
            "name_tokens": {key: _summary(value) for key, value in name_tokens.items()},
            "name_scripts": {key: dict(value) for key, value in scripts.items()},
            "name_shapes": {key: dict(value) for key, value in name_shapes.items()},
        },
        "tokenizer": tokenizer_name if token_counter is not None else None,
        "context_cues": {key: {"documents": positive_negative[key], "counts": dict(value),
                              "rates": {cue: count / positive_negative[key] for cue, count in value.items()}}
                         for key, value in sorted(cues.items())},
        "limitations": [
            "Structural checks cannot adjudicate whether every occurrence follows the semantic policy.",
            "Grouped identity and template disjointness is metadata-dependent and does not prove independent authorship.",
            "Masked template overlap and lexical cue rates are diagnostics, not universal pass/fail thresholds.",
            "The Wilson interval assumes binomial span outcomes; repeated synthetic templates reduce effective independence.",
            "Even 200/200 recalled spans cannot demonstrate a 99% two-sided 95% Wilson lower bound; 381/381 is the minimum.",
            "No sealed qualification input is read; compatibility with its annotation policy remains unverified.",
        ],
    }
