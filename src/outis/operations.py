"""Bounded, process-local counters containing no source text or patient IDs."""

import hashlib
import platform
import time
from collections import Counter
from contextlib import contextmanager
from threading import Lock


class Usage:
    def __init__(self):
        self._lock = Lock()
        self._projects = {}
        self._active = 0
        self._started = time.monotonic()

    @contextmanager
    def track(self, projects, characters, operation):
        started = time.monotonic()
        with self._lock:
            self._active += 1
        success = False
        try:
            yield
            success = True
        finally:
            with self._lock:
                self._active -= 1
                counts = Counter(projects)
                for project in counts:
                    # Caller-controlled IDs never become raw metric labels. Bound
                    # cardinality independently of request volume.
                    digest = hashlib.sha256(project.encode()).hexdigest()[:16]
                    key = (
                        digest
                        if digest in self._projects or len(self._projects) < 128
                        else "overflow"
                    )
                    row = self._projects.setdefault(key, Counter())
                    row["requests"] += 1
                    row["documents"] += counts[project]
                    row["characters"] += sum(
                        n for p, n in zip(projects, characters) if p == project
                    )
                    row["successes" if success else "failures"] += 1
                    row["seconds"] += time.monotonic() - started

    def snapshot(self):
        with self._lock:
            return {
                "uptime_seconds": time.monotonic() - self._started,
                "active_requests": self._active,
                "projects": {key: dict(row) for key, row in self._projects.items()},
            }

    def prometheus(self):
        snapshot = self.snapshot()
        lines = [
            "# TYPE deid_active_requests gauge",
            f"deid_active_requests {snapshot['active_requests']}",
        ]
        for field in (
            "requests",
            "documents",
            "characters",
            "successes",
            "failures",
            "seconds",
        ):
            metric = f"deid_{field}_total"
            lines.append(f"# TYPE {metric} counter")
            for key, row in sorted(snapshot["projects"].items()):
                lines.append(f'{metric}{{project="{key}"}} {row.get(field, 0)}')
        return "\n".join(lines) + "\n"


def diagnostics(service):
    result = {
        "api_version": "0.2.0",
        "model_version": service.version,
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "device": str(getattr(service.detector, "device", "injected")),
        "language_scope": ["en"],
        "usage_persistence": "process_local",
    }
    # Names, counts and content hashes only; never dictionary terms.
    site = getattr(service, "site_dictionaries", None)
    if site is not None:
        result["site_dictionaries"] = site.summary()
    return result
