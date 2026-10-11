"""Simultaneous requests, detector-lock recovery, patient isolation, and usage.

A blocking fake detector holds each call open until the test releases it, so
overlap is forced with events and barriers rather than hoped for with sleeps.
Every wait has a timeout: a lock that is never released fails, not hangs.
"""

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from outis.api import create_app
from outis.service import Deidentifier
from outis.types import Span

SECRET = b"test-only-secret-never-use-in-production-12345"
TIMEOUT = 10
NAMES = re.compile(r"Mara Calder|Ivo Brandt|Ines Duarte|Tomas Lindqvist")


def name_spans(texts):
    return [
        [Span(m.start(), m.end(), "NAME") for m in NAMES.finditer(text)]
        for text in texts
    ]


class BlockingDetector:
    """Counts overlapping calls; each call waits for ``release`` when gated."""

    def __init__(self, gated=True, fail=()):
        self.gated = gated
        self.fail = set(fail)
        self.release = threading.Event()
        self.entered = threading.Semaphore(0)
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = 0

    def detect_batch(self, texts):
        with self.lock:
            self.calls += 1
            call = self.calls
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.entered.release()
            if self.gated:
                assert self.release.wait(TIMEOUT), "detector was never released"
            if call in self.fail:
                raise RuntimeError("simulated accelerator failure")
            return name_spans(texts)
        finally:
            with self.lock:
                self.active -= 1


def wait_entered(detector, count=1):
    for _ in range(count):
        assert detector.entered.acquire(timeout=TIMEOUT), "detector not entered"


def eventually(condition):
    deadline = time.monotonic() + TIMEOUT
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


def run_threads(count, target):
    with ThreadPoolExecutor(count) as pool:
        futures = [pool.submit(target, index) for index in range(count)]
        return [future.result(timeout=TIMEOUT) for future in futures]


def test_detector_calls_are_serialized_and_waiters_are_counted_active():
    detector = BlockingDetector()
    service = Deidentifier(detector=detector, secret=SECRET)
    with ThreadPoolExecutor(3) as pool:
        futures = [
            pool.submit(service.deidentify, f"Mara Calder visit {i}.", project_id="p")
            for i in range(3)
        ]
        wait_entered(detector)
        # One request is inside the detector; the other two wait on its lock
        # but are already in flight.
        eventually(lambda: service.usage.snapshot()["active_requests"] == 3)
        assert not detector.entered.acquire(timeout=0.1)
        detector.release.set()
        results = [future.result(timeout=TIMEOUT) for future in futures]
    assert detector.max_active == 1 and detector.calls == 3
    assert sorted(r["text"] for r in results) == [
        f"[NAME] visit {i}." for i in range(3)
    ]
    snapshot = service.usage.snapshot()
    assert snapshot["active_requests"] == 0
    ((row),) = snapshot["projects"].values()
    assert (row["requests"], row["successes"], row.get("failures", 0)) == (3, 3, 0)


@pytest.mark.parametrize("failure", ["raise", "malformed"])
def test_detector_lock_is_released_after_a_failed_call(failure):
    detector = BlockingDetector(gated=False, fail=[1] if failure == "raise" else ())
    service = Deidentifier(detector=detector, secret=SECRET)
    if failure == "malformed":
        original = detector.detect_batch
        malformed = [True]
        # The first call returns no rows for a one-text batch.
        detector.detect_batch = lambda texts: (
            [] if malformed and malformed.pop() else original(texts)
        )
        with pytest.raises(RuntimeError, match="invalid batch output"):
            service.deidentify("Mara Calder seen.")
    else:
        with pytest.raises(RuntimeError, match="simulated"):
            service.deidentify("Mara Calder seen.")
    assert not service.lock.locked()
    # A second thread must acquire the lock; a leaked lock would block here.
    (result,) = run_threads(1, lambda _: service.deidentify("Ivo Brandt seen."))
    assert result["text"] == "[NAME] seen."
    snapshot = service.usage.snapshot()
    ((row),) = snapshot["projects"].values()
    assert (row["successes"], row["failures"]) == (1, 1)
    assert snapshot["active_requests"] == 0


def test_failure_while_others_wait_does_not_strand_them():
    detector = BlockingDetector(fail=[1])
    service = Deidentifier(detector=detector, secret=SECRET)

    def call(index):
        try:
            return service.deidentify(f"Mara Calder visit {index}.")["text"]
        except RuntimeError as error:
            return str(error)

    with ThreadPoolExecutor(4) as pool:
        futures = [pool.submit(call, index) for index in range(4)]
        wait_entered(detector)
        detector.release.set()
        outcomes = [future.result(timeout=TIMEOUT) for future in futures]
    assert outcomes.count("simulated accelerator failure") == 1
    successes = [o for o in outcomes if o != "simulated accelerator failure"]
    assert len(successes) == 3
    assert all(re.fullmatch(r"\[NAME\] visit [0-3]\.", o) for o in successes)
    assert not service.lock.locked()


PATIENTS = {
    "patient-a": ["Mara Calder seen with Ivo Brandt.", "Mara Calder returned."],
    "patient-b": ["Ines Duarte seen with Tomas Lindqvist.", "Tomas Lindqvist called."],
}


def synthetic_batch(service, patient):
    return service.deidentify_batch(
        [
            {"text": text, "mode": "synthetic", "patient_id": patient}
            for text in PATIENTS[patient]
        ],
        shared_entities=True,
    )


def test_concurrent_patients_with_shared_context_stay_isolated():
    detector = BlockingDetector(gated=False)
    service = Deidentifier(detector=detector, secret=SECRET)
    expected = {patient: synthetic_batch(service, patient) for patient in PATIENTS}
    assert expected["patient-a"] != expected["patient-b"]
    order = list(PATIENTS) * 8
    barrier = threading.Barrier(len(order))

    def call(index):
        barrier.wait(TIMEOUT)
        return order[index], synthetic_batch(service, order[index])

    for patient, results in run_threads(len(order), call):
        assert results == expected[patient]
        released = " ".join(result["text"] for result in results)
        for other in PATIENTS:
            for text in PATIENTS[other]:
                for name in NAMES.findall(text):
                    assert name not in released


def test_usage_counters_are_exact_under_contention():
    detector = BlockingDetector(gated=False, fail=range(1, 200, 5))
    service = Deidentifier(detector=detector, secret=SECRET)
    barrier = threading.Barrier(8)

    def call(index):
        barrier.wait(TIMEOUT)
        outcomes = []
        for _ in range(10):
            project = f"project-{index % 3}"
            try:
                service.deidentify("Mara Calder seen.", project_id=project)
                outcomes.append((project, True))
            except RuntimeError:
                outcomes.append((project, False))
        return outcomes

    outcomes = [item for result in run_threads(8, call) for item in result]
    snapshot = service.usage.snapshot()
    assert snapshot["active_requests"] == 0
    rows = list(snapshot["projects"].values())
    assert sum(row["requests"] for row in rows) == 80
    assert sum(row["documents"] for row in rows) == 80
    assert sum(row["characters"] for row in rows) == 80 * len("Mara Calder seen.")
    assert sum(row.get("successes", 0) for row in rows) == sum(ok for _, ok in outcomes)
    assert sum(row.get("failures", 0) for row in rows) == 16
    assert sorted(row["requests"] for row in rows) == sorted(
        [sum(1 for p, _ in outcomes if p == f"project-{k}") for k in range(3)]
    )


def test_http_requests_run_concurrently_and_report_consistent_usage():
    detector = BlockingDetector()
    service = Deidentifier(detector=detector, secret=SECRET)
    # Four inference workers admit all four requests at once; the detector
    # lock still serializes the model calls themselves.
    with TestClient(create_app(service, inference_workers=4)) as client:

        def post(index):
            return client.post(
                "/v1/deidentify", json={"text": f"Ines Duarte, visit {index}."}
            )

        with ThreadPoolExecutor(4) as pool:
            futures = [pool.submit(post, index) for index in range(4)]
            wait_entered(detector)
            # Health and usage stay responsive while inference is blocked.
            assert client.get("/health").json() == {"status": "ok"}
            eventually(lambda: client.get("/v1/usage").json()["active_requests"] == 4)
            assert detector.calls == 1
            detector.release.set()
            responses = [future.result(timeout=TIMEOUT) for future in futures]
        assert [r.status_code for r in responses] == [200] * 4
        assert sorted(r.json()["text"] for r in responses) == [
            f"[NAME], visit {i}." for i in range(4)
        ]
        usage = client.get("/v1/usage").json()
    assert usage["active_requests"] == 0
    ((row),) = usage["projects"].values()
    assert (row["requests"], row["successes"]) == (4, 4)
    assert detector.max_active == 1
