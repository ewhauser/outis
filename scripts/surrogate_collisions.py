"""Measure project-scope surrogate collisions on invented values.

In project scope one value maps to one surrogate across a whole release, so two
distinct source values that draw the same surrogate merge in the output. This
reports, per label and corpus size, the share of distinct invented values whose
surrogate is shared with at least one other value. ``--scope patient`` shows
what the narrower patient-scope spaces would give if one patient anchor covered
the same number of values. ``--style realistic`` measures the format-preserving
style; its spaces do not depend on scope. No real PHI is used; the secret,
project ID and patient anchor are fixed test values.
"""

import argparse
import random
import string
from collections import Counter

from outis.replacement.primitives import project_key
from outis.replacement.synthetic import synthetic_value

SECRET = b"surrogate-collision-measurement-secret-0123"
PROJECT_ID = "collision-measurement"
PATIENT = "collision-measurement-patient"


def _word(rng, length):
    return rng.choice(string.ascii_uppercase) + "".join(
        rng.choices(string.ascii_lowercase, k=length - 1)
    )


def _digits(rng, length):
    return "".join(rng.choices(string.digits, k=length))


GENERATORS = {
    "NAME": lambda rng: (
        f"{_word(rng, rng.randint(4, 8))} {_word(rng, rng.randint(5, 9))}"
    ),
    "FACILITY": lambda rng: f"{_word(rng, 8)} Medical Center",
    "LOCATION": lambda rng: _word(rng, 9),
    "ADDRESS": lambda rng: f"{rng.randint(1, 9999)} {_word(rng, 7)} Street",
    "PHONE": lambda rng: f"{rng.randint(200, 999)}-{_digits(rng, 3)}-{_digits(rng, 4)}",
    "EMAIL": lambda rng: f"{_word(rng, 8).lower()}@example.org",
    "URL": lambda rng: f"https://{_word(rng, 8).lower()}.example.org/",
    "IP": lambda rng: "10." + ".".join(str(rng.randint(0, 255)) for _ in range(3)),
    "SSN": lambda rng: f"{_digits(rng, 3)}-{_digits(rng, 2)}-{_digits(rng, 4)}",
    **{
        f"MRN ({length} digits)": lambda rng, length=length: _digits(rng, length)
        for length in range(3, 13)
    },
    "MRN (AA-999999)": lambda rng: (
        "".join(rng.choices(string.ascii_uppercase, k=2)) + "-" + _digits(rng, 6)
    ),
    "ACCOUNT": lambda rng: _digits(rng, 9),
    "ID": lambda rng: "ID-" + _digits(rng, 7),
}


def distinct_values(label, count, seed=0):
    """Up to ``count`` distinct values; a short MRN domain is used whole."""
    rng = random.Random(f"{label}:{seed}")
    generate = GENERATORS[label]
    if label.startswith("MRN (") and label[5:].split(" ")[0].isdigit():
        length = int(label[5:].split(" ")[0])
        count = min(count, 10**length)
    values = set()
    while len(values) < count:
        values.add(generate(rng))
    return sorted(values)


def collision_share(
    label,
    values,
    *,
    scope="project",
    style="obvious",
    secret=SECRET,
    project_id=PROJECT_ID,
):
    """Share of distinct values whose surrogate is not unique in one scope."""
    key, patient = (
        (project_key(secret, project_id), None)
        if scope == "project"
        else (secret, PATIENT)
    )
    entity = label.split(" ")[0]
    surrogates = [
        synthetic_value(value, entity, key, patient, {}, style=style)
        for value in values
    ]
    counts = Counter(surrogates)
    shared = sum(1 for surrogate in surrogates if counts[surrogate] > 1)
    return shared / len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[2_000, 20_000])
    parser.add_argument("--scope", choices=["project", "patient"], default="project")
    parser.add_argument("--style", choices=["obvious", "realistic"], default="obvious")
    args = parser.parse_args()
    print("| Label | " + " | ".join(f"{n:,} values" for n in args.sizes) + " |")
    print("| --- |" + " ---: |" * len(args.sizes))
    for label in GENERATORS:
        rates = [
            collision_share(
                label, distinct_values(label, n), scope=args.scope, style=args.style
            )
            for n in args.sizes
        ]
        print(f"| {label} | " + " | ".join(f"{rate:.2%}" for rate in rates) + " |")


if __name__ == "__main__":
    main()
