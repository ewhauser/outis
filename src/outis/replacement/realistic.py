"""Opt-in realistic surrogates: format-preserving identifiers and plausible places.

Every choice is drawn from the shared keyed digest, through a ``keyed(label,
value, salt)`` callable the caller binds to its secret and identity scope. No
original character survives except punctuation, layout, whitelisted generic
terms, and the state and country of a place when the caller keeps them.

Identifiers use a keyed Feistel permutation over the exact format domain, so
distinct identifiers of one label and format never share a surrogate. Contact
values stay inside reserved fictional ranges; places and streets come from the
bundled word lists in ``realistic_data``.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import string
import unicodedata
from collections.abc import Callable
from math import prod
from urllib.parse import urlsplit

from outis.replacement import realistic_data as data
from outis.replacement.primitives import canonical_form

Keyed = Callable[[str, str, int], bytes]

REALISTIC_LABELS = frozenset(
    {
        "MRN",
        "ACCOUNT",
        "ID",
        "SSN",
        "PHONE",
        "EMAIL",
        "URL",
        "IP",
        "LOCATION",
        "ADDRESS",
        "FACILITY",
    }
)
_ROUNDS = 16
_DIGITS = string.digits
_UPPER = string.ascii_uppercase
_LOWER = string.ascii_lowercase


class _Draws:
    """Nearly unbiased choices from one keyed digest; draws never share bytes."""

    def __init__(self, digest: bytes):
        self._digest = digest
        self._count = 0

    def below(self, bound: int) -> int:
        self._count += 1
        # 256 bits beyond the bound make the modulo bias negligible.
        size = 32 + (bound.bit_length() + 7) // 8
        block = hashlib.shake_256(self._digest + self._count.to_bytes(4, "big")).digest(
            size
        )
        return int.from_bytes(block, "big") % bound

    def choice(self, items):
        return items[self.below(len(items))]

    def digits(self, count: int) -> str:
        return "".join(_DIGITS[self.below(10)] for _ in range(count))


def _styled(word: str, original: str) -> str:
    """Follow the original's all-caps or all-lowercase spelling."""
    letters = [char for char in original if char.isalpha()]
    if len(letters) > 1 and all(char.isupper() for char in letters):
        return word.upper()
    if letters and all(char.islower() for char in letters):
        return word.lower()
    return word


def _slots(value: str):
    """Per character: (alphabet, index) for replaced positions, None if copied.

    ASCII digits and letters carry their index. Other letters and digits are
    replaced by ASCII of the same class, but have no index in that alphabet.
    """
    slots = []
    for char in value:
        if char in _DIGITS:
            slots.append((_DIGITS, _DIGITS.index(char)))
        elif char in _UPPER:
            slots.append((_UPPER, _UPPER.index(char)))
        elif char in _LOWER:
            slots.append((_LOWER, _LOWER.index(char)))
        elif char.isalpha():
            slots.append((_UPPER if char.isupper() else _LOWER, None))
        elif char.isalnum():
            slots.append((_DIGITS, None))
        else:
            slots.append(None)
    return slots


def _random_shape(value: str, draws: _Draws, *, letters: bool = True) -> str:
    """Replace each digit (and letter) with a random one of the same class."""
    return "".join(
        char
        if slot is None or (not letters and slot[0] is not _DIGITS)
        else draws.choice(slot[0])
        for char, slot in zip(value, _slots(value))
    )


def _expand(digest: bytes, bits: int) -> int:
    if bits <= 8 * len(digest):
        return int.from_bytes(digest, "big") >> (8 * len(digest) - bits)
    block = hashlib.shake_256(digest).digest((bits + 7) // 8)
    return int.from_bytes(block, "big") >> (8 * len(block) - bits)


def _identifier(value: str, label: str, keyed: Keyed, salt: int) -> str:
    """Keyed derangement of the value's exact format domain.

    The domain is every string with the same length, character class per
    position (digit, uppercase, lowercase) and punctuation. A balanced Feistel
    network whose round functions are the shared keyed digest permutes
    ``ceil(log2 size)`` bits; cycle walking restricts it to the domain. The
    surrogate is ``P^-1(P(x) + 1 mod size)``: one cycle through the whole
    domain, so it is injective and never returns the original.
    """
    slots = _slots(value)
    used = [slot for slot in slots if slot is not None]
    if not used:
        return value
    if any(index is None for _, index in used):
        # Non-ASCII characters have no index; fall back to keyed shaped draws.
        return _random_shape(value, _Draws(keyed(label, value, salt)))
    size = prod(len(alphabet) for alphabet, _ in used)
    number = 0
    for alphabet, index in used:
        number = number * len(alphabet) + index
    pattern = "".join(
        char if slot is None else {_DIGITS: "9", _UPPER: "A", _LOWER: "a"}[slot[0]]
        for char, slot in zip(value, slots)
    )
    bits = max(2, (size - 1).bit_length())
    bits += bits & 1
    half = bits // 2
    rounds: dict[tuple[int, int], int] = {}

    def round_value(index, block):
        if (index, block) not in rounds:
            message = json.dumps([pattern, index, block], separators=(",", ":"))
            digest = keyed(label + ":FORMAT", message, salt)
            rounds[index, block] = _expand(digest, half)
        return rounds[index, block]

    def forward(block):
        left, right = block >> half, block & ((1 << half) - 1)
        for index in range(_ROUNDS):
            left, right = right, left ^ round_value(index, right)
        return (left << half) | right

    def backward(block):
        left, right = block >> half, block & ((1 << half) - 1)
        for index in reversed(range(_ROUNDS)):
            left, right = right ^ round_value(index, left), left
        return (left << half) | right

    def walk(block, step):
        block = step(block)
        while block >= size:
            block = step(block)
        return block

    number = walk((walk(number, forward) + 1) % size, backward)
    characters = []
    for alphabet, _ in reversed(used):
        number, index = divmod(number, len(alphabet))
        characters.append(alphabet[index])
    replaced = iter(reversed(characters))
    return "".join(
        char if slot is None else next(replaced) for char, slot in zip(value, slots)
    )


def _ssn(value: str, identity: str, keyed: Keyed, salt: int) -> str:
    """Keep the layout; nine-digit values use only never-issued SSN areas."""
    draws = _Draws(keyed("SSN", identity, salt))
    # Any decimal digit counts, so fullwidth or Arabic-Indic SSNs also get a
    # never-issued (ASCII) number rather than unrestricted digits.
    positions = [
        index
        for index, char in enumerate(value)
        if unicodedata.decimal(char, None) is not None
    ]
    if len(positions) != 9:
        return _random_shape(value, draws, letters=False)
    special = len(data.SSN_NEVER_ISSUED_AREAS) - 100
    groups = len(data.SSN_NON_ITIN_GROUPS)
    combination = draws.below(special * 100 + 100 * groups)
    if combination < special * 100:
        area, group = divmod(combination, 100)
    else:
        area, group = divmod(combination - special * 100, groups)
        area, group = special + area, data.SSN_NON_ITIN_GROUPS[group]
    digits = iter(f"{data.SSN_NEVER_ISSUED_AREAS[area]}{group:02d}{draws.digits(4)}")
    characters = list(value)
    for position in positions:
        characters[position] = next(digits)
    return "".join(characters)


_EXTENSION_WORDS = frozenset({"ext", "extn", "x", "extension"})
# Generic words copied verbatim; any other word makes the shape unrecognized.
_PHONE_WORDS = _EXTENSION_WORDS | {"tel", "phone", "ph", "fax", "cell", "mobile"}


def _phone(value: str, identity: str, keyed: Keyed, salt: int) -> str:
    """NANP layout with a 555-0100..0199 fictional line in any area code."""
    draws = _Draws(keyed("PHONE", identity, salt))
    fictional = "555" + "01" + draws.digits(2)
    area = draws.choice(data.AREA_CODES)
    fallback = f"({area}) 555-{fictional[3:]}"
    positions = [index for index, char in enumerate(value) if char in _DIGITS]
    words = re.findall(r"[^\W\d_]+", value)
    if (
        not positions
        or any(word.casefold() not in _PHONE_WORDS for word in words)
        or any(char.isdigit() and char not in _DIGITS for char in value)
    ):
        return fallback
    # The main number ends at the first extension marker after its first
    # digit; it must then be exactly a 7-, 10- or 1+10-digit NANP number.
    boundary = next(
        (
            match.start()
            for match in re.finditer(r"[^\W\d_]+|[#,;]", value)
            if match.start() > positions[0]
            and (match[0] in "#,;" or match[0].casefold() in _EXTENSION_WORDS)
        ),
        len(value),
    )
    digits = "".join(value[index] for index in positions if index < boundary)
    international = "+" in value[: positions[0]]
    if len(digits) == 11 and digits[0] == "1":
        number = "1" + area + fictional
    elif len(digits) == 10 and digits[0] in "23456789" and not international:
        number = area + fictional
    elif len(digits) == 7 and digits[0] in "23456789" and not international:
        number = fictional
    else:
        return fallback
    number += draws.digits(len(positions) - len(digits))
    characters = list(value)
    for position, digit in zip(positions, number):
        characters[position] = digit
    return "".join(characters)


def _surname(draws: _Draws) -> str:
    stem = draws.choice(data.PLACE_STEMS)
    ending = draws.choice(data.PLACE_ENDINGS)
    if ending in stem.casefold():
        ending = "ton"
    if len(stem) > 1 and stem[-1] == stem[-2] == ending[0]:
        stem = stem[:-1]
    return stem + ending


def _feature(draws: _Draws, stem: str) -> str:
    feature = draws.choice(data.PLACE_FEATURES)
    return feature if feature not in stem else data.PLACE_FEATURES[0]


def _town_name(draws: _Draws) -> str:
    """Ashford, Cedar Falls, Ashford Falls or West Ashford, weighted 4:2:7:7.

    The rarer short forms keep names plausible; the long forms widen the space
    to tens of thousands of names so distinct places seldom merge.
    """
    form = draws.below(20)
    if form < 4:
        return _surname(draws)
    if form < 6:
        stem = draws.choice(data.PLACE_STEMS)
        return f"{stem} {_feature(draws, stem)}"
    if form < 13:
        town = _surname(draws)
        return f"{town} {_feature(draws, town)}"
    modifier, town = draws.choice(data.PLACE_MODIFIERS), _surname(draws)
    return f"{modifier if modifier not in town else 'New'} {town}"


def _reserved_host(host: str, draws: _Draws, *, site: bool = False) -> str:
    """A host under example.com, example.org or example.net.

    With ``site``, a keyed name stands in for the original site label, so
    distinct sites stay distinct below the reserved domain.
    """
    labels = [label for label in host.split(".") if label]
    tld = labels[-1].casefold() if labels else ""
    base = "example." + (tld if tld in {"com", "org", "net"} else "com")
    subdomains = [
        label
        if label.casefold() in data.GENERIC_SUBDOMAINS
        # Keyed words, not source labels: subdomains can name an institution.
        else _styled(draws.choice(data.URL_WORDS), label)
        for label in labels[:-2]
    ]
    if site and labels:
        subdomains.append(_styled(_surname(draws).casefold(), labels[-2:][0]))
    return ".".join([*subdomains, _styled(base, ".".join(labels[-2:]))])


def _email(value: str, identity: str, keyed: Keyed, salt: int) -> str:
    draws = _Draws(keyed("EMAIL", identity, salt))
    local, at, domain = value.rpartition("@")
    if not at or not local:
        local, domain = "first.last", "example.com"
    tokens = re.findall(r"[^\W\d_]+|\d+|[\W_]", local)
    letter_runs = sum(1 for token in tokens if token[0].isalpha())
    parts, seen_letters = [], 0
    for token in tokens:
        if token.isdigit():
            parts.append(draws.digits(len(token)))
        elif token[0].isalpha():
            if letter_runs == 1:
                word = draws.choice(data.FIRST_NAMES) + _surname(draws).casefold()
            elif len(token) == 1:
                word = draws.choice(_UPPER)
            else:
                word = (
                    draws.choice(data.FIRST_NAMES)
                    if seen_letters == 0
                    else _surname(draws)
                )
            parts.append(_styled(word, token))
            seen_letters += 1
        else:
            parts.append(token if token in string.punctuation else ".")
    return "".join(parts) + "@" + _reserved_host(domain, draws)


_URL = re.compile(
    r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://|//)?(?P<authority>[^/?#]*)(?P<rest>.*)",
    re.S,
)
_SCHEMES = frozenset(
    {"//", "http://", "https://", "ftp://", "sftp://", "ws://", "wss://"}
)
_RESERVED_HOST = re.compile(r"(?:[a-z0-9-]+\.)*example\.(?:com|org|net)", re.I)


def _url_tail(rest: str, draws: _Draws) -> str:
    """Path, query and fragment with keyed words; punctuation and layout kept."""
    path = re.split(r"[?#]", rest, maxsplit=1)[0]
    extension = re.search(r"\.([A-Za-z0-9]+)$", path)
    keep = (
        extension.start(1)
        if extension and extension[1].casefold() in data.URL_EXTENSIONS
        else None
    )
    parts = []
    for match in re.finditer(r"%[0-9A-Fa-f]{2}|[^\W\d_]+|\d+|.", rest, re.S):
        token = match[0]
        if match.start() == keep:
            parts.append(token)
        elif token.startswith("%") and len(token) == 3:
            # Percent-encoded punctuation is layout; encoded letters are not.
            parts.append(
                token if not chr(int(token[1:], 16)).isalnum() else draws.choice(_LOWER)
            )
        elif token.isdigit():
            parts.append(draws.digits(len(token)))
        elif token[0].isalpha():
            word = (
                draws.choice(data.URL_WORDS) if len(token) > 1 else draws.choice(_LOWER)
            )
            parts.append(_styled(word, token))
        else:
            parts.append(token if token in string.punctuation else "-")
    return "".join(parts)


def _url(value: str, identity: str, keyed: Keyed, salt: int) -> str:
    draws = _Draws(keyed("URL", identity, salt))
    match = _URL.fullmatch(value)
    scheme = match["scheme"] or ""
    if scheme and scheme.casefold() not in _SCHEMES:
        scheme = "https://"
    host = match["authority"].rpartition("@")[2]
    host, colon, port = host.partition(":")
    port = colon + port if port.isdigit() else ""
    authority = _reserved_host(host, draws, site=True) + port if host or scheme else ""
    rendered = scheme + authority + _url_tail(match["rest"], draws)
    # Browsers read "\" as "/"; any host a parser can find must be reserved.
    try:
        parsed = urlsplit(rendered.replace("\\", "/")).hostname
    except ValueError:
        parsed = ""
    if parsed is None or _RESERVED_HOST.fullmatch(parsed):
        return rendered
    return "https://" + _reserved_host("", draws, site=True) + "/"


_IPV4 = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}(?P<tail>/\d{1,2}|:\d{1,5})?")
_IPV6 = re.compile(
    r"(?P<open>\[)?(?P<address>[0-9A-Fa-f:.]+)(?:%[\w.-]+)?(?P<close>\])?(?P<tail>/\d{1,3}|:\d{1,5})?"
)
_DOCUMENTATION_V4 = ("192.0.2", "198.51.100", "203.0.113")


def _ip(value: str, identity: str, keyed: Keyed, salt: int) -> str:
    """Documentation ranges only: three IPv4 /24s and 2001:db8::/32."""
    draws = _Draws(keyed("IP", identity, salt))
    v4 = f"{draws.choice(_DOCUMENTATION_V4)}.{1 + draws.below(254)}"
    if match := _IPV4.fullmatch(value):
        return v4 + (match["tail"] or "")
    match = _IPV6.fullmatch(value)
    try:
        if match is None or ":" not in match["address"]:
            raise ValueError
        ipaddress.IPv6Address(match["address"])
    except ValueError:
        return v4
    address = ipaddress.IPv6Address((0x20010DB8 << 96) | draws.below(1 << 96))
    groups = match["address"].split(":")
    text = (
        address.exploded
        if len(groups) == 8 and all(len(group) == 4 for group in groups)
        else address.compressed
    )
    if any(char.isupper() for char in match["address"]):
        text = text.upper()
    return (match["open"] or "") + text + (match["close"] or "") + (match["tail"] or "")


_STATE_NAMES = {name.casefold(): code for code, (name, _) in data.US_STATES.items()}
_COUNTRY_NAMES = frozenset(country.casefold() for country in data.COUNTRIES)


def _trailing(text: str, end: int, accept) -> tuple[int, str] | None:
    """The longest run of up to three trailing words accepted by ``accept``."""
    words = list(re.finditer(r"[^\s,]+", text[:end]))
    for count in (3, 2, 1):
        if len(words) >= count:
            start = words[-count].start()
            found = accept(text[start:end])
            if found is not None:
                return start, found
    return None


def _state_code(candidate: str) -> str | None:
    if candidate in data.US_STATES:
        return candidate
    return _STATE_NAMES.get(" ".join(candidate.casefold().split()))


def _country(candidate: str) -> str | None:
    if candidate in data.COUNTRY_ABBREVIATIONS:
        return candidate
    folded = " ".join(candidate.casefold().split())
    return candidate if folded in _COUNTRY_NAMES else None


_NOT_BEFORE_ZIP = data.UNIT_DESIGNATORS | {"box"}


def _peel_region(text: str):
    """Split trailing ``[state] [ZIP] [country]`` off a place.

    Returns the end of the remaining locality and the peeled items as
    ``(kind, start, end, detail)``, where detail is a state's postal code.
    """
    items = []
    end = len(text)
    while True:
        end = len(text[:end].rstrip(" \t\r\n,"))
        kinds = {item[0] for item in items}
        zip_code = re.search(r"(?<![\w-])\d{5}(?:-\d{4})?$", text[:end])
        if zip_code:
            before = text[: zip_code.start()].rstrip().rsplit(None, 1)[-1:]
            # "PO Box 12345" and "Suite 10000" end in numbers that are not ZIPs.
            if before and (before[0] == "#" or _bare(before[0]) in _NOT_BEFORE_ZIP):
                zip_code = None
        if not items and (found := _trailing(text, end, _country)):
            items.append(("country", found[0], end, None))
        elif zip_code and "zip" not in kinds and "state" not in kinds:
            items.append(("zip", zip_code.start(), end, None))
        elif "state" not in kinds and (found := _trailing(text, end, _state_code)):
            items.append(("state", found[0], end, found[1]))
        else:
            return end, items
        end = items[-1][1]


def _town(original: str, keyed: Keyed, salt: int) -> str:
    """A keyed town name, consistent for one place across LOCATION and ADDRESS."""
    words = original.split()
    designator = ""
    if len(words) > 1 and words[-1].casefold() in data.LOCALITY_DESIGNATORS:
        designator = " " + words[-1]
        original = original[: original.rindex(words[-1])].rstrip()
    key = canonical_form(original)
    originals = set(key.split())
    for attempt in range(salt, salt + 100):
        name = _town_name(_Draws(keyed("TOWN", key, attempt)))
        if not originals & set(canonical_form(name).split()):
            break
    return _styled(name, original) + designator


def _replace_runs(text: str, replace) -> str:
    """Apply ``replace`` to each comma-free, trimmed run; separators kept."""
    return re.sub(r"[^\s,](?:[^,\r\n]*[^\s,])?", lambda m: replace(m[0]), text)


def _region(text, items, *, replace_region, keyed, salt):
    """Rendered (start, end, text) edits for peeled state, ZIP and country."""
    edits = []
    emitted_state = None
    for kind, start, end, code in items:
        if kind != "state":
            continue
        emitted_state = code
        original = text[start:end]
        if replace_region:
            draws = _Draws(keyed("STATE", code, salt))
            emitted_state = draws.choice([c for c in data.US_STATES if c != code])
            name = data.US_STATES[emitted_state][0]
            # A postal code stays a postal code; a state name stays a name.
            rendered = (
                emitted_state if original in data.US_STATES else _styled(name, original)
            )
            edits.append((start, end, rendered))
    for kind, start, end, _ in items:
        original = text[start:end]
        if kind == "zip":
            draws = _Draws(keyed("ZIP", json.dumps([emitted_state, original]), salt))
            ranges = (
                data.US_STATES[emitted_state][1]
                if emitted_state
                else data.UNALLOCATED_ZIP_PREFIXES
            )
            prefixes = [p for low, high in ranges for p in range(low, high + 1)]
            rendered = f"{draws.choice(prefixes):03d}{draws.digits(2)}"
            if "-" in original:
                rendered += "-" + draws.digits(4)
            edits.append((start, end, rendered))
        elif kind == "country" and replace_region:
            draws = _Draws(keyed("COUNTRY", canonical_form(original), salt))
            if original in data.COUNTRY_ABBREVIATIONS:
                options = [c for c in ("USA", "UK", "UAE") if c != original]
                edits.append((start, end, draws.choice(options)))
            else:
                folded = canonical_form(original)
                options = [c for c in data.COUNTRIES if c.casefold() != folded]
                edits.append((start, end, _styled(draws.choice(options), original)))
    return edits


def _apply(text: str, edits) -> str:
    parts, cursor = [], 0
    for start, end, rendered in sorted(edits):
        parts.extend((text[cursor:start], rendered))
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _location(value: str, keyed: Keyed, salt: int, keep_region: bool) -> str:
    end, items = _peel_region(value)
    locality = value[:end]
    # A span that is only a state or country has nothing else to replace, and
    # the original is never echoed, so its region is replaced instead.
    replace_region = not keep_region or not locality.strip(" \t\r\n,")
    edits = _region(value, items, replace_region=replace_region, keyed=keyed, salt=salt)
    edits.append((0, end, _replace_runs(locality, lambda run: _town(run, keyed, salt))))
    return _apply(value, edits)


_PO_BOX = re.compile(
    r"(?P<prefix>(?:p\.?\s*o\.?|post\s+office)\s*box|box)\s+(?P<number>\S+)", re.I
)


def _bare(token: str) -> str:
    return token.strip(".,").casefold()


def _is_unit(token: str) -> bool:
    return token.startswith("#") or _bare(token) in data.UNIT_DESIGNATORS


def _unit_edits(tokens, start, draws):
    """Keep unit designators; shape-replace the identifiers that follow them."""
    edits = []
    for index in range(start, len(tokens)):
        token = tokens[index]
        if token[0].startswith("#") and len(token[0]) > 1:
            edits.append(
                (token.start() + 1, token.end(), _random_shape(token[0][1:], draws))
            )
        elif not _is_unit(token[0]):
            edits.append((token.start(), token.end(), _random_shape(token[0], draws)))
    return edits


_SUFFIXES = {
    spelling.casefold(): (full, abbreviation)
    for full, abbreviation in data.STREET_SUFFIXES
    for spelling in (full, abbreviation)
}


def _street(name: str, keyed: Keyed, salt: int) -> tuple[str, tuple[str, str]]:
    """A keyed street name and suffix, consistent for one original street."""
    key = canonical_form(name)
    originals = set(key.split())
    for attempt in range(salt, salt + 100):
        draws = _Draws(keyed("STREET", key, attempt))
        street = draws.choice(data.STREET_NAMES) if draws.below(2) else _surname(draws)
        suffix = draws.choice(data.STREET_SUFFIXES)
        if not originals & {street.casefold()}:
            break
    return _styled(street, name), suffix


def _street_line(core: str, draws: _Draws, keyed: Keyed, salt: int) -> str:
    tokens = list(re.finditer(r"\S+", core))
    edits = [(tokens[0].start(), tokens[0].end(), _house_number(tokens[0][0], draws))]
    index = 1
    if index < len(tokens) - 1 and _bare(tokens[index][0]) in data.DIRECTIONALS:
        index += 1
    name_start = index
    unit = next(
        (i for i in range(name_start, len(tokens)) if _is_unit(tokens[i][0])),
        len(tokens),
    )
    suffix = next(
        (
            i
            for i in range(name_start + 1, unit)
            if _bare(tokens[i][0]) in _SUFFIXES
            or _bare(tokens[i][0]) in data.STREET_SUFFIX_ALIASES
        ),
        None,
    )
    name_end = suffix if suffix is not None else unit
    if name_end > name_start:
        first, last = tokens[name_start], tokens[name_end - 1]
        street, (full, abbreviation) = _street(
            core[first.start() : last.end()], keyed, salt
        )
        if suffix is None:
            edits.append((first.start(), last.end(), f"{street} {full}"))
        else:
            original = tokens[suffix][0]
            bare = original.rstrip(".,")
            spelled = (
                abbreviation
                if _SUFFIXES.get(bare.casefold(), (bare,))[0].casefold()
                != bare.casefold()
                else full
            )
            edits.append((first.start(), last.end(), street))
            edits.append(
                (
                    tokens[suffix].start(),
                    tokens[suffix].start() + len(bare),
                    _styled(spelled, bare),
                )
            )
            after = suffix + 1
            if after < unit and _bare(tokens[after][0]) in data.DIRECTIONALS:
                after += 1
            if after < unit:
                # Words after the street suffix name the locality.
                start, end = tokens[after].start(), tokens[unit - 1].end()
                edits.append((start, end, _town(core[start:end], keyed, salt)))
    edits.extend(_unit_edits(tokens, unit, draws))
    return _apply(core, edits)


def _house_number(token: str, draws: _Draws) -> str:
    rendered = _random_shape(token, draws)
    if rendered[:1] == "0":
        rendered = draws.choice("123456789") + rendered[1:]
    return rendered


def _address_chunk(core: str, draws: _Draws, keyed: Keyed, salt: int) -> str:
    if match := _PO_BOX.fullmatch(core):
        return core[: match.start("number")] + _random_shape(match["number"], draws)
    if core[0].isdigit():
        return _street_line(core, draws, keyed, salt)
    tokens = list(re.finditer(r"\S+", core))
    if _is_unit(tokens[0][0]):
        return _apply(core, _unit_edits(tokens, 0, draws))
    return _town(core, keyed, salt)


def _address(
    value: str, identity: str, keyed: Keyed, salt: int, keep_region: bool
) -> str:
    """Street, unit, box, locality and region parts in the original layout."""
    draws = _Draws(keyed("ADDRESS", identity, salt))
    end, items = _peel_region(value)
    head = value[:end]
    replace_region = not keep_region or not head.strip(" \t\r\n,")
    edits = _region(value, items, replace_region=replace_region, keyed=keyed, salt=salt)
    edits.append(
        (
            0,
            end,
            _replace_runs(head, lambda run: _address_chunk(run, draws, keyed, salt)),
        )
    )
    return _apply(value, edits)


def _facility(value: str, setting: str, identity: str, keyed: Keyed, salt: int) -> str:
    """A fictional institution name that keeps the whitelisted setting terms."""
    draws = _Draws(keyed("FACILITY", identity, salt))
    stem = _town_name(draws)
    modifier = draws.choice(data.FACILITY_MODIFIERS)
    if modifier and modifier in stem.split():
        modifier = ""
    return _styled(" ".join(part for part in (stem, modifier, setting) if part), value)


def realistic_value(
    value: str,
    label: str,
    identity: str,
    keyed: Keyed,
    salt: int,
    *,
    keep_region: bool,
    facility_setting: str = "Medical Center",
) -> str:
    """One candidate surrogate; the caller retries with the next salt if it
    canonically equals the original."""
    if label in {"MRN", "ACCOUNT", "ID"}:
        return _identifier(value, label, keyed, salt)
    if label == "SSN":
        return _ssn(value, identity, keyed, salt)
    if label == "PHONE":
        return _phone(value, identity, keyed, salt)
    if label == "EMAIL":
        return _email(value, identity, keyed, salt)
    if label == "URL":
        return _url(value, identity, keyed, salt)
    if label == "IP":
        return _ip(value, identity, keyed, salt)
    if label == "LOCATION":
        return _location(value, keyed, salt, keep_region)
    if label == "ADDRESS":
        return _address(value, identity, keyed, salt, keep_region)
    if label == "FACILITY":
        return _facility(value, facility_setting, identity, keyed, salt)
    raise ValueError("No realistic surrogate for this label")
