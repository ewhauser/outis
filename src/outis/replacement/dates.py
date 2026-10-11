"""Patient-wide date shifting that redacts dates it cannot shift exactly."""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta

_MONTHS = tuple(calendar.month_name)
_MONTH_ABBREVIATIONS = tuple(calendar.month_abbr)
_MONTH_LOOKUP = {
    variant.casefold(): number
    for number in range(1, 13)
    for variant in (_MONTHS[number], _MONTH_ABBREVIATIONS[number])
}
_MONTH_LOOKUP["sept"] = 9
_MONTH_PATTERN = "(?:" + "|".join(sorted(_MONTH_LOOKUP, key=len, reverse=True)) + ")"


def _numeric_date(value: str, offset: int) -> tuple[str | None, str | None]:
    iso = re.fullmatch(
        r"(?P<y>\d{4})(?P<s>[-/.])(?P<m>\d{1,2})(?P=s)(?P<d>\d{1,2})", value
    )
    american = re.fullmatch(
        r"(?P<m>\d{1,2})(?P<s>[-/.])(?P<d>\d{1,2})(?P=s)(?P<y>\d{4}|\d{2})", value
    )
    match = iso or american
    if not match:
        return None, None
    # With year last, an initial value above 12 establishes day/month order.
    # Values that could be either convention retain the documented US policy.
    day_first = american is not None and int(match["m"]) > 12
    month = int(match["d"] if day_first else match["m"])
    day = int(match["m"] if day_first else match["d"])
    two_digit_year = len(match["y"]) == 2
    if two_digit_year:
        # The century is unknown, but shifting only needs the calendar. Leap
        # years agree in the 1900s and 2000s except for year 00, so accept the
        # shift only when both centuries give the same two-digit result.
        results = set()
        for century in (1900, 2000):
            try:
                shifted = date(century + int(match["y"]), month, day) + timedelta(
                    days=offset
                )
            except (ValueError, OverflowError):
                results.add(None)
                continue
            results.add((shifted.year % 100, shifted.month, shifted.day))
        if len(results) != 1 or None in results:
            return (
                None,
                "Two-digit year date is ambiguous across centuries and was redacted",
            )
        year, shifted_month, shifted_day = results.pop()
        shifted = date(2000 + year, shifted_month, shifted_day)
    else:
        try:
            shifted = date(int(match["y"]), month, day) + timedelta(days=offset)
        except (ValueError, OverflowError):
            return None, "Invalid or out-of-range date was redacted"
    month_width = len(match["d"] if day_first else match["m"])
    source_day = match["m"] if day_first else match["d"]
    day_width = (
        2
        if len(source_day) == 2 and (source_day.startswith("0") or month_width == 2)
        else 1
    )
    values = {
        "y": str(shifted.year % 100).zfill(2)
        if two_digit_year
        else str(shifted.year).zfill(4),
        "m": str(shifted.month).zfill(month_width),
        "d": str(shifted.day).zfill(day_width),
    }
    order = (
        ("y", "m", "d") if iso else ("d", "m", "y") if day_first else ("m", "d", "y")
    )
    warning = None
    if (
        day_first
        and shifted.day <= 12
        and shifted.day != shifted.month
        and two_digit_year
    ):
        # Year-first output is not readable with a two-digit year.
        return (
            None,
            "Shifted day/month date with a two-digit year would be ambiguous and was redacted",
        )
    if day_first and shifted.day <= 12 and shifted.day != shifted.month:
        # A D/M input can be unambiguous while the shifted D/M output is not.
        # The documented M/D default would then change its calendar meaning.
        # Retain the separator/padding but make the date order explicit.
        order = ("y", "m", "d")
        warning = "Shifted day/month/year date was emitted as year/month/day to avoid ambiguity"
    elif american and int(match["m"]) <= 12 and int(match["d"]) <= 12:
        warning = "Ambiguous numeric date was interpreted as month/day/year"
    rendered = match["s"].join(values[part] for part in order)
    return rendered, warning


def _datetime_date(value: str, offset: int) -> tuple[str | None, str | None]:
    match = re.fullmatch(
        r"(?P<date>\d{4}-\d{2}-\d{2})"
        r"(?P<tail>[Tt ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
        r"(?P<zone>[Zz]|[+-]\d{2}:?\d{2})?)",
        value,
    )
    if not match:
        return None, None
    try:
        datetime.fromisoformat(
            value.removesuffix("z") + ("Z" if value.endswith("z") else "")
        )
        zone = match["zone"]
        if zone and zone[0] in "+-":
            digits = zone[1:].replace(":", "")
            if int(digits[:2]) > 23 or int(digits[2:]) > 59:
                raise ValueError("Invalid time zone")
    except ValueError:
        return None, "Invalid or out-of-range date-time was redacted"
    rendered, warning = _numeric_date(match["date"], offset)
    if rendered is None:
        return None, warning or "Invalid or out-of-range date-time was redacted"
    return rendered + match["tail"], warning


def _case_like(value: str, example: str) -> str:
    if example.isupper():
        return value.upper()
    if example.islower():
        return value.lower()
    return value


def _month_text(month: int, source: str, dot: str, hyphenated: bool = False) -> str:
    abbreviated = len(source) <= 4 and (source.casefold() != "may" or hyphenated)
    text = _MONTH_ABBREVIATIONS[month] if abbreviated else _MONTHS[month]
    if source.casefold() == "sept" and month == 9:
        text = "Sept"
    return _case_like(text, source) + dot


def _day_text(
    day: int, source: str, ordinal: str | None, hyphenated: bool = False
) -> str:
    text = str(day).zfill(
        2 if source.startswith("0") or (hyphenated and len(source) == 2) else 1
    )
    if ordinal:
        suffix = (
            "th"
            if 10 <= day % 100 <= 20
            else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        )
        text += _case_like(suffix, ordinal)
    return text


def _word_date(value: str, offset: int) -> tuple[str | None, str | None]:
    month_first = re.fullmatch(
        rf"(?P<m>{_MONTH_PATTERN})(?P<dot>\.?)"
        r"(?P<space>\s+|-)(?P<d>\d{1,2})(?P<ordinal>st|nd|rd|th)?"
        r"(?P<join>,\s*|\s+|-)(?P<y>\d{4})",
        value,
        re.I,
    )
    day_first = re.fullmatch(
        rf"(?P<d>\d{{1,2}})(?P<ordinal>st|nd|rd|th)?(?P<space>\s+|-)"
        rf"(?P<m>{_MONTH_PATTERN})(?P<dot>\.?)(?P<join>,\s*|\s+|-)(?P<y>\d{{4}})",
        value,
        re.I,
    )
    match = month_first or day_first
    if not match:
        return None, None
    month = _MONTH_LOOKUP[match["m"].casefold()]
    try:
        shifted = date(int(match["y"]), month, int(match["d"])) + timedelta(days=offset)
    except (ValueError, OverflowError):
        return None, "Invalid or out-of-range date was redacted"
    hyphenated = match["space"] == "-" or match["join"] == "-"
    month_text = _month_text(shifted.month, match["m"], match["dot"], hyphenated)
    # Month-name styles ordinarily use an unpadded day.  Preserve an explicit
    # zero-padded style when the source actually gives evidence of that style.
    day_text = _day_text(shifted.day, match["d"], match["ordinal"], hyphenated)
    year_text = str(shifted.year).zfill(4)
    if month_first:
        return month_text + match["space"] + day_text + match["join"] + year_text, None
    return day_text + match["space"] + month_text + match["join"] + year_text, None


_RANGE_CONNECTOR = r"(?:\s*[-–—]\s*|\s+(?:to|through|thru|until)\s+)"
_COMPRESSED_MONTH_FIRST = re.compile(
    rf"(?P<m1>{_MONTH_PATTERN})(?P<dot1>\.?)(?P<sp1>\s+)(?P<d1>\d{{1,2}})(?P<o1>st|nd|rd|th)?"
    rf"(?P<conn>{_RANGE_CONNECTOR})"
    rf"(?:(?P<m2>{_MONTH_PATTERN})(?P<dot2>\.?)(?P<sp2>\s+))?(?P<d2>\d{{1,2}})(?P<o2>st|nd|rd|th)?"
    r"(?P<join>,\s*|\s+)(?P<y>\d{4})",
    re.I,
)
_COMPRESSED_DAY_FIRST = re.compile(
    r"(?P<d1>\d{1,2})(?P<o1>st|nd|rd|th)?"
    rf"(?:(?P<sp1>\s+)(?P<m1>{_MONTH_PATTERN})(?P<dot1>\.?))?"
    rf"(?P<conn>{_RANGE_CONNECTOR})"
    rf"(?P<d2>\d{{1,2}})(?P<o2>st|nd|rd|th)?(?P<sp2>\s+)(?P<m2>{_MONTH_PATTERN})(?P<dot2>\.?)"
    r"(?P<join>,\s*|\s+)(?P<y>\d{4})",
    re.I,
)


def _single_date(value: str, offset: int) -> tuple[str | None, str | None]:
    for parser in (_datetime_date, _numeric_date, _word_date):
        rendered, warning = parser(value, offset)
        if rendered is not None or warning is not None:
            return rendered, warning
    return None, None


def _compressed_range(value: str, offset: int) -> tuple[str | None, str | None]:
    """Shift a range whose endpoints share one written year, e.g. March 2-9, 2024."""
    for pattern, month_first in (
        (_COMPRESSED_MONTH_FIRST, True),
        (_COMPRESSED_DAY_FIRST, False),
    ):
        match = pattern.fullmatch(value)
        if not match:
            continue
        first_month = match["m1"] or match["m2"]
        second_month = match["m2"] or match["m1"]
        first_dot = match["dot1"] if match["m1"] else match["dot2"]
        second_dot = match["dot2"] if match["m2"] else match["dot1"]
        space = match["sp1"] or match["sp2"] or " "
        year = int(match["y"])
        try:
            start = date(year, _MONTH_LOOKUP[first_month.casefold()], int(match["d1"]))
            end = date(year, _MONTH_LOOKUP[second_month.casefold()], int(match["d2"]))
        except ValueError:
            return None, "Invalid or out-of-range date was redacted"
        if end < start:
            # A shared written year cannot express a range that crosses a
            # year boundary; the intended years are not knowable.
            return None, None
        try:
            start, end = start + timedelta(days=offset), end + timedelta(days=offset)
        except OverflowError:
            return None, "Invalid or out-of-range date was redacted"
        start_month = _month_text(start.month, first_month, first_dot)
        end_month = _month_text(end.month, second_month, second_dot)
        start_day = _day_text(start.day, match["d1"], match["o1"])
        end_day = _day_text(end.day, match["d2"], match["o2"])
        conn, join = match["conn"], match["join"]
        if start.year != end.year:
            if month_first:
                return (
                    f"{start_month}{space}{start_day}{join}{start.year:04d}{conn}"
                    f"{end_month}{space}{end_day}{join}{end.year:04d}"
                ), None
            return (
                f"{start_day}{space}{start_month}{join}{start.year:04d}{conn}"
                f"{end_day}{space}{end_month}{join}{end.year:04d}"
            ), None
        same_month = start.month == end.month
        if month_first:
            tail = (
                end_day
                if same_month and not match["m2"]
                else f"{end_month}{space}{end_day}"
            )
            return (
                f"{start_month}{space}{start_day}{conn}{tail}{join}{end.year:04d}",
                None,
            )
        head = (
            start_day
            if same_month and not match["m1"]
            else f"{start_day}{space}{start_month}"
        )
        return f"{head}{conn}{end_day}{space}{end_month}{join}{end.year:04d}", None
    return None, None


def _date_range(value: str, offset: int) -> tuple[str | None, str | None]:
    """Shift both endpoints of a written date range by the same offset.

    A range is annotated as one DATE span. Shifting each endpoint separately
    keeps the interval and the connector, so chronology survives. A range whose
    endpoints cannot both be parsed is left to the caller to redact.
    """
    rendered, warning = _compressed_range(value, offset)
    if rendered is not None or warning is not None:
        return rendered, warning
    for connector in re.finditer(_RANGE_CONNECTOR, value, re.I):
        left, right = value[: connector.start()], value[connector.end() :]
        if not left or not right:
            continue
        start, start_warning = _single_date(left, offset)
        end, end_warning = _single_date(right, offset)
        if start is not None and end is not None:
            warnings = list(dict.fromkeys(w for w in (start_warning, end_warning) if w))
            return start + connector.group() + end, "; ".join(warnings) or None
        if (start is None and start_warning) or (end is None and end_warning):
            return None, start_warning or end_warning
    return None, None


def shift_date(value: str, offset: int) -> tuple[str, str | None, bool]:
    # Outer spacing is part of the selected range and can be safely preserved.
    leading = value[: len(value) - len(value.lstrip())]
    trailing = value[len(value.rstrip()) :]
    stripped = value.strip()
    for parser in (_datetime_date, _numeric_date, _word_date, _date_range):
        rendered, warning = parser(stripped, offset)
        if rendered is not None:
            return leading + rendered + trailing, warning, True
        if warning is not None:
            return "[DATE]", warning, False
    return (
        "[DATE]",
        "Incomplete or unsupported date was redacted; chronology cannot be preserved",
        False,
    )
