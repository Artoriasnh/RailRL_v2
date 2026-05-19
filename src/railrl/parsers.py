"""Parsers for Derby route_id and UK 4-character headcode.

Both come from Open Rail Data Wiki Signalling Nomenclature.
The route format observed in Derby TD data:
    R<prefix><signal>[<letter>](-<sub>)?(<class>)
where:
    prefix : DW | TD | DC | EC | DY    Derby workstation lines of route
    signal : digits, optionally with trailing letter (e.g. 5045, 564)
    letter : 1+ uppercase letters (A, B, ..., AB, ...)
    sub    : optional hyphenated integer for sub-position routes
             (RTD5045B-1(M), RTD5045B-2(M)) — observed at TD-5045 / TD-5054
    class  : (M) Main, (C) Call-on, (S) Shunt, (W) Warner, (PS) Proceed-on-Sight

The 4-character UK train reporting number ("headcode"):
    <class_digit><dest_letter><serial_2digit>
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional

from . import config as C


# Compiled regex for parse_route_id (str -> RouteId | None)
_ROUTE_RE = re.compile(
    r"^R(?P<prefix>DW|TD|DC|EC|DY)"
    r"(?P<signal>\d+[A-Za-z]*)"
    r"(?P<letter>[A-Z]+)"
    r"(?:-(?P<sub>\d+))?"
    r"\((?P<cls>M|C|S|W|PS|SP)\)$"
)

# Same pattern as a string for pandas Series.str.extract — returns 5 columns:
#   prefix, signal, letter, sub, cls
ROUTE_RE_PATTERN = (
    r"^R(?P<prefix>DW|TD|DC|EC|DY)"
    r"(?P<signal>\d+[A-Za-z]*)"
    r"(?P<letter>[A-Z]+)"
    r"(?:-(?P<sub>\d+))?"
    r"\((?P<cls>M|C|S|W|PS|SP)\)$"
)

_HEADCODE_RE = re.compile(r"^(?P<cls>[0-9])(?P<dest>[A-Z])(?P<serial>\d{2})$")
HEADCODE_RE_PATTERN = r"^(?P<hc_cls>[0-9])(?P<hc_dest>[A-Z])(?P<hc_serial>\d{2})$"


@dataclass(frozen=True)
class RouteId:
    """Parsed components of a Derby route_id string."""
    raw: str
    prefix: str
    signal_no: str
    letter: str
    sub: Optional[str]   # None when there's no hyphenated sub-position
    cls: str             # 'cls' avoids the reserved keyword `class`

    @property
    def prefix_description(self) -> str:
        return C.PREFIX_DESCRIPTION.get(self.prefix, "")

    @property
    def class_description(self) -> str:
        return C.ROUTE_CLASS.get(self.cls, "")


def parse_route_id(route_id: str) -> Optional[RouteId]:
    """Return a RouteId or None if the string does not match Derby format."""
    if not isinstance(route_id, str):
        return None
    m = _ROUTE_RE.match(route_id.strip())
    if not m:
        return None
    return RouteId(
        raw=route_id,
        prefix=m.group("prefix"),
        signal_no=m.group("signal"),
        letter=m.group("letter"),
        sub=m.group("sub"),
        cls=m.group("cls"),
    )


@dataclass(frozen=True)
class Headcode:
    """Parsed components of a UK 4-character train reporting number."""
    raw: str
    cls_digit: str
    dest_letter: str
    serial: str

    @property
    def class_meaning(self) -> str:
        return C.HEADCODE_CLASS.get(self.cls_digit, "(non-standard)")


def parse_headcode(headcode: str) -> Optional[Headcode]:
    """Return a Headcode or None if not a valid 4-char UK reporting number."""
    if not isinstance(headcode, str):
        return None
    m = _HEADCODE_RE.match(headcode.strip())
    if not m:
        return None
    return Headcode(
        raw=headcode,
        cls_digit=m.group("cls"),
        dest_letter=m.group("dest"),
        serial=m.group("serial"),
    )
