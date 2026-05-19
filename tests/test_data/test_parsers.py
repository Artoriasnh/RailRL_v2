"""Unit tests for railrl.parsers."""
from __future__ import annotations
import pytest

from railrl.parsers import (
    parse_route_id, parse_headcode,
    ROUTE_RE_PATTERN, HEADCODE_RE_PATTERN,
)


# ---------- parse_route_id ----------

@pytest.mark.parametrize("raw,prefix,signal,letter,sub,cls", [
    ("RTD5045A(M)",   "TD", "5045", "A", None, "M"),
    ("RTD5054D(S)",   "TD", "5054", "D", None, "S"),
    ("RDY564B(M)",    "DY", "564",  "B", None, "M"),
    ("RDC5104B(M)",   "DC", "5104", "B", None, "M"),
    ("RDW5319C(M)",   "DW", "5319", "C", None, "M"),
    ("REC5474A(M)",   "EC", "5474", "A", None, "M"),
    ("RDW5316E(M)",   "DW", "5316", "E", None, "M"),
    # hyphenated sub-position routes (observed at TD-5045 / TD-5054)
    ("RTD5045B-1(M)", "TD", "5045", "B", "1",  "M"),
    ("RTD5045B-2(M)", "TD", "5045", "B", "2",  "M"),
    ("RTD5054C-1(M)", "TD", "5054", "C", "1",  "M"),
    # all classes
    ("RDC5099A(C)",   "DC", "5099", "A", None, "C"),
])
def test_parse_route_id_valid(raw, prefix, signal, letter, sub, cls):
    r = parse_route_id(raw)
    assert r is not None
    assert (r.prefix, r.signal_no, r.letter, r.sub, r.cls) == (prefix, signal, letter, sub, cls)
    assert r.raw == raw


@pytest.mark.parametrize("bad", ["", "invalid", "R5045A(M)", "RTD5045A", "RTD5045A(X)", None, 12345])
def test_parse_route_id_invalid(bad):
    assert parse_route_id(bad) is None


# ---------- parse_headcode ----------

@pytest.mark.parametrize("raw,cls,dest,serial", [
    ("1S49", "1", "S", "49"),
    ("2A26", "2", "A", "26"),
    ("6Z42", "6", "Z", "42"),
    ("5T15", "5", "T", "15"),
    ("0B91", "0", "B", "91"),
])
def test_parse_headcode_valid(raw, cls, dest, serial):
    h = parse_headcode(raw)
    assert h is not None
    assert (h.cls_digit, h.dest_letter, h.serial) == (cls, dest, serial)


@pytest.mark.parametrize("bad", ["343R", "1234", "S49", "1S4", None, 12])
def test_parse_headcode_invalid(bad):
    assert parse_headcode(bad) is None


# ---------- vectorised regex compatibility ----------

def test_vectorised_route_pattern_extract():
    import pandas as pd
    s = pd.Series(["RTD5045A(M)", "RTD5045B-2(M)", "junk", None])
    out = s.astype(str).str.extract(ROUTE_RE_PATTERN)
    assert out.loc[0, "prefix"] == "TD" and out.loc[0, "letter"] == "A" and pd.isna(out.loc[0, "sub"])
    assert out.loc[1, "sub"] == "2"
    assert pd.isna(out.loc[2, "prefix"])
    assert pd.isna(out.loc[3, "prefix"])


def test_vectorised_headcode_pattern_extract():
    import pandas as pd
    s = pd.Series(["1S49", "343R", "6Z42", None])
    out = s.astype(str).str.extract(HEADCODE_RE_PATTERN)
    assert out.loc[0, "hc_cls"] == "1" and out.loc[0, "hc_dest"] == "S"
    assert pd.isna(out.loc[1, "hc_cls"])
    assert out.loc[2, "hc_cls"] == "6"
