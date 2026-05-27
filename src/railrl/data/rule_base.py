"""P2.5 rule base — load the Hao-approved Training Plan rules + context matcher (spec 05 §13.5).

The 19 rules were AI-drafted from `Training_Plan_2022.docx`, then **reviewed and approved
per-rule by Hao** (2026-05-27; see `outputs/rule_base/rules_full_draft.md`). This module is
the single machine-readable source: `RULES` (the approved rows), `load_rule_base()`, and
`rule_matches()`. `scripts/rules/03_finalize.py` writes these to `rules.parquet`.

WHAT IS / ISN'T CHECKABLE (verified against snapshots_v2 schema, 2026-05-27)
--------------------------------------------------------------------------
The snapshot IDENTITY columns (never fed to the model, but usable for offline analysis) give
us per decision:  `focal_signal` (the decision signal = origin/approach), `focal_train`
(train_id -> headcode -> class), `chosen_route_id`, `candidate_route_ids`. The static route
catalog (`route_to_tc_all.csv` + `platform_tc_map.csv`) gives each route_id its END PLATFORM
and END SIGNAL. So:

  * HARD rules (confidence=high) key on the DECISION SIGNAL (origin) + the audited route's end
    platform/route_id -- FULLY checkable. These gate (section 12).
  * SOFT traffic-flow rules (confidence=med) key on the train's *destination direction*
    (Sheffield / West / North ...). The state deliberately HIDES destination (leak audit), and
    the repo has no headcode-letter -> direction map. So a soft rule only matches when
    `resolve_direction(sample)` returns a direction; the default resolver returns None ->
    soft rules report 'undetermined' (NOT fabricated). Supplying a headcode->direction map
    later activates them. Soft rules NEVER gate regardless.

DECISION-SIGNAL FORMAT (critical, fixed 2026-05-27): `focal_signal` is stored as the BARE
number ('5045') because the route index keys on route_to_tc_all.csv `start` (also bare). Rule
anchors are PREFIXED ('TD5045'). All candidate routes at a decision share the SAME start
signal, and the route_id encodes it WITH prefix (RTD5045... -> 'TD5045'), so `decision_signal`
recovers the prefixed form from the candidates (format-independent).

Pure-python + stdlib csv only (no torch / no pyarrow) -> importable in the sandbox and on
Windows alike. The route catalog is read lazily from `data/` on first use.
"""
from __future__ import annotations

import csv
import re
from typing import Optional

from .. import config as C

# ------------------------------------------------------------------
# Direction anchors -- Hao-verified + AI-cross-checked vs route data (2026-05-27).
# Used ONLY by the (currently inert) soft-rule direction resolver and for documentation.
# ------------------------------------------------------------------
DIRECTION_ANCHORS = {
    "North":      {"end_signals": ["DC5061", "DC5062", "DC5063", "DC5064", "DC5065", "DC5066"],
                   "tcs": ["T884", "T883"], "note": "Duffield/Chesterfield corridor"},
    "South":      {"end_signals": ["DW5319"], "tcs": ["TYVR"], "note": "Pear Tree (Birmingham/Crewe/Stenson)"},
    "West":       {"end_signals": ["DW5302"], "tcs": ["TYTW", "TYTV", "TYTS"], "note": "platform 2"},
    "Nottingham": {"end_signals": ["TD5030"], "tcs": ["TFPV"], "note": "Spondon side (east)"},
}

# Platform sub-section TCs (from platform_tc_map.csv) -- for 3B/4B layover (C15c) etc.
PLATFORM_SUB_TC = {"3B": "TNGU", "4B": "TRJV"}

# Branch detection: a route is "onto the branch" if its end signal is in these sets.
SINFIN_SIGNALS = {"DW5320", "DW5321", "DW5323"}
MATLOCK_SIGNALS = {"DY571", "DY572"}

# ------------------------------------------------------------------
# THE 19 APPROVED RULES (Hao-signed 2026-05-27).
#   match: machine-readable condition. Keys (all optional):
#     focal_signal_in : list[str]  -- DECISION signal (PREFIXED) must be one of these (origin)
#     target_platform : int        -- rule applies only when the AUDITED route ends at this platform
#     branch          : str        -- 'sinfin'|'matlock' (audited route leads onto that branch)
#     direction_in    : list[str]  -- destination direction (SOFT; needs resolve_direction)
#     train_class_in  : list[str]  -- headcode class digit(s), e.g. ['1','2'] = passenger
#   kind: 'route_choice' | 'platform_set' | 'policy_fact'
#   pref: preferred_route_id / preferred_platforms(set) / non_preferred_route_ids
# ------------------------------------------------------------------
RULES = [
    # ---- HARD: preferred/non-preferred route (section 5) ----
    dict(rule_id="S5-TD5045-platform4", source_section="§5", confidence="high",
         kind="route_choice",
         cond_origin="East/Spondon (TD5045)", cond_destination="platform_4", cond_train_class=None,
         match=dict(focal_signal_in=["TD5045"], target_platform=4),
         pref=dict(preferred_route_id="RTD5045B-1(M)",
                   non_preferred_route_ids=["RTD5045B-2(M)"], preferred_platforms=None),
         user_approved=True,
         notes="preferred via 306(TDPA)+311(TNGK); non-pref via 303rev(TFPB)+307(TFMW) when 311 locked. 4x verified."),

    # ---- HARD: access constraints / facts (section 5/6/9) ----
    dict(rule_id="S5-TD5049-platform3or4only", source_section="§5", confidence="high",
         kind="platform_set",
         cond_origin="RTC North sidings (TD5049)", cond_destination="platform_3_or_4", cond_train_class=None,
         match=dict(focal_signal_in=["TD5049"]),
         pref=dict(preferred_route_id=None, non_preferred_route_ids=[], preferred_platforms=[3, 4]),
         user_approved=True,
         notes="TD5049 -> platform 3 or 4 ONLY (cannot signal into platform 5). para426/434."),

    dict(rule_id="S5-platform1-north-aline", source_section="§5", confidence="high",
         kind="policy_fact",
         cond_origin="platform_1 (DC5061 north)", cond_destination="North", cond_train_class=None,
         match=dict(focal_signal_in=["DC5061"]),
         pref=dict(preferred_route_id=None, non_preferred_route_ids=[], preferred_platforms=None),
         user_approved=True,
         notes="platform 1 northbound = A line/down fast, the ONLY route (informational; one option). para478."),

    dict(rule_id="S9-rtcnorth-platform3or4", source_section="§9", confidence="high",
         kind="platform_set",
         cond_origin="North", cond_destination="RTC North sidings", cond_train_class=None,
         match=dict(focal_signal_in=["TD5049"]),
         pref=dict(preferred_route_id=None, non_preferred_route_ids=[], preferred_platforms=[3, 4]),
         user_approved=True,
         notes="Access to RTC North by trains from the North via platform 3 or 4. para749. "
               "(Shares anchor TD5049 with C8 -- both constrain to {3,4}.)"),

    dict(rule_id="S9-rtcsouth-access", source_section="§9", confidence="high",
         kind="platform_set",
         cond_origin="any", cond_destination="RTC South sidings (TD5043)", cond_train_class=None,
         match=dict(focal_signal_in=["TD5043"]),
         pref=dict(preferred_route_id=None, non_preferred_route_ids=[],
                   preferred_platforms=[3, 4, 5, 6]),
         user_approved=True,
         notes="Access to RTC South via pilot line/platform 6/5/3/4. para751. Anchor=exit signal TD5043 "
               "(Hao's TC 'TDWV' had 0 route hits -> dropped). pilot-line acceptance handled separately."),

    dict(rule_id="S9-platform5-depart-301", source_section="§9", confidence="high",
         kind="policy_fact",
         cond_origin="platform_5 (DC5065)", cond_destination=None, cond_train_class=None,
         match=dict(focal_signal_in=["DC5065"]),
         pref=dict(preferred_route_id=None, non_preferred_route_ids=[], preferred_platforms=None),
         user_approved=True,
         notes="platform 5 departure runs along down main reverse to 301pts (valid but slower). para753."),

    dict(rule_id="S6-platform6-callon-north", source_section="§6", confidence="high",
         kind="platform_set",
         cond_origin="North (DC5076)", cond_destination="platform_6", cond_train_class=None,
         match=dict(focal_signal_in=["DC5076"], target_platform=6),
         pref=dict(preferred_route_id=None, non_preferred_route_ids=[], preferred_platforms=[6]),
         user_approved=True,
         notes="call-on into occupied platform 6 from DC5076 (stop+advise driver); except EC5486/EC5488 "
               "on Chad curve. para711. Linked to f_call_on stratum."),

    dict(rule_id="S9-litchurch-platform3or4", source_section="§9", confidence="high",
         kind="platform_set",
         cond_origin="Litchurch Lane (DW5310/202pts)", cond_destination="platform_3_or_4", cond_train_class=None,
         match=dict(focal_signal_in=["DW5310"]),
         pref=dict(preferred_route_id=None, non_preferred_route_ids=[], preferred_platforms=[3, 4]),
         user_approved=True,
         notes="Litchurch Lane in/out via platform 3 or 4 (gauge-cleared). para757."),

    # ---- HARD: branch safety policy (section 11/14) ----
    dict(rule_id="S11-sinfin-singleline", source_section="§11", confidence="high",
         kind="policy_fact",
         cond_origin="any", cond_destination="Sinfin branch", cond_train_class=None,
         match=dict(branch="sinfin"),
         pref=dict(preferred_route_id=None, non_preferred_route_ids=[], preferred_platforms=None),
         user_approved=True,
         notes="single-line: one train at a time; must not approach DW5320 while another signalled "
               "toward DW5323. para779. Policy (no per-decision preferred action)."),

    dict(rule_id="S14-matlock-token", source_section="§14", confidence="high",
         kind="policy_fact",
         cond_origin="any", cond_destination="Matlock branch", cond_train_class=None,
         match=dict(branch="matlock"),
         pref=dict(preferred_route_id=None, non_preferred_route_ids=[], preferred_platforms=None),
         user_approved=True,
         notes="No-Signaller Token (TS7); token released on 912TC at Ambergate; DY572 out / DY571 set. para806-808."),

    # ---- SOFT: section 3 traffic-flow platform preferences (med; reference only; need destination direction) ----
    dict(rule_id="S3-sheffieldmatlock-platform5", source_section="§3", confidence="med",
         kind="platform_set",
         cond_origin=None, cond_destination="Sheffield/Matlock", cond_train_class="passenger",
         match=dict(direction_in=["Sheffield", "Matlock"], train_class_in=["1", "2"]),
         pref=dict(preferred_route_id=None, non_preferred_route_ids=[], preferred_platforms=[5]),
         user_approved=True,
         notes="'trains to Sheffield and Matlock will use platform 5 predominantly' (soft). "
               "CONFLICTS with C17 (->platform 6); both med, neither gates; if both match -> ambiguous. para364."),

    dict(rule_id="S3-west-platform2", source_section="§3", confidence="med",
         kind="platform_set",
         cond_origin=None, cond_destination="West", cond_train_class="passenger",
         match=dict(direction_in=["West"], train_class_in=["1", "2"]),
         pref=dict(preferred_route_id=None, non_preferred_route_ids=[], preferred_platforms=[2]),
         user_approved=True,
         notes="'all passenger trains to the West will use platform 2' (option 3/4). para370/379."),

    dict(rule_id="S3-north-fastest-platform5", source_section="§3", confidence="med",
         kind="platform_set",
         cond_origin=None, cond_destination="North", cond_train_class=None,
         match=dict(direction_in=["North"]),
         pref=dict(preferred_route_id=None, non_preferred_route_ids=[], preferred_platforms=[5]),
         user_approved=True,
         notes="North quickest = platform 5 (3/4/pilot also; 6 slower) (soft). "
               "overlaps C15a (North passenger->platform 1); if both match -> ambiguous. para376."),

    dict(rule_id="S3-south-optimum-platform6", source_section="§3", confidence="med",
         kind="platform_set",
         cond_origin=None, cond_destination="South", cond_train_class=None,
         match=dict(direction_in=["South"]),
         pref=dict(preferred_route_id=None, non_preferred_route_ids=[], preferred_platforms=[6]),
         user_approved=True,
         notes="South optimum = platform 6 (3/4/pilot also) (soft). para382."),

    dict(rule_id="S3-north-passenger-platform1", source_section="§3", confidence="med",
         kind="platform_set",
         cond_origin=None, cond_destination="North", cond_train_class="passenger",
         match=dict(direction_in=["North"], train_class_in=["1", "2"]),
         pref=dict(preferred_route_id=None, non_preferred_route_ids=[], preferred_platforms=[1]),
         user_approved=True,
         notes="'trains for the North will use platform 1' (soft). overlaps C13. para367."),

    dict(rule_id="S3-nottingham-platform3or4", source_section="§3", confidence="med",
         kind="platform_set",
         cond_origin=None, cond_destination="Nottingham", cond_train_class="passenger",
         match=dict(direction_in=["Nottingham"], train_class_in=["1", "2"]),
         pref=dict(preferred_route_id=None, non_preferred_route_ids=[], preferred_platforms=[3, 4]),
         user_approved=True,
         notes="'Trains for Nottingham will use platform 3 or 4' (soft). para367."),

    dict(rule_id="S3-crewe-layover-platform3b4b", source_section="§3", confidence="med",
         kind="platform_set",
         cond_origin=None, cond_destination="Crewe", cond_train_class="passenger",
         match=dict(direction_in=["Crewe"], train_class_in=["1", "2"]),
         pref=dict(preferred_route_id=None, non_preferred_route_ids=[], preferred_platforms=[3, 4]),
         user_approved=True,
         notes="Crewe layover = platform 3B or 4B (TC TNGU/TRJV) (soft). para367."),

    dict(rule_id="S3-birmingham-platform3or4", source_section="§3", confidence="med",
         kind="platform_set",
         cond_origin=None, cond_destination="Birmingham", cond_train_class="passenger",
         match=dict(direction_in=["Birmingham"], train_class_in=["1", "2"]),
         pref=dict(preferred_route_id=None, non_preferred_route_ids=[], preferred_platforms=[3, 4]),
         user_approved=True,
         notes="'trains for Birmingham using platform 3 or 4' (soft). para364."),

    dict(rule_id="S3-sheffieldmatlocknorthern-platform6", source_section="§3", confidence="med",
         kind="platform_set",
         cond_origin=None, cond_destination="Sheffield/Matlock/Northern", cond_train_class="passenger",
         match=dict(direction_in=["Sheffield", "Matlock", "Northern"], train_class_in=["1", "2"]),
         pref=dict(preferred_route_id=None, non_preferred_route_ids=[], preferred_platforms=[6]),
         user_approved=True,
         notes="'Matlock/Sheffield/Northern will use platform 6' (option 3/4) (soft). "
               "CONFLICTS with R2 (->platform 5) -- Plan itself contradicts; both med, neither gates. para370."),
]

assert len(RULES) == 19, f"expected 19 approved rules, got {len(RULES)}"


# ------------------------------------------------------------------
# Static route catalog: route_id -> end_platform / end_signal / tracks.
# ------------------------------------------------------------------
_CATALOG: Optional[dict] = None


def _platform_tc_map() -> dict:
    """TC id -> platform int (from platform_tc_map.csv)."""
    out = {}
    p = C.REFERENCE_DIR / "platform_tc_map.csv"
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            tc, plat = r.get("tc_id"), r.get("platform_id")
            if tc and plat and str(plat).strip().isdigit():
                out[tc.strip()] = int(plat)
    return out


def _route_catalog() -> dict:
    """route_id -> dict(end_platform:int|None, end_signal:str|None, tracks:list[str]).
    Built from route_to_tc_all.csv (route, end, track). end_platform = platform of the LAST
    platform-TC found in the track list (per platform_tc_map)."""
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG
    tc2plat = _platform_tc_map()
    cat: dict = {}
    p = C.ROUTE_TO_TC_CSV
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            rid = (r.get("route") or "").strip()
            if not rid:
                continue
            tracks = re.findall(r"'([A-Z0-9]+)'", r.get("track", "") or "")
            end_plat = None
            for tc in tracks:                       # last platform-TC wins (destination platform)
                if tc in tc2plat:
                    end_plat = tc2plat[tc]
            end_sig_no = (r.get("end") or "").strip()
            m = re.match(r"R([A-Z]{2})", rid)       # prefix from route id (e.g. RTD... -> TD)
            end_signal = (m.group(1) + end_sig_no) if (m and end_sig_no.isdigit()) else None
            if rid not in cat or (cat[rid]["end_platform"] is None and end_plat is not None):
                cat[rid] = dict(end_platform=end_plat, end_signal=end_signal, tracks=tracks)
    _CATALOG = cat
    return cat


def route_end_platform(route_id: str) -> Optional[int]:
    return _route_catalog().get(str(route_id), {}).get("end_platform")


def route_on_branch(route_id: str) -> Optional[str]:
    """'sinfin'|'matlock'|None -- does the route's end signal touch a branch?"""
    info = _route_catalog().get(str(route_id))
    if not info:
        return None
    sig = info.get("end_signal")
    if sig in SINFIN_SIGNALS:
        return "sinfin"
    if sig in MATLOCK_SIGNALS:
        return "matlock"
    return None


def headcode_class_digit(focal_train) -> Optional[str]:
    """class digit of the focal train's 4-char headcode (train_id[2:6][0]); None if unparseable."""
    s = str(focal_train or "")
    if len(s) >= 6 and s[2:6][0].isdigit():
        return s[2]
    m = re.search(r"[0-9][A-Z][0-9]{2}", s)        # standard headcode pattern anywhere
    return m.group(0)[0] if m else None


def is_passenger(focal_train) -> Optional[bool]:
    d = headcode_class_digit(focal_train)
    if d is None:
        return None
    return d in ("1", "2")


_SIG_FROM_RID = re.compile(r"^R([A-Z]{2})(\d+)")


def signal_from_route_id(route_id) -> Optional[str]:
    """'RTD5045B-1(M)' -> 'TD5045' (the route's START/decision signal, PREFIXED form)."""
    m = _SIG_FROM_RID.match(str(route_id or ""))
    return (m.group(1) + m.group(2)) if m else None


def decision_signal(sample: dict) -> Optional[str]:
    """The decision (approach) signal in canonical PREFIXED form (e.g. 'TD5045').

    CRITICAL: `focal_signal` in snapshots_v2 is stored as the BARE number ('5045') because
    the route index keys on route_to_tc_all.csv `start` (also bare). Our rule anchors are
    PREFIXED ('TD5045'). All candidate routes at a decision share the SAME start signal, and
    the route_id encodes it WITH prefix (RTD5045... -> 'TD5045'), so we recover the prefixed
    decision signal from any candidate. Falls back to raw focal_signal if no candidates."""
    for rid in (sample.get("candidate_route_ids") or []):
        s = signal_from_route_id(rid)
        if s:
            return s
    fs = sample.get("focal_signal")
    return str(fs) if fs is not None else None


def resolve_direction(sample: dict) -> Optional[str]:
    """Destination DIRECTION for soft traffic-flow rules. The state hides destination
    (leak audit) and no headcode-letter->direction map exists in-repo -> returns None by
    default, so soft rules stay 'undetermined' rather than fabricated. Override by passing
    sample['_direction'] (e.g., from a future headcode map) to activate soft rules."""
    return sample.get("_direction")


# ------------------------------------------------------------------
# Matching
# ------------------------------------------------------------------
def rule_matches(rule: dict, sample: dict, audited_route_id: Optional[str] = None) -> bool:
    """Does this rule's context apply to this decision?
    sample needs: focal_signal (str), focal_train (str), candidate_route_ids (list[str]).
    audited_route_id = the route whose compliance we test (model argmax or signaller chosen);
    used for target_platform / branch conditions. Returns True iff ALL present conditions hold.
    """
    m = rule.get("match", {})

    fs = m.get("focal_signal_in")
    if fs is not None:
        ds = decision_signal(sample)        # PREFIXED form derived from candidates (robust)
        raw = str(sample.get("focal_signal"))
        if ds not in fs and raw not in fs:
            return False

    tc = m.get("train_class_in")
    if tc is not None:
        d = headcode_class_digit(sample.get("focal_train"))
        if d is None or d not in tc:
            return False

    di = m.get("direction_in")
    if di is not None:
        direction = resolve_direction(sample)
        if direction is None or direction not in di:
            return False

    tp = m.get("target_platform")
    if tp is not None:
        if audited_route_id is None or route_end_platform(audited_route_id) != tp:
            return False

    br = m.get("branch")
    if br is not None:
        if audited_route_id is None or route_on_branch(audited_route_id) != br:
            return False

    return True


def load_rule_base():
    """Return the approved rules. DataFrame if pandas is importable, else list[dict]."""
    try:
        import pandas as pd
        return pd.DataFrame(RULES)
    except Exception:
        return list(RULES)
