"""Excel rendering for market_state. Presentation only — no maths lives here.

Split from market_state.py so the data layer can be tested without openpyxl and so
a formatting change can never alter a number.
"""
from __future__ import annotations

import logging
import os
import time

import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from . import market_state as ms

logger = logging.getLogger("qlib_lab.market_state_xlsx")

H1 = Font(bold=True, size=14)
H2 = Font(bold=True, size=11)
HDR = Font(bold=True, color="FFFFFF")
HDR_FILL = PatternFill("solid", fgColor="44546A")
FIRED_FILL = PatternFill("solid", fgColor="FCE4D6")
WARN_FILL = PatternFill("solid", fgColor="FFF2CC")
MUTED = Font(italic=True, color="808080", size=9)
THIN = Border(bottom=Side(style="thin", color="D0D0D0"))

# green (low) -> white -> red (high); percentiles read the same way everywhere
SCALE = ColorScaleRule(
    start_type="num", start_value=0, start_color="C6EFCE",
    mid_type="num", mid_value=0.5, mid_color="FFFFFF",
    end_type="num", end_value=1, end_color="F8CBAD")

PCT2 = "0.00"
PCT_HUMAN = "0.0%"
NUM2 = "#,##0.00"
NUM1 = "#,##0.1"


def _headers(ws, cols, row=1):
    for i, c in enumerate(cols, start=1):
        cell = ws.cell(row=row, column=i, value=str(c))
        cell.font = HDR
        cell.fill = HDR_FILL
        cell.alignment = Alignment(horizontal="center")
    ws.freeze_panes = ws.cell(row=row + 1, column=1)


def _autosize(ws, minw=9, maxw=44):
    for col in ws.columns:
        letter = get_column_letter(col[0].column)
        longest = max((len(str(c.value)) for c in col if c.value is not None),
                      default=0)
        ws.column_dimensions[letter].width = max(minw, min(maxw, longest + 2))


def _write_df(ws, df: pd.DataFrame, index_name: str | None = None, start_row=1):
    """Write a DataFrame as a header row + data rows. Returns the last row index."""
    if df is None or df.empty:
        ws.cell(row=start_row, column=1, value="no data available (fail-safe)")
        return start_row
    cols = ([index_name] if index_name else []) + [str(c) for c in df.columns]
    _headers(ws, cols, row=start_row)
    r = start_row
    for idx, row in df.iterrows():
        r += 1
        c = 1
        if index_name:
            val = idx.date() if isinstance(idx, pd.Timestamp) else idx
            ws.cell(row=r, column=c, value=val)
            c += 1
        for v in row:
            if isinstance(v, pd.Timestamp):
                v = v.date()
            elif isinstance(v, (list, dict)):
                v = str(v)
            elif pd.isna(v) if not isinstance(v, (str, bool)) else False:
                v = None
            ws.cell(row=r, column=c, value=v)
            c += 1
    return r


def _fmt_cols(ws, df, mapping, first_data_row=2, offset=1):
    """Apply number formats by column NAME, so inserting a column cannot misalign."""
    for name, fmt in mapping.items():
        if name not in df.columns:
            continue
        col = list(df.columns).index(name) + 1 + offset
        letter = get_column_letter(col)
        for cell in ws[letter][first_data_row - 1:]:
            cell.number_format = fmt


def _scale_cols(ws, df, names, first_data_row=2, offset=1, last_row=None):
    last = last_row or (len(df) + first_data_row - 1)
    for name in names:
        if name not in df.columns:
            continue
        letter = get_column_letter(list(df.columns).index(name) + 1 + offset)
        ws.conditional_formatting.add(f"{letter}{first_data_row}:{letter}{last}",
                                      SCALE)


# ----------------------------------------------------------------------- sheets

def _sheet_today(wb, state):
    ws = wb.create_sheet("TODAY")
    ws["A1"] = "MARKET STATE"
    ws["A1"].font = H1
    ws["C1"] = f"as of {state.get('as_of') or 'unknown'} close"
    ws["A2"] = f"generated {state.get('generated_at', '')[:19]}Z"
    ws["A2"].font = MUTED

    r = 4
    ws.cell(row=r, column=1, value="SUMMARY").font = H2
    r += 1
    for line in state["narrative"].split("\n"):
        ws.cell(row=r, column=1, value=line)
        r += 1

    vol, liq, fund = state["vol"], state["liq"], state["funding"]
    struct, cx = state["struct"], state["crowding"]

    def block(title, rows, row):
        ws.cell(row=row, column=1, value=title).font = H2
        row += 1
        for label, value, fmt in rows:
            ws.cell(row=row, column=1, value=label)
            c = ws.cell(row=row, column=2, value=value)
            if fmt:
                c.number_format = fmt
            row += 1
        return row + 1

    r += 1
    r = block("VOLATILITY", [
        ("VIX", vol.get("vix"), NUM2),
        ("VRP", vol.get("vrp"), NUM2),
        ("VRP percentile (1y)", vol.get("vrp_pct"), PCT_HUMAN),
        ("VIX9D / VIX3M", vol.get("term_ratio"), "0.000"),
        ("structure", vol.get("structure"), None),
        ("VVIX", vol.get("vvix"), NUM2),
        ("SKEW", vol.get("skew"), NUM2),
        ("realised vol 20d", vol.get("rv20"), NUM2),
    ], r)

    crowd_rows = []
    if not cx.empty:
        for _, row in cx[cx["side"] != "neutral"].iterrows():
            crowd_rows.append((f"{row['asset']} ({row['side']})",
                               row["idx52"], PCT2))
    if not crowd_rows:
        crowd_rows = [("no positioning extreme", None, None)]
    crowd_rows.append(("BTC funding (ann. %)",
                       fund.get("funding_1d", 0) * 365 * 100 if fund else None, NUM2))
    r = block("POSITIONING (COT idx52)", crowd_rows, r)

    r = block("LIQUIDITY", [
        ("net liquidity ($bn)", liq.get("netliq"), NUM2),
        ("13w change ($bn)", liq.get("netliq_chg13w"), NUM2),
        ("13w change z", liq.get("netliq_chg13w_z"), NUM2),
        ("WALCL ($bn)", liq.get("walcl"), NUM2),
        ("TGA ($bn)", liq.get("tga"), NUM2),
        ("RRP ($bn)", liq.get("rrp"), NUM2),
    ], r)

    r = block("STRUCTURE", [
        ("absorption (top-k eigen share)", struct.get("absorption"), PCT2),
        ("top-1 eigen share", struct.get("top1_share"), PCT2),
        ("correlation window (sessions)", struct.get("window"), None),
        ("assets in matrix", struct.get("n_assets"), None),
    ], r)

    ws.cell(row=r, column=1, value="CONDITIONS FIRING TODAY").font = H2
    r += 1
    hits = [c for c in state["conditions"] if c["fired"]]
    ws.cell(row=r, column=1,
            value=f"{len(hits)} of {len(state['conditions'])} firing")
    r += 1
    for c in state["conditions"]:
        ws.cell(row=r, column=1, value="FIRING" if c["fired"] else "-")
        ws.cell(row=r, column=2, value=c["id"])
        ws.cell(row=r, column=3, value=c["detail"])
        if c["fired"]:
            for col in (1, 2, 3):
                ws.cell(row=r, column=col).fill = FIRED_FILL
        r += 1

    r += 1
    stale = [k for k, v in state["stale"].items() if v]
    if stale:
        cell = ws.cell(row=r, column=1,
                       value=f"STALE / MISSING INPUTS: {', '.join(stale)} "
                             f"- those blocks read neutral, not zero")
        cell.fill = WARN_FILL
        r += 1
    ws.cell(row=r, column=1, value=ms.CONDITIONS_NOTE).font = MUTED

    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 18
    ws.column_dimensions["C"].width = 76
    return ws


def _sheet_volatility(wb, state):
    ws = wb.create_sheet("Volatility")
    df = state["vol_hist"]
    last = _write_df(ws, df, index_name="date")
    if df.empty:
        return ws
    _fmt_cols(ws, df, {c: NUM2 for c in
                       ("spy", "vix", "vix9d", "vix3m", "vvix", "skew", "rv20", "vrp")})
    _fmt_cols(ws, df, {"term_ratio": "0.000"})
    _fmt_cols(ws, df, {c: PCT_HUMAN for c in df.columns if c.endswith("_pct")})
    _scale_cols(ws, df, [c for c in df.columns if c.endswith("_pct")], last_row=last)
    _autosize(ws)
    return ws


def _sheet_positioning(wb, state):
    ws = wb.create_sheet("Positioning")
    pos = state["pos_hist"]
    if pos.empty:
        ws["A1"] = "no COT history available (fail-safe)"
        return ws

    ws["A1"] = "LATEST BY ASSET"
    ws["A1"].font = H2
    latest = state["pos_latest"][
        ["asset", "report_date", "net_pct", "z156", "idx52", "chg4w"]
    ].sort_values("idx52", ascending=False, na_position="last")
    last = _write_df(ws, latest.reset_index(drop=True), start_row=2)
    _fmt_cols(ws, latest, {"net_pct": "0.0000", "z156": NUM2, "idx52": PCT2,
                           "chg4w": "0.0000"}, first_data_row=3, offset=0)
    _scale_cols(ws, latest, ["idx52"], first_data_row=3, offset=0, last_row=last)

    r = last + 2
    ws.cell(row=r, column=1, value="idx52 HISTORY (weekly, assets as columns)").font = H2
    wide = pos.pivot_table(index="effective_date", columns="asset", values="idx52")
    ws2 = wb.create_sheet("Positioning_idx52")
    last2 = _write_df(ws2, wide, index_name="date")
    _fmt_cols(ws2, wide, {c: PCT2 for c in wide.columns})
    _scale_cols(ws2, wide, list(wide.columns), last_row=last2)
    _autosize(ws2)

    fund = state["fund_hist"]
    if not fund.empty:
        ws3 = wb.create_sheet("Funding")
        last3 = _write_df(ws3, fund, index_name="date")
        _fmt_cols(ws3, fund, {"funding_1d": "0.000000", "z156": NUM2, "idx52w": PCT2})
        _scale_cols(ws3, fund, ["idx52w"], last_row=last3)
        _autosize(ws3)

    _autosize(ws)
    return ws


def _sheet_liquidity(wb, state):
    ws = wb.create_sheet("Liquidity")
    df = state["liq_hist"]
    _write_df(ws, df, index_name="date")
    if not df.empty:
        _fmt_cols(ws, df, {c: NUM2 for c in df.columns})
        _autosize(ws)
    return ws


def _sheet_structure(wb, state):
    ws = wb.create_sheet("Structure")
    struct = state["struct"]
    ws["A1"] = f"ROLLING {struct.get('window')}-SESSION CORRELATION"
    ws["A1"].font = H2
    corr = struct.get("corr", pd.DataFrame())
    last = _write_df(ws, corr, index_name="asset", start_row=2)
    if not corr.empty:
        _fmt_cols(ws, corr, {c: PCT2 for c in corr.columns}, first_data_row=3)
        for i in range(len(corr.columns)):
            letter = get_column_letter(i + 2)
            ws.conditional_formatting.add(
                f"{letter}3:{letter}{last}",
                ColorScaleRule(start_type="num", start_value=-1,
                               start_color="9BC2E6", mid_type="num", mid_value=0,
                               mid_color="FFFFFF", end_type="num", end_value=1,
                               end_color="F8CBAD"))

    r = last + 2
    ws.cell(row=r, column=1, value="ABSORPTION").font = H2
    r += 1
    for label, val in (("top-k eigenvalue share", struct.get("absorption")),
                       ("top-1 eigenvalue share", struct.get("top1_share")),
                       ("assets in matrix", struct.get("n_assets"))):
        ws.cell(row=r, column=1, value=label)
        ws.cell(row=r, column=2, value=val).number_format = PCT2
        r += 1
    ws.cell(row=r, column=1,
            value="high absorption = risk concentrated in few factors, so "
                  "diversification is weaker than the position count suggests"
            ).font = MUTED

    betas = struct.get("betas", pd.DataFrame())
    if not betas.empty:
        r += 2
        ws.cell(row=r, column=1, value="BETAS").font = H2
        _write_df(ws, betas, index_name="asset", start_row=r + 1)
        _fmt_cols(ws, betas, {c: NUM2 for c in betas.columns},
                  first_data_row=r + 2)
    _autosize(ws)
    return ws


def _sheet_crowding(wb, state):
    ws = wb.create_sheet("Crowding")
    ws["A1"] = "CROWDING x STRUCTURE"
    ws["A1"].font = H1
    ws["A2"] = ("COT says WHO is positioned; correlation says WHAT moves together. "
                "An asset that is crowded AND correlated with the other crowded names "
                "is not diversifying - it is the same bet wearing another ticker.")
    ws["A3"] = ("corr_to_crowded = mean correlation to the OTHER crowded names. "
                "same_bet = crowded and |corr_to_crowded| >= 0.35.")
    ws["A3"].font = MUTED

    cx = state["crowding"]
    last = _write_df(ws, cx, start_row=5)
    if not cx.empty:
        _fmt_cols(ws, cx, {"idx52": PCT2, "z156": NUM2, "chg4w": "0.0000",
                           "corr_to_crowded": PCT2}, first_data_row=6, offset=0)
        _scale_cols(ws, cx, ["idx52"], first_data_row=6, offset=0, last_row=last)
    ws.column_dimensions["A"].width = 12
    for col in "BCDEFG":
        ws.column_dimensions[col].width = 17
    return ws


def _sheet_conditions(wb, state):
    ws = wb.create_sheet("Conditions")
    ws["A1"] = "REGISTERED CONDITIONS"
    ws["A1"].font = H1
    ws["A2"] = f"registered {ms.CONDITIONS_REGISTERED}"
    cell = ws["A3"]
    cell.value = ms.CONDITIONS_NOTE
    cell.fill = WARN_FILL
    cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("A3:D6")

    _headers(ws, ["id", "rule", "firing today", "detail"], row=8)
    r = 8
    for c in state["conditions"]:
        r += 1
        ws.cell(row=r, column=1, value=c["id"])
        ws.cell(row=r, column=2, value=c["rule"])
        ws.cell(row=r, column=3, value="FIRING" if c["fired"] else "-")
        ws.cell(row=r, column=4, value=c["detail"])
        if c["fired"]:
            for col in range(1, 5):
                ws.cell(row=r, column=col).fill = FIRED_FILL

    log = _read_fire_log()
    if log is not None and not log.empty:
        r += 2
        ws.cell(row=r, column=1, value="FIRE LOG (accumulates forward)").font = H2
        _write_df(ws, log, index_name="as_of", start_row=r + 1)

    ws.column_dimensions["A"].width = 32
    ws.column_dimensions["B"].width = 56
    ws.column_dimensions["C"].width = 13
    ws.column_dimensions["D"].width = 72
    return ws


def _read_fire_log():
    try:
        if not ms.FIRE_LOG.exists():
            return None
        rows = [__import__("json").loads(ln)
                for ln in ms.FIRE_LOG.read_text(encoding="utf-8").splitlines() if ln]
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["fired"] = df["fired"].apply(lambda v: ", ".join(v) if v else "(none)")
        return df.set_index("as_of")[["generated_at", "fired"]].tail(400)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"fire log unreadable: {e}")
        return None


STORY_FILL = {
    "HOLDING": PatternFill("solid", fgColor="C6EFCE"),
    "WEAK": PatternFill("solid", fgColor="FFF2CC"),
    "BROKEN": PatternFill("solid", fgColor="F8CBAD"),
    "NO_DATA": PatternFill("solid", fgColor="E7E6E6"),
}


def _sheet_stories(wb, state):
    """Stated market stories and whether the relationship each implies still holds."""
    ws = wb.create_sheet("Narratives")
    ws["A1"] = "MARKET STORIES vs THE TAPE"
    ws["A1"].font = H1
    ws["A2"] = (f"registered {ms.STORIES_REGISTERED}; state read on the "
                f"{ms.STORY_CORR_FAST}d correlation, {ms.STORY_CORR_SLOW}d shown beside it")
    cell = ws["A3"]
    cell.value = ms.STORIES_NOTE
    cell.fill = WARN_FILL
    cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("A3:F8")

    _headers(ws, ["id", "the story", "implies", f"corr {ms.STORY_CORR_FAST}d",
                  f"corr {ms.STORY_CORR_SLOW}d", "state", "state (slow)"], row=10)
    r = 10
    for st in (state.get("stories") or []):
        r += 1
        ws.cell(row=r, column=1, value=st["id"])
        ws.cell(row=r, column=2, value=st["claim"])
        ws.cell(row=r, column=3, value=st["implies"])
        for col, key in ((4, "corr_fast"), (5, "corr_slow")):
            c = ws.cell(row=r, column=col, value=st[key])
            c.number_format = PCT2
        c = ws.cell(row=r, column=6, value=st["state"])
        c.fill = STORY_FILL.get(st["state"], VERDICT_FILL["neutral"])
        ws.cell(row=r, column=7, value=st["state_slow"])
    if r == 10:
        ws.cell(row=11, column=1, value="no story table in this run (framework "
                                        "unavailable)").font = MUTED

    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 58
    ws.column_dimensions["C"].width = 26
    for col in "DEFG":
        ws.column_dimensions[col].width = 13
    return ws


VERDICT_FILL = {
    "bullish": PatternFill("solid", fgColor="C6EFCE"),
    "bearish": PatternFill("solid", fgColor="F8CBAD"),
    "neutral": PatternFill("solid", fgColor="FFFFFF"),
    "NO_DATA": PatternFill("solid", fgColor="E7E6E6"),
}


def _sheet_framework(wb, table, layers, missing):
    """Trend -> ... -> Risk for every instrument, one row each."""
    ws = wb.create_sheet("Framework")
    ws["A1"] = "PER-ASSET FRAMEWORK"
    ws["A1"].font = H1
    ws["A2"] = ("Each layer answers one question. One metric alone is weak; the point "
                "is WHICH layers agree. The confluence column is a COUNT of agreeing "
                "measurements - not a score, not a ranking, and never tested against "
                "forward returns.")
    ws["A4"] = (
        "HOW TO READ: positioning 'bearish' means CROWDED = vulnerable, not a bearish "
        "price call. risk 'bearish' means ELEVATED HAZARD (high realised vol / deep "
        "drawdown), not price down. INVERSE rows (VIX family) read backwards: bullish "
        "= rising volatility = risk-OFF. NO_DATA means this box cannot measure that "
        "layer - it is NOT a neutral reading. " +
        " | ".join(f"{k}: {v}" for k, v in missing.items()))
    ws["A4"].font = MUTED
    for cell in ("A2", "A4"):
        ws[cell].alignment = Alignment(wrap_text=True, vertical="top")
    # Write values BEFORE merging: the top-left cell keeps the value, but every other
    # cell in the range becomes a read-only MergedCell. Excel also does NOT auto-fit
    # row height on merged cells, so a long disclaimer in a single merged row renders
    # clipped to one line -- and these two blocks are the whole reason this sheet
    # stays on the descriptive side of the line, so they must actually be readable.
    ws.merge_cells("A2:P3")
    ws.merge_cells("A4:P6")
    for r in (2, 3, 4, 5, 6):
        ws.row_dimensions[r].height = 30

    keys = [k for k, _q in layers]
    cols = ["asset", "price", "reads", "universes"] + keys + ["confluence"]
    _headers(ws, cols, row=8)
    r = 8
    for _, row in table.iterrows():
        r += 1
        ws.cell(row=r, column=1, value=row["asset"])
        ws.cell(row=r, column=2, value=row["price"]).number_format = "#,##0.00"
        ws.cell(row=r, column=3,
                value="INVERSE" if row.get("inverse") else "")
        ws.cell(row=r, column=4, value=row.get("universes", ""))
        for i, k in enumerate(keys):
            c = ws.cell(row=r, column=5 + i, value=row[k])
            c.fill = VERDICT_FILL.get(row[k], VERDICT_FILL["neutral"])
            c.alignment = Alignment(horizontal="center")
        # n_bullish/n_bearish are deliberately NOT written as sortable numbers: the
        # tally sums incommensurable units (an uncrowded `bullish` and a low-hazard
        # `bullish` mean different things) so ranking assets by it would be meaningless
        # -- and would turn a thermometer into an ungated model.
        ws.cell(row=r, column=5 + len(keys), value=row["confluence"])
    ws.freeze_panes = ws.cell(row=9, column=2)
    ws.column_dimensions["A"].width = 10
    ws.column_dimensions["B"].width = 12
    ws.column_dimensions["C"].width = 9
    ws.column_dimensions["D"].width = 30
    for i in range(len(keys)):
        ws.column_dimensions[get_column_letter(5 + i)].width = 13
    ws.column_dimensions[get_column_letter(5 + len(keys))].width = 16
    return ws


def _sheet_framework_detail(wb, table, layers, focus):
    """The readable version: every layer's actual reading, in framework order."""
    ws = wb.create_sheet("Framework Detail")
    ws["A1"] = "FRAMEWORK - READINGS"
    ws["A1"].font = H1
    r = 2
    idx = table.set_index("asset")
    for name in focus:
        if name not in idx.index:
            continue
        row = idx.loc[name]
        r += 1
        ws.cell(row=r, column=1,
                value=f"{name}   {row['price']:,.2f}   ({row['reads']})").font = H2
        ws.cell(row=r, column=4, value=row["confluence"]).font = H2
        r += 1
        for key, question in layers:
            ws.cell(row=r, column=1, value=key)
            v = ws.cell(row=r, column=2, value=row[key])
            v.fill = VERDICT_FILL.get(row[key], VERDICT_FILL["neutral"])
            ws.cell(row=r, column=3, value=question).font = MUTED
            ws.cell(row=r, column=4, value=row[f"{key}_detail"])
            r += 1
        r += 1
    ws.column_dimensions["A"].width = 15
    ws.column_dimensions["B"].width = 11
    ws.column_dimensions["C"].width = 42
    ws.column_dimensions["D"].width = 104
    return ws


HISTORY_COLS = ["vix", "vrp", "vrp_pct", "term_ratio", "term_stale_days", "vvix",
                "skew", "skew_pct", "rv20", "netliq_bn", "netliq_chg13w_bn",
                "netliq_chg13w_z", "rrp_bn", "absorption", "top1_share",
                "btc_funding_z"]


def _sheet_history(wb):
    """The recorded state series: one row per session, appended forward.

    This is the progress record. It holds MEASUREMENTS ONLY -- no forward return, no
    outcome, no scoring of a past reading. That boundary is what keeps the workbook
    descriptive; joining these rows to future prices would make it a model.
    """
    log = _read_state_log()
    ws = wb.create_sheet("History")
    ws["A1"] = "RECORDED STATE HISTORY"
    ws["A1"].font = H1
    ws["A2"] = ("One row per session, appended forward and idempotent per date. "
                "Measurements only - deliberately NO forward returns and no scoring "
                "of past readings, because joining this to future prices would turn "
                "the workbook into a model that the PBO/DSR gate would then apply to. "
                "Rows before " + ms.CONDITIONS_REGISTERED + " (if any) are backfilled "
                "and are not evidence.")
    ws["A2"].font = MUTED
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("A2:R4")
    for r in (2, 3, 4):
        ws.row_dimensions[r].height = 26

    if log is None or log.empty:
        ws["A6"] = "no history recorded yet - it accumulates one row per build"
        return ws

    cols = ["as_of"] + [c for c in HISTORY_COLS if c in log.columns] +            ["n_fired", "fired", "crowded_long", "crowded_short"]
    _headers(ws, cols, row=6)
    r = 6
    for _, row in log.iterrows():
        r += 1
        for i, c in enumerate(cols):
            v = row.get(c)
            if isinstance(v, list):
                v = ", ".join(map(str, v)) or "(none)"
            elif isinstance(v, float) and pd.isna(v):
                v = None
            cell = ws.cell(row=r, column=i + 1, value=v)
            if c.endswith("_pct") or c in ("absorption", "top1_share"):
                cell.number_format = PCT2
            elif c in ("vix", "vrp", "rv20", "netliq_bn", "netliq_chg13w_bn",
                       "netliq_chg13w_z", "btc_funding_z", "vvix", "skew", "rrp_bn"):
                cell.number_format = NUM2
    last = r
    for name in ("vrp_pct", "skew_pct", "absorption"):
        if name in cols:
            letter = get_column_letter(cols.index(name) + 1)
            ws.conditional_formatting.add(f"{letter}7:{letter}{last}", SCALE)
    ws.freeze_panes = ws.cell(row=7, column=2)
    _autosize(ws, maxw=34)
    return ws


def _read_state_log():
    import json
    try:
        if not ms.FIRE_LOG.exists():
            return None
        rows = []
        for ln in ms.FIRE_LOG.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            d.update(d.pop("framework", {}) or {})
            d.pop("story_corr", None)      # in the log, too wide for the sheet
            for sid, sstate in (d.pop("stories", {}) or {}).items():
                d[sid] = sstate            # one column per story, so a turn gets a date
            d["n_fired"] = len(d.get("fired") or [])
            rows.append(d)
        return pd.DataFrame(rows).sort_values("as_of").tail(500) if rows else None
    except Exception as e:  # noqa: BLE001
        logger.warning(f"state log unreadable: {e}")
        return None


def _read_registry():
    try:
        from . import registry_corr as rc
        if not rc.RESULT_PATH.exists():
            return None
        return __import__("json").loads(rc.RESULT_PATH.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"registry report unreadable: {e}")
        return None


def _sheet_registry(wb):
    """How much of what this workbook prints is actually independent.

    Read-only view of `journal/exports/registry_corr.json`. Absent report -> absent
    sheet, like every other reader here.
    """
    res = _read_registry()
    if not res:
        return None
    ws = wb.create_sheet("Registry")
    ws["A1"] = "HOW MUCH OF THIS IS INDEPENDENT?"
    ws["A1"].font = H1
    ws["A2"] = (f"{res['sessions']} sessions {res['span'][0]} -> {res['span'][1]}; "
                f"the confluence tally is only meaningful if the readings differ")
    cell = ws["A3"]
    cell.value = ("LEVELS answers 'do two readings say the same thing about today', which "
                  "is what the tally depends on. CHANGES answers 'do they move together'. "
                  "A pair can look independent in one and redundant in the other, so "
                  "neither number is allowed to stand alone. Complete-case, so the "
                  "latest-starting reading truncates the window for all of them.")
    cell.fill = WARN_FILL
    cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("A3:F6")

    _headers(ws, ["matrix", "readings", "rows used", "independent signals",
                  "n for 90% var", "absorption"], row=8)
    r = 8
    for scope, label in (("market_level", "market-level"),
                         ("with_per_asset_cot", "+ per-asset COT")):
        for mode in ("level", "change"):
            d = res.get(scope, {}).get(mode) or {}
            r += 1
            ws.cell(row=r, column=1, value=f"{label}, {mode}s")
            ws.cell(row=r, column=2, value=d.get("n_columns"))
            ws.cell(row=r, column=3, value=d.get("rows_used"))
            ws.cell(row=r, column=4, value=d.get("participation_ratio"))
            ws.cell(row=r, column=5, value=d.get("n_for_90pct"))
            ws.cell(row=r, column=6, value=d.get("absorption")).number_format = PCT2
    ca = res.get("cross_asset_89")
    if ca:
        r += 1
        ws.cell(row=r, column=1, value=f"the market itself ({ca.get('n_assets')} assets)")
        ws.cell(row=r, column=2, value=ca.get("n_columns"))
        ws.cell(row=r, column=4, value=ca.get("participation_ratio"))
        ws.cell(row=r, column=5, value=ca.get("n_for_90pct"))
        ws.cell(row=r, column=6, value=ca.get("absorption")).number_format = PCT2

    ep = res.get("independent_episodes") or {}
    if ep:
        r += 2
        ws.cell(row=r, column=1, value="INDEPENDENT EPISODES, NOT SESSIONS").font = H2
        r += 1
        ws.cell(row=r, column=1, value=(
            f"median decorrelation {ep.get('median_decorrelation_days')} sessions -> about "
            f"{ep.get('episodes_at_median')} independent looks in the whole panel. Every "
            f"number on this sheet rests on roughly that much information."))
        never = ep.get("never_decorrelated_within_500d") or []
        if never:
            r += 1
            ws.cell(row=r, column=1, value=(
                f"never decorrelated inside the 500-session search cap: {', '.join(never)}"
                " - that is a cap, not a measurement")).font = MUTED

    pairs = res.get("top_redundant_pairs") or []
    if pairs:
        r += 2
        ws.cell(row=r, column=1, value="MOST REDUNDANT PAIRS (levels)").font = H2
        _headers(ws, ["reading", "reading", "|corr|"], row=r + 1)
        for pr in pairs:
            r += 1
            ws.cell(row=r + 1, column=1, value=pr["a"])
            ws.cell(row=r + 1, column=2, value=pr["b"])
            ws.cell(row=r + 1, column=3, value=pr["abs_corr"]).number_format = PCT2
        r += 1

    ws.column_dimensions["A"].width = 46
    ws.column_dimensions["B"].width = 46
    for col in "CDEF":
        ws.column_dimensions[col].width = 15
    return ws


def build_workbook(state: dict, path=None):
    """Render every sheet. Returns the path written."""
    path = path or ms.EXPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    wb.remove(wb.active)
    _sheet_today(wb, state)
    _sheet_volatility(wb, state)
    _sheet_positioning(wb, state)
    _sheet_liquidity(wb, state)
    _sheet_structure(wb, state)
    _sheet_crowding(wb, state)
    fw = state.get("framework")
    if fw is not None and not fw.empty:
        from . import asset_frame as af
        _sheet_framework(wb, fw, af.LAYERS, af.MISSING)
        _sheet_framework_detail(wb, fw, af.LAYERS, state.get("framework_focus", []))
    _sheet_history(wb)
    _sheet_conditions(wb, state)
    _sheet_stories(wb, state)
    _sheet_registry(wb)

    # PID-scoped temp name: a manual rebuild overlapping MarketStateDaily would
    # otherwise race on one shared .tmp and one run dies with WinError 32 (observed).
    # PID-scoped temp name: a manual rebuild overlapping MarketStateDaily would
    # otherwise race on one shared .tmp and one run dies with WinError 32 (observed).
    tmp = path.with_suffix(f".{os.getpid()}.xlsx.tmp")
    try:
        wb.save(tmp)
        # Excel takes an exclusive lock on an open workbook, so os.replace raises
        # WinError 5/32 whenever the user happens to have it open -- which at 11:45
        # is the NORMAL case, not an edge case. Retry briefly (covers a Norton lock
        # or a mid-save moment), then fall back to a sidecar file rather than losing
        # the run. The caller reports which path it actually wrote.
        for attempt in range(5):
            try:
                tmp.replace(path)
                return path
            except PermissionError:
                if attempt == 4:
                    break
                time.sleep(0.5)
        pending = path.with_name(f"{path.stem}.pending{path.suffix}")
        tmp.replace(pending)
        logger.warning(f"{path.name} is locked (open in Excel?) - wrote {pending.name} "
                       f"instead; close the workbook and re-run to refresh in place")
        return pending
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
