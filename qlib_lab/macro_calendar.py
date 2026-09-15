"""Scheduled macro-event calendar — FOMC decision days + NFP Fridays.

FOMC: embedded list of SCHEDULED decision dates 2011-2026 (the second day of
each two-day meeting), transcribed from the Fed's published calendars. 2020's
March meeting is omitted (replaced by emergency actions — Lucca-Moench drift
is documented for *scheduled* meetings only). A yearly sanity check flags any
year outside 7-9 meetings. Scraping federalreserve.gov was considered and
rejected: brittle HTML, and the schedule is published years ahead anyway —
extend FOMC_DATES each December from
https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm.

NFP: deterministic first-Friday-of-month rule (rare exceptions exist around
holidays; acceptable for event-study granularity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FOMC_DATES = [
    # 2011
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22",
    "2011-08-09", "2011-09-21", "2011-11-02", "2011-12-13",
    # 2012
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20",
    "2012-08-01", "2012-09-13", "2012-10-24", "2012-12-12",
    # 2013
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19",
    "2013-07-31", "2013-09-18", "2013-10-30", "2013-12-18",
    # 2014
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18",
    "2014-07-30", "2014-09-17", "2014-10-29", "2014-12-17",
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17",
    "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15",
    "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14",
    "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020 (scheduled only; March meeting replaced by emergency actions)
    "2020-01-29", "2020-04-29", "2020-06-10",
    "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 (published schedule)
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]


def fomc_dates() -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in FOMC_DATES))
    counts = pd.Series(idx.year).value_counts()
    bad = counts[(counts < 7) | (counts > 9)]
    if not bad.empty:
        import logging
        logging.getLogger("qlib_lab.macro_calendar").warning(
            f"FOMC calendar sanity: unusual meeting counts {bad.to_dict()}")
    return idx


def nfp_dates(start: str, end: str) -> pd.DatetimeIndex:
    """First Friday of each month in [start, end]."""
    months = pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS")
    out = []
    for m in months:
        days = pd.date_range(m, m + pd.offsets.MonthEnd(0), freq="D")
        fridays = days[days.dayofweek == 4]
        if len(fridays):
            out.append(fridays[0])
    return pd.DatetimeIndex(out)


def days_to_next_fomc(index: pd.DatetimeIndex, cap: int = 45) -> pd.Series:
    """Calendar days until the next scheduled FOMC decision, capped.

    NaN once the embedded schedule runs out (fail-soft feature, no fake zeros)."""
    fomc = fomc_dates()
    pos = np.searchsorted(fomc.values, index.values, side="left")
    out = np.full(len(index), np.nan)
    ok = pos < len(fomc)
    out[ok] = (fomc.values[pos[ok]] - index.values[ok]) / np.timedelta64(1, "D")
    return pd.Series(np.minimum(out, cap), index=index)
