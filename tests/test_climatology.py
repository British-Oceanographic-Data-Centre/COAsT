from coast.CLIMATOLOGY import CLIMATOLOGY, Season
from datetime import date


YEARS = [2000, 2001]
PERIOD = Season.ALL
# Date ranges for WINTER 2000 -> 2003
DATE_RANGES = [
    (date(2000, 3, 1), date(2000, 5, 31)),
    (date(2000, 6, 1), date(2000, 9, 30)),
    (date(2000, 10, 1), date(2000, 11, 30)),
    (date(2000, 12, 1), date(2001, 2, 28)),
    (date(2001, 3, 1), date(2001, 5, 31)),
    (date(2001, 6, 1), date(2001, 9, 30)),
    (date(2001, 10, 1), date(2001, 11, 30)),
    (date(2001, 12, 1), date(2002, 2, 28))
]


def test_get_date_ranges():
    result = CLIMATOLOGY._get_date_ranges(YEARS, PERIOD)
    assert result == DATE_RANGES
