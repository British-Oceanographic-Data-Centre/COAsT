from coast.CLIMATOLOGY import CLIMATOLOGY, Season
from datetime import date


YEARS = [2000, 2001, 2002, 2003]
PERIOD = Season.WINTER
# Date ranges for WINTER 2000 -> 2003
DATE_RANGES = [
    (date(2000, 12, 1), date(2001, 2, 28)),
    (date(2001, 12, 1), date(2002, 2, 28)),
    (date(2002, 12, 1), date(2003, 2, 28)),
    (date(2003, 12, 1), date(2004, 2, 29))
    ] 


def test_get_date_ranges():
    result = CLIMATOLOGY._get_date_ranges(YEARS, PERIOD)
    assert result == DATE_RANGES
