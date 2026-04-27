"""Season date utilities."""

import datetime
from seasonalgeo.models.schema import Season


def get_season_dates(
    season: Season, year: int
) -> tuple[datetime.date, datetime.date]:
    """Return (start_date, end_date) for a season in a given year.

    Winter spans Dec of the given year through Feb of the next year.
    """
    month_ranges = {
        Season.SPRING: (3, 5),
        Season.SUMMER: (6, 8),
        Season.AUTUMN: (9, 11),
        Season.WINTER: (12, 2),
    }
    start_month, end_month = month_ranges[season]

    if season == Season.WINTER:
        start = datetime.date(year, 12, 1)
        end = datetime.date(year + 1, 2, 28)
        # Handle leap year
        if (year + 1) % 4 == 0 and ((year + 1) % 100 != 0 or (year + 1) % 400 == 0):
            end = datetime.date(year + 1, 2, 29)
    else:
        start = datetime.date(year, start_month, 1)
        # End of the last month
        if end_month == 12:
            end = datetime.date(year, 12, 31)
        else:
            end = datetime.date(year, end_month + 1, 1) - datetime.timedelta(days=1)

    return start, end


def get_all_season_windows(
    year_range: tuple[int, int],
    seasons: list[Season] | None = None,
) -> list[tuple[Season, int, datetime.date, datetime.date]]:
    """Generate all (season, year, start_date, end_date) combinations."""
    if seasons is None:
        seasons = list(Season)

    windows = []
    for year in range(year_range[0], year_range[1] + 1):
        for season in seasons:
            start, end = get_season_dates(season, year)
            windows.append((season, year, start, end))

    return windows
