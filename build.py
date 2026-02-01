from __future__ import annotations

import argparse
import json
import re
import io
import urllib.request
from html.parser import HTMLParser
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import csv
from typing import Iterable

try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:  # pragma: no cover - optional dependency
    pdf_extract_text = None

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

try:
    import markdown
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False


def is_term_line(line: str) -> bool:
    line = line.strip()
    return " until " in line


def normalize_date_text(value: str) -> str:
    return " ".join(word.capitalize() if word.isalpha() else word for word in value.split())


def parse_month(value: str) -> int:
    return datetime.strptime(value.strip().capitalize(), "%B").month


def parse_date_text(value: str) -> date:
    for fmt in ("%A %d %B %Y", "%d %B %Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: {value}")


def get_term_boundaries(line: str) -> tuple[date, date]:
    line = line.strip()
    if " until " in line:
        start_text, end_text = line.split(" until ")
    elif " to " in line:
        start_text, end_text = line.split(" to ")
    else:
        raise ValueError(f"Unrecognized term range: {line}")
    start_text = normalize_date_text(start_text)
    end_text = normalize_date_text(end_text)
    end_date = parse_date_text(end_text)
    try:
        start_date = parse_date_text(start_text)
    except ValueError:
        start_date = parse_date_text(f"{start_text} {end_date.year}")
    return start_date, end_date


def get_term_dates(line: str) -> list[date]:
    start_date, end_date = get_term_boundaries(line)
    dates: list[date] = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


@dataclass
class AcademicYear:
    name: str
    provisional: bool
    term_ranges: list[tuple[date, date]]
    term_dates: list[date]
    holiday_dates: list[date]


@dataclass(frozen=True)
class PDDay:
    date: date
    label: str


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return value or "school"


def load_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is not installed")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def resolve_config_path(value: object, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else base_dir / path


def extract_academic_year_sections(text: str) -> list[dict[str, object]]:
    clean_text = re.sub(r"\s+", " ", text)
    header_re = re.compile(
        r"(School term dates|School calendar)\s+(\d{4})\s+to\s+(\d{4})(?:\s+consultation)?",
        re.IGNORECASE,
    )
    term_re = re.compile(
        r"((?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+)?\d{1,2}\s+\w+(?:\s+\d{4})?)\s+"
        r"(?:until|to)\s+"
        r"((?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+)?\d{1,2}\s+\w+\s+\d{4})",
        re.IGNORECASE,
    )
    matches = list(header_re.finditer(clean_text))
    sections: list[dict[str, object]] = []

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(clean_text)
        section_text = clean_text[start:end]
        start_year = int(match.group(2))
        end_year = int(match.group(3))
        year_name = f"{start_year}-{str(end_year)[-2:]}"
        provisional = "consultation" in match.group(0).lower()
        lower_text = section_text.lower()
        term_block_start = None
        for label in ("term dates:", "term time:"):
            index_start = lower_text.find(label)
            if index_start != -1:
                term_block_start = index_start + len(label)
                break
        term_block_end = None
        if term_block_start is not None:
            for label in ("holiday dates:", "school calendar", "school term dates"):
                index_end = lower_text.find(label, term_block_start)
                if index_end != -1:
                    term_block_end = index_end
                    break
        term_block = (
            section_text[term_block_start:term_block_end]
            if term_block_start is not None
            else section_text
        )
        term_lines = [
            f"{normalize_date_text(m.group(1))} until {normalize_date_text(m.group(2))}"
            for m in term_re.finditer(term_block)
        ]

        holiday_block = ""
        holiday_index = lower_text.find("holiday dates:")
        if holiday_index != -1:
            holiday_start = holiday_index + len("holiday dates:")
            holiday_end = None
            for label in ("school calendar", "school term dates"):
                index_end = lower_text.find(label, holiday_start)
                if index_end != -1:
                    holiday_end = index_end
                    break
            holiday_block = section_text[holiday_start:holiday_end]
        holiday_dates = extract_holiday_dates(holiday_block)
        if term_lines:
            sections.append(
                {
                    "name": year_name,
                    "provisional": provisional,
                    "term_lines": term_lines,
                    "holiday_dates": holiday_dates,
                }
            )

    return sections


def extract_term_lines(text: str) -> list[str]:
    clean_text = re.sub(r"\s+", " ", text)
    term_re = re.compile(
        r"((?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+)?\d{1,2}\s+\w+(?:\s+\d{4})?)\s+"
        r"(?:until|to)\s+"
        r"((?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+)?\d{1,2}\s+\w+\s+\d{4})",
        re.IGNORECASE,
    )
    return [
        f"{normalize_date_text(m.group(1))} until {normalize_date_text(m.group(2))}"
        for m in term_re.finditer(clean_text)
    ]


def extract_holiday_dates(text: str) -> list[date]:
    if not text:
        return []
    clean_text = re.sub(r"\s+", " ", text)
    dates: set[date] = set()

    def add_range(start_day: int, start_month: str, end_day: int, end_month: str, year: int) -> None:
        start = date(year, parse_month(start_month), start_day)
        end = date(year, parse_month(end_month), end_day)
        current = start
        while current <= end:
            dates.add(current)
            current += timedelta(days=1)

    range_re = re.compile(
        r"(\d{1,2})\s+(\w+)\s+to\s+(\d{1,2})\s+(\w+)\s+(\d{4})",
        re.IGNORECASE,
    )
    same_month_re = re.compile(
        r"(\d{1,2})\s+to\s+(\d{1,2})\s+(\w+)\s+(\d{4})",
        re.IGNORECASE,
    )
    and_re = re.compile(
        r"(\d{1,2})\s+and\s+(\d{1,2})\s+(\w+)\s+(\d{4})",
        re.IGNORECASE,
    )

    working = clean_text
    for match in range_re.finditer(clean_text):
        add_range(
            int(match.group(1)),
            match.group(2),
            int(match.group(3)),
            match.group(4),
            int(match.group(5)),
        )
    working = range_re.sub(" ", working)

    for match in same_month_re.finditer(working):
        add_range(
            int(match.group(1)),
            match.group(3),
            int(match.group(2)),
            match.group(3),
            int(match.group(4)),
        )
    working = same_month_re.sub(" ", working)

    for match in and_re.finditer(working):
        year = int(match.group(4))
        month = match.group(3)
        dates.add(date(year, parse_month(month), int(match.group(1))))
        dates.add(date(year, parse_month(month), int(match.group(2))))
    working = and_re.sub(" ", working)

    single_re = re.compile(r"\b(\d{1,2})\s+(\w+)\s+(\d{4})\b", re.IGNORECASE)
    for match in single_re.finditer(working):
        dates.add(date(int(match.group(3)), parse_month(match.group(2)), int(match.group(1))))

    return sorted(dates)


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self._parts.append(data)

    def get_text(self) -> str:
        return "\n".join(self._parts)


def extract_text_from_html(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


class _HTMLLinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._links: list[dict[str, str]] = []
        self._current_href: str | None = None
        self._current_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = None
        for key, value in attrs:
            if key.lower() == "href" and value:
                href = value
                break
        self._current_href = href
        self._current_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a":
            return
        if self._current_href:
            text = " ".join(self._current_text).strip()
            self._links.append({"href": self._current_href, "text": text})
        self._current_href = None
        self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._current_text.append(data)

    def get_links(self) -> list[dict[str, str]]:
        return self._links


def extract_pdf_links(html: str, base_url: str) -> list[str]:
    parser = _HTMLLinkExtractor()
    parser.feed(html)
    links = []
    excluded = {
        "approved_sheffield_school_calendar_2026_to_2027.pdf",
    }
    for item in parser.get_links():
        href = item.get("href", "")
        text = item.get("text", "").lower()
        if not href or ".pdf" not in href.lower():
            continue
        if any(name in href.lower() for name in excluded):
            continue
        if "school calendar" in text or "bank holiday" in text or "term dates" in text:
            full_url = urllib.request.urljoin(base_url, href)
            links.append(full_url)
    return links


def read_source_text(source: str, scan_pdfs: bool = True) -> str:
    if source.startswith("http://") or source.startswith("https://"):
        request = urllib.request.Request(
            source,
            headers={"User-Agent": "datepatterns/1.0"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            content = response.read().decode("utf-8", errors="ignore")
        text = extract_text_from_html(content)
        if scan_pdfs:
            pdf_texts: list[str] = []
            for pdf_url in extract_pdf_links(content, source):
                try:
                    pdf_texts.append(fetch_pdf_text(pdf_url))
                except Exception as exc:
                    print(f"Warning: failed to read PDF {pdf_url}: {exc}")
            if pdf_texts:
                text = text + "\n" + "\n".join(pdf_texts)
        return text
    return Path(source).read_text(encoding="utf-8")


def fetch_pdf_text(pdf_url: str) -> str:
    if pdf_extract_text is None:
        raise RuntimeError("pdfminer.six not installed")
    request = urllib.request.Request(
        pdf_url,
        headers={"User-Agent": "datepatterns/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()
    return pdf_extract_text(io.BytesIO(data))


def fetch_bank_holidays(source: str) -> set[date]:
    request = urllib.request.Request(
        source,
        headers={"User-Agent": "datepatterns/1.0"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8", errors="ignore"))
    events = payload.get("england-and-wales", {}).get("events", [])
    return {date.fromisoformat(item["date"]) for item in events if "date" in item}


def parse_academic_years(text: str) -> list[AcademicYear]:
    sections = extract_academic_year_sections(text)
    years: list[AcademicYear] = []

    if not sections:
        term_lines = extract_term_lines(text)
        if not term_lines:
            return []
        grouped: dict[str, list[str]] = {}
        for line in term_lines:
            start_date, _ = get_term_boundaries(line)
            start_year = start_date.year if start_date.month >= 8 else start_date.year - 1
            year_name = f"{start_year}-{str(start_year + 1)[-2:]}"
            grouped.setdefault(year_name, []).append(line)
        sections = [
            {"name": name, "provisional": False, "term_lines": lines, "holiday_dates": []}
            for name, lines in sorted(grouped.items())
        ]

    for section in sections:
        provisional = bool(section["provisional"])
        term_lines = [line for line in section["term_lines"] if is_term_line(str(line))]
        if not term_lines:
            continue
        term_ranges = [get_term_boundaries(line) for line in term_lines]
        try:
            start_year = int(str(section["name"]).split("-")[0])
            academic_start = date(start_year, 8, 1)
            academic_end = date(start_year + 1, 7, 31)
            term_ranges = [
                (start, end)
                for start, end in term_ranges
                if end >= academic_start and start <= academic_end
            ]
        except ValueError:
            academic_start = None
            academic_end = None
        if len(term_ranges) > 6:
            term_ranges = sorted(
                term_ranges,
                key=lambda r: (r[1] - r[0]).days,
                reverse=True,
            )[:6]
            term_ranges = sorted(term_ranges, key=lambda r: r[0])
        term_dates: list[date] = []
        for start_date, end_date in term_ranges:
            term_dates.extend(get_term_dates(f"{start_date:%A %d %B %Y} until {end_date:%A %d %B %Y}"))
        term_dates = sorted(set(term_dates))
        holiday_dates = sorted(set(section.get("holiday_dates", [])))
        years.append(
            AcademicYear(
                name=str(section["name"]),
                provisional=provisional,
                term_ranges=term_ranges,
                term_dates=term_dates,
                holiday_dates=holiday_dates,
            )
        )

    if years:
        deduped: dict[str, AcademicYear] = {}
        for year in years:
            existing = deduped.get(year.name)
            if not existing or len(year.term_dates) > len(existing.term_dates):
                deduped[year.name] = year
        years = list(deduped.values())

    return years


def calculate_school_holidays(term_ranges: list[tuple[date, date]], academic_start: date, academic_end: date) -> list[tuple[date, date, str]]:
    """Calculate school holiday periods between terms."""
    holidays: list[tuple[date, date, str]] = []
    
    if not term_ranges:
        return holidays
    
    sorted_terms = sorted(term_ranges, key=lambda r: r[0])
    
    # Holiday before first term
    first_term_start = sorted_terms[0][0]
    if academic_start < first_term_start:
        holidays.append((academic_start, first_term_start - timedelta(days=1), "Summer holiday"))
    
    # Holidays between terms
    for i in range(len(sorted_terms) - 1):
        current_end = sorted_terms[i][1]
        next_start = sorted_terms[i + 1][0]
        gap_days = (next_start - current_end).days - 1
        
        if gap_days > 0:
            holiday_start = current_end + timedelta(days=1)
            holiday_end = next_start - timedelta(days=1)
            
            # Determine holiday name based on timing
            month = holiday_start.month
            if month >= 10 and gap_days <= 7:
                label = "October half-term"
            elif month >= 12 or month <= 1:
                if gap_days > 7:
                    label = "Christmas holiday"
                else:
                    label = "Christmas break"
            elif month == 2:
                label = "February half-term"
            elif month >= 3 and month <= 4:
                if gap_days > 7:
                    label = "Easter holiday"
                else:
                    label = "Spring break"
            elif month == 5 or month == 6:
                label = "May half-term"
            else:
                label = "School holiday"
            
            holidays.append((holiday_start, holiday_end, label))
    
    # Holiday after last term
    last_term_end = sorted_terms[-1][1]
    if last_term_end < academic_end:
        holidays.append((last_term_end + timedelta(days=1), academic_end, "Summer holiday"))
    
    return holidays


def week_numbers(dates: Iterable[date]) -> dict[date, int]:
    week_map: dict[date, int] = {}
    date_week: dict[date, int] = {}
    for current in sorted(set(dates)):
        monday = current - timedelta(days=current.weekday())
        if monday not in week_map:
            week_map[monday] = len(week_map) + 1
        date_week[current] = week_map[monday]
    return date_week


def parse_pd_days(pd_days: str | None, pd_file: Path | None) -> list[PDDay]:
    values: dict[date, str] = {}
    if pd_days:
        for part in pd_days.split(","):
            part = part.strip()
            if part:
                values[date.fromisoformat(part)] = "PD day"
    if pd_file and pd_file.exists():
        with pd_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row:
                    continue
                date_text = (row.get("date") or "").strip()
                if not date_text:
                    continue
                label = (row.get("label") or "PD day").strip() or "PD day"
                values[date.fromisoformat(date_text)] = label
    return [PDDay(date=day, label=label) for day, label in sorted(values.items())]


def parse_pd_days_from_config(entries: object) -> list[PDDay]:
    if not entries:
        return []
    if not isinstance(entries, list):
        raise ValueError("pd_days must be a list of {date,label}")
    days: list[PDDay] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        date_text = str(item.get("date", "")).strip()
        if not date_text:
            continue
        label = str(item.get("label", "PD day")).strip() or "PD day"
        days.append(PDDay(date=date.fromisoformat(date_text), label=label))
    return days


def apply_bank_holidays_to_years(
    years: list[AcademicYear],
    bank_holidays: set[date],
) -> None:
    if not bank_holidays:
        return
    for year in years:
        holiday_set = set(year.holiday_dates)
        try:
            start_year = int(year.name.split("-")[0])
            academic_start = date(start_year, 8, 1)
            academic_end = date(start_year + 1, 7, 31)
            holiday_set.update(
                d for d in bank_holidays if academic_start <= d <= academic_end
            )
        except ValueError:
            holiday_set.update(bank_holidays)
        year.holiday_dates = sorted(holiday_set)


def validate_years(
    years: list[AcademicYear],
    pd_days: list[PDDay],
    debug: bool = False,
    label: str = "",
    expected_days: int | None = None,
) -> tuple[list[AcademicYear], list[str]]:
    pd_set = {pd.date for pd in pd_days}
    invalid_years: list[str] = []
    valid_years: list[AcademicYear] = []

    for year in years:
        holiday_set = set(year.holiday_dates)
        school_days = [
            d for d in year.term_dates if d not in pd_set and d not in holiday_set
        ]
        if debug:
            pd_in_year = [d for d in pd_set if d in year.term_dates]
            prefix = f"{label}: " if label else ""
            print(
                f"{prefix}{year.name}: terms={len(year.term_ranges)} term_days={len(year.term_dates)} "
                f"holidays={len(year.holiday_dates)} pd_in_term={len(pd_in_year)} "
                f"school_days={len(school_days)} provisional={year.provisional}"
            )
        if expected_days is None:
            valid_years.append(year)
        else:
            if not school_days:
                invalid_years.append(f"{year.name} (0 school days)")
            elif len(school_days) != expected_days:
                invalid_years.append(
                    f"{year.name} ({len(school_days)} school days, expected {expected_days})"
                )
            else:
                valid_years.append(year)
    return valid_years, invalid_years


def merge_academic_years(
    existing_data: dict[str, object] | None,
    new_years: list[AcademicYear],
    year_formatter: callable,
    pd_days: list[PDDay] | None = None,
) -> list[dict[str, object]]:
    """Merge existing academic years with new ones, preserving confirmed years.
    
    Regenerates confirmed years if PD days have changed.
    """
    if not existing_data:
        return [year_formatter(year) for year in new_years]
    
    existing_years = existing_data.get("academic_years", [])
    if not isinstance(existing_years, list):
        existing_years = []
    
    # Build a map of existing years by name
    existing_map: dict[str, dict[str, object]] = {}
    for year_data in existing_years:
        if isinstance(year_data, dict) and "name" in year_data:
            existing_map[str(year_data["name"])] = year_data
    
    # Calculate PD days signature for comparison
    pd_set = {pd.date for pd in (pd_days or [])}
    
    def get_year_pd_dates(year_name: str) -> set[str]:
        """Get PD dates that fall within this academic year."""
        try:
            start_year = int(year_name.split("-")[0])
            academic_start = date(start_year, 8, 1)
            academic_end = date(start_year + 1, 7, 31)
            year_pds = {
                pd.date.isoformat() for pd in (pd_days or [])
                if academic_start <= pd.date <= academic_end
            }
            return year_pds
        except (ValueError, IndexError):
            return set()
    
    # Build result list
    result_years: dict[str, dict[str, object]] = {}
    
    # Add all existing confirmed years (preserve if PD dates unchanged)
    for name, year_data in existing_map.items():
        if not year_data.get("provisional", False):
            # Check if PD dates have changed
            existing_pd_dates = set(year_data.get("pd_days", []))
            if isinstance(existing_pd_dates, list):
                existing_pd_dates = {d.get("date") if isinstance(d, dict) else d for d in existing_pd_dates}
            new_pd_dates = get_year_pd_dates(name)
            
            if existing_pd_dates == new_pd_dates:
                # PD dates unchanged, keep existing
                result_years[name] = year_data
            else:
                # PD dates changed, mark for regeneration (don't add yet)
                pass
    
    # Add or update years from new data
    for year in new_years:
        existing = existing_map.get(year.name)
        
        if year.name in result_years:
            # Already have a confirmed version with same PD dates, skip
            continue
        elif existing and not existing.get("provisional", False):
            # Was confirmed but PD dates changed, regenerate it
            result_years[year.name] = year_formatter(year)
        elif existing and existing.get("provisional", False) and not year.provisional:
            # Update provisional to confirmed
            result_years[year.name] = year_formatter(year)
        elif existing and existing.get("provisional", False) and year.provisional:
            # Both provisional, use new data (might have updates)
            result_years[year.name] = year_formatter(year)
        elif not existing:
            # New year, add it
            result_years[year.name] = year_formatter(year)
    
    # Return sorted by name
    return [result_years[name] for name in sorted(result_years.keys())]


def build_term_json(
    years: list[AcademicYear],
    school_name: str,
    pd_days: list[PDDay],
    existing_data: dict[str, object] | None = None,
) -> dict[str, object]:
    now = datetime.now(timezone.utc).isoformat()
    
    def format_year(year: AcademicYear) -> dict[str, object]:
        week_map = week_numbers(year.term_dates)
        return {
            "name": year.name,
            "provisional": year.provisional,
            "terms": [
                {"start": start.isoformat(), "end": end.isoformat()}
                for start, end in year.term_ranges
            ],
            "holidays": [d.isoformat() for d in year.holiday_dates],
            "dates": [
                {"date": d.isoformat(), "week": week_map[d]}
                for d in year.term_dates
            ],
        }
    
    merged_years = merge_academic_years(existing_data, years, format_year, pd_days)
    
    payload: dict[str, object] = {
        "generated_at": now,
        "source": "https://www.sheffield.gov.uk/schools-childcare/school-information-term-dates",
        "school": school_name,
        "pd_days": [
            {"date": pd.date.isoformat(), "label": pd.label} for pd in pd_days
        ],
        "academic_years": merged_years,
    }
    return payload


def build_school_json(
    years: list[AcademicYear],
    school_name: str,
    pd_days: list[PDDay],
    existing_data: dict[str, object] | None = None,
) -> dict[str, object]:
    now = datetime.now(timezone.utc).isoformat()
    
    pd_set = {pd.date for pd in pd_days}
    
    def format_year(year: AcademicYear) -> dict[str, object]:
        week_map = week_numbers(year.term_dates)
        bank_holidays_set = set(year.holiday_dates)
        school_dates = [
            d for d in year.term_dates if d not in pd_set and d not in bank_holidays_set
        ]
        
        # Calculate school holiday periods
        try:
            start_year = int(year.name.split("-")[0])
            academic_start = date(start_year, 8, 1)
            academic_end = date(start_year + 1, 7, 31)
        except ValueError:
            academic_start = date(2025, 8, 1)
            academic_end = date(2026, 7, 31)
        
        school_holiday_periods = calculate_school_holidays(year.term_ranges, academic_start, academic_end)
        
        return {
            "name": year.name,
            "provisional": year.provisional,
            "bank_holidays": [d.isoformat() for d in sorted(bank_holidays_set)],
            "school_holidays": [
                {
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "label": label
                }
                for start, end, label in school_holiday_periods
            ],
            "dates": [
                {"date": d.isoformat(), "week": week_map[d]}
                for d in school_dates
            ],
        }
    
    merged_years = merge_academic_years(existing_data, years, format_year, pd_days)
    
    # Calculate all_schooldays from merged years
    all_schooldays: list[str] = []
    for year_data in merged_years:
        if isinstance(year_data, dict) and "dates" in year_data:
            dates = year_data.get("dates", [])
            if isinstance(dates, list):
                for d in dates:
                    if isinstance(d, dict) and "date" in d:
                        all_schooldays.append(str(d["date"]))
    
    payload: dict[str, object] = {
        "generated_at": now,
        "source": "https://www.sheffield.gov.uk/schools-childcare/school-information-term-dates",
        "school": school_name,
        "pd_days": [
            {"date": pd.date.isoformat(), "label": pd.label} for pd in pd_days
        ],
        "academic_years": merged_years,
        "all_schooldays": sorted(all_schooldays),
    }
    return payload


def format_ics_date(value: date) -> str:
    return value.strftime("%Y%m%d")


def write_ics(
    path: Path,
    years: list[AcademicYear],
    school_name: str,
    pd_days: list[PDDay],
    label: str,
    include_pd: bool,
    shortname: str | None = None,
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//datepatterns//term dates//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
    ]

    pd_set = {pd.date for pd in pd_days}
    
    # Use shortname if provided for summary, otherwise use full name
    display_name = shortname if shortname else school_name
    
    for year in years:
        week_map = week_numbers(year.term_dates)
        holiday_set = set(year.holiday_dates)
        if include_pd:
            dates = year.term_dates
        else:
            dates = [
                d
                for d in year.term_dates
                if d not in pd_set and d not in holiday_set
            ]
        
        # Create day counter for school days (only for label == "School day")
        day_counter = {day: idx + 1 for idx, day in enumerate(dates)} if label == "School day" else {}
        
        for day in dates:
            week = week_map[day]
            
            # Format summary based on whether we're counting days
            if label == "School day" and day in day_counter:
                day_num = day_counter[day]
                event_label = f"Day {day_num} (Week {week})"
            else:
                event_label = f"{label} (Week {week})"
            
            if year.provisional:
                event_label = f"PROVISIONAL {event_label}"
            
            if display_name:
                summary = f"{display_name}: {event_label}"
            else:
                summary = event_label
            
            status = "TENTATIVE" if year.provisional else "CONFIRMED"
            lines.extend(
                [
                    "BEGIN:VEVENT",
                    f"UID:{label.lower().replace(' ', '-')}-{day:%Y%m%d}-{year.name}@datepatterns",
                    f"DTSTAMP:{now}",
                    f"DTSTART;VALUE=DATE:{format_ics_date(day)}",
                    f"DTEND;VALUE=DATE:{format_ics_date(day + timedelta(days=1))}",
                    f"SUMMARY:{summary}",
                    f"DESCRIPTION:Academic year {year.name}",
                    f"STATUS:{status}",
                    "END:VEVENT",
                ]
            )

    lines.append("END:VCALENDAR")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_school_holidays_ics(
    path: Path,
    years: list[AcademicYear],
    school_name: str,
    pd_days: list[PDDay],
    bank_holidays: set[date] | None = None,
    shortname: str | None = None,
) -> None:
    """Write iCalendar file with school holiday periods, bank holidays, and PD days."""
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//datepatterns//school holidays//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
    ]

    bank_holidays_set = bank_holidays or set()
    display_name = shortname if shortname else school_name

    for year in years:
        # Calculate school holiday periods
        try:
            start_year = int(year.name.split("-")[0])
            academic_start = date(start_year, 8, 1)
            academic_end = date(start_year + 1, 7, 31)
        except ValueError:
            academic_start = date(2025, 8, 1)
            academic_end = date(2026, 7, 31)
        
        school_holiday_periods = calculate_school_holidays(year.term_ranges, academic_start, academic_end)
        status = "TENTATIVE" if year.provisional else "CONFIRMED"
        
        # Add school holiday periods as multi-day events
        for start, end, label in school_holiday_periods:
            summary = label
            if year.provisional:
                summary = f"PROVISIONAL {summary}"
            if display_name:
                summary = f"{display_name}: {summary}"
            
            lines.extend(
                [
                    "BEGIN:VEVENT",
                    f"UID:holiday-{start:%Y%m%d}-{end:%Y%m%d}-{year.name}@datepatterns",
                    f"DTSTAMP:{now}",
                    f"DTSTART;VALUE=DATE:{format_ics_date(start)}",
                    f"DTEND;VALUE=DATE:{format_ics_date(end + timedelta(days=1))}",
                    f"SUMMARY:{summary}",
                    f"STATUS:{status}",
                    "END:VEVENT",
                ]
            )
        
        # Add bank holidays as single-day events
        for bank_holiday in sorted(bank_holidays_set):
            if academic_start <= bank_holiday <= academic_end:
                summary = "Bank holiday"
                if year.provisional:
                    summary = f"PROVISIONAL {summary}"
                if display_name:
                    summary = f"{display_name}: {summary}"
                
                lines.extend(
                    [
                        "BEGIN:VEVENT",
                        f"UID:bank-{bank_holiday:%Y%m%d}-{year.name}@datepatterns",
                        f"DTSTAMP:{now}",
                        f"DTSTART;VALUE=DATE:{format_ics_date(bank_holiday)}",
                        f"DTEND;VALUE=DATE:{format_ics_date(bank_holiday + timedelta(days=1))}",
                        f"SUMMARY:{summary}",
                        f"STATUS:{status}",
                        "END:VEVENT",
                    ]
                )
        
        # Add PD days that fall within this academic year
        for pd in pd_days:
            if academic_start <= pd.date <= academic_end:
                summary = pd.label
                if year.provisional:
                    summary = f"PROVISIONAL {summary}"
                if display_name:
                    summary = f"{display_name}: {summary}"
                
                lines.extend(
                    [
                        "BEGIN:VEVENT",
                        f"UID:pd-{pd.date:%Y%m%d}-{year.name}@datepatterns",
                        f"DTSTAMP:{now}",
                        f"DTSTART;VALUE=DATE:{format_ics_date(pd.date)}",
                        f"DTEND;VALUE=DATE:{format_ics_date(pd.date + timedelta(days=1))}",
                        f"SUMMARY:{summary}",
                        f"STATUS:{status}",
                        "END:VEVENT",
                    ]
                )

    lines.append("END:VCALENDAR")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def json_data_changed(new_data: dict[str, object], existing_data: dict[str, object] | None) -> bool:
    """Check if JSON data has changed, ignoring generated_at timestamp."""
    if existing_data is None:
        return True
    
    # Create copies without generated_at for comparison
    new_copy = {k: v for k, v in new_data.items() if k != "generated_at"}
    existing_copy = {k: v for k, v in existing_data.items() if k != "generated_at"}
    
    return new_copy != existing_copy


def build_index_html(output_dir: Path, readme_path: Path, config: dict[str, object] | None = None) -> None:
    """Build an index.html file with README content and links to generated files."""
    # Build mapping of directory names to full school/LEA names
    name_mapping = {}
    if config:
        # Add schools
        schools_config = config.get("schools", [])
        if isinstance(schools_config, list):
            for school in schools_config:
                if isinstance(school, dict):
                    name = school.get("name", "")
                    dir_name = school.get("dir", "")
                    if name and dir_name:
                        name_mapping[str(dir_name)] = str(name)
        
        # Add LEAs
        leas_config = config.get("leas", [])
        if isinstance(leas_config, list):
            for lea in leas_config:
                if isinstance(lea, dict):
                    name = lea.get("name", "")
                    if name:
                        name_mapping[slugify(str(name))] = str(name)
    
    readme_html = ""
    if readme_path.exists():
        readme_content = readme_path.read_text(encoding="utf-8")
        if HAS_MARKDOWN:
            # Convert markdown to HTML
            md = markdown.Markdown(extensions=['fenced_code', 'tables', 'nl2br'])
            readme_html = md.convert(readme_content)
        else:
            # Fallback: wrap in pre tag
            readme_html = f"<pre>{readme_content}</pre>"
    
    # Collect all generated files grouped by school/LEA name
    files_by_school = {}
    
    # Scan LEAs directory
    leas_dir = output_dir / "leas"
    if leas_dir.exists():
        for lea_dir in sorted(leas_dir.iterdir()):
            if lea_dir.is_dir():
                # Use mapping to get full name, fallback to formatted dir name
                lea_name = name_mapping.get(lea_dir.name, lea_dir.name.replace("-", " ").title())
                if lea_name not in files_by_school:
                    files_by_school[lea_name] = []
                for file in sorted(lea_dir.iterdir()):
                    if file.is_file():
                        rel_path = file.relative_to(output_dir)
                        files_by_school[lea_name].append((str(rel_path), file.name, file.stat().st_size))
    
    # Scan schools directory
    schools_dir = output_dir / "schools"
    if schools_dir.exists():
        for school_dir in sorted(schools_dir.iterdir()):
            if school_dir.is_dir():
                # Use mapping to get full name, fallback to formatted dir name
                school_name = name_mapping.get(school_dir.name, school_dir.name.replace("-", " ").title())
                if school_name not in files_by_school:
                    files_by_school[school_name] = []
                for file in sorted(school_dir.iterdir()):
                    if file.is_file():
                        rel_path = file.relative_to(output_dir)
                        files_by_school[school_name].append((str(rel_path), file.name, file.stat().st_size))
    
    # Root outputs (without school grouping)
    root_files = []
    for file in sorted(output_dir.iterdir()):
        if file.is_file() and file.name != "index.html":
            rel_path = file.relative_to(output_dir)
            root_files.append((str(rel_path), file.name, file.stat().st_size))
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Term Dates - Generated Outputs</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 8px;
        }}
        h3 {{
            color: #555;
            margin-top: 20px;
        }}
        .readme {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
            overflow-x: auto;
        }}
        .readme h1 {{
            margin-top: 0;
            font-size: 1.8em;
        }}
        .readme h2 {{
            margin-top: 1.5em;
            font-size: 1.4em;
        }}
        .readme h3 {{
            margin-top: 1.2em;
            font-size: 1.2em;
        }}
        .readme pre {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        .readme code {{
            background: #e8f4f8;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        .readme pre code {{
            background: transparent;
            padding: 0;
        }}
        .readme ul, .readme ol {{
            padding-left: 30px;
        }}
        .readme li {{
            margin: 5px 0;
        }}
        .readme a {{
            color: #3498db;
            text-decoration: underline;
        }}
        .file-list {{
            list-style: none;
            padding: 0;
        }}
        .file-list li {{
            padding: 10px;
            margin: 5px 0;
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 4px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .file-list li:hover {{
            background: #f0f7ff;
            border-color: #3498db;
        }}
        .file-list a {{
            color: #3498db;
            text-decoration: none;
            font-weight: 500;
            flex-grow: 1;
        }}
        .file-list a:hover {{
            text-decoration: underline;
        }}
        .file-size {{
            color: #888;
            font-size: 0.9em;
            margin-left: 10px;
        }}
        .category {{
            margin: 30px 0;
        }}
        .timestamp {{
            color: #888;
            font-size: 0.9em;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
        }}
        code {{
            background: #e8f4f8;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    <h1>📅 Term Dates - Generated Outputs</h1>
    
    <div class="readme">
{readme_html}
    </div>
    
    <h2>Generated Files</h2>
"""
    
    # Add file listings by school
    for school_name in sorted(files_by_school.keys()):
        files = files_by_school[school_name]
        if files:
            html += f"""
    <div class="category">
        <h3>{school_name}</h3>
        <ul class="file-list">
"""
            for rel_path, filename, size in files:
                size_kb = size / 1024
                size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
                
                # Use webcal:// for ICS files, otherwise use regular https:// link
                if filename.endswith('.ics'):
                    # Convert to webcal:// URL for calendar subscription
                    link_url = f"webcal://utcsheffield.github.io/term_dates/{rel_path}"
                    link_text = f"{rel_path} (Subscribe)"
                else:
                    link_url = rel_path
                    link_text = rel_path
                
                html += f"""            <li>
                <a href="{link_url}">{link_text}</a>
                <span class="file-size">{size_str}</span>
            </li>
"""
            html += """        </ul>
    </div>
"""
    
    # Add root outputs section if there are any
    if root_files:
        html += f"""
    <div class="category">
        <h3>General</h3>
        <ul class="file-list">
"""
        for rel_path, filename, size in root_files:
            size_kb = size / 1024
            size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
            
            # Use webcal:// for ICS files, otherwise use regular https:// link
            if filename.endswith('.ics'):
                # Convert to webcal:// URL for calendar subscription
                link_url = f"webcal://utcsheffield.github.io/term_dates/{rel_path}"
                link_text = f"{rel_path} (Subscribe)"
            else:
                link_url = rel_path
                link_text = rel_path
            
            html += f"""            <li>
                <a href="{link_url}">{link_text}</a>
                <span class="file-size">{size_str}</span>
            </li>
"""
        html += """        </ul>
    </div>
"""
    
    # Add timestamp
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    html += f"""
</body>
</html>
"""
    
    # Write index.html
    index_path = output_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build JSON and iCal term/school dates for Sheffield term dates."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="YAML config file (optional)",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Source URL or path to term_dates.txt",
    )
    parser.add_argument(
        "--scan-pdfs",
        action="store_true",
        help="Scan linked PDFs for term dates and bank holidays",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write outputs",
    )
    parser.add_argument(
        "--school-name",
        default=None,
        help="School name to include in school outputs",
    )
    parser.add_argument(
        "--pd-days",
        help="Comma-separated ISO dates for PD days (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--pd-days-file",
        type=Path,
        default=None,
        help="CSV file with PD days (columns: date,label)",
    )
    parser.add_argument(
        "--bank-holidays-source",
        default=None,
        help="Bank holiday JSON source (default: GOV.UK)",
    )
    parser.add_argument(
        "--skip-bank-holidays",
        action="store_true",
        help="Do not apply GOV.UK bank holidays",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print term/holiday/PD counts per academic year",
    )
    args = parser.parse_args()

    config = load_config(args.config) if args.config else {}
    config_base = args.config.parent if args.config else Path.cwd()
    source_value = args.source or config.get("source_url", "https://www.sheffield.gov.uk/schools-childcare/school-information-term-dates")
    source = str(source_value)
    scan_pdfs = args.scan_pdfs or bool(config.get("scan_pdfs", False))
    output_dir = args.output_dir or resolve_config_path(
        config.get("output_dir", Path(__file__).with_name("output")),
        config_base,
    )
    general_name = "Sheffield"
    school_name = args.school_name or config.get("school_name", "The Sheffield UTC Academy Trust")
    pd_days_file = args.pd_days_file or resolve_config_path(
        config.get("pd_days_file", Path(__file__).with_name("pd_days.csv")),
        config_base,
    )
    bank_source = args.bank_holidays_source or config.get(
        "bank_holidays_source", "https://www.gov.uk/bank-holidays.json"
    )
    skip_bank_holidays = args.skip_bank_holidays or bool(
        config.get("skip_bank_holidays", False)
    )
    debug = args.debug or bool(config.get("debug", False))

    bank_holidays: set[date] = set()
    if not skip_bank_holidays:
        try:
            bank_holidays = fetch_bank_holidays(bank_source)
        except Exception as exc:
            print(f"Warning: failed to load bank holidays: {exc}")

    leas_config = config.get("leas", [])
    if isinstance(leas_config, list) and leas_config:
        lea_years_map: dict[str, list[AcademicYear]] = {}
        invalid_leas: list[str] = []

        for lea in leas_config:
            if not isinstance(lea, dict):
                continue
            lea_name = str(lea.get("name", "Sheffield")).strip() or "Sheffield"
            lea_source = lea.get("source_url", source)
            lea_scan = bool(lea.get("scan_pdfs", scan_pdfs))
            lea_text = read_source_text(str(lea_source), scan_pdfs=lea_scan)
            lea_years = parse_academic_years(lea_text)
            if not lea_years:
                invalid_leas.append(f"{lea_name} (no academic years)")
                continue
            apply_bank_holidays_to_years(lea_years, bank_holidays)
            valid_years, invalid_years = validate_years(
                lea_years,
                [],
                debug=debug,
                label=f"LEA {lea_name}",
                expected_days=None,
            )
            if invalid_years:
                invalid_leas.append(f"{lea_name}: " + ", ".join(sorted(invalid_years)))
                continue
            lea_years_map[lea_name] = valid_years

        if invalid_leas:
            print("Warning: date totals invalid for LEAs:")
            for detail in invalid_leas:
                print(f"- {detail}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        leas_dir = output_dir / "leas"
        leas_dir.mkdir(parents=True, exist_ok=True)

        for lea_name, valid_years in lea_years_map.items():
            lea_dir = leas_dir / slugify(lea_name)
            lea_dir.mkdir(parents=True, exist_ok=True)
            
            # Load existing data if present
            lea_json_path = lea_dir / "term_dates.json"
            existing_lea_data = None
            if lea_json_path.exists():
                try:
                    existing_lea_data = json.loads(lea_json_path.read_text(encoding="utf-8"))
                except Exception:
                    pass  # If we can't parse it, start fresh
            
            lea_json = build_school_json(valid_years, lea_name, [], existing_lea_data)
            
            # Only write ICS files if JSON has changed
            if json_data_changed(lea_json, existing_lea_data):
                (lea_dir / "term_dates.json").write_text(
                    json.dumps(lea_json, indent=2) + "\n", encoding="utf-8"
                )
                write_ics(
                    lea_dir / "term_dates.ics",
                    valid_years,
                    lea_name,
                    [],
                    label="Term day",
                    include_pd=False,
                    shortname=None,
                )
                write_school_holidays_ics(
                    lea_dir / "school_holidays.ics",
                    valid_years,
                    lea_name,
                    [],
                    bank_holidays=bank_holidays,
                    shortname=None,
                )

        schools_config = config.get("schools", [])
        if isinstance(schools_config, list) and schools_config:
            invalid_by_school: list[str] = []
            for school in schools_config:
                if not isinstance(school, dict):
                    continue
                name = str(school.get("name", "School")).strip() or "School"
                lea_name = str(school.get("lea", "Sheffield")).strip() or "Sheffield"
                base_years = lea_years_map.get(lea_name)
                if not base_years:
                    invalid_by_school.append(f"{name}: unknown LEA {lea_name}")
                    continue
                pd_file_value = school.get("pd_days_file", pd_days_file)
                pd_file = resolve_config_path(pd_file_value, config_base) if pd_file_value else None
                pd_days = parse_pd_days(None, pd_file) if pd_file else []
                pd_days.extend(parse_pd_days_from_config(school.get("pd_days")))
                valid_years, invalid_years = validate_years(
                    base_years,
                    pd_days,
                    debug=debug,
                    label=name,
                    expected_days=None,
                )
                if invalid_years:
                    invalid_by_school.append(
                        f"{name}: " + ", ".join(sorted(invalid_years))
                    )
                    continue
                schools_dir = output_dir / "schools"
                schools_dir.mkdir(parents=True, exist_ok=True)
                dir_name = str(school.get("dir", "")).strip()
                school_dir = schools_dir / (dir_name or slugify(name))
                school_dir.mkdir(parents=True, exist_ok=True)
                
                # Load existing data if present
                school_json_path = school_dir / "school_dates.json"
                existing_school_data = None
                if school_json_path.exists():
                    try:
                        existing_school_data = json.loads(school_json_path.read_text(encoding="utf-8"))
                    except Exception:
                        pass  # If we can't parse it, start fresh
                
                school_json = build_school_json(valid_years, name, pd_days, existing_school_data)
                
                # Only write ICS files if JSON has changed
                if json_data_changed(school_json, existing_school_data):
                    (school_dir / "school_dates.json").write_text(
                        json.dumps(school_json, indent=2) + "\n", encoding="utf-8"
                    )
                    # Get shortname from config if available
                    school_shortname = str(school.get("shortname", "")).strip() or None
                    write_ics(
                        school_dir / "school_dates.ics",
                        valid_years,
                        name,
                        pd_days,
                        label="School day",
                        include_pd=False,
                        shortname=school_shortname,
                    )
                    write_school_holidays_ics(
                        school_dir / "school_holidays.ics",
                        valid_years,
                        name,
                        pd_days,
                        bank_holidays=bank_holidays,
                        shortname=school_shortname,
                    )

            if invalid_by_school:
                print("Warning: date totals invalid for schools:")
                for detail in invalid_by_school:
                    print(f"- {detail}")

        # Build index.html
        readme_path = Path(__file__).parent / "README.md"
        build_index_html(output_dir, readme_path, config)

        print(f"Wrote outputs to {output_dir}")
        return

    source_text = read_source_text(str(source), scan_pdfs=scan_pdfs)
    years = parse_academic_years(source_text)

    if not years:
        print("Warning: no academic years parsed from source.")
        return

    apply_bank_holidays_to_years(years, bank_holidays)

    pd_days = parse_pd_days(args.pd_days, pd_days_file)
    pd_days.extend(parse_pd_days_from_config(config.get("pd_days")))

    valid_years, invalid_years = validate_years(
        years,
        pd_days,
        debug=debug,
        expected_days=None,
    )
    if invalid_years:
        print("Warning: date totals invalid:")
        for detail in invalid_years:
            print(f"- {detail}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing data if present
    term_json_path = output_dir / "term_dates.json"
    existing_term_data = None
    if term_json_path.exists():
        try:
            existing_term_data = json.loads(term_json_path.read_text(encoding="utf-8"))
        except Exception:
            pass  # If we can't parse it, start fresh
    
    school_json_path = output_dir / "school_dates.json"
    existing_school_data = None
    if school_json_path.exists():
        try:
            existing_school_data = json.loads(school_json_path.read_text(encoding="utf-8"))
        except Exception:
            pass  # If we can't parse it, start fresh

    term_json = build_term_json(valid_years, general_name, [], existing_term_data)
    school_json = build_school_json(valid_years, school_name, pd_days, existing_school_data)

    # Only write ICS files if JSON has changed
    if json_data_changed(term_json, existing_term_data):
        (output_dir / "term_dates.json").write_text(
            json.dumps(term_json, indent=2) + "\n", encoding="utf-8"
        )
        write_ics(
            output_dir / "term_dates.ics",
            valid_years,
            general_name,
            [],
            label="Term day",
            include_pd=True,
            shortname=None,
        )
    
    if json_data_changed(school_json, existing_school_data):
        (output_dir / "school_dates.json").write_text(
            json.dumps(school_json, indent=2) + "\n", encoding="utf-8"
        )
        write_ics(
            output_dir / "school_dates.ics",
            valid_years,
            school_name,
            pd_days,
            label="School day",
            include_pd=False,
            shortname=None,
        )
        write_school_holidays_ics(
            output_dir / "school_holidays.ics",
            valid_years,
            school_name,
            pd_days,
            bank_holidays=bank_holidays,
            shortname=None,
        )

    # Build index.html
    readme_path = Path(__file__).parent / "README.md"
    build_index_html(output_dir, readme_path, config)

    if pd_days and len(pd_days) != 5:
        print(f"Warning: expected 5 PD days, got {len(pd_days)}")
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()