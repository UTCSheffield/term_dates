# term_dates

Builds JSON and iCal outputs for Sheffield term dates, with optional school-specific PD days, bank holiday removal, and academic week numbering.

Week exports are generated as a single Excel workbook per output folder, with one worksheet per academic year and a notes column describing closures and holiday periods.

Source data: https://www.sheffield.gov.uk/schools-childcare/school-information-term-dates

## Usage

Install the dependencies and run the build from the repository directory:

```sh
python -m pip install -r requirements.txt
python build.py
```

The build reads `config.yaml` by default. Update a school's `pd_days` there when
INSET or other training days change, then rebuild. These days are excluded from
school days but are recorded separately from school holiday periods.

Check the debug counts, warnings and generated JSON and calendars against the
source before using them. The build does not verify that all term ranges were
found: a missing term can be reported as a long holiday and omit school days.
Holiday start and end dates in JSON are inclusive.

## Using a local source

The council website may return a security challenge or otherwise be unavailable
to the build. If the page is accessible but incomplete, PDF scanning is enabled
by `scan_pdfs` in the supplied configuration. It cannot help if the page itself
is blocked. Alternatively, obtain the dates from a trusted source and prepare a
plain text file containing every term range for each academic year needed.

For example, the following format describes six term ranges. Verify the dates
against the current source before relying on them:

```text
School term dates 2026 to 2027
Term dates:
1 September 2026 until 23 October 2026
2 November 2026 until 18 December 2026
4 January 2027 until 12 February 2027
22 February 2027 until 25 March 2027
12 April 2027 until 28 May 2027
7 June 2027 until 21 July 2027
```

Use full month names and include the year on both dates. Ranges are inclusive;
enter term dates, not holiday dates. School-specific INSET days belong in
`pd_days`, not in the term ranges.

Save the text as, for example, `term_dates.txt` in the repository directory.
In `config.yaml`, change the relevant LEA's `source_url` to `term_dates.txt`.
Relative source paths are resolved from the directory in which the build runs.
With an LEA-specific `source_url`, `--source` does not override that value.

First build into a fresh directory to check the results without changing the
existing outputs:

```sh
python build.py --output-dir /tmp/term-dates-check
```

Use a new empty directory for each check. Inspect the LEA and school JSON and
calendar files there, including school days and holiday boundaries. Existing
confirmed years can be retained during a normal build, so changing the source
does not guarantee existing years are refreshed. Do not rely on a rebuild alone
to repair previously generated data. A fresh build only contains years present
in its source; replacing existing outputs with it can discard historical years.
Keep the existing outputs until all required years have been verified and a
preservation strategy is in place. Restore the council URL when it becomes
accessible again.

## YAML config (multiple schools)

Use config.yaml to run multiple LEAs and schools in one pass:

```yaml
scan_pdfs: true
debug: false
bank_holidays_source: https://www.gov.uk/bank-holidays.json
output_dir: output
leas:
  - name: Sheffield
    source_url: https://www.sheffield.gov.uk/schools-childcare/school-information-term-dates
    scan_pdfs: true
schools:
  - name: The Sheffield UTC Academy Trust
    shortname: SUTC
    lea: Sheffield
    dir: sutc
    pd_days:
      - date: 2025-09-01
        label: INSET day
      - date: 2025-10-22
        label: INSET day
```


