# term_dates

Builds JSON and iCal outputs for Sheffield term dates, with optional school-specific PD days, bank holiday removal, and academic week numbering.

Source data: https://www.sheffield.gov.uk/schools-childcare/school-information-term-dates

## Usage

Generate outputs from the Sheffield term dates URL (default) or a local text file.
If the web page content is unreliable, use --scan-pdfs to parse linked PDF calendars (requires pdfminer.six).

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
		lea: Sheffield
		dir: sutc
		pd_days:
			- date: 2025-09-01
				label: INSET day
			- date: 2025-10-22
				label: INSET day
```


