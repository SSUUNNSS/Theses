"""Download and validate the official Queensland 2024 hourly PM2.5 CSV."""

import csv
import io
from pathlib import Path
import sys

import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / 'data' / 'raw' / 'pm2-5-qld-2024.csv'
DATASET_PAGE = 'https://www.data.qld.gov.au/dataset/air-quality-monitoring-2024-grouped-by-pollutant'
CSV_URL = 'https://files.science-data.qld.gov.au/air_quality/pollutant/pm2-5-qld-2024.csv'
DATASET_API = 'https://www.data.qld.gov.au/api/3/action/package_show?id=air-quality-monitoring-2024-grouped-by-pollutant'
MINIMUM_FILE_BYTES = 10_000


def validate_csv(content: bytes) -> None:
    """Reject error pages and files without the expected wide measurement schema."""
    if len(content) < MINIMUM_FILE_BYTES:
        raise ValueError('Raw file is unexpectedly small.')
    header = next(csv.reader(io.StringIO(content.decode('utf-8-sig'))), [])
    if len(header) < 3 or header[:2] != ['Date', 'Time']:
        raise ValueError('Expected Date, Time and station columns; response may be HTML.')
    if not any('ug/m' in column or 'µg/m' in column for column in header[2:]):
        raise ValueError('Expected PM2.5 measurement units in station headers.')


def find_csv_url() -> str:
    """Discover only the named PM2.5 resource, never an arbitrary pollutant."""
    response = requests.get(DATASET_API, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not payload.get('success'):
        raise ValueError('Dataset lookup failed.')
    matches = [r['url'] for r in payload['result']['resources']
               if 'pm2-5-qld-2024.csv' in r.get('url', '').lower()]
    if len(matches) != 1:
        raise ValueError('Could not unambiguously locate the 2024 PM2.5 CSV.')
    return matches[0]


def download() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        try:
            validate_csv(OUTPUT_PATH.read_bytes())
            print(f'Using existing raw file: {OUTPUT_PATH}')
            return
        except (ValueError, UnicodeError):
            print('Existing raw file failed validation; downloading again.')
    try:
        # Prefer the confirmed direct government URL; consult metadata on failure.
        try:
            response = requests.get(CSV_URL, timeout=60)
            response.raise_for_status()
            validate_csv(response.content)
        except (requests.RequestException, ValueError):
            response = requests.get(find_csv_url(), timeout=60)
            response.raise_for_status()
            validate_csv(response.content)
        if 'text/html' in response.headers.get('content-type', '').lower():
            raise ValueError('Server returned HTML instead of CSV.')
        temporary = OUTPUT_PATH.with_suffix('.csv.part')
        temporary.write_bytes(response.content)
        temporary.replace(OUTPUT_PATH)
        print(f'Downloaded {len(response.content):,} bytes from {response.url}')
        print(f'Saved {OUTPUT_PATH}')
    except (requests.RequestException, ValueError, KeyError, OSError) as exc:
        print(f'Automatic download failed: {exc}\nManually download Queensland 2024 '
              f'hourly PM2.5 CSV from {DATASET_PAGE}\nand save it as {OUTPUT_PATH}',
              file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == '__main__':
    download()
