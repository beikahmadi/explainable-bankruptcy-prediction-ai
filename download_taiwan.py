"""Download the UCI Taiwanese Bankruptcy Prediction dataset for local use."""
from pathlib import Path
from urllib.request import urlopen
from shutil import copyfileobj

URL = 'https://archive.ics.uci.edu/static/public/572/data.csv'
DEST = Path('taiwan_bankruptcy.csv')

def main():
    if DEST.exists():
        raise SystemExit(f'{DEST} already exists; refusing to overwrite it.')
    temp = DEST.with_suffix('.csv.part')
    try:
        with urlopen(URL, timeout=60) as response, temp.open('wb') as out:
            copyfileobj(response, out)
        temp.replace(DEST)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    print(f'Downloaded {DEST} from UCI; see README for attribution and license.')

if __name__ == '__main__': main()
