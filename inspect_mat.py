"""Inspect MATLAB variable names, shapes and types without printing data values."""
import argparse
from pathlib import Path
from scipy.io import loadmat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path, help='Path to a MATLAB .mat file')
    args = parser.parse_args()
    try:
        variables = loadmat(args.file)
    except (OSError, ValueError, NotImplementedError) as exc:
        parser.exit(1, f'Could not read MATLAB file: {exc}\n')
    for name, value in variables.items():
        if not name.startswith('__'):
            print(f'{name}: shape={value.shape}, dtype={value.dtype}')


if __name__ == '__main__':
    main()
