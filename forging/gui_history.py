"""Read partial, live worker output without coupling plotting to validation cadence."""
import csv
import math
import re


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def read_history(directory):
    rows = []
    path = directory / 'train_log.csv'
    if path.exists():
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = [row for row in csv.DictReader(stream) if finite(row.get('epoch')) is not None]
    epochs = {float(r['epoch']): finite(r.get('train_total')) for r in rows}
    batches = {}
    path = directory / 'worker.log'
    if path.exists():
        text = path.read_text(encoding='utf-8', errors='replace')
        for match in re.finditer(r'E(\d+) batch=(\d+)/(\d+) loss=([\d.eE+\-]+)', text):
            epoch, batch, count = map(int, match.group(1, 2, 3))
            value = finite(match[4])
            if value is not None and count:
                batches[epoch - 1 + batch / count] = value
        for match in re.finditer(r'E(\d+) train=([\d.eE+\-]+) validation=deferred', text):
            epochs[float(match[1])] = finite(match[2])
    return rows, sorted((x, y) for x, y in epochs.items() if y is not None), sorted(batches.items())
