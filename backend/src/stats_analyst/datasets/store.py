"""In-memory registry of uploaded datasets."""

from __future__ import annotations

import csv
import io
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass

import pandas as pd


class DatasetNotFoundError(KeyError):
    pass


class DatasetParseError(ValueError):
    pass


@dataclass(frozen=True)
class Dataset:
    id: str
    name: str
    frame: pd.DataFrame


class DatasetStore:
    """Keeps the most recent uploads in memory, evicting the oldest past `max_datasets`."""

    def __init__(self, max_datasets: int = 20) -> None:
        self._max = max_datasets
        self._items: OrderedDict[str, Dataset] = OrderedDict()
        self._lock = threading.Lock()

    def add_csv(self, name: str, content: bytes) -> Dataset:
        try:
            frame = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
        except (pd.errors.ParserError, csv.Error, UnicodeDecodeError, ValueError) as exc:
            raise DatasetParseError(f"Could not parse CSV: {exc}") from exc
        if frame.empty or frame.shape[1] == 0:
            raise DatasetParseError("The CSV has no rows or no columns.")
        frame.columns = [str(column).strip() for column in frame.columns]
        return self.add(name, frame)

    def add(self, name: str, frame: pd.DataFrame) -> Dataset:
        dataset = Dataset(id=uuid.uuid4().hex[:12], name=name, frame=frame)
        with self._lock:
            self._items[dataset.id] = dataset
            while len(self._items) > self._max:
                self._items.popitem(last=False)
        return dataset

    def get(self, dataset_id: str) -> Dataset:
        with self._lock:
            try:
                return self._items[dataset_id]
            except KeyError as exc:
                raise DatasetNotFoundError(dataset_id) from exc
