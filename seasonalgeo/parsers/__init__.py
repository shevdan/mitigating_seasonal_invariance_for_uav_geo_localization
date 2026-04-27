"""Dataset parsers for extracting GeoRecords from UAV geo-localization benchmarks."""

from seasonalgeo.parsers.base import BaseParser
from seasonalgeo.parsers.denseuav import DenseUAVParser
from seasonalgeo.parsers.uavvisloc import UAVVisLocParser
from seasonalgeo.parsers.university1652 import University1652Parser
from seasonalgeo.parsers.sues200 import SUES200Parser

__all__ = [
    "BaseParser",
    "DenseUAVParser",
    "UAVVisLocParser",
    "University1652Parser",
    "SUES200Parser",
]
