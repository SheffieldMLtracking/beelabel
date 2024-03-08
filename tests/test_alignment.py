"""
Unit tests for the code in beelabel/alignment.py
"""

import datetime
import logging
import time
import tempfile
from pathlib import Path

import pytest

import beelabel.alignment


class TestPhotosInTimeRange:
    """
    Tests for the grab_photos_in_timerange() function
    """

    @staticmethod
    @pytest.fixture(scope='class', autouse=True)
    def photo_dir():
        with tempfile.TemporaryDirectory() as temp_dir:
            timestamp = datetime.datetime.now()

            for i in range(3):
                timestamp + datetime.timedelta(hours=i)
                # TODO revert to colons '%H:%M:%S'
                time_of_day = timestamp.time().strftime('%H-%M-%S')
                filename = f"file_{i}_{time_of_day}.np"
                path = Path(temp_dir).joinpath(filename)
                path.touch(exist_ok=True)

            yield path

    def test_grab_photos_in_timerange_str(self, photo_dir):
        # TODO generate empty files in a temp dir

        beelabel.alignment.grab_photos_in_timerange(
            path=str(photo_dir),
            starttime='08:00:00',
            endtime='08:10:00'
        )

        # TODO delete temp files

    def test_grab_photos_in_timerange_struct_time(self, photo_dir):
        beelabel.alignment.grab_photos_in_timerange(
            path=str(photo_dir),
            starttime=time.strptime('08:00:00', '%H:%M:%S'),
            endtime=time.strptime('08:10:00', '%H:%M:%S')
        )
