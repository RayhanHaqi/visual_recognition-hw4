import unittest

from train_hw4 import TimeEstimateProgressBar


class TimeEstimateProgressBarTest(unittest.TestCase):
    def test_formats_seconds_under_one_minute(self):
        self.assertEqual(TimeEstimateProgressBar._format_seconds(0), "0:00")

    def test_formats_seconds_one_minute_five_seconds(self):
        self.assertEqual(TimeEstimateProgressBar._format_seconds(65), "1:05")

    def test_formats_seconds_one_hour_one_minute_one_second(self):
        self.assertEqual(TimeEstimateProgressBar._format_seconds(3661), "1:01:01")

    def test_formats_seconds_two_hours(self):
        self.assertEqual(TimeEstimateProgressBar._format_seconds(7200), "2:00:00")


if __name__ == "__main__":
    unittest.main()
