"""Regression checks for hourly semantics and strict evaluator alignment."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import prepare_data
import evaluate
import fetch_data


class BenchmarkTests(unittest.TestCase):
    def test_midnight_and_malformed(self):
        frame = pd.DataFrame({'Date': ['31/01/2024', 'bad', '01/02/2024'],
                              'Time': ['24:00', '12:00', '01:00:00']})
        values = prepare_data.parse_timestamp(frame)
        self.assertEqual(values[0], pd.Timestamp('2024-02-01'))
        self.assertTrue(pd.isna(values[1]))
        self.assertEqual(values[2], pd.Timestamp('2024-02-01 01:00'))

    def test_gaps_and_station_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.csv'
            pd.DataFrame({'Date': ['01/01/2024'] * 4,
                          'Time': ['00:00', '01:00', '03:00', '04:00'],
                          'A (ug/m^3)': [1, 2, 4, 5],
                          'B (ug/m^3)': [10, 20, 40, 50]}).to_csv(path, index=False)
            result = prepare_data.build_dataset(path).set_index(['station', 'timestamp'])
            self.assertNotIn(('A', pd.Timestamp('2024-01-01 01:00')), result.index)
            gap = result.loc[('A', pd.Timestamp('2024-01-01 02:00'))]
            self.assertTrue(pd.isna(gap.pm25_current))
            self.assertEqual(gap.pm25_lag_1h, 2)
            self.assertEqual(gap.target_pm25_next_hour, 4)
            row = result.loc[('B', pd.Timestamp('2024-01-01 03:00'))]
            self.assertTrue(pd.isna(row.pm25_lag_1h))
            self.assertEqual(row.pm25_rollmean_24h, 15)

    def test_evaluator_valid_and_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / 'labels.csv'
            preds = Path(directory) / 'predictions.csv'
            labels.write_text('row_id,target_pm25_next_hour\na,1\nb,3\n')
            cases = [('b,5\na,1\n', False), ('a,1\n', True),
                     ('a,1\nb,3\nc,4\n', True), ('a,1\na,3\n', True),
                     ('a,inf\nb,3\n', True), ('a,no\nb,3\n', True),
                     (',1\nb,3\n', True), ('', True)]
            for rows, invalid in cases:
                with self.subTest(rows=rows):
                    preds.write_text('row_id,prediction\n' + rows)
                    with patch.object(sys, 'argv', ['evaluate', '--predictions', str(preds), '--labels', str(labels)]):
                        if invalid:
                            with self.assertRaises(ValueError):
                                evaluate.main()
                        else:
                            output = io.StringIO()
                            with contextlib.redirect_stdout(output):
                                evaluate.main()
                            self.assertIn('RMSE: 1.414214', output.getvalue())

    def test_html_rejected(self):
        with self.assertRaises(ValueError):
            fetch_data.validate_csv(b'<html>' + b'x' * 20000)


if __name__ == '__main__':
    unittest.main()
