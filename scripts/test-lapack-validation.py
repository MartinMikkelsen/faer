#!/usr/bin/env python3
"""Small offline checks for selection, continuation and status accounting."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import sys
sys.dont_write_bytecode = True

spec = importlib.util.spec_from_file_location('validation', Path(__file__).with_name('run-lapack-validation.py'))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


class Runner(unittest.TestCase):
    def test_classification_and_empty_selection(self):
        self.assertEqual(v.classify('linalg::evd::lapack::f64::lapack_schur_cases'), 'schur')
        self.assertEqual(v.classify('linalg::evd::lapack::f64::lapack_schur_reorder_cases'), 'schur_reorder')
        self.assertEqual(v.classify('linalg::qr::lapack::c64::lapack_qr_reflector_checkers'), 'qr')
        self.assertEqual(v.classify('linalg::svd::tests::lapack_svd_f32::lapack::diagonal_families'), 'svd')
        with self.assertRaisesRegex(ValueError, 'empty'):
            v.selected_tests('')
        with self.assertRaisesRegex(ValueError, 'unclassified'):
            v.classify('linalg::evd::lapack::f64::unmapped')

    def test_continues_after_failure_timeout_and_incomplete_output(self):
        calls = []
        def fake(command, log, timeout):
            calls.append(command)
            family = log.stem
            text = 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;'
            if family == 'svd':
                text = 'test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out;'
            log.write_text('timeout' if family == 'lu' else text)
            return (101 if family == 'svd' else 124 if family == 'lu' else 0), 0.1
        groups = {family: [family + '_test'] for family in v.FAMILIES}
        with tempfile.TemporaryDirectory() as directory:
            rows = v.run_families('fake-binary', groups, Path(directory), fake)
        self.assertEqual(len(calls), len(v.FAMILIES))
        self.assertEqual(next(r for r in rows if r['family'] == 'svd')['exit_code'], 101)
        self.assertFalse(next(r for r in rows if r['family'] == 'lu')['complete'])
        self.assertEqual(rows[-1]['family'], 'lstsq')
        self.assertTrue(rows[-1]['complete'])
        self.assertTrue(all('--exact' in c and '--include-ignored' in c for c in calls))

    def test_svd_calibration_is_not_a_numeric_group(self):
        groups = {'svd': [
            'linalg::svd::tests::lapack_svd_f32::lapack::panics_and_errors_do_not_skip_later_cases',
            'linalg::svd::tests::lapack_svd_f32::lapack::diagonal_families'],
            'lu': ['linalg::lu::lapack::f32::lapack_lu_cases']}
        self.assertEqual(v.expected_records(groups), 2)

    def test_case_counts(self):
        row, = v.case_records('LAPACK_CASES family=lu dtype=f32 planned=2 attempted=2 call_failures=1 diagnostics=1 auxiliary_attempted=4')
        self.assertEqual((row['planned'], row['attempted'], row['auxiliary_attempted']), (2, 2, 4))


if __name__ == '__main__':
    unittest.main()
