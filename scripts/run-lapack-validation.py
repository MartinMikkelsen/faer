#!/usr/bin/env python3
"""Build once, run every bounded LAPACK family, and retain failures/evidence."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

FAMILIES = (
    'common', 'svd', 'hessenberg', 'schur', 'nonsymmetric_evd',
    'schur_reorder', 'balance', 'tridiagonal_evd', 'self_adjoint_evd',
    'lu', 'cholesky', 'indefinite', 'qr', 'triangular', 'lstsq',
)
# Local measurements are documented in docs/lapack-validation.md. Hosted
# timings remain unmeasured; these bounds leave several times the local runtime.
LIMITS = dict(common=30, svd=120, hessenberg=60, schur=90,
              nonsymmetric_evd=60, schur_reorder=30, balance=30,
              tridiagonal_evd=90, self_adjoint_evd=120, lu=180,
              cholesky=60, indefinite=120, qr=90, triangular=60, lstsq=60)
CASE = re.compile(
    r'LAPACK_CASES family=(\S+) dtype=(\S+) planned=(\d+) attempted=(\d+) '
    r'call_failures=(\d+) diagnostics=(\d+)(?: auxiliary_attempted=(\d+))?'
)
SUMMARY = re.compile(r'test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;')


def classify(name):
    if name.startswith('linalg::svd::tests::'):
        return 'svd'
    leaf = name.rsplit('::', 1)[-1]
    if leaf.startswith(('lapack_common_', 'lapack_lin_')):
        return 'common'
    for family in sorted(FAMILIES, key=len, reverse=True):
        if leaf.startswith('lapack_' + family + '_'):
            return family
    if '::lapack::' in name:
        raise ValueError('unclassified validation test: ' + name)
    return None


def selected_tests(listing):
    groups = {family: [] for family in FAMILIES}
    for line in listing.splitlines():
        if line.endswith(': test'):
            name = line[:-6]
            family = classify(name)
            if family:
                groups[family].append(name)
    empty = [family for family, names in groups.items() if not names]
    if empty:
        raise ValueError('empty test selection: ' + ', '.join(empty))
    return groups


def run_command(command, log, timeout):
    start = time.perf_counter()
    with log.open('w') as stream:
        try:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
            code = result.returncode
        except subprocess.TimeoutExpired:
            stream.write('\nValidation command timed out.\n')
            code = 124
    return code, time.perf_counter() - start


def run_families(binary, groups, output, command=run_command):
    rows = []
    for family in FAMILIES:
        names = groups[family]
        if not names:
            raise ValueError('empty test selection: ' + family)
        log = output / (family + '.log')
        code, seconds = command(
            [str(binary), '--exact', *names, '--include-ignored', '--nocapture', '--test-threads=1'],
            log, LIMITS[family],
        )
        contents = log.read_text()
        summaries = SUMMARY.findall(contents)
        executed = sum(map(int, summaries[-1])) if summaries else 0
        rows.append(dict(family=family, selected=len(names), executed=executed,
                         exit_code=code, seconds=round(seconds, 3),
                         complete=executed == len(names), log=log.name))
        print(f'{family}: selected={len(names)} executed={executed} exit={code} {seconds:.2f}s', flush=True)
    return rows


def case_records(contents):
    records = []
    for family, dtype, *numbers in CASE.findall(contents):
        values = [int(n or 0) for n in numbers]
        records.append(dict(family=family, dtype=dtype, **dict(zip(
            ('planned', 'attempted', 'call_failures', 'diagnostics', 'auxiliary_attempted'), values))))
    return records


def expected_records(groups):
    expected = sum('::lapack::' in name and name.endswith(('_cases', '_parallel', '_subnormal', '_range'))
                       for family,names in groups.items() if family != 'svd' for name in names)
    svd_expected = sum('::lapack_svd_' in name and name.endswith((
            'dchkbd_15_to_19', 'diagonal_families', 'rotated_families', 'bidiagonal_family',
            'rank_one_at_recursion_splits', 'prefilled_outputs', 'layouts_and_reused_buffers', 'parallel_paths'))
            for name in groups['svd'])
    return expected + svd_expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('target/lapack-validation'))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('CARGO_PROFILE_DEV_DEBUG', '0')
    os.environ.setdefault('CARGO_INCREMENTAL', '0')
    envlog = output / 'environment.log'
    with envlog.open('w') as stream:
        for command in (['git', 'rev-parse', 'HEAD'], ['git', 'status', '--short'],
                        ['uname', '-a'], ['rustc', '-Vv']):
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    code, seconds = run_command(
        ['cargo', 'test', '-p', 't4a-faer', '--lib', '--locked', '--no-run', '--message-format=json'],
        output / 'build.log', 1200,
    )
    report = dict(build_exit_code=code, build_seconds=round(seconds, 3), families=[], cases=[])
    try:
        if code:
            raise ValueError(f'test build failed: exit {code}')
        binaries = []
        for line in (output / 'build.log').read_text().splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (item.get('reason') == 'compiler-artifact' and item.get('executable')
                    and item.get('target', {}).get('name') == 'faer'
                    and item.get('profile', {}).get('test')):
                binaries.append(item['executable'])
        if len(binaries) != 1:
            raise ValueError(f'expected one faer test binary, got {binaries}')
        binary = Path(binaries[0])
        listing = subprocess.run([str(binary), '--list', '--format', 'terse'],
                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout
        (output / 'selection.log').write_text(listing)
        groups = selected_tests(listing)
        report['families'] = run_families(binary, groups, output)
        for row in report['families']:
            report['cases'].extend(case_records((output / row['log']).read_text()))
        report['execution_seconds'] = round(sum(row['seconds'] for row in report['families']), 3)
        # Every selected numeric group must emit a record, including failing groups.
        expected = expected_records(groups)
        if len(report['cases']) != expected:
            raise ValueError(f'case records missing: expected {expected}, got {len(report["cases"])}')
        if any(row['planned'] != row['attempted'] for row in report['cases']):
            raise ValueError('incomplete configured case loop')
    except (ValueError, subprocess.SubprocessError, OSError) as error:
        report['error'] = str(error)
        print(error, flush=True)
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return int(bool(report.get('error')) or any(
        row['exit_code'] != 0 or not row['complete'] for row in report['families']))


if __name__ == '__main__':
    raise SystemExit(main())
