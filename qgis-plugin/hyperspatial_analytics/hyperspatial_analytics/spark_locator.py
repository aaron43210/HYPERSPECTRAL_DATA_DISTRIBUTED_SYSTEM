# -*- coding: utf-8 -*-
"""Locate ``spark-submit`` (and a usable JAVA_HOME) at run time, and build the
environment the Sedona jobs need.

Why this exists: on macOS a QGIS.app launched from Finder does NOT inherit the
login shell's PATH, so ``shutil.which('spark-submit')`` returns None and the
analytics button silently fails even though Spark is installed. This mirrors the
processing plugin's engine ``Browse...`` pattern: resolve a path explicitly, in
priority order, and remember the user's choice in QSettings.

Resolution order for spark-submit:
    1. An explicit path saved in QSettings('hyperspatial_analytics/spark_submit')
       (set via the dialog's Spark "Browse..." button).
    2. ``$SPARK_HOME/bin/spark-submit``.
    3. ``spark-submit`` on the process PATH (shutil.which).
    4. Common install locations (Homebrew, the active conda env, /usr/local).
    5. The user's login shell, which sources their profile (conda, SDKMAN, etc.).

``spark_env()`` returns an environment with HADOOP_CONF_DIR unset (a populated
value crashes the JVM on macOS), JAVA_HOME filled in when resolvable, and the
spark-submit / java / conda bin dirs prepended to PATH.
"""
import os
import sys
import shutil
import subprocess  # nosec B404 - used only to query the user's own login shell
import contextlib

from qgis.PyQt.QtCore import QSettings

SPARK_SETTINGS_KEY = 'hyperspatial_analytics/spark_submit'


def _exe_name():
    """Return the spark-submit executable name for the host OS."""
    return 'spark-submit.cmd' if sys.platform.startswith('win') else 'spark-submit'


def _login_shell_which(exe):
    """Ask the user's login shell where `exe` is (POSIX only).

    A login shell sources the user's profile, so it sees PATH entries added by
    conda/SDKMAN/Homebrew that a Finder-launched GUI process never gets.
    """
    if sys.platform.startswith('win'):
        return None
    shell = os.environ.get('SHELL') or '/bin/bash'
    with contextlib.suppress(Exception):
        out = subprocess.run(  # nosec B603 - fixed argv, no shell=True, user's own shell
            [shell, '-l', '-c', 'command -v %s' % exe],
            capture_output=True, text=True, timeout=15,
        )
        cand = out.stdout.strip().splitlines()
        if cand and os.path.isfile(cand[0]):
            return cand[0]
    return None


def _candidate_dirs():
    """Common directories that hold spark-submit, most specific first."""
    dirs = []
    spark_home = os.environ.get('SPARK_HOME')
    if spark_home:
        dirs.append(os.path.join(spark_home, 'bin'))
    # Active conda/virtualenv (pip-installed pyspark drops spark-submit here).
    for prefix_var in ('CONDA_PREFIX', 'VIRTUAL_ENV'):
        p = os.environ.get(prefix_var)
        if p:
            dirs.append(os.path.join(p, 'bin'))
    dirs.append(os.path.join(sys.prefix, 'bin'))
    # Homebrew (Apple Silicon + Intel) and the classic /usr/local install.
    dirs += ['/opt/homebrew/bin', '/usr/local/bin', '/opt/spark/bin']
    return dirs


def resolve_spark_submit():
    """Find ``spark-submit``.

    Returns a 3-tuple ``(path, source, error)``:
      * success -> (absolute_path, human_source_label, None)
      * failure -> (None, None, help_message)
    """
    exe = _exe_name()

    # (1) Explicit path saved by the user via the Spark "Browse..." button.
    with contextlib.suppress(Exception):
        saved = QSettings().value(SPARK_SETTINGS_KEY)
        if saved:
            saved = str(saved)
            if os.path.isfile(saved):
                return saved, 'configured path (QSettings)', None

    # (2) $SPARK_HOME/bin.
    spark_home = os.environ.get('SPARK_HOME')
    if spark_home:
        cand = os.path.join(spark_home, 'bin', exe)
        if os.path.isfile(cand):
            return cand, '$SPARK_HOME', None

    # (3) Process PATH.
    found = shutil.which(exe)
    if found:
        return found, 'process PATH', None

    # (4) Common install locations.
    for d in _candidate_dirs():
        cand = os.path.join(d, exe)
        if os.path.isfile(cand):
            return cand, d, None

    # (5) Login shell (sources the profile; sees conda/SDKMAN PATH).
    cand = _login_shell_which(exe)
    if cand:
        return cand, 'login shell PATH', None

    error = (
        "spark-submit not found.\n\n"
        "To run Sedona analytics, do ONE of the following:\n"
        "  1. Click 'Browse...' on the Spark row and select spark-submit "
        "(spark-submit.cmd on Windows); the choice is remembered.\n"
        "  2. Set SPARK_HOME, or put spark-submit on your PATH.\n"
        "  3. Install the conda env: conda env create -f environment.yml, then "
        "launch QGIS from that activated environment."
    )
    return None, None, error


def _detect_java_home():
    """Best-effort JAVA_HOME, preferring a Spark 3.4-supported JDK (8/11/17).

    On macOS we ask ``/usr/libexec/java_home -v`` for 17 then 11 before falling
    back to an existing JAVA_HOME or the default JVM - this avoids silently
    handing Spark 3.4 a Java 21/25 it cannot run on.
    """
    if sys.platform == 'darwin':
        for ver in ('17', '11', '1.8'):
            with contextlib.suppress(Exception):
                out = subprocess.run(  # nosec B603 - fixed system tool, no shell
                    ['/usr/libexec/java_home', '-v', ver],
                    capture_output=True, text=True, timeout=10,
                )
                cand = out.stdout.strip()
                if cand and os.path.isdir(cand):
                    return cand
    jh = os.environ.get('JAVA_HOME')
    if jh and os.path.isdir(jh):
        return jh
    if sys.platform == 'darwin':
        with contextlib.suppress(Exception):
            out = subprocess.run(  # nosec B603 - fixed system tool, no shell
                ['/usr/libexec/java_home'], capture_output=True, text=True, timeout=10,
            )
            cand = out.stdout.strip()
            if cand and os.path.isdir(cand):
                return cand
    return None


def spark_env(spark_submit_path=None):
    """Return an ``os.environ`` copy ready for a Sedona spark-submit child.

    * HADOOP_CONF_DIR = "" (critical: a populated value crashes the JVM on macOS).
    * JAVA_HOME filled in when resolvable, with $JAVA_HOME/bin on PATH.
    * The spark-submit dir and the active conda/env bin prepended to PATH so the
      child finds python/pyspark even under a Finder-launched QGIS.
    Fully defensive: any failure still returns a usable environment.
    """
    try:
        env = os.environ.copy()
        env['HADOOP_CONF_DIR'] = ''

        path_prepends = []
        if spark_submit_path and os.path.isfile(spark_submit_path):
            bindir = os.path.dirname(spark_submit_path)
            path_prepends.append(bindir)
            # The job does `import pyspark`/`import sedona`, so Spark's driver &
            # executor Python must be the interpreter that sits next to this
            # spark-submit (the conda/pip env it was installed into) - NOT QGIS's
            # bundled python. Pin it explicitly so a Finder-launched QGIS works.
            for py in ('python3', 'python'):
                cand = os.path.join(bindir, py)
                if os.path.isfile(cand):
                    env['PYSPARK_PYTHON'] = cand
                    env['PYSPARK_DRIVER_PYTHON'] = cand
                    break
        path_prepends.append(os.path.join(sys.prefix, 'bin'))

        jh = _detect_java_home()
        if jh:
            env['JAVA_HOME'] = jh
            path_prepends.append(os.path.join(jh, 'bin'))

        path_prepends = [d for d in path_prepends if d and os.path.isdir(d)]
        if path_prepends:
            existing = env.get('PATH', '')
            env['PATH'] = os.pathsep.join(path_prepends + ([existing] if existing else []))
        return env
    except Exception:
        env = os.environ.copy()
        env['HADOOP_CONF_DIR'] = ''
        return env
