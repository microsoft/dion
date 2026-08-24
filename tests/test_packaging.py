"""Tests for the packaging metadata that PyPI validates at upload time.

These guard failure modes that every local check passes:

- PyPI rejects an unparseable ``author_email``, but ``twine check`` only
  renders the long description and never looks at the field. The address
  shipped here for the package's first year (``{user1, user2}@microsoft.com``,
  obfuscated against harvesters) builds and passes ``twine check`` cleanly and
  fails only against the upload API.
- ``setup.py`` reads ``requirements_*.txt`` at build time. If those files are
  absent from the sdist, ``read_requirements`` warns and returns ``[]``, so
  building from the sdist succeeds and produces a wheel declaring *no*
  dependencies. ``MANIFEST.in`` is what keeps them in.
- ``find_packages`` only picks up directories that have an ``__init__.py``, so
  a new subpackage without one is dropped from the distribution silently.

Metadata is read out of ``setup.py`` with ``ast`` so the tests need neither a
build step nor torch. The artifact tests run only when ``DION_DIST_DIR`` points
at a built ``dist/`` directory; CI sets it after ``python -m build``. That bare
``python -m build`` builds the wheel *from the sdist*, so the wheel assertions
below also cover what the sdist carries.
"""

import ast
import fnmatch
import os
import tarfile
import zipfile
from email.utils import parseaddr
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent
SETUP_PY = REPO_ROOT / "setup.py"
MANIFEST_IN = REPO_ROOT / "MANIFEST.in"
PACKAGE_DIR = REPO_ROOT / "dion"
REQUIREMENTS_FILES = sorted(p.name for p in REPO_ROOT.glob("requirements_*.txt"))


def _setup_call():
    tree = ast.parse(SETUP_PY.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup":
            return {kw.arg: kw.value for kw in node.keywords}
    raise AssertionError("no setup() call found in setup.py")


def _module_assignment(name):
    tree = ast.parse(SETUP_PY.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"no module-level assignment to {name} in setup.py")


def _keyword(name):
    node = _setup_call().get(name)
    if node is None:
        raise AssertionError(f"setup() has no {name} keyword")
    return ast.literal_eval(node)


def _manifest_include_patterns():
    patterns = []
    for line in MANIFEST_IN.read_text().splitlines():
        line = line.strip()
        if line.startswith("include "):
            patterns.extend(line.split()[1:])
    return patterns


def test_author_email_is_a_deliverable_address():
    """PyPI's upload API rejects an address that does not parse."""
    email = _keyword("author_email")
    name, addr = parseaddr(email)
    assert addr, f"author_email {email!r} does not parse as an email address"
    assert "@" in addr and not addr.startswith("@") and not addr.endswith("@")
    assert "," not in addr and "{" not in addr and "}" not in addr


def test_version_is_pep440():
    """A tag-driven release publishes this string; PyPI requires PEP 440."""
    Version(_module_assignment("version"))


def test_project_urls_point_at_the_repository():
    """Without these the PyPI page has no link back to the source."""
    urls = _keyword("project_urls")
    assert "Issues" in urls, "no issue tracker link; bug reports land in a maintainer inbox"
    for label, url in urls.items():
        assert url.startswith("https://github.com/microsoft/dion"), (
            f"project_urls[{label!r}] does not point at the repository: {url}"
        )


def test_classifiers_declare_the_license():
    """setup.py's license="MIT" is free text; the classifier is what PyPI facets on."""
    classifiers = _keyword("classifiers")
    assert "License :: OSI Approved :: MIT License" in classifiers


def test_python_classifiers_agree_with_python_requires():
    """A classifier PyPI advertises but python_requires excludes is a false claim."""
    supported = SpecifierSet(_keyword("python_requires"))
    declared = [
        c.rsplit(" :: ", 1)[1]
        for c in _keyword("classifiers")
        if c.startswith("Programming Language :: Python :: ")
    ]
    versions = [v for v in declared if "." in v]
    assert versions, "no Programming Language :: Python :: X.Y classifiers to check"
    excluded = [v for v in versions if not supported.contains(v)]
    assert not excluded, (
        f"classifiers advertise Python {excluded} but python_requires "
        f"{str(supported)!r} rejects them"
    )


def test_repo_has_requirements_files():
    """The manifest and sdist tests below are parametrized on this glob."""
    assert REQUIREMENTS_FILES, "no requirements_*.txt in the repo root; the tests below are vacuous"


@pytest.mark.parametrize("requirements_file", REQUIREMENTS_FILES)
def test_manifest_ships_every_requirements_file(requirements_file):
    """Any requirements file setup.py may read has to survive into the sdist."""
    patterns = _manifest_include_patterns()
    assert any(fnmatch.fnmatch(requirements_file, pattern) for pattern in patterns), (
        f"{requirements_file} matches no include line in MANIFEST.in, so an sdist build "
        f"would silently drop the dependencies it declares"
    )


def _dist_dir():
    dist = os.environ.get("DION_DIST_DIR")
    if not dist:
        pytest.skip("DION_DIST_DIR not set; run `python -m build` first")
    return Path(dist)


def _one(pattern):
    matches = sorted(_dist_dir().glob(pattern))
    assert len(matches) == 1, f"expected exactly one {pattern} in dist/, found {matches}"
    return matches[0]


def test_sdist_contains_the_requirements_files():
    with tarfile.open(_one("*.tar.gz")) as tar:
        names = {Path(n).name for n in tar.getnames()}
    missing = [name for name in REQUIREMENTS_FILES if name not in names]
    assert not missing, f"sdist is missing {missing}; a build from it would declare no deps"


def test_wheel_declares_its_runtime_dependencies():
    with zipfile.ZipFile(_one("*.whl")) as wheel:
        metadata_name = next(n for n in wheel.namelist() if n.endswith(".dist-info/METADATA"))
        metadata = wheel.read(metadata_name).decode()
    requires = [
        line.split(":", 1)[1].strip()
        for line in metadata.splitlines()
        if line.startswith("Requires-Dist:") and "extra ==" not in line
    ]
    assert any(r.startswith("torch") for r in requires), f"wheel declares no torch: {requires}"
    assert any(r.startswith("numpy") for r in requires), f"wheel declares no numpy: {requires}"


def test_wheel_ships_every_package_module():
    """find_packages() drops a subdirectory that has no __init__.py, without complaint."""
    source = {p.relative_to(REPO_ROOT).as_posix() for p in PACKAGE_DIR.rglob("*.py")}
    assert source, f"no modules found under {PACKAGE_DIR}"
    with zipfile.ZipFile(_one("*.whl")) as wheel:
        shipped = set(wheel.namelist())
    missing = sorted(source - shipped)
    assert not missing, (
        f'the wheel is missing {missing}; find_packages(include=["dion", "dion.*"]) '
        f"only picks up directories with an __init__.py"
    )
