"""Pin what the text readers accept, return and raise, whatever parser is used."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import pytest

import iorec
from iorec import _utils

if TYPE_CHECKING:
    from pathlib import Path

POINTS = "0.5 1e-5 -0.0\n1 2 3\n4.5 5 6\n7 8 9\n"
TRIANGLES = "2\n0 1 2 1 0\n1 2 3 2 1\n"
MESH = "4\n" + POINTS + TRIANGLES
EXPECTED_POINTS = np.array([[0.5, 1e-5, -0.0], [1, 2, 3], [4.5, 5, 6], [7, 8, 9]])
EXPECTED_TRIANGLES = np.array([[0, 1, 2], [1, 2, 3]])
EXPECTED_LABELS = np.array([[1, 0], [2, 1]])

Reader = Callable[..., tuple]


@pytest.fixture(params=["default", "no_size_threshold"])
def _threshold(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each test with and without the minimum file size of any fast parser."""
    if request.param == "no_size_threshold":
        monkeypatch.setattr(_utils, "_FAST_TEXT_MIN_BYTES", 0, raising=False)


def _write(tmp_path: Path, text: str | bytes) -> Path:
    filename = tmp_path / "mesh.rec"
    filename.write_bytes(text if isinstance(text, bytes) else text.encode())
    return filename


def _readers() -> list[Reader]:
    def with_metadata(filename: Path, *, validate: bool = False) -> tuple:
        return iorec.load_rec_with_metadata(filename, validate=validate)[:3]

    return [iorec.load_rec, with_metadata]


@pytest.fixture(params=_readers(), ids=["load_rec", "load_rec_with_metadata"])
def reader(request: pytest.FixtureRequest) -> Reader:
    """Both public entry points of the 3D text reader."""
    return request.param


def _check_expected(result: tuple) -> None:
    points, triangles, labels = result
    assert points.dtype == np.float64
    assert triangles.dtype == np.int64
    assert labels.dtype == np.int64
    np.testing.assert_array_equal(points, EXPECTED_POINTS)
    np.testing.assert_array_equal(triangles, EXPECTED_TRIANGLES)
    np.testing.assert_array_equal(labels, EXPECTED_LABELS)
    assert np.signbit(points[0, 2])  # negative zero survives


ACCEPTED = {
    "plain": MESH,
    "windows_newlines": MESH.replace("\n", "\r\n"),
    "tabs": MESH.replace(" ", "\t"),
    "repeated_spaces": MESH.replace(" ", "   "),
    "leading_and_trailing_spaces": MESH.replace("\n", " \n").replace("0.5", " 0.5"),
    "trailing_blank_lines": MESH + "\n\n",
    "no_final_newline": MESH.rstrip("\n"),
    "blank_line_inside_block": MESH.replace("1 2 3\n", "1 2 3\n\n"),
    "comment_at_end_of_row": MESH.replace("1 2 3\n", "1 2 3 # c\n"),
    "explicit_plus_signs": MESH.replace("1 2 3", "+1 +2 +3"),
    "exponent_forms": MESH.replace("1e-5", "1E-05").replace("4.5", "45e-1"),
    "leading_zeros_in_integers": MESH.replace("0 1 2 1 0", "00 01 02 01 00"),
    "trailing_garbage_without_validate": MESH + "garbage\n",
}


@pytest.mark.usefixtures("_threshold")
@pytest.mark.parametrize("text", ACCEPTED.values(), ids=ACCEPTED.keys())
def test_accepted_variants(reader: Reader, tmp_path: Path, text: str) -> None:
    """Formatting variants that read today give the same arrays."""
    _check_expected(reader(_write(tmp_path, text)))


@pytest.mark.usefixtures("_threshold")
def test_non_finite_values(reader: Reader, tmp_path: Path) -> None:
    """`nan`, `inf` and their signed or capitalised forms are read as floats."""
    text = MESH.replace("4.5 5 6", "nan -inf Infinity").replace("7 8 9", "-nan NaN inf")
    points = reader(_write(tmp_path, text))[0]
    assert np.isnan(points[2, 0])
    assert points[2, 1] == -np.inf
    assert points[2, 2] == np.inf
    assert np.isnan(points[3, 0])
    assert np.isnan(points[3, 1])
    assert points[3, 2] == np.inf
    with pytest.raises(iorec.RecFormatError, match="non-finite"):
        reader(_write(tmp_path, text), validate=True)


@pytest.mark.usefixtures("_threshold")
def test_appendix_after_geometry(tmp_path: Path) -> None:
    """The metadata appendix after the text geometry is read and not parsed."""
    filename = tmp_path / "mesh.rec"
    iorec.save_rec(
        filename,
        EXPECTED_POINTS,
        EXPECTED_TRIANGLES,
        EXPECTED_LABELS,
        metadata={"coordinate_frame": "voxel_index", "spacing": (1.0, 0.5, 0.5)},
    )
    _check_expected(iorec.load_rec(filename, validate=True))
    points, triangles, labels, metadata = iorec.load_rec_with_metadata(
        filename,
        validate=True,
    )
    _check_expected((points, triangles, labels))
    assert metadata is not None
    assert metadata["coordinate_frame"] == "voxel_index"


ERRORS = {
    "point_count_too_large": (
        MESH.replace("4\n", "5\n", 1),
        "the number of columns changed from 3 to 1 at row 5",
    ),
    "non_numeric_point": (
        MESH.replace("4.5 5 6", "4.5 x 6"),
        "could not convert string 'x' to float64 at row 2, column 2",
    ),
    "float_in_integer_block": (
        MESH.replace("1 2 3 2 1", "1 2 3.0 2 1"),
        "could not convert string '3.0' to int64 at row 1, column 3",
    ),
    "integer_overflow": (
        MESH.replace("0 1 2 1 0", "0 1 2 1 99999999999999999999"),
        "could not convert string '99999999999999999999' to int64",
    ),
    "ragged_points": (
        MESH.replace("4.5 5 6", "4.5 5"),
        "the number of columns changed from 3 to 2 at row 3",
    ),
    "hexadecimal_float": (
        MESH.replace("0.5", "0x1p-1"),
        "could not convert string '0x1p-1' to float64",
    ),
    "hexadecimal_integer": (
        MESH.replace("0 1 2 1 0", "0 0x1 2 1 0"),
        "could not convert string '0x1' to int64",
    ),
    "nan_with_payload": (
        MESH.replace("4.5 5 6", "4.5 nan(1) 6"),
        "could not convert string 'nan\\(1\\)' to float64",
    ),
    "underscore_in_number": (
        MESH.replace("1 2 3\n", "1 2 1_0\n"),
        "could not convert string '1_0' to float64",
    ),
}


@pytest.mark.usefixtures("_threshold")
@pytest.mark.parametrize(
    ("text", "message"),
    ERRORS.values(),
    ids=ERRORS.keys(),
)
def test_errors_without_validate(
    reader: Reader,
    tmp_path: Path,
    text: str,
    message: str,
) -> None:
    """Without `validate`, numpy's own `ValueError` reaches the caller."""
    with pytest.raises(ValueError, match=message) as info:
        reader(_write(tmp_path, text))
    assert not isinstance(info.value, iorec.RecFormatError)


@pytest.mark.usefixtures("_threshold")
@pytest.mark.parametrize(
    ("text", "message"),
    ERRORS.values(),
    ids=ERRORS.keys(),
)
def test_errors_with_validate(
    reader: Reader,
    tmp_path: Path,
    text: str,
    message: str,
) -> None:
    """With `validate`, the same problems raise `RecFormatError` naming the file."""
    filename = _write(tmp_path, text)
    pattern = rf"mesh\.rec: invalid (point|triangle) block: .*{message}"
    with pytest.raises(iorec.RecFormatError, match=pattern):
        reader(filename, validate=True)


@pytest.mark.usefixtures("_threshold")
def test_short_triangle_block(reader: Reader, tmp_path: Path) -> None:
    """A short block is returned short, or rejected with `validate`."""
    filename = _write(tmp_path, MESH.replace("2\n0 1", "3\n0 1"))
    triangles = reader(filename)[1]
    np.testing.assert_array_equal(triangles, EXPECTED_TRIANGLES)
    with pytest.raises(iorec.RecFormatError, match=r"triangles have shape \(2, 3\)"):
        reader(filename, validate=True)


@pytest.mark.usefixtures("_threshold")
def test_extra_column_in_triangles(reader: Reader, tmp_path: Path) -> None:
    """Six-column triangle records are read without `validate`, rejected with it."""
    text = MESH.replace("0 1 2 1 0", "0 1 2 1 0 9").replace("1 2 3 2 1", "1 2 3 2 1 9")
    filename = _write(tmp_path, text)
    assert reader(filename)[1].shape == (2, 3)
    with pytest.raises(iorec.RecFormatError, match="6 columns, expected 5"):
        reader(filename, validate=True)


@pytest.mark.usefixtures("_threshold")
def test_trailing_garbage_with_validate(reader: Reader, tmp_path: Path) -> None:
    """Content after the last record that is not an appendix is rejected."""
    filename = _write(tmp_path, MESH + "garbage\n")
    with pytest.raises(iorec.RecFormatError, match="unexpected content"):
        reader(filename, validate=True)


@pytest.mark.usefixtures("_threshold")
@pytest.mark.parametrize(
    ("text", "plain_message", "validate_message"),
    [
        (
            "x\n" + POINTS + TRIANGLES,
            "invalid literal for int\\(\\) with base 10: 'x'",
            "invalid point count 'x'",
        ),
        (
            "-1\n" + POINTS + TRIANGLES,
            "invalid literal for int\\(\\) with base 10: '0.5 1e-5 -0.0'",
            "point count must be non-negative",
        ),
        (
            "4\n" + POINTS + "y\n",
            "invalid literal for int\\(\\) with base 10: 'y'",
            "invalid triangle count 'y'",
        ),
    ],
)
def test_bad_counts(
    reader: Reader,
    tmp_path: Path,
    text: str,
    plain_message: str,
    validate_message: str,
) -> None:
    """A bad count line raises `ValueError`, a `RecFormatError` with `validate`."""
    filename = _write(tmp_path, text)
    with pytest.raises(ValueError, match=plain_message):
        reader(filename)
    with pytest.raises(iorec.RecFormatError, match=validate_message):
        reader(filename, validate=True)


@pytest.mark.usefixtures("_threshold")
def test_empty_mesh(reader: Reader, tmp_path: Path) -> None:
    """Zero points and zero triangles read as empty arrays of the right shape."""
    points, triangles, labels = reader(_write(tmp_path, "0\n0\n"))
    assert points.shape == (0, 3)
    assert triangles.shape == (0, 3)
    assert labels.shape == (0, 2)


@pytest.mark.usefixtures("_threshold")
def test_2d_text_variants(tmp_path: Path) -> None:
    """2D text files read the same under formatting variants."""
    text = "3\n0.5 1e-5\n1 2\n-0.0 nan\n2\n0 1 1 0\n1 2 2 1\n"
    variants = (text, text.replace("\n", "\r\n"), text.replace(" ", "\t"), text + "\n")
    for variant in variants:
        points, segments, labels = iorec.load_2d_rec(_write(tmp_path, variant))
        assert points.dtype == np.float64
        assert points.shape == (3, 2)
        assert np.isnan(points[2, 1])
        np.testing.assert_array_equal(segments, [[0, 1], [1, 2]])
        np.testing.assert_array_equal(labels, [[1, 0], [2, 1]])
    with pytest.raises(iorec.RecFormatError, match="non-finite"):
        iorec.load_2d_rec(_write(tmp_path, text), validate=True)


@pytest.mark.usefixtures("_threshold")
def test_2d_errors(tmp_path: Path) -> None:
    """2D parse errors follow the 3D ones."""
    filename = _write(tmp_path, "2\n0.5 x\n1 2\n0\n")
    with pytest.raises(ValueError, match="could not convert string 'x'"):
        iorec.load_2d_rec(filename)
    with pytest.raises(iorec.RecFormatError, match="invalid point block"):
        iorec.load_2d_rec(filename, validate=True)


@pytest.mark.usefixtures("_threshold")
def test_float_parsing_is_exact(tmp_path: Path) -> None:
    """Every float64 `repr` reads back to the identical bits."""
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2**63, size=6000, dtype=np.uint64)
    values = bits.view(np.float64)
    values = values[np.isfinite(values)]
    values = np.concatenate(
        [
            values,
            rng.normal(size=3000) * 10.0 ** rng.integers(-20, 20, size=3000),
            [5e-324, -5e-324, 2.2250738585072014e-308, 1.7976931348623157e308],
            [-0.0, 0.0],
            [0.1, 0.30000000000000004, 1e22, 1e23, 9007199254740993.0],
        ],
    )
    values = values[: len(values) // 3 * 3]
    points = values.reshape(-1, 3)
    triangles = np.zeros((1, 3), dtype=np.int64)
    labels = np.array([[1, 0]])
    filename = tmp_path / "exact.rec"
    iorec.save_rec(filename, points, triangles, labels)
    loaded = iorec.load_rec(filename)[0]
    np.testing.assert_array_equal(loaded.view(np.uint64), points.view(np.uint64))
    tokens = filename.read_text().split("\n")[1 : 1 + len(points)]
    expected = np.array([[float(t) for t in row.split()] for row in tokens])
    np.testing.assert_array_equal(loaded.view(np.uint64), expected.view(np.uint64))


@pytest.mark.usefixtures("_threshold")
def test_returned_arrays_layout(tmp_path: Path) -> None:
    """Points are C-contiguous; triangles and labels are column views of one block."""
    points, triangles, labels = iorec.load_rec(_write(tmp_path, MESH))
    assert points.flags.c_contiguous
    assert triangles.base is labels.base
    assert triangles.base is not None
    assert triangles.base.shape == (2, 5)


def _bits(array: np.ndarray) -> np.ndarray:
    return array.view(np.uint64) if array.dtype == np.float64 else array


def _random_mesh(count: int, dimension: int) -> tuple:
    rng = np.random.default_rng(2)
    points = (
        rng.normal(size=(count, dimension))
        * 10.0
        ** rng.integers(-8, 8, count)[
            :,
            None,
        ]
    )
    points[0, 0] = np.nan
    points[1, 0] = -np.inf
    points[2, 0] = -0.0
    cells = rng.integers(0, count, size=(2 * count, dimension))
    labels = np.stack([rng.integers(1, 5, 2 * count), np.zeros(2 * count)], axis=1)
    return points, cells.astype(np.int64), labels.astype(np.int64)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("validate", [False, True])
def test_large_file_matches_general_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dimension: int,
    validate: bool,
) -> None:
    """A large file reads bit-identically with and without pyarrow, by both paths."""
    pytest.importorskip("pyarrow")
    points, cells, labels = _random_mesh(20000, dimension)
    filename = tmp_path / "large.rec"
    save = iorec.save_rec if dimension == 3 else iorec.save_2d_rec
    load = iorec.load_rec if dimension == 3 else iorec.load_2d_rec
    load_meta = (
        iorec.load_rec_with_metadata
        if dimension == 3
        else iorec.load_2d_rec_with_metadata
    )
    save(filename, points, cells, labels, metadata={"spacing": (1.0,) * dimension})
    assert filename.stat().st_size > _utils._FAST_TEXT_MIN_BYTES
    kwargs = {"validate": validate}
    if validate:  # the random mesh has non-finite points, so only compare errors
        with pytest.raises(iorec.RecFormatError, match="non-finite"):
            load(filename, **kwargs)
        points = np.nan_to_num(points)
        save(filename, points, cells, labels, metadata={"x": 1})
        cells %= len(points)
    fast = (load(filename, **kwargs), load_meta(filename, **kwargs)[:3])
    monkeypatch.setattr(_utils, "_FAST_TEXT_MIN_BYTES", 10**12)
    slow = (load(filename, **kwargs), load_meta(filename, **kwargs)[:3])
    for fast_result, slow_result in zip(fast, slow):
        for a, b in zip(fast_result, slow_result):
            assert a.dtype == b.dtype
            assert a.shape == b.shape
            np.testing.assert_array_equal(_bits(a), _bits(b))


def test_fast_parser_is_used_on_plain_text() -> None:
    """The pyarrow parser accepts plain files and declines the unusual ones."""
    fast_text = pytest.importorskip("iorec._fast_text")
    pytest.importorskip("pyarrow")
    for name in ("plain", "trailing_blank_lines", "no_final_newline", "exponent_forms"):
        assert fast_text.parse_text_geometry(ACCEPTED[name].encode(), 3, 5)
    for name in (
        "windows_newlines",
        "tabs",
        "repeated_spaces",
        "comment_at_end_of_row",
    ):
        assert fast_text.parse_text_geometry(ACCEPTED[name].encode(), 3, 5) is None
    for text, _ in ERRORS.values():
        assert fast_text.parse_text_geometry(text.encode(), 3, 5) is None


def test_fast_parser_matches_numpy_on_special_values() -> None:
    """Signed zeros, NaN signs, overflow and subnormals have the same bits as numpy."""
    import io

    fast_text = pytest.importorskip("iorec._fast_text")
    pytest.importorskip("pyarrow")
    for row in ("-nan NaN Inf", "1e400 -1e-400 5e-324", "-0.0 1e-310 2.5e-320"):
        text = MESH.replace("4.5 5 6", row)
        parsed = fast_text.parse_text_geometry(text.encode(), 3, 5)
        assert parsed is not None
        expected = np.loadtxt(
            io.StringIO(text.split("\n", 1)[1]),
            max_rows=4,
            dtype=np.float64,
            ndmin=2,
        )
        np.testing.assert_array_equal(_bits(parsed[0]), _bits(expected))


def test_without_pyarrow_the_general_reader_is_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With pyarrow missing, big files read as before."""
    fast_text = pytest.importorskip("iorec._fast_text")
    monkeypatch.setattr(fast_text, "_pyarrow", lambda: None)
    monkeypatch.setattr(_utils, "_FAST_TEXT_MIN_BYTES", 0)
    _check_expected(iorec.load_rec(_write(tmp_path, MESH)))
