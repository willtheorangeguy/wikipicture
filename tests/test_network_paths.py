"""Tests for wikipicture.network_paths."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from wikipicture.network_paths import (
    normalize_path_argument,
    parse_unc_server_root,
    resolve_scan_roots,
)


# ---------------------------------------------------------------------------
# normalize_path_argument
# ---------------------------------------------------------------------------


class TestNormalizePathArgument:
    def test_plain_path_unchanged(self) -> None:
        assert normalize_path_argument(r"C:\Photos") == r"C:\Photos"

    def test_strips_whitespace(self) -> None:
        assert normalize_path_argument("  C:\\Photos  ") == "C:\\Photos"

    def test_strips_stray_trailing_quote_from_cmd(self) -> None:
        # cmd.exe turns "\\nas\share\" into \\nas\share" on the argv.
        assert normalize_path_argument('\\\\nas\\share"') == "\\\\nas\\share"

    def test_strips_surrounding_quotes(self) -> None:
        assert normalize_path_argument('"\\\\nas\\share"') == "\\\\nas\\share"


# ---------------------------------------------------------------------------
# parse_unc_server_root
# ---------------------------------------------------------------------------


class TestParseUncServerRoot:
    def test_server_root_with_trailing_backslash(self) -> None:
        assert parse_unc_server_root("\\\\192.168.68.100\\") == "192.168.68.100"

    def test_server_root_without_trailing_backslash(self) -> None:
        assert parse_unc_server_root("\\\\192.168.68.100") == "192.168.68.100"

    def test_forward_slash_form(self) -> None:
        assert parse_unc_server_root("//nas/") == "nas"

    def test_share_path_is_not_a_server_root(self) -> None:
        assert parse_unc_server_root("\\\\nas\\Photos") is None

    def test_local_path_is_not_a_server_root(self) -> None:
        assert parse_unc_server_root(r"C:\Photos") is None

    def test_relative_path_is_not_a_server_root(self) -> None:
        assert parse_unc_server_root("photos") is None


# ---------------------------------------------------------------------------
# resolve_scan_roots
# ---------------------------------------------------------------------------


class TestResolveScanRoots:
    def test_local_directory(self, tmp_path: Path) -> None:
        assert resolve_scan_roots(str(tmp_path)) == [tmp_path]

    def test_missing_local_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(OSError, match="does not exist"):
            resolve_scan_roots(str(tmp_path / "nope"))

    def test_missing_unc_share_mentions_access(self) -> None:
        with pytest.raises(OSError, match="not.*accessible|does not exist"):
            resolve_scan_roots("\\\\no-such-host-xyz\\share")

    def test_server_root_expands_to_accessible_shares(self) -> None:
        with (
            patch(
                "wikipicture.network_paths.enumerate_shares",
                return_value=["Archival", "Personal"],
            ),
            patch.object(Path, "is_dir", return_value=True),
        ):
            roots = resolve_scan_roots("\\\\nas\\")
        assert roots == [Path("\\\\nas\\Archival"), Path("\\\\nas\\Personal")]

    def test_server_root_skips_inaccessible_shares(self) -> None:
        def fake_is_dir(self: Path) -> bool:
            return "Archival" in str(self)

        with (
            patch(
                "wikipicture.network_paths.enumerate_shares",
                return_value=["Archival", "Locked"],
            ),
            patch.object(Path, "is_dir", fake_is_dir),
        ):
            roots = resolve_scan_roots("\\\\nas\\")
        assert roots == [Path("\\\\nas\\Archival")]

    def test_server_root_with_no_shares_raises(self) -> None:
        with patch(
            "wikipicture.network_paths.enumerate_shares", return_value=[]
        ):
            with pytest.raises(OSError, match="No browsable disk shares"):
                resolve_scan_roots("\\\\nas\\")

    def test_server_root_with_no_accessible_shares_raises(self) -> None:
        with (
            patch(
                "wikipicture.network_paths.enumerate_shares",
                return_value=["Locked"],
            ),
            patch.object(Path, "is_dir", return_value=False),
        ):
            with pytest.raises(OSError, match="none are accessible"):
                resolve_scan_roots("\\\\nas\\")
