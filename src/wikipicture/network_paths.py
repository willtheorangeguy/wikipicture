"""Support for Windows UNC network paths (\\\\server\\share)."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Matches a bare UNC server root such as \\192.168.68.100, \\nas\ or //nas/
# (a host but no share name).
_UNC_SERVER_ROOT = re.compile(r"^[\\/]{2}([^\\/]+)[\\/]*$")

# Share-type bits from lmshare.h: anything besides a plain disk share
# (printers, IPC, devices) or marked special/temporary (ADMIN$, C$, IPC$).
_STYPE_NON_DISK_MASK = 0x3FFFFFFF
_STYPE_SPECIAL_MASK = 0xC0000000

_NERR_SUCCESS = 0
_ERROR_ACCESS_DENIED = 5
_ERROR_BAD_NETPATH = 53
_ERROR_MORE_DATA = 234


def normalize_path_argument(raw: str) -> str:
    """Clean up shell artifacts in a path argument.

    cmd.exe turns a quoted path with a trailing backslash ("\\\\nas\\share\\")
    into an argument ending in a stray double quote; strip that, along with
    surrounding whitespace and quotes.
    """
    return raw.strip().strip('"').rstrip('"')


def parse_unc_server_root(raw: str) -> str | None:
    """Return the host if *raw* is a bare UNC server root, else None.

    \\\\192.168.68.100\\ -> '192.168.68.100'; \\\\nas\\share -> None.
    """
    match = _UNC_SERVER_ROOT.match(raw)
    return match.group(1) if match else None


def enumerate_shares(server: str) -> list[str]:
    """Return the browsable disk shares on *server* via NetShareEnum.

    Hidden/administrative shares (names ending in '$', e.g. C$, IPC$) and
    non-disk shares (printers, IPC) are excluded.

    Raises:
        OSError: If the server cannot be reached, denies access, or share
            enumeration is unsupported on this platform.
    """
    if os.name != "nt":
        raise OSError(
            "Enumerating shares on a UNC server root is only supported on "
            "Windows; pass a full //server/share path instead."
        )

    import ctypes
    from ctypes import wintypes

    class SHARE_INFO_1(ctypes.Structure):
        _fields_ = [
            ("netname", wintypes.LPWSTR),
            ("type", wintypes.DWORD),
            ("remark", wintypes.LPWSTR),
        ]

    netapi32 = ctypes.WinDLL("netapi32")
    shares: list[str] = []
    resume = wintypes.DWORD(0)

    while True:
        buf = ctypes.POINTER(SHARE_INFO_1)()
        entries_read = wintypes.DWORD()
        total_entries = wintypes.DWORD()
        status = netapi32.NetShareEnum(
            ctypes.c_wchar_p(server),
            1,
            ctypes.byref(buf),
            wintypes.DWORD(0xFFFFFFFF),  # MAX_PREFERRED_LENGTH
            ctypes.byref(entries_read),
            ctypes.byref(total_entries),
            ctypes.byref(resume),
        )
        if status not in (_NERR_SUCCESS, _ERROR_MORE_DATA):
            if status == _ERROR_ACCESS_DENIED:
                raise OSError(
                    f"Access denied enumerating shares on \\\\{server}. "
                    f"Authenticate first, e.g.: net use \\\\{server} /user:<name>"
                )
            if status == _ERROR_BAD_NETPATH:
                raise OSError(f"Network server \\\\{server} could not be reached.")
            raise OSError(
                f"Could not enumerate shares on \\\\{server} (error {status})."
            )
        try:
            for i in range(entries_read.value):
                info = buf[i]
                if info.type & _STYPE_SPECIAL_MASK:
                    continue
                if info.type & _STYPE_NON_DISK_MASK:
                    continue
                if info.netname:
                    shares.append(info.netname)
        finally:
            netapi32.NetApiBufferFree(buf)
        if status == _NERR_SUCCESS:
            break

    return shares


def resolve_scan_roots(raw: str) -> list[Path]:
    """Expand a path argument into one or more existing directories to scan.

    Accepts a local directory, a full UNC share path (\\\\server\\share[\\dir]),
    or a bare UNC server root (\\\\server\\), which is expanded to every
    browsable disk share on that server.

    Raises:
        OSError: If the path does not exist or no accessible shares are found.
    """
    cleaned = normalize_path_argument(raw)
    server = parse_unc_server_root(cleaned)

    if server is None:
        path = Path(cleaned)
        if not path.is_dir():
            if cleaned.startswith(("\\\\", "//")):
                raise OSError(
                    f"Network path '{cleaned}' does not exist or is not "
                    f"accessible. Check the share name and that you have "
                    f"access (e.g.: net use {cleaned} /user:<name>)."
                )
            raise OSError(f"Directory '{cleaned}' does not exist.")
        return [path]

    share_names = enumerate_shares(server)
    if not share_names:
        raise OSError(
            f"No browsable disk shares found on \\\\{server}. "
            f"Pass a share path directly, e.g. \\\\{server}\\<share>."
        )

    roots: list[Path] = []
    for name in share_names:
        share_path = Path(f"\\\\{server}\\{name}")
        if share_path.is_dir():
            roots.append(share_path)
        else:
            logger.warning(
                "Skipping share %s — not accessible with current credentials.",
                share_path,
            )

    if not roots:
        raise OSError(
            f"Found {len(share_names)} share(s) on \\\\{server} but none are "
            f"accessible. Authenticate first, e.g.: "
            f"net use \\\\{server}\\{share_names[0]} /user:<name>"
        )
    return roots
