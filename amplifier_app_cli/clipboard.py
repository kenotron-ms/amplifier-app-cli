"""Clipboard content reading utilities.

Provides cross-platform support for reading content from the system clipboard,
returning base64-encoded data and a media type suitable for use in multimodal
LLM messages.

Supported formats
-----------------
- Images  : image/png, image/jpeg, image/gif, image/webp
- Documents: application/pdf  (Claude 3.5+)
- Text    : text/plain

Supported platforms
-------------------
- macOS : AppKit subprocess (images), osascript JXA (PDF + images), pbpaste (text)
- Linux : xclip (image/png, application/pdf, text/plain) with xsel raw fallback
- Remote: OSC 52 terminal clipboard query — reads from the local terminal emulator
          (e.g. Ghostty) so it works transparently over SSH + tmux.
          Requires ``set -g allow-passthrough on`` in tmux ≥ 3.3.
- Windows: not yet supported

Usage::

    from .clipboard import get_clipboard_content

    result = get_clipboard_content()
    if result:
        base64_data, media_type = result
        # media_type is one of: image/png, image/jpeg, image/gif, image/webp,
        #                        application/pdf, text/plain
"""

from __future__ import annotations

import base64
import logging
import os
import re
import select
import subprocess
import sys
import tempfile
import termios
import time
import tty
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Binary magic bytes for format detection ────────────────────────────────────

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8"
_GIF87_MAGIC = b"GIF87a"
_GIF89_MAGIC = b"GIF89a"
_WEBP_RIFF = b"RIFF"
_WEBP_MARKER = b"WEBP"
_PDF_MAGIC = b"%PDF-"


def _detect_media_type(data: bytes) -> str | None:
    """Return the MIME type of *data* based on magic-byte signatures, or None.

    Checked in priority order:
    1. PNG, JPEG, GIF, WebP  — image formats
    2. PDF                   — document format
    3. UTF-8 text            — plain-text fallback (only if non-empty)
    """
    if data.startswith(_PNG_MAGIC):
        return "image/png"
    if data[:2] == _JPEG_MAGIC:
        return "image/jpeg"
    if data[:6] in (_GIF87_MAGIC, _GIF89_MAGIC):
        return "image/gif"
    if len(data) >= 12 and data[:4] == _WEBP_RIFF and data[8:12] == _WEBP_MARKER:
        return "image/webp"
    if data.startswith(_PDF_MAGIC):
        return "application/pdf"
    # Last resort: valid UTF-8 and non-empty → treat as plain text
    try:
        if data.decode("utf-8").strip():
            return "text/plain"
    except UnicodeDecodeError:
        pass
    return None


# ── Public API ─────────────────────────────────────────────────────────────────


def get_clipboard_content() -> tuple[str, str] | None:
    """Read content from the system clipboard.

    Tries methods in this order:

    1. **OSC 52** terminal query — works locally *and* over SSH/tmux because
       the escape sequence travels through the PTY wire to the real terminal
       emulator that holds the clipboard.
    2. **Platform-native** methods (macOS / Linux).

    Returns:
        A ``(base64_data, media_type)`` tuple when supported content is found,
        or ``None``.  *base64_data* is a standard (padded) base64 string.
        *media_type* is one of ``image/png``, ``image/jpeg``, ``image/gif``,
        ``image/webp``, ``application/pdf``, or ``text/plain``.
    """
    # OSC 52 first — transparent over SSH + tmux
    result = _get_clipboard_content_osc52()
    if result:
        # On macOS, terminals (e.g. Ghostty) answering OSC 52 may return the
        # text-plain flavour even when the clipboard also has a richer PDF or
        # image representation (e.g. after copying from Preview.app).  When
        # that happens, try the native macOS path so the higher-fidelity type
        # is preferred over a plain-text downgrade.
        if result[1] == "text/plain" and sys.platform == "darwin":
            native = _get_clipboard_content_macos()
            if native and native[1] != "text/plain":
                return native
        return result

    if sys.platform == "darwin":
        return _get_clipboard_content_macos()
    if sys.platform.startswith("linux"):
        return _get_clipboard_content_linux()

    logger.debug("Clipboard reading not supported on platform: %s", sys.platform)
    return None


# Backward-compatible alias kept so any external callers don't break.
get_clipboard_image = get_clipboard_content


# ── OSC 52 terminal clipboard query ───────────────────────────────────────────


def _get_clipboard_content_osc52() -> tuple[str, str] | None:
    """Read clipboard from the local terminal emulator via OSC 52 query.

    Sends ``ESC ] 52 ; c ; ? BEL``; the terminal responds with the clipboard
    contents as base64.  Because the exchange travels over the PTY wire, it
    works transparently over SSH and through tmux (when tmux has
    ``set -g allow-passthrough on``).

    Supported terminals: Ghostty, iTerm2, WezTerm, foot, Alacritty ≥ 0.13,
    kitty.  Returns ``None`` in ≤ 1 s when the terminal doesn't respond.
    """
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return None

    fd = sys.stdin.fileno()
    try:
        old_settings = termios.tcgetattr(fd)
    except termios.error:
        return None  # not a real TTY

    response = b""
    try:
        tty.setraw(fd)
        # ESC ] 52 ; c ; ? BEL  — "c" selects the system clipboard
        sys.stdout.write("\033]52;c;?\007")
        sys.stdout.flush()

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            ready, _, _ = select.select([fd], [], [], 0.05)
            if not ready:
                if response and (
                    response.endswith(b"\007") or response.endswith(b"\033\\")
                ):
                    break
                continue
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            response += chunk
            if response.endswith(b"\007") or response.endswith(b"\033\\"):
                break
    except Exception as exc:
        logger.debug("OSC 52 clipboard query error: %s", exc)
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    if not response:
        logger.debug(
            "OSC 52: no response from terminal (may not support clipboard reads)"
        )
        return None

    match = re.search(rb"\033\]52;[^;]*;([A-Za-z0-9+/=]+)(?:\007|\033\\)", response)
    if not match:
        logger.debug("OSC 52: could not parse response payload: %r", response[:80])
        return None

    try:
        raw = base64.standard_b64decode(match.group(1))
    except Exception as exc:
        logger.debug("OSC 52: base64 decode failed: %s", exc)
        return None

    media_type = _detect_media_type(raw)
    if media_type is None:
        logger.debug("OSC 52: unrecognised content (first bytes: %r)", raw[:8])
        return None

    logger.debug("OSC 52: got %s (%d bytes)", media_type, len(raw))
    return base64.standard_b64encode(raw).decode("ascii"), media_type


# ── macOS ──────────────────────────────────────────────────────────────────────


def _get_clipboard_content_macos() -> tuple[str, str] | None:
    """Read content from the macOS clipboard.

    Priority: **PDF → Image → File URL → Text**

    PDF is checked before image because a clipboard holding both a PDF and its
    rendered image preview (e.g. after copying from Preview.app) should yield
    the richer PDF, not a thumbnail.  Screenshots never have PDF data so the
    priority is safe for the common screenshot-paste workflow.

    File URL is checked before plain text so that copying a file in Finder
    (⌘C) attaches the actual file content rather than the near-empty filename
    string that ``pbpaste`` returns for ``public.file-url`` clipboard entries.
    """
    # 1. PDF
    result = _try_osascript_pdf()
    if result:
        return result

    # 2. Images
    result = _try_appkit()
    if result:
        return result
    result = _try_osascript_image()
    if result:
        return result
    result = _try_pngpaste()
    if result:
        return result

    # 3. Filesystem file copied in Finder (⌘C on a file)
    result = _try_macos_file_url()
    if result:
        return result

    # 4. Plain text
    result = _try_pbpaste()
    if result:
        return result

    logger.debug("No supported content found in macOS clipboard")
    return None


def _try_appkit() -> tuple[str, str] | None:
    """Read image from macOS clipboard via a PyObjC AppKit subprocess.

    Runs in a child process to avoid import-time side-effects in the main
    process (AppKit triggers an event loop if imported directly).
    """
    script = (
        "import sys\n"
        "try:\n"
        "    from AppKit import NSPasteboard, NSPasteboardTypePNG, NSPasteboardTypeTIFF\n"
        "    pb = NSPasteboard.generalPasteboard()\n"
        "    png = pb.dataForType_(NSPasteboardTypePNG)\n"
        "    if png:\n"
        "        sys.stdout.buffer.write(bytes(png)); sys.exit(0)\n"
        "    tiff = pb.dataForType_(NSPasteboardTypeTIFF)\n"
        "    if tiff:\n"
        "        from AppKit import NSBitmapImageRep, NSPNGFileType\n"
        "        rep = NSBitmapImageRep.imageRepWithData_(tiff)\n"
        "        if rep:\n"
        "            png2 = rep.representationUsingType_properties_(NSPNGFileType, None)\n"
        "            if png2:\n"
        "                sys.stdout.buffer.write(bytes(png2)); sys.exit(0)\n"
        "    sys.exit(1)\n"
        "except Exception:\n"
        "    sys.exit(2)\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=5.0,
        )
        if proc.returncode == 0 and proc.stdout:
            return base64.standard_b64encode(proc.stdout).decode("ascii"), "image/png"
    except Exception as exc:
        logger.debug("AppKit clipboard read failed: %s", exc)
    return None


def _try_osascript_image() -> tuple[str, str] | None:
    """Read image from macOS clipboard via osascript JXA (PNG or TIFF→PNG)."""
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = Path(f.name)

        script = (
            "ObjC.import('AppKit');"
            "ObjC.import('Foundation');"
            f"var outPath={repr(str(tmp))};"
            "var pb=$.NSPasteboard.generalPasteboard;"
            "var wrote=false;"
            "var pngData=pb.dataForType('public.png');"
            "if(pngData&&pngData.length>0){"
            "  pngData.writeToFileAtomically(outPath,true);"
            "  wrote=true;"
            "}"
            "if(!wrote){"
            "  var tiffData=pb.dataForType('public.tiff');"
            "  if(tiffData&&tiffData.length>0){"
            "    var rep=$.NSBitmapImageRep.imageRepWithData(tiffData);"
            "    if(rep){"
            "      var pngOut=rep.representationUsingTypeProperties($.NSPNGFileType,null);"
            "      if(pngOut&&pngOut.length>0){"
            "        pngOut.writeToFileAtomically(outPath,true);"
            "        wrote=true;"
            "      }"
            "    }"
            "  }"
            "}"
            "wrote"
        )
        proc = subprocess.run(
            ["osascript", "-l", "JavaScript", "-e", script],
            capture_output=True,
            timeout=5.0,
        )
        if proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            return (
                base64.standard_b64encode(tmp.read_bytes()).decode("ascii"),
                "image/png",
            )
    except FileNotFoundError:
        pass  # osascript not available
    except Exception as exc:
        logger.debug("osascript image clipboard read failed: %s", exc)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return None


def _try_osascript_pdf() -> tuple[str, str] | None:
    """Read PDF from macOS clipboard via osascript JXA (com.adobe.pdf)."""
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp = Path(f.name)

        script = (
            "ObjC.import('AppKit');"
            "ObjC.import('Foundation');"
            f"var outPath={repr(str(tmp))};"
            "var pb=$.NSPasteboard.generalPasteboard;"
            "var pdfData=pb.dataForType('com.adobe.pdf');"
            "if(pdfData&&pdfData.length>0){"
            "  pdfData.writeToFileAtomically(outPath,true);"
            "  true;"
            "}else{"
            "  false;"
            "}"
        )
        proc = subprocess.run(
            ["osascript", "-l", "JavaScript", "-e", script],
            capture_output=True,
            timeout=5.0,
        )
        if proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            raw = tmp.read_bytes()
            if raw.startswith(_PDF_MAGIC):
                return (
                    base64.standard_b64encode(raw).decode("ascii"),
                    "application/pdf",
                )
    except FileNotFoundError:
        pass  # osascript not available
    except Exception as exc:
        logger.debug("osascript PDF clipboard read failed: %s", exc)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return None


def _try_pngpaste() -> tuple[str, str] | None:
    """Read image from macOS clipboard via the pngpaste CLI (brew install pngpaste)."""
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = Path(f.name)
        proc = subprocess.run(
            ["pngpaste", str(tmp)],
            capture_output=True,
            timeout=5.0,
        )
        if proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            return (
                base64.standard_b64encode(tmp.read_bytes()).decode("ascii"),
                "image/png",
            )
    except FileNotFoundError:
        pass  # pngpaste not installed — silently skip
    except Exception as exc:
        logger.debug("pngpaste clipboard read failed: %s", exc)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return None


def _try_macos_file_url() -> tuple[str, str] | None:
    """Read a filesystem item copied in Finder from the macOS clipboard.

    When the user copies a file with ⌘C in Finder, macOS places a
    ``public.file-url`` (a ``file://`` URI) into the clipboard rather than the
    file's raw bytes.  ``pbpaste`` then returns either an empty string or the
    filename, producing the confusing "Text attached (~0 KB)" result.

    This function reads the ``public.file-url`` entry via osascript JXA,
    resolves it to an absolute path, reads the file, and returns its content
    with proper type detection — so images, PDFs, and text files all attach
    correctly.

    Supported types: ``image/*``, ``application/pdf``, ``text/plain``.
    Directories and unrecognised binary files return ``None``.
    """
    # Resolve the public.file-url entry to an absolute filesystem path.
    # Finder puts two kinds of file URLs on the clipboard:
    #   • Path URLs:      file:///Users/ken/Downloads/doc.pdf
    #   • Reference URLs: file:///.file/id=6571367.161995492
    # Python's urlparse cannot resolve reference URLs — the path component
    # (/.file/id=...) is not a real filesystem path.  NSURL.filePathURL
    # handles both forms and returns a real path URL we can use directly.
    script = (
        "ObjC.import('AppKit');"
        "ObjC.import('Foundation');"
        "var pb=$.NSPasteboard.generalPasteboard;"
        "var d=pb.dataForType('public.file-url');"
        "if(!d||d.length===0){''};"
        "else{"
        "var s=$.NSString.alloc.initWithDataEncoding(d,$.NSUTF8StringEncoding).js;"
        "var url=$.NSURL.URLWithString(s);"
        "var p=url.filePathURL;"
        "p?p.path.js:'';}"
    )
    try:
        proc = subprocess.run(
            ["osascript", "-l", "JavaScript", "-e", script],
            capture_output=True,
            timeout=5.0,
        )
        if proc.returncode != 0:
            return None

        # JXA returns the resolved filesystem path directly (e.g. /Users/ken/Downloads/doc.pdf)
        file_path_str = proc.stdout.decode("utf-8", errors="replace").strip()
        if not file_path_str:
            return None

        file_path = Path(file_path_str)
        if not file_path.is_file():
            logger.debug("File URL clipboard: not a regular file: %s", file_path)
            return None

        raw = file_path.read_bytes()
        media_type = _detect_media_type(raw)
        if media_type is None:
            logger.debug(
                "File URL clipboard: unsupported type for %s (first bytes: %r)",
                file_path.name,
                raw[:8],
            )
            return None

        logger.debug(
            "File URL clipboard: got %s from %s (%d bytes)",
            media_type,
            file_path.name,
            len(raw),
        )
        return base64.standard_b64encode(raw).decode("ascii"), media_type

    except FileNotFoundError:
        pass  # osascript not available
    except Exception as exc:
        logger.debug("File URL clipboard read failed: %s", exc)
    return None


def _try_pbpaste() -> tuple[str, str] | None:
    """Read plain text from macOS clipboard via the built-in pbpaste utility."""
    try:
        proc = subprocess.run(
            ["pbpaste"],
            capture_output=True,
            timeout=5.0,
        )
        if proc.returncode == 0 and proc.stdout:
            try:
                if proc.stdout.decode("utf-8").strip():
                    return (
                        base64.standard_b64encode(proc.stdout).decode("ascii"),
                        "text/plain",
                    )
            except UnicodeDecodeError:
                pass
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.debug("pbpaste clipboard read failed: %s", exc)
    return None


# ── Linux ──────────────────────────────────────────────────────────────────────


def _get_clipboard_content_linux() -> tuple[str, str] | None:
    """Read content from the Linux X11/Wayland clipboard.

    Strategy
    --------
    1. Ask **xclip** for each supported MIME type in priority order.
    2. Fall back to **xsel** raw output with magic-byte type detection.

    Note: both tools require a running display server (``$DISPLAY`` for X11,
    ``$WAYLAND_DISPLAY`` for Wayland).  Over a headless SSH session the OSC 52
    path (handled earlier in :func:`get_clipboard_content`) is the right one.
    """
    # Try explicit MIME types via xclip (most accurate)
    for mime in ("image/png", "application/pdf", "image/jpeg", "text/plain"):
        result = _try_xclip(mime)
        if result:
            return result

    # xsel raw-bytes fallback with magic detection
    result = _try_xsel_raw()
    if result:
        return result

    logger.debug("No supported content found in Linux clipboard (tried xclip, xsel)")
    return None


def _try_xclip(mime: str) -> tuple[str, str] | None:
    """Request a specific MIME type from the Linux clipboard via xclip."""
    try:
        proc = subprocess.run(
            ["xclip", "-selection", "clipboard", "-t", mime, "-o"],
            capture_output=True,
            timeout=5.0,
        )
        if proc.returncode == 0 and proc.stdout:
            # Validate the returned bytes actually match the requested type
            detected = _detect_media_type(proc.stdout)
            if detected == mime:
                return (
                    base64.standard_b64encode(proc.stdout).decode("ascii"),
                    mime,
                )
    except FileNotFoundError:
        pass  # xclip not installed
    except Exception as exc:
        logger.debug("xclip read failed for %s: %s", mime, exc)
    return None


def _try_xsel_raw() -> tuple[str, str] | None:
    """Read raw bytes from the Linux clipboard via xsel and auto-detect type."""
    try:
        proc = subprocess.run(
            ["xsel", "--clipboard", "--output"],
            capture_output=True,
            timeout=5.0,
        )
        if proc.returncode == 0 and proc.stdout:
            media_type = _detect_media_type(proc.stdout)
            if media_type:
                return (
                    base64.standard_b64encode(proc.stdout).decode("ascii"),
                    media_type,
                )
    except FileNotFoundError:
        pass  # xsel not installed
    except Exception as exc:
        logger.debug("xsel clipboard read failed: %s", exc)
    return None
