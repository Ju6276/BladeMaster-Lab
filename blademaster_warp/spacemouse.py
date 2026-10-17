"""USB SpaceMouse input. Never sends commands to a real robot.

HID protocol: 3Dconnexion report 1 is six signed little-endian axes (or XYZ
only), report 2 is rotation on older devices, report 3 is the button bits.
Status and battery reports do not refresh motion. Axis signs follow the
operator convention used by the teleop client: raw [1, -1, -1, 1, -1, 1],
then mapped to world axes by hid_twist.
"""

from collections import deque
from dataclasses import dataclass
import sys
import threading
import time

import numpy as np


@dataclass(frozen=True)
class MouseSnapshot:
    axes: np.ndarray
    received: np.ndarray
    buttons: int
    presses: tuple
    error: str = ""


class MouseState:
    def __init__(self):
        self.lock = threading.Lock()
        self.axes = np.zeros(6)
        self.received = np.full(2, -np.inf)
        self.buttons = 0
        self.presses = deque(maxlen=16)
        self.error = ""

    def feed(self, report, now=None):
        now = time.monotonic() if now is None else now
        if not report:
            return
        with self.lock:
            if report[0] == 1 and len(report) in (7, 13):
                values = np.frombuffer(bytes(report[1:]), dtype="<i2").astype(float)
                self.axes[: len(values)] = np.clip(values / 350.0, -1.0, 1.0)
                self.received[0] = now
                if len(values) == 6:
                    self.received[1] = now
            elif report[0] == 2 and len(report) == 7:
                self.axes[3:] = np.clip(np.frombuffer(bytes(report[1:]), dtype="<i2") / 350.0, -1.0, 1.0)
                self.received[1] = now
            elif report[0] == 3 and len(report) >= 2:
                buttons = int(report[1]) & 3
                rising = buttons & ~self.buttons
                # Both buttons together hold rather than choose a direction.
                if buttons == 3 and rising:
                    self.presses.append("hold")
                elif rising:
                    self.presses.append("close" if rising & 1 else "open")
                self.buttons = buttons

    def sample(self):
        with self.lock:
            result = MouseSnapshot(
                self.axes.copy(), self.received.copy(), self.buttons, tuple(self.presses), self.error
            )
            self.presses.clear()
        return result


def hid_module():
    """hidapi. On Linux use the hidraw backend: the default `hid` module opens
    the device through libusb and contends with the kernel driver."""
    if sys.platform.startswith("linux"):
        import hidraw

        sys.modules.setdefault("hid", hidraw)
    import hid

    return hid


class SpaceMouse:
    """One HID reader. Keeps the latest axes, not a motion queue."""

    def __init__(self):
        self.state = MouseState()
        self.stop = threading.Event()
        self.thread = None
        self.device = None
        self.info = None

    def __enter__(self):
        hid = hid_module()
        # Multi-axis controller usage excludes duplicate vendor interfaces.
        devices = {
            d["path"]: d
            for d in hid.enumerate()
            if d["vendor_id"] in (0x256F, 0x046D)
            and "spacemouse" in (d.get("product_string") or "").lower()
            and d.get("usage_page") == 1
            and d.get("usage") == 8
        }
        if not devices and sys.platform.startswith("linux") and self._open_spacenavd():
            return self
        if len(devices) != 1:
            raise RuntimeError(
                f"Expected one USB SpaceMouse, found {len(devices)}. Connect its USB data cable."
            )
        path, description = next(iter(devices.items()))
        self.info = {k: description.get(k) for k in ("product_string", "vendor_id", "product_id")}
        # macOS otherwise may route the same relative-axis reports to the desktop pointer.
        if sys.platform == "darwin":
            import ctypes

            backend = ctypes.CDLL(hid.__file__)
            exclusive = backend.hid_darwin_set_open_exclusive
            exclusive.argtypes = [ctypes.c_int]
            exclusive.restype = None
            exclusive(1)
            self.info["exclusive_hid"] = True
        self.device = hid.device()
        self.device.open_path(path)
        self.thread = threading.Thread(target=self._read, name="SpaceMouse-HID", daemon=True)
        self.thread.start()
        return self

    def _read(self):
        try:
            while not self.stop.is_set():
                report = self.device.read(64, 50)
                self.state.feed(report)
        except Exception as exc:
            with self.state.lock:
                self.state.error = f"SpaceMouse disconnected: {exc}"

    def _open_spacenavd(self):
        """Linux fallback when spacenavd grabs the device and hidraw is root-only."""
        import ctypes
        import os

        try:
            lib = ctypes.CDLL("libspnav.so.0")
        except OSError:
            return False
        if not os.path.exists("/var/run/spnav.sock") or lib.spnav_open() == -1:
            return False
        vendor, product = ctypes.c_uint(), ctypes.c_uint()
        name = ctypes.create_string_buffer(128)
        # Raw axis and button events bypass spacenavd's inversion, axis map and sensitivity.
        if (
            lib.spnav_protocol() < 1
            or lib.spnav_evmask(0x34) == -1
            or lib.spnav_dev_usbid(ctypes.byref(vendor), ctypes.byref(product)) == -1
            or vendor.value not in (0x256F, 0x046D)
        ):
            lib.spnav_close()
            return False
        lib.spnav_dev_name(name, len(name))
        self.device = lib
        self.info = dict(
            product_string=name.value.decode(errors="replace"),
            vendor_id=vendor.value,
            product_id=product.value,
            backend="spacenavd",
        )
        self.thread = threading.Thread(target=self._read_spacenavd, name="SpaceMouse-spnav", daemon=True)
        self.thread.start()
        return True

    def _read_spacenavd(self):
        import ctypes
        import os
        import select
        import socket

        lib, event = self.device, (ctypes.c_int * 32)()
        raw, buttons = np.zeros(6, dtype="<i2"), 0
        conn = socket.socket(fileno=os.dup(lib.spnav_fd()))
        try:
            while not self.stop.is_set():
                if select.select([conn], [], [], 0.05)[0] and not conn.recv(
                    1, socket.MSG_PEEK | socket.MSG_DONTWAIT
                ):
                    raise ConnectionError("spacenavd closed the connection")
                while lib.spnav_poll_event(ctypes.byref(event)):
                    kind, a, b = event[0], event[1], event[2]
                    if kind == 5 and 0 <= a < 6:
                        raw[a] = np.clip(b, -32768, 32767)
                    elif kind == 6 and b in (0, 1):
                        buttons = buttons | (1 << b) if a else buttons & ~(1 << b)
                        self.state.feed([3, buttons])
                    elif kind == 3 and a == 1:
                        raise ConnectionError("device removed")
                # Absolute evdev axes only report changes, so a held deflection stays current.
                self.state.feed(bytes([1]) + raw.tobytes())
        except Exception as exc:
            with self.state.lock:
                self.state.error = f"SpaceMouse disconnected: {exc}"
        finally:
            conn.close()

    def __exit__(self, *exc):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.device and (self.info or {}).get("backend") == "spacenavd":
            self.device.spnav_close()
        elif self.device:
            self.device.close()
