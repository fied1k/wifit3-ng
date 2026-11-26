"""Regression tests for upstream derv82/wifit3 v0.7.2...v0.7.3 backported features."""
from __future__ import annotations

import struct

from wifit3.crack.hc22000_format import parse_hc22000, pmkid_hashline
from wifit3.crack.passwords import ssid_variants
from wifit3.dot11.ie import ssid_ie, ssid_text
from wifit3.dot11.packet import AssocRequestPacket, AssocRespPacket, BeaconPacket
from wifit3.dot11.probe import wpa2_beacon
from wifit3.dot11.wpa import derive_pmk, is_valid_psk, psk_bytes
from wifit3.models import AccessPoint, Handshake
from wifit3.persist import config as config_mod
from wifit3.persist.config import Config
from wifit3.persist.pcap import (
    LINKTYPE_IEEE802_11,
    LINKTYPE_IEEE802_11_RADIOTAP,
    PCAP_MAGIC_NANOSECONDS,
    PCAPNG_BYTE_ORDER_MAGIC,
    PCAPNG_ENHANCED_PACKET_BLOCK,
    PCAPNG_INTERFACE_DESCRIPTION_BLOCK,
    PCAPNG_SECTION_HEADER_BLOCK,
    read_access_points,
    read_pcap,
    write_pcap,
)
from wifit3.ui.screens.filter import ScanFilter
from wifit3.wlan.sink import WlanSink


def test_config_nicknames_round_trip(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr(config_mod, "_PATH", cfg_path)
    monkeypatch.setattr(Config, "nicknames", None)

    Config.set_nickname("AA:BB:CC:DD:EE:FF", "Office Router")
    assert Config.nickname("aa:bb:cc:dd:ee:ff") == "Office Router"
    Config.save()

    Config.nicknames = None
    Config.load()
    assert Config.nickname("AA:BB:CC:DD:EE:FF") == "Office Router"

    Config.set_nickname("AA:BB:CC:DD:EE:FF", "")
    assert Config.nickname("aa:bb:cc:dd:ee:ff") is None


def test_scan_filter_matches_nickname(monkeypatch):
    monkeypatch.setattr(Config, "nicknames", {"aa:bb:cc:dd:ee:ff": "BasementAP"})
    ap = AccessPoint(bssid="aa:bb:cc:dd:ee:ff", ssid="MyWiFi")
    assert ScanFilter(text="basement").matches(ap)
    assert not ScanFilter(text="attic").matches(ap)


def test_non_utf8_ssid_bytes_hashline_and_hex_password():
    raw_ssid = b"Caf\xe9"
    assert ssid_text(raw_ssid) == "Caf"
    assert ssid_text(b"\xd6\xd0\xce\xc4") == "\\xd6\\xd0\\xce\\xc4"
    assert ssid_ie(raw_ssid) == b"\x00\x04Caf\xe9"

    hs = Handshake(bssid="aa:bb:cc:dd:ee:ff", client_mac="11:22:33:44:55:66", pmkid=b"\xab" * 16)
    line = pmkid_hashline(raw_ssid, hs)
    assert line is not None
    parsed = parse_hc22000(line)
    assert parsed is not None
    assert parsed.ssid_bytes == raw_ssid
    assert parsed.ssid_displayable == "Café"

    variants = ssid_variants(raw_ssid)
    assert any(v.startswith("$HEX[") and v.endswith("]") for v in variants)
    hex_psk = f"$HEX[{(raw_ssid + b'1234').hex()}]"
    assert is_valid_psk(hex_psk)
    assert psk_bytes(hex_psk) == raw_ssid + b"1234"
    assert derive_pmk(hex_psk, raw_ssid) == derive_pmk(
        (raw_ssid + b"1234").decode("utf-8", errors="surrogateescape"), raw_ssid
    )


def test_read_pcap_microseconds_nanoseconds_radiotap_and_pcapng(tmp_path):
    frame = wpa2_beacon(b"\xaa\xbb\xcc\xdd\xee\xff", b"Caf\xe9", 6)

    # Standard microsecond pcap
    p1 = tmp_path / "std.pcap"
    write_pcap(p1, [(frame, 1700000000.25)])
    recs = list(read_pcap(p1))
    assert len(recs) == 1
    assert recs[0][0] == frame
    assert abs(recs[0][1] - 1700000000.25) < 1e-5

    # Big-endian nanosecond pcap with radiotap + FCS flag
    radiotap = struct.pack("<BBHI", 0, 0, 9, 0x00000002) + b"\x10"  # flags=FCS present
    payload = radiotap + frame + b"\xde\xad\xbe\xef"
    p2 = tmp_path / "ns_rt.pcap"
    hdr = struct.pack(">IHHiIII", PCAP_MAGIC_NANOSECONDS, 2, 4, 0, 0, 65535, LINKTYPE_IEEE802_11_RADIOTAP)
    rec = struct.pack(">IIII", 1700000001, 500_000_000, len(payload), len(payload))
    p2.write_bytes(hdr + rec + payload)
    recs2 = list(read_pcap(p2))
    assert len(recs2) == 1
    assert recs2[0][0] == frame
    assert abs(recs2[0][1] - 1700000001.5) < 1e-6

    # Minimal pcapng
    def _block(btype: int, body: bytes) -> bytes:
        pad = (-len(body)) % 4
        total = 12 + len(body) + pad
        return struct.pack("<II", btype, total) + body + (b"\x00" * pad) + struct.pack("<I", total)

    shb = _block(PCAPNG_SECTION_HEADER_BLOCK, struct.pack("<IHHq", PCAPNG_BYTE_ORDER_MAGIC, 1, 0, -1))
    idb = _block(PCAPNG_INTERFACE_DESCRIPTION_BLOCK, struct.pack("<HHI", LINKTYPE_IEEE802_11, 0, 65535))
    ts_us = 1700000002_000_000
    epb = _block(
        PCAPNG_ENHANCED_PACKET_BLOCK,
        struct.pack("<IIIII", 0, ts_us >> 32, ts_us & 0xFFFFFFFF, len(frame), len(frame)) + frame,
    )
    p3 = tmp_path / "cap.pcapng"
    p3.write_bytes(shb + idb + epb)
    recs3 = list(read_pcap(p3))
    assert len(recs3) == 1
    assert recs3[0][0] == frame
    assert abs(recs3[0][1] - 1700000002.0) < 1e-5

    # read_access_points preserves non-UTF-8 SSID bytes and displayable_ssid
    aps = read_access_points(p3)
    assert len(aps) == 1
    assert aps[0].ssid_bytes == b"Caf\xe9"
    assert aps[0].ssid_displayable == "Café"


def test_sink_assoc_decloak_requires_status_0_within_window():
    sink = WlanSink()
    bssid = "aa:bb:cc:dd:ee:ff"
    sta = "12:22:33:44:55:66"
    sink.update(
        BeaconPacket(
            type="beacon", type_id=0, subtype_id=8, bssid=bssid, source=bssid,
            dest="ff:ff:ff:ff:ff:ff", to_ds=False, from_ds=False, rssi=-50,
            raw=b"\x80" * 36, ssid=None, ssid_bytes=b"",
        ),
        "wlan0",
        timestamp=10.0,
    )
    sink.update(
        AssocRequestPacket(
            type="assoc_req", type_id=0, subtype_id=0, bssid=bssid, source=sta,
            dest=bssid, to_ds=False, from_ds=False, rssi=-50,
            raw=b"\x00" * 36, ssid="Secret", ssid_bytes=b"Secret",
        ),
        "wlan0",
        timestamp=10.01,
    )
    # Rejected status != 0 does not decloak
    sink.update(
        AssocRespPacket(
            type="assoc_resp", type_id=0, subtype_id=1, bssid=bssid, source=bssid,
            dest=sta, to_ds=False, from_ds=False, rssi=-50,
            raw=b"\x10" * 30, status=12,
        ),
        "wlan0",
        timestamp=10.02,
    )
    assert sink.access_points[bssid].is_hidden
