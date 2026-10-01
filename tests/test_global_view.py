"""Network Rail Global View 4G / 5G scanner logs (2026 on): date and time apart, a field more per row than the header."""
from tcs.config import load_settings
from tcs.sources.measurements import load_measurements

HEADER_4G = ("seq_id,Latitude,Longitude,speed,gps_speed,train,date,time,mnc,mcc,operator,earfcn,dlfreq,cellid,pci,rsrp,WB_Rsrp,"
             "ptotal,rsrq,WB_Rsrq,WB_Rssi,sinr,nr,reading_count")
ROWS_4G = [   # as delivered: an empty field before the last, so every row is one longer than the header
    "1,52.8135,-1.63294,,46.2,PLPR2,20/03/2026,13:07:43,10,234,O2 (Telefónica UK),3725,952.5,153856890,246,-80.35,-74.8,-58.3,-22.07,-11.1,-46.9,-9.45,no,,1",
    "2,52.8135,-1.63294,,46.2,PLPR2,20/03/2026,13:07:43,10,234,O2 (TelefÃ³nica UK),6400,811,153856891,247,-90.00,-88.8,-60.0,-20.00,-11.0,-47.0,-5.00,no,,1",
    "3,52.8135,-1.63294,,46.2,PLPR2,20/03/2026,13:07:43,20,234,Three,6175,793.5,670984,140,-89.42,-84.91,-70.8,-18.15,-8.46,-62.09,-2.77,no,,2",
    "4,52.8136,-1.63290,,46.2,NMT,20/03/2026,13:07:43,30,234,EE,1617,1815,1,10,0.00,0,-60,-12.0,-9.0,-50,5.0,no,,1",  # no reading
    "5,54.9729,-1.60543,,37.2,NMT,22/03/2026,16:16:36,15,234,Vodafone UK,6300,806,131328624,23,-68.56,-65.22,-51.31,-17.33,-9.54,-38.64,-5.19,yes,,2",
]
HEADER_5G = "seq_id,Latitude,Longitude,speed,gps_speed,train,date,time,mnc,mcc,operator,nrarfcn,dlfreq,rssi,pci,rsrp,ssb_idx,rsrq,sinr,reading_count"
ROWS_5G = [
    "7,51.5847,-2.76454,,26.5,NMT,27/03/2026,12:03:23,10,234,O2 (TelefÃ³nica UK),152690,763.45,-67.34,612,-111.94,0,-31.16,-20.78,,1",
    "8,51.5847,-2.76454,,26.5,NMT,27/03/2026,12:03:23,10,234,O2 (Telefónica UK),152690,763.45,-67.34,613,-104.50,1,-25.00,-12.00,,1",
    "9,51.4968,-0.070002,,47.2,PLPR2,24/03/2026,05:52:30,30,234,EE,156510,782.55,-47.19,264,-204.60,1,-37.55,-27.28,,1",
]


def _write(tmp_path, name, header, rows):
    f = tmp_path / name
    f.write_bytes(("﻿" + header + "\r\n" + "\r\n".join(rows) + "\r\n").encode("utf-8"))
    return f


def test_global_view_4g_reads_the_columns_where_they_are_and_keeps_the_strongest_carrier(tmp_path):
    ops = load_settings(offline=True).operators
    m = load_measurements(_write(tmp_path, "gv4.csv", HEADER_4G, ROWS_4G), "global_view_4g", ops)
    assert sorted(m["provider_id"]) == ["o2", "three", "vodafone"]           # EE's 0 dBm is "no reading", not a signal
    o2 = m[m["provider_id"] == "o2"].iloc[0]
    assert o2["rsrp_dbm"] == -80.35 and o2["channel"] == 3725                # the stronger of O2's two carriers, either spelling
    assert str(o2["timestamp"]) == "2026-03-20 13:07:43+00:00"               # 20 March, day first
    assert set(m["radio"]) == {"4G"} and set(m["device"]) == {"PLPR2", "NMT"}
    assert m.loc[m["provider_id"] == "three", "sinr_db"].item() == -2.77     # not shifted by the extra field


def test_global_view_5g_keeps_the_strongest_beam_and_drops_impossible_readings(tmp_path):
    ops = load_settings(offline=True).operators
    m = load_measurements(_write(tmp_path, "gv5.csv", HEADER_5G, ROWS_5G), "global_view_5g", ops)
    assert m["provider_id"].tolist() == ["o2"] and m["rsrp_dbm"].item() == -104.5   # EE's -204.6 dBm is not a reading
    assert set(m["radio"]) == {"5G"} and m["channel"].item() == 152690
