"""Network Rail Yellow Train LTE scanner logs: import, best server, chunked reading."""
import pandas as pd

from tcs.config import load_settings
from tcs.sources.measurements import load_measurements

HEADER = "latitude,longitude,eastings,northings,speed,train,datetime,mnc,operator,earfcn,dlfreq,phylayercellid,pci,rsrp,cal_rsrp,total_power,rsrq,sinr"
ROWS = [
    # two EE carriers in the same second: the stronger (calibrated) one is what a modem would use
    "51.60,-1.22,453726,189936,22,Train1,2018-06-01 00:00:01,30,EE,1667,1815,1,10,-90.0,-86.0,-60,-12.0,5.0",
    "51.60,-1.22,453726,189936,22,Train1,2018-06-01 00:00:01,30,EE,6225,2680,1,11,-80.0,-76.0,-60,-11.0,9.0",
    "51.60,-1.22,453726,189936,22,Train1,2018-06-01 00:00:01,20,Three,1392,1842,1,12,-95.0,-91.0,-60,-14.0,3.0",
    "51.60,-1.22,453726,189936,22,Train2,2018-06-01 00:00:01,20,Three,1392,1842,1,12,-85.0,-81.0,-60,-14.0,3.0",   # another train
    "51.61,-1.21,453800,190000,30,Train1,2018-06-01 00:00:02,10,O2,6400,816,1,19,-77.5,-73.8,-56,-13.2,14.9",
    "55.95,-3.19,325000,673000,30,Train1,2018-06-01 00:00:03,15,Vodafone,347,2144,1,19,-76.6,-71.5,-56,-13.1,19.6",   # far away
]


def _write(tmp_path, rows):
    f = tmp_path / "yt.csv"
    f.write_bytes(("﻿" + HEADER + "\r\n" + "\r\n".join(rows) + "\r\n").encode("utf-8"))   # as delivered: BOM, CRLF
    return f


def test_yellow_train_keeps_the_strongest_carrier_per_second_and_network(tmp_path):
    ops = load_settings(offline=True).operators
    m = load_measurements(_write(tmp_path, ROWS), "yellow_train", ops)
    assert len(m) == 5                                                      # the weaker EE carrier is dropped
    ee = m[m["provider_id"] == "ee"]
    assert ee["rsrp_dbm"].tolist() == [-76.0] and ee["cell_id"].tolist() == [11]   # calibrated RSRP, not the raw -80
    assert sorted(m.loc[m["provider_id"] == "three", "device"]) == ["Train1", "Train2"]   # each train keeps its own reading
    assert set(m["provider_id"]) == {"ee", "three", "o2", "vodafone"} and set(m["radio"]) == {"4G"}


def test_large_files_are_read_in_chunks_near_the_route(tmp_path):
    ops = load_settings(offline=True).operators
    f = _write(tmp_path, ROWS)
    bbox = (-1.3, 51.5, -1.1, 51.7)                                         # the Didcot rows, not Edinburgh
    whole = load_measurements(f, "yellow_train", ops, bbox=bbox)
    chunked = load_measurements(f, "yellow_train", ops, bbox=bbox, chunksize=2)   # a second straddles two chunks
    assert len(whole) == 4 and "vodafone" not in set(whole["provider_id"])
    key = ["timestamp", "device", "provider_id", "rsrp_dbm"]
    assert chunked[key].sort_values(key).reset_index(drop=True).equals(whole[key].sort_values(key).reset_index(drop=True))

    pq = tmp_path / "yt.parquet"                                            # the condensed form reads the same
    pd.read_csv(f, encoding="utf-8-sig").to_parquet(pq)
    assert load_measurements(pq, "yellow_train", ops, bbox=bbox)[key].sort_values(key).reset_index(drop=True).equals(
        whole[key].sort_values(key).reset_index(drop=True))
