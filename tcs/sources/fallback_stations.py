"""Approximate station coordinates (lat, lon) for offline mode, keyed by CRS code.

Covers every station in config/route.yaml and config/routes/*.yaml, so `tcs run --offline --route <id>` and
`tcs build-all --offline` work for the whole catalogue. Offline routes are synthetic splines through these points
(flagged synthetic everywhere downstream), so ~100 m precision is plenty. A new route file needs its stations here
too; tests/test_units.py checks the catalogue is covered.

Sources: the reference route's (ECML) entries are hand-entered. All others are from uk-railway-stations 1.11.1 by
David Wheatley (https://github.com/davwheat/uk-railway-stations), derived from Trainline EU's stations dataset
(https://github.com/trainline-eu/stations) and its sources, used under the Open Database License (ODbL-1.0);
rounded to 4 decimal places.
"""

STATIONS: dict[str, tuple[float, float]] = {
    "AFK": (51.1434, 0.8752),  # Ashford International
    "AGV": (51.8174, -3.0090),  # Abergavenny
    "ALM": (55.3927, -1.6367),  # Alnmouth (hand-entered)
    "APP": (54.5803, -2.4867),  # Appleby
    "AVM": (57.1883, -3.8289),  # Aviemore
    "BAN": (52.0609, -1.3275),  # Banbury
    "BCS": (51.9033, -1.1498),  # Bicester North
    "BCU": (50.8167, -1.5741),  # Brockenhurst
    "BDM": (52.1362, -0.4794),  # Bedford
    "BGN": (51.5071, -3.5753),  # Bridgend
    "BHM": (52.4776, -1.8987),  # Birmingham New Street
    "BMH": (50.7275, -1.8640),  # Bournemouth
    "BMO": (52.4791, -1.8925),  # Birmingham Moor Street
    "BNG": (53.2229, -4.1355),  # Bangor
    "BOD": (50.4458, -4.6630),  # Bodmin Parkway
    "BPW": (51.5139, -2.5430),  # Bristol Parkway
    "BRI": (51.4491, -2.5804),  # Bristol Temple Meads
    "BSK": (51.2684, -1.0886),  # Basingstoke
    "BTN": (50.8289, -0.1407),  # Brighton
    "BUG": (50.9536, -0.1277),  # Burgess Hill
    "BUT": (52.8058, -1.6425),  # Burton-on-Trent
    "BWK": (55.7742, -2.0110),  # Berwick-upon-Tweed (hand-entered)
    "CAR": (54.8903, -2.9330),  # Carlisle
    "CBN": (50.2104, -5.2979),  # Camborne
    "CDF": (51.4755, -3.1797),  # Cardiff Central
    "CHD": (53.2382, -1.4201),  # Chesterfield
    "CHM": (51.7366, 0.4693),  # Chelmsford
    "CLJ": (51.4641, -0.1703),  # Clapham Junction
    "CNM": (51.8970, -2.1001),  # Cheltenham Spa
    "COL": (51.9006, 0.8925),  # Colchester
    "CRE": (53.0890, -2.4326),  # Crewe
    "CRO": (55.9557, -4.0357),  # Croy
    "CTR": (53.1968, -2.8802),  # Chester
    "CWB": (53.2968, -3.7259),  # Colwyn Bay
    "CWM": (51.6571, -3.0160),  # Cwmbran
    "DAR": (54.5205, -1.5474),  # Darlington (hand-entered)
    "DBY": (52.9164, -1.4626),  # Derby
    "DCH": (50.7088, -2.4376),  # Dorchester South
    "DEW": (53.6921, -1.6335),  # Dewsbury
    "DHM": (54.7793, -1.5817),  # Durham (hand-entered)
    "DID": (51.6113, -1.2426),  # Didcot Parkway
    "DIS": (52.3736, 1.1235),  # Diss
    "DON": (53.5220, -1.1400),  # Doncaster (hand-entered)
    "DUN": (55.9985, -2.5139),  # Dunbar (hand-entered)
    "DVP": (51.1260, 1.3042),  # Dover Priory
    "EBD": (51.4435, 0.3214),  # Ebbsfleet International
    "ECR": (51.3757, -0.0933),  # East Croydon
    "EDB": (55.9520, -3.1883),  # Edinburgh Waverley (hand-entered)
    "EMD": (52.8620, -1.2630),  # East Midlands Parkway
    "EUS": (51.5289, -0.1342),  # London Euston
    "EXD": (50.7292, -3.5436),  # Exeter St Davids
    "FKC": (51.0830, 1.1683),  # Folkestone Central
    "FKK": (55.9919, -3.7923),  # Falkirk High
    "FLN": (53.2495, -3.1330),  # Flint
    "GLC": (55.8583, -4.2584),  # Glasgow Central
    "GLQ": (55.8626, -4.2511),  # Glasgow Queen Street
    "GRA": (52.9111, -0.6420),  # Grantham (hand-entered)
    "GTW": (51.1564, -0.1610),  # Gatwick Airport
    "HFD": (52.0612, -2.7081),  # Hereford
    "HHD": (53.3077, -4.6311),  # Holyhead
    "HHE": (51.0049, -0.1054),  # Haywards Heath
    "HUD": (53.6484, -1.7851),  # Huddersfield
    "HWY": (51.6294, -0.7446),  # High Wycombe
    "HYM": (55.9452, -3.2194),  # Haymarket
    "INV": (57.4802, -4.2232),  # Inverness
    "IPS": (52.0506, 1.1445),  # Ipswich
    "KEI": (53.8679, -1.9011),  # Keighley
    "KET": (52.3932, -0.7317),  # Kettering
    "KGX": (51.5308, -0.1238),  # London King's Cross (hand-entered)
    "KIN": (57.0778, -4.0532),  # Kingussie
    "KSW": (54.4551, -2.3686),  # Kirkby Stephen
    "LAN": (54.0486, -2.8079),  # Lancaster
    "LBO": (52.7796, -1.1965),  # Loughborough
    "LDS": (53.7944, -1.5486),  # Leeds
    "LEI": (52.6321, -1.1236),  # Leicester
    "LIN": (55.9764, -3.5958),  # Linlithgow
    "LLJ": (53.2840, -3.8091),  # Llandudno Junction
    "LMS": (52.2846, -1.5358),  # Leamington Spa
    "LOC": (55.1224, -3.3538),  # Lockerbie
    "LSK": (50.4468, -4.4695),  # Liskeard
    "LST": (51.5182, -0.0814),  # London Liverpool Street
    "LTV": (52.6869, -1.8000),  # Lichfield Trent Valley
    "LUD": (52.3711, -2.7160),  # Ludlow
    "LUT": (51.8825, -0.4141),  # Luton
    "MAC": (53.2593, -2.1214),  # Macclesfield
    "MAN": (53.4772, -2.2301),  # Manchester Piccadilly
    "MHR": (52.4804, -0.9089),  # Market Harborough
    "MKC": (52.0342, -0.7748),  # Milton Keynes Central
    "MNG": (51.9491, 1.0449),  # Manningtree
    "MPT": (55.1622, -1.6828),  # Morpeth (hand-entered)
    "MTH": (55.7919, -3.9952),  # Motherwell
    "MYB": (51.5244, -0.1636),  # London Marylebone
    "NCL": (54.9683, -1.6174),  # Newcastle (hand-entered)
    "NNG": (53.0817, -0.7986),  # Newark North Gate (hand-entered)
    "NRW": (52.6263, 1.3077),  # Norwich
    "NTA": (50.5302, -3.5992),  # Newton Abbot
    "NTH": (51.6623, -3.8073),  # Neath
    "NTR": (54.3330, -1.4410),  # Northallerton (hand-entered)
    "NUN": (52.5268, -1.4642),  # Nuneaton
    "NWP": (51.5888, -3.0004),  # Newport
    "OXN": (54.3052, -2.7222),  # Oxenholme Lake District
    "PAD": (51.5171, -0.1773),  # London Paddington
    "PAR": (50.3557, -4.7046),  # Par
    "PBO": (52.5746, -0.2500),  # Peterborough (hand-entered)
    "PIT": (56.7026, -3.7361),  # Pitlochry
    "PLY": (50.3778, -4.1434),  # Plymouth
    "PMT": (55.9845, -3.7145),  # Polmont
    "PNR": (54.6620, -2.7588),  # Penrith North Lakes
    "PNZ": (50.1224, -5.5320),  # Penzance
    "POO": (50.7193, -1.9836),  # Poole
    "PRE": (53.7557, -2.7072),  # Preston
    "PRT": (53.3365, -3.4072),  # Prestatyn
    "PTA": (51.5924, -3.7817),  # Port Talbot Parkway
    "PTH": (56.3915, -3.4384),  # Perth
    "RDG": (51.4592, -0.9723),  # Reading
    "RED": (50.2331, -5.2259),  # Redruth
    "RET": (53.3153, -0.9475),  # Retford (hand-entered)
    "RHD": (54.2058, -2.3609),  # Ribblehead
    "RHL": (53.3184, -3.4891),  # Rhyl
    "RUG": (52.3790, -1.2503),  # Rugby
    "SAU": (50.3396, -4.7893),  # St Austell
    "SER": (50.1706, -5.4439),  # St Erth
    "SET": (54.0669, -2.2807),  # Settle
    "SFA": (51.5448, -0.0087),  # Stratford International
    "SHF": (53.3784, -1.4621),  # Sheffield
    "SHR": (52.7119, -2.7494),  # Shrewsbury
    "SHY": (53.8336, -1.7736),  # Shipley
    "SKI": (53.9587, -2.0259),  # Skipton
    "SMK": (52.1901, 1.0007),  # Stowmarket
    "SOL": (52.4146, -1.7889),  # Solihull
    "SOT": (53.0080, -2.1811),  # Stoke-on-Trent
    "SOU": (50.9077, -1.4140),  # Southampton Central
    "SPT": (53.4055, -2.1630),  # Stockport
    "SRA": (51.5413, -0.0035),  # Stratford
    "STA": (52.8036, -2.1226),  # Stafford
    "STG": (56.1197, -3.9343),  # Stirling
    "STP": (51.5327, -0.1270),  # London St Pancras
    "SVG": (51.9017, -0.2069),  # Stevenage (hand-entered)
    "SWA": (51.6257, -3.9404),  # Swansea
    "SWI": (51.5657, -1.7853),  # Swindon
    "SYB": (53.4841, -2.0642),  # Stalybridge
    "TAM": (52.6374, -1.6871),  # Tamworth
    "TAU": (51.0233, -3.1027),  # Taunton
    "TBD": (51.1175, -0.1611),  # Three Bridges
    "TRU": (50.2640, -5.0642),  # Truro
    "VIC": (51.4947, -0.1446),  # London Victoria
    "WAT": (51.5028, -0.1128),  # London Waterloo
    "WBQ": (53.3859, -2.6032),  # Warrington Bank Quay
    "WEY": (50.6159, -2.4549),  # Weymouth
    "WFJ": (51.6639, -0.3961),  # Watford Junction
    "WGN": (53.5434, -2.6330),  # Wigan North Western
    "WIN": (51.0673, -1.3206),  # Winchester
    "WKF": (53.6822, -1.5056),  # Wakefield Westgate
    "WML": (53.3271, -2.2260),  # Wilmslow
    "WOK": (51.3185, -0.5578),  # Woking
    "WRM": (50.6929, -2.1153),  # Wareham
    "WRP": (52.2860, -1.6123),  # Warwick Parkway
    "YRK": (53.9581, -1.0931),  # York (hand-entered)
}

# United States, keyed by Amtrak station code (routes with `country: US`). From the US DOT's NTAD Amtrak Stations
# layer (public domain; tcs/sources/ntad_amtrak.py), rounded to 4 decimal places.
US_STATIONS: dict[str, tuple[float, float]] = {
    "BAL": (39.3073, -76.6157),  # Baltimore Penn Station, MD
    "BBY": (42.3473, -71.0758),  # Boston Back Bay, MA
    "BOS": (42.3523, -71.0553),  # Boston South Station, MA
    "BWI": (39.1924, -76.6943),  # BWI Marshall Airport, MD
    "KIN": (41.4840, -71.5606),  # Kingston, RI
    "MET": (40.5681, -74.3296),  # Metropark, NJ
    "MYS": (41.3509, -71.9631),  # Mystic, CT
    "NHV": (41.2977, -72.9267),  # New Haven Union Station, CT
    "NLC": (41.3543, -72.0932),  # New London, CT
    "NWK": (40.7347, -74.1648),  # Newark Penn Station, NJ
    "NYP": (40.7510, -73.9963),  # New York Penn Station, NY
    "OSB": (41.3004, -72.3768),  # Old Saybrook, CT
    "PHL": (39.9556, -75.1810),  # Philadelphia 30th Street, PA
    "PVD": (41.8295, -71.4135),  # Providence, RI
    "RTE": (42.2102, -71.1479),  # Route 128, MA
    "STM": (41.0471, -73.5422),  # Stamford, CT
    "WAS": (38.8970, -77.0064),  # Washington Union Station, DC
    "WIL": (39.7373, -75.5511),  # Wilmington, DE
    "WLY": (41.3811, -71.8298),  # Westerly, RI
}

BY_COUNTRY: dict[str, dict[str, tuple[float, float]]] = {"GB": STATIONS, "US": US_STATIONS}


def for_country(country: str | None) -> dict[str, tuple[float, float]]:
    """The offline coordinate table for a route's country (GB when unset, as before)."""
    c = (country or "GB").upper()
    return BY_COUNTRY.get("GB" if c == "UK" else c, STATIONS)
