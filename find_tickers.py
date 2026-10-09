"""Find the Trading 212 tickers for the ETFs in the index universe.

Yahoo Finance and Trading 212 use different ticker formats, so before the
index can be traded each ETF has to be matched to its Trading 212 instrument.
This only reads the instrument list; it places no orders.
"""

from t212_client import T212Client

SYMBOLS = ["VWRL", "IGLT", "INXG", "SGLN", "COMM"]

client = T212Client()  # demo account
instruments = client.get_instruments()

for symbol in SYMBOLS:
    matches = [i for i in instruments
               if i.get("shortName") == symbol or i["ticker"].startswith(symbol)]
    print(f"\n{symbol}: {len(matches)} match(es)")
    for i in matches:
        print(f"  {i['ticker']:16s} {i.get('currencyCode', ''):4s} "
              f"{i.get('type', ''):6s} {i.get('isin', ''):13s} {i.get('name', '')}")
