from t212_client import T212Client

client = T212Client()  # demo by default

print("Cash:", client.get_cash())
print("Open positions:", len(client.get_portfolio()))

instruments = client.get_instruments()
print("Tradable instruments:", len(instruments))
print("Example:", instruments[0])

client.place_market_order("AAPL_US_EQ", 1)  # dry run: prints, sends nothing