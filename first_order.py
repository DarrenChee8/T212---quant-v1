import time
from t212_client import T212Client

client = T212Client()  # demo account

order = client.place_market_order("AAPL_US_EQ", 1, dry_run=False)
print("Placed:", order)

time.sleep(3)
pending = client.get_orders()
print("Pending orders:", pending)

for o in pending:
    client.cancel_order(o["id"])
    print("Cancelled order", o["id"])

time.sleep(3)
print("Pending after cancel:", client.get_orders())