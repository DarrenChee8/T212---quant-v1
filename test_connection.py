import os
import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.environ["T212_API_KEY"]
SECRET = os.getenv("T212_API_SECRET")

BASE = "https://demo.trading212.com/api/v0"  # practice account

auth = (KEY, SECRET) if SECRET else None
headers = {} if SECRET else {"Authorization": KEY}

r = requests.get(f"{BASE}/equity/account/cash", auth=auth, headers=headers, timeout=10)
print(r.status_code)
print(r.json() if r.ok else r.text)