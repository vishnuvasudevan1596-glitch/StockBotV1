import os
import requests
import time

print("DEBUG: bot.py started (minimal test)", flush=True)

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

print(f"TOKEN present: {bool(TOKEN)}", flush=True)
print(f"CHAT_ID present: {bool(CHAT_ID)}", flush=True)

if TOKEN and CHAT_ID:
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": CHAT_ID, "text": "✅ Bot test message from Railway"}, timeout=10)
        print(f"Telegram response: {r.status_code} {r.text}", flush=True)
    except Exception as e:
        print(f"Telegram error: {e}", flush=True)
else:
    print("Missing TOKEN or CHAT_ID - cannot send", flush=True)

print("Test complete. Sleeping forever.", flush=True)
while True:
    time.sleep(60)
