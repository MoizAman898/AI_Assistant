"""
reminders.py - message customers before their booking.

  python reminders.py                  check all agent databases once
  python reminders.py healthcare       check one agent only
  python reminders.py --loop           keep running, check every 15 minutes

A reminder is sent once per booking, REMINDER_HOURS (default 24) before the start
time, by email or SMS/WhatsApp (see .env.example) and always logged to
customer_messages.log. Schedule it with cron / Windows Task Scheduler every
15-30 minutes, or just leave `--loop` running.
"""
import argparse
import sqlite3
import time
from pathlib import Path

from actions import send_due_reminders

DOMAINS = ["ecommerce", "education", "healthcare", "marketing", "realestate", "restaurant"]
BASE = Path(__file__).resolve().parent


def run_once(domains):
    total = 0
    for d in domains:
        path = BASE / f"{d}.db"
        if not path.exists():
            continue
        db = sqlite3.connect(path)
        try:
            n = send_due_reminders(db)
        except sqlite3.OperationalError:
            n = 0  # no bookings table yet (agent never booked anything)
        finally:
            db.close()
        if n:
            print(f"[{d}] sent {n} reminder(s)")
        total += n
    return total


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("domain", nargs="?", choices=DOMAINS)
    p.add_argument("--loop", action="store_true")
    a = p.parse_args()
    targets = [a.domain] if a.domain else DOMAINS
    if a.loop:
        print("Reminder loop running (Ctrl+C to stop)...")
        while True:
            run_once(targets)
            time.sleep(15 * 60)
    else:
        print(f"Done. {run_once(targets)} reminder(s) sent.")
