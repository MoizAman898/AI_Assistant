"""
owner_panel.py - tiny owner/admin view for any agent database.

  python owner_panel.py restaurant list bookings
  python owner_panel.py restaurant list orders
  python owner_panel.py restaurant list tickets
  python owner_panel.py restaurant list leads
  python owner_panel.py restaurant status orders 3 preparing
  python owner_panel.py healthcare status bookings 5 completed

Order statuses you can use: placed, preparing, shipped, delivered, cancelled.
Customers see the new status when they ask the agent to check their order.
"""
import argparse
import sqlite3
import sys
from pathlib import Path

DOMAINS = ["ecommerce", "education", "healthcare", "marketing", "realestate", "restaurant"]
COLUMNS = {
    "bookings": "id, kind, name, detail, date, time, size, contact, status",
    "orders": "id, kind, name, items, total, notes, status, created_at",
    "tickets": "id, name, issue, contact, status, created_at",
    "leads": "id, kind, name, detail, contact, created_at",
}


def main():
    p = argparse.ArgumentParser(description="Owner panel for agent databases")
    p.add_argument("domain", choices=DOMAINS)
    sub = p.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("table", choices=COLUMNS)
    st = sub.add_parser("status")
    st.add_argument("table", choices=["bookings", "orders", "tickets"])
    st.add_argument("row_id", type=int)
    st.add_argument("new_status")
    a = p.parse_args()

    path = Path(__file__).resolve().parent / f"{a.domain}.db"
    if not path.exists():
        sys.exit(f"{path.name} not found. Run the {a.domain} agent once first.")
    db = sqlite3.connect(path)
    try:
        if a.cmd == "list":
            rows = db.execute(f"SELECT {COLUMNS[a.table]} FROM {a.table} ORDER BY id DESC").fetchall()
            print(COLUMNS[a.table])
            print("-" * 60)
            for r in rows:
                print(" | ".join(str(x) for x in r))
            if not rows:
                print("(empty)")
        else:
            cur = db.execute(f"UPDATE {a.table} SET status=? WHERE id=?", (a.new_status, a.row_id))
            db.commit()
            print("Updated." if cur.rowcount else "No such id.")
    except sqlite3.OperationalError as e:
        sys.exit(f"Table not ready yet ({e}). Run the agent once first.")


if __name__ == "__main__":
    main()
