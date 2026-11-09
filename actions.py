"""
actions.py - Real task tools shared by all Zuvayra Systems agents.

Every agent gets:
  get_current_datetime      resolves "today" / "tomorrow" for the LLM
  create_support_ticket     human handoff (saved in DB + owner notified)
plus domain tools (bookings, orders, enrollments, quotes, returns ...).

Data lives in the agent's own SQLite file (tables: bookings, orders, leads,
tickets). The owner is notified on every new booking / order / lead / ticket:
  - always appended to notifications.log and printed to the console
  - emailed too when SMTP_HOST, SMTP_USER, SMTP_PASSWORD, OWNER_EMAIL are set

Customers are messaged on booking, reschedule and cancel, and reminded before the
appointment (run reminders.py). Channel: email, or SMS/WhatsApp via Twilio, always
logged to customer_messages.log.

Safety: cancel / delete tools refuse to run unless confirmed=True, and the
system prompt tells the agent to ask the user first.
"""

import datetime as dt
import json
import os
import re
import base64
import smtplib
import sqlite3
import threading
import urllib.parse
import urllib.request
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.tools import StructuredTool, tool

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "notifications.log"
CUSTOMER_LOG = BASE_DIR / "customer_messages.log"

BUSINESS = {
    "restaurant": "Restaurant", "healthcare": "Clinic", "ecommerce": "Online Store",
    "education": "Academy", "realestate": "Real Estate", "marketing": "Marketing Agency",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS bookings (
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, name TEXT NOT NULL,
    detail TEXT, date TEXT NOT NULL, time TEXT NOT NULL, size INTEGER DEFAULT 1,
    contact TEXT, status TEXT DEFAULT 'confirmed', created_at TEXT,
    reminder_sent INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, name TEXT NOT NULL,
    items TEXT, total REAL, notes TEXT, status TEXT DEFAULT 'placed', created_at TEXT);
CREATE TABLE IF NOT EXISTS leads (
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, name TEXT NOT NULL,
    detail TEXT, contact TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, issue TEXT NOT NULL,
    contact TEXT, status TEXT DEFAULT 'open', created_at TEXT);
"""


# ------------------------------------------------------------ notifications

def _a(noun: str) -> str:
    return "an" if noun[0] in "aeiou" else "a"


def _stamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def notify_owner(business: str, subject: str, body: str) -> str:
    """Log + print every notification; also email it (in the background) when SMTP is configured."""
    subject = f"[{business}] {subject}"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"--- {_stamp()} | {subject}\n{body}\n\n")
    except OSError:
        pass
    print(f"\n[NOTIFY OWNER] {subject}")

    host, user = os.getenv("SMTP_HOST"), os.getenv("SMTP_USER")
    pwd, to = os.getenv("SMTP_PASSWORD"), os.getenv("OWNER_EMAIL")
    if not (host and user and pwd and to):
        return "logged"
    threading.Thread(target=_send_email, args=(host, user, pwd, to, subject, body)).start()
    return "queued"


def _send_email(host, user, pwd, to, subject, body) -> None:
    """Runs in a background thread so the chat reply is never blocked by SMTP."""
    try:
        msg = EmailMessage()
        msg["Subject"], msg["To"] = subject, to
        msg["From"] = os.getenv("SMTP_FROM", user)
        msg.set_content(body)
        port = int(os.getenv("SMTP_PORT", "587"))
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=15) as s:
                s.login(user, pwd)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=15) as s:
                s.starttls()
                s.login(user, pwd)
                s.send_message(msg)
    except Exception as e:
        print(f"[NOTIFY] email failed, kept in log: {e}")


# ------------------------------------------------------------ scheduling

def _m(h: int, mins: int = 0) -> int:
    return h * 60 + mins


class Schedule:
    """Weekly opening hours + slot rules. hours: {weekday(Mon=0): (start_min, end_min)}"""

    def __init__(self, hours, step=30, last_offset=30, capacity=1):
        self.hours, self.step = hours, step
        self.last_offset, self.capacity = last_offset, capacity


SCHEDULES = {
    "restaurant": Schedule({**{d: (_m(11), _m(23)) for d in range(7)},
                            4: (_m(11), _m(24)), 5: (_m(11), _m(24))},
                           step=30, last_offset=60, capacity=40),
    "healthcare": Schedule({**{d: (_m(9), _m(17)) for d in range(5)}, 5: (_m(9), _m(13))},
                           step=30, last_offset=30, capacity=2),
    "realestate": Schedule({d: (_m(10), _m(18)) for d in range(6)},
                           step=60, last_offset=60, capacity=2),
    "marketing": Schedule({d: (_m(10), _m(17)) for d in range(5)},
                          step=30, last_offset=30, capacity=1),
    "education": Schedule({d: (_m(16), _m(20)) for d in range(6)},
                          step=60, last_offset=60, capacity=15),
}


def _fmt_t(mins: int) -> str:
    return f"{(mins // 60) % 24:02d}:{mins % 60:02d}"


def _parse_date(s: str) -> dt.date:
    try:
        return dt.date.fromisoformat(str(s).strip())
    except ValueError:
        raise ValueError("Date must be YYYY-MM-DD (call get_current_datetime to resolve words like 'tomorrow').")


def _parse_time(s: str) -> int:
    s = str(s).strip().upper().replace(".", "")
    for fmt in ("%H:%M", "%I:%M %p", "%I %p", "%I:%M%p", "%I%p"):
        try:
            t = dt.datetime.strptime(s, fmt)
            return t.hour * 60 + t.minute
        except ValueError:
            pass
    raise ValueError("Time must look like 19:30 (24-hour).")


def _used(db, kind, d_str, tmin, exclude=None) -> int:
    q = ("SELECT COALESCE(SUM(size),0) FROM bookings "
         "WHERE kind=? AND date=? AND time=? AND status='confirmed'")
    params = [kind, d_str, _fmt_t(tmin)]
    if exclude:
        q += " AND id!=?"
        params.append(exclude)
    return db.execute(q, params).fetchone()[0]


def _free_slots(db, kind, sch, d: dt.date, size=1, exclude=None) -> List[str]:
    hrs = sch.hours.get(d.weekday())
    if not hrs:
        return []
    now = dt.datetime.now()
    now_min = now.hour * 60 + now.minute
    out, t, last = [], hrs[0], hrs[1] - sch.last_offset
    while t <= last:
        passed = d == dt.date.today() and t <= now_min
        if not passed and _used(db, kind, d.isoformat(), t, exclude) + size <= sch.capacity:
            out.append(_fmt_t(t))
        t += sch.step
    return out


def _check_slot(db, kind, sch, d_str, t_str, size, exclude=None) -> Tuple[dt.date, int]:
    d, tm = _parse_date(d_str), _parse_time(t_str)
    if d < dt.date.today():
        raise ValueError("That date is in the past.")
    hrs = sch.hours.get(d.weekday())
    if not hrs:
        raise ValueError(f"We are closed on {d.strftime('%A')}s.")
    last = hrs[1] - sch.last_offset
    if not hrs[0] <= tm <= last:
        raise ValueError(f"On {d.strftime('%A')}s bookings run from {_fmt_t(hrs[0])} to {_fmt_t(last)}.")
    if (tm - hrs[0]) % sch.step:
        raise ValueError(f"Slots start every {sch.step} minutes from {_fmt_t(hrs[0])}.")
    now = dt.datetime.now()
    if d == dt.date.today() and tm <= now.hour * 60 + now.minute:
        raise ValueError("That time has already passed today.")
    if _used(db, kind, d.isoformat(), tm, exclude) + size > sch.capacity:
        free = _free_slots(db, kind, sch, d, size, exclude)
        raise ValueError("That slot is full. " + (
            f"Free slots that day: {', '.join(free)}." if free else "No free slots that day."))
    return d, tm


def _insert_booking(db, kind, name, detail, d, tm, size, contact) -> int:
    dup = db.execute(
        "SELECT id FROM bookings WHERE kind=? AND lower(name)=lower(?) AND date=? "
        "AND time=? AND status='confirmed'", (kind, name, d.isoformat(), _fmt_t(tm))).fetchone()
    if dup:
        raise ValueError(f"{name} already has booking #{dup[0]} at that time.")
    cur = db.execute(
        "INSERT INTO bookings (kind,name,detail,date,time,size,contact,status,created_at,reminder_sent) "
        "VALUES (?,?,?,?,?,?,?, 'confirmed', ?, ?)",
        (kind, name, detail, d.isoformat(), _fmt_t(tm), size, contact, _stamp(),
         _reminder_flag(d.isoformat(), _fmt_t(tm))))
    db.commit()
    _booking_message(kind, "confirmed", cur.lastrowid, name, d.isoformat(), _fmt_t(tm), contact)
    return cur.lastrowid



# ------------------------------------------------------------ customer messages

KIND_DOMAIN = {"reservation": "restaurant", "appointment": "healthcare", "demo_class": "education",
               "viewing": "realestate", "consultation": "marketing"}
KIND_LABEL = {"reservation": "table reservation", "appointment": "appointment", "demo_class": "demo class",
              "viewing": "property viewing", "consultation": "consultation"}


def _biz_name(kind: str) -> str:
    return os.getenv("BUSINESS_NAME") or BUSINESS[KIND_DOMAIN[kind]]


def _pretty_when(d_str: str, t_str: str) -> str:
    when = dt.datetime.fromisoformat(f"{d_str} {t_str}")
    return f"{when.strftime('%A, %B')} {when.day} at {when.strftime('%I:%M %p').lstrip('0')}"


def _reminder_flag(d_str: str, t_str: str) -> int:
    """1 if the booking is already inside the reminder window (the confirmation is enough)."""
    hours = float(os.getenv("REMINDER_HOURS", "24"))
    when = dt.datetime.fromisoformat(f"{d_str} {t_str}")
    return 1 if when - dt.datetime.now() <= dt.timedelta(hours=hours) else 0


def _send_sms(sid, token, from_, to, body) -> None:
    try:
        to_ = f"whatsapp:{to}" if from_.startswith("whatsapp:") else to
        data = urllib.parse.urlencode({"From": from_, "To": to_, "Body": body}).encode()
        req = urllib.request.Request(
            f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json", data=data)
        auth = base64.b64encode(f"{sid}:{token}".encode()).decode()
        req.add_header("Authorization", f"Basic {auth}")
        urllib.request.urlopen(req, timeout=15)
    except Exception as e:
        print(f"[MESSAGE] SMS/WhatsApp failed (phone must be like +15551234567): {e}")


def send_customer_message(business: str, contact: str, subject: str, body: str) -> str:
    """
    Message the customer. Channel is chosen from `contact`:
      email address -> SMTP email (background thread)
      phone number  -> Twilio SMS/WhatsApp if TWILIO_* is set
    Everything is also written to customer_messages.log (works with no setup).
    """
    contact = (contact or "").strip()
    if not contact:
        return "skipped"
    channel = "log"
    host, user, pwd = os.getenv("SMTP_HOST"), os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD")
    sid, token, from_ = (os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"),
                         os.getenv("TWILIO_FROM"))
    if "@" in contact and host and user and pwd:
        threading.Thread(target=_send_email,
                         args=(host, user, pwd, contact, f"[{business}] {subject}", body)).start()
        channel = "email"
    elif "@" not in contact and sid and token and from_:
        threading.Thread(target=_send_sms, args=(sid, token, from_, contact, body)).start()
        channel = "sms/whatsapp"
    try:
        with open(CUSTOMER_LOG, "a", encoding="utf-8") as f:
            f.write(f"--- {_stamp()} | to {contact} | via {channel}\n{body}\n\n")
    except OSError:
        pass
    print(f"[MESSAGE -> {contact}] ({channel}) {body}")
    return channel


def _booking_message(kind: str, event: str, bid: int, name: str, d_str: str, t_str: str,
                     contact: str) -> None:
    biz, label = _biz_name(kind), KIND_LABEL[kind]
    when = _pretty_when(d_str, t_str)
    texts = {
        "confirmed": (f"{label.title()} confirmed",
                      f"Hi {name}, your {label} #{bid} with {biz} is confirmed for {when}. "
                      "To change or cancel it, just message us."),
        "rescheduled": (f"{label.title()} rescheduled",
                        f"Hi {name}, your {label} #{bid} with {biz} has been moved to {when}."),
        "cancelled": (f"{label.title()} cancelled",
                      f"Hi {name}, your {label} #{bid} with {biz} on {when} has been cancelled."),
    }
    subject, body = texts[event]
    send_customer_message(biz, contact, subject, body)


def send_due_reminders(db, hours: Optional[float] = None, now: Optional[dt.datetime] = None) -> int:
    """Send one reminder for every confirmed booking starting within `hours` (default 24)."""
    hours = hours if hours is not None else float(os.getenv("REMINDER_HOURS", "24"))
    now = now or dt.datetime.now()
    rows = db.execute("SELECT id,kind,name,date,time,contact FROM bookings "
                      "WHERE status='confirmed' AND COALESCE(reminder_sent,0)=0").fetchall()
    sent = 0
    for bid, kind, name, d, t, contact in rows:
        when = dt.datetime.fromisoformat(f"{d} {t}")
        if when < now or when - now > dt.timedelta(hours=hours):
            continue
        if when.date() == now.date():
            day = "today"
        elif when.date() == now.date() + dt.timedelta(days=1):
            day = "tomorrow"
        else:
            day = f"on {when.strftime('%A')}"
        biz, label = _biz_name(kind), KIND_LABEL[kind]
        send_customer_message(
            biz, contact, f"Reminder: your {label}",
            f"Reminder: Hi {name}, you have a {label} with {biz} {day} - {_pretty_when(d, t)}. "
            "Reply or contact us if you need to change it.")
        db.execute("UPDATE bookings SET reminder_sent=1 WHERE id=?", (bid,))
        sent += 1
    db.commit()
    return sent


# ------------------------------------------------------------ order helpers

def _price_num(v) -> float:
    m = re.search(r"[\d,]+(?:\.\d+)?", str(v))
    return float(m.group().replace(",", ""))


def _match_item(part: str, prices: Dict[str, str]) -> Optional[str]:
    part = part.strip().lower()
    if part in prices:
        return part
    hits = [k for k in prices if part and (part in k or k in part)]
    return hits[0] if len(hits) == 1 else None


def _parse_items(text: str, prices: Dict[str, str]) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    for part in re.split(r",|\band\b|\n", text):
        part = part.strip().lower()
        if not part:
            continue
        qty = 1
        m = re.match(r"^(\d+)\s*(?:x|\u00d7)?\s*(.+)$", part)
        if m:
            qty, part = int(m.group(1)), m.group(2).strip()
        else:
            m = re.match(r"^(.+?)\s*(?:x|\u00d7)\s*(\d+)$", part)
            if m:
                part, qty = m.group(1).strip(), int(m.group(2))
        key = _match_item(part, prices)
        if not key:
            raise ValueError(f"'{part}' is not available. Available: {', '.join(prices)}.")
        if not 1 <= qty <= 20:
            raise ValueError("Quantity must be between 1 and 20.")
        items.append((key, qty))
    if not items:
        raise ValueError("No items given.")
    return items


def _place_order(db, business, kind, noun, name, items_text, prices, notes=""):
    items = _parse_items(items_text, prices)
    total = sum(q * _price_num(prices[k]) for k, q in items)
    summary = ", ".join(f"{q} x {k.title()}" for k, q in items)
    cur = db.execute(
        "INSERT INTO orders (kind,name,items,total,notes,status,created_at) "
        "VALUES (?,?,?,?,?, 'placed', ?)",
        (kind, name, json.dumps(items), total, notes, _stamp()))
    db.commit()
    oid = cur.lastrowid
    notify_owner(business, f"New {noun} #{oid} from {name}",
                 f"Items: {summary}\nTotal: ${total:,.2f}\n{notes}")
    return oid, total, summary


# ------------------------------------------------------------ generic tool factories

def _booking_tools(db, business, kind, noun, sch) -> List[StructuredTool]:
    def check_available_slots(date: str, size: int = 1) -> str:
        try:
            d = _parse_date(date)
        except ValueError as e:
            return str(e)
        if d < dt.date.today():
            return "That date is in the past."
        free = _free_slots(db, kind, sch, d, size)
        if not free:
            return f"No free slots on {d.strftime('%A %Y-%m-%d')}."
        return f"Free slots on {d.strftime('%A %Y-%m-%d')}: {', '.join(free)}"

    def reschedule(booking_id: int, new_date: str, new_time: str) -> str:
        row = db.execute("SELECT name,date,time,size,status,contact FROM bookings WHERE id=? AND kind=?",
                         (booking_id, kind)).fetchone()
        if not row or row[4] != "confirmed":
            return f"No active {noun} #{booking_id} found."
        try:
            d, tm = _check_slot(db, kind, sch, new_date, new_time, row[3], exclude=booking_id)
        except ValueError as e:
            return f"Could not reschedule: {e}"
        db.execute("UPDATE bookings SET date=?, time=?, reminder_sent=? WHERE id=?",
                   (d.isoformat(), _fmt_t(tm), _reminder_flag(d.isoformat(), _fmt_t(tm)), booking_id))
        db.commit()
        _booking_message(kind, "rescheduled", booking_id, row[0], d.isoformat(), _fmt_t(tm), row[5])
        notify_owner(business, f"{noun.title()} #{booking_id} rescheduled",
                     f"{row[0]}: {row[1]} {row[2]} -> {d.isoformat()} {_fmt_t(tm)}")
        return f"{noun.title()} #{booking_id} moved to {d.strftime('%A %Y-%m-%d')} at {_fmt_t(tm)}."

    def cancel(booking_id: int, confirmed: bool = False) -> str:
        row = db.execute("SELECT name,date,time,status,contact FROM bookings WHERE id=? AND kind=?",
                         (booking_id, kind)).fetchone()
        if not row or row[3] != "confirmed":
            return f"No active {noun} #{booking_id} found."
        if not confirmed:
            return (f"Confirmation required: ask the user to confirm cancelling {noun} "
                    f"#{booking_id} ({row[0]}, {row[1]} {row[2]}), then call again with confirmed=true.")
        db.execute("UPDATE bookings SET status='cancelled' WHERE id=?", (booking_id,))
        db.commit()
        _booking_message(kind, "cancelled", booking_id, row[0], row[1], row[2], row[4])
        notify_owner(business, f"{noun.title()} #{booking_id} cancelled",
                     f"{row[0]}: {row[1]} {row[2]}")
        return f"{noun.title()} #{booking_id} has been cancelled."

    def status(booking_id: int) -> str:
        row = db.execute("SELECT name,detail,date,time,size,status FROM bookings WHERE id=? AND kind=?",
                         (booking_id, kind)).fetchone()
        if not row:
            return f"No {noun} #{booking_id} found."
        return (f"{noun.title()} #{booking_id}: {row[0]}, {row[1] or '-'}, {row[2]} at {row[3]}, "
                f"size {row[4]}, status {row[5]}.")

    mk = StructuredTool.from_function
    return [
        mk(check_available_slots, name="check_available_slots",
           description=f"List free {noun} slots on a date (YYYY-MM-DD). Optional size = party size."),
        mk(reschedule, name=f"reschedule_{noun}",
           description=f"Move an existing {noun} to a new date (YYYY-MM-DD) and time (24h HH:MM)."),
        mk(cancel, name=f"cancel_{noun}",
           description=f"Cancel {_a(noun)} {noun} by id. Call with confirmed=true ONLY after the user explicitly agrees."),
        mk(status, name=f"check_{noun}_status", description=f"Look up {_a(noun)} {noun} by its id."),
    ]


def _order_tools(db, business, kind, noun) -> List[StructuredTool]:
    def status(order_id: int) -> str:
        row = db.execute("SELECT name,items,total,notes,status FROM orders WHERE id=? AND kind=?",
                         (order_id, kind)).fetchone()
        if not row:
            return f"No {noun} #{order_id} found."
        items = ", ".join(f"{q} x {k.title()}" for k, q in json.loads(row[1] or "[]"))
        return f"{noun.title()} #{order_id} for {row[0]}: {items}. Total ${row[2]:,.2f}. Status: {row[4]}."

    def cancel(order_id: int, confirmed: bool = False) -> str:
        row = db.execute("SELECT name,status FROM orders WHERE id=? AND kind=?",
                         (order_id, kind)).fetchone()
        if not row:
            return f"No {noun} #{order_id} found."
        if row[1] != "placed":
            return (f"{noun.title()} #{order_id} is already '{row[1]}' and cannot be cancelled here. "
                    "Offer to create a support ticket.")
        if not confirmed:
            return (f"Confirmation required: ask the user to confirm cancelling {noun} "
                    f"#{order_id}, then call again with confirmed=true.")
        db.execute("UPDATE orders SET status='cancelled' WHERE id=?", (order_id,))
        db.commit()
        notify_owner(business, f"{noun.title()} #{order_id} cancelled", f"Customer: {row[0]}")
        return f"{noun.title()} #{order_id} has been cancelled."

    mk = StructuredTool.from_function
    return [
        mk(status, name=f"check_{noun}_status", description=f"Look up the status of {_a(noun)} {noun} by its id."),
        mk(cancel, name=f"cancel_{noun}",
           description=f"Cancel {_a(noun)} {noun} that is still 'placed'. Use confirmed=true ONLY after the user agrees."),
    ]


# ------------------------------------------------------------ domain tools

def _restaurant(db, biz, prices):
    sch = SCHEDULES["restaurant"]

    def book_table(name: str, date: str, time: str, guests: int, phone: str = "",
                   special_requests: str = "") -> str:
        """Reserve a table. date=YYYY-MM-DD, time=24h HH:MM. Parties of 9+ must phone the restaurant."""
        if guests < 1:
            return "Number of guests must be at least 1."
        if guests > 8:
            return "For 9 or more guests please call the restaurant to arrange seating."
        try:
            d, tm = _check_slot(db, "reservation", sch, date, time, guests)
            rid = _insert_booking(db, "reservation", name, special_requests, d, tm, guests, phone)
        except ValueError as e:
            return f"Could not book: {e}"
        notify_owner(biz, f"New reservation #{rid} from {name}",
                     f"{d.strftime('%A %Y-%m-%d')} at {_fmt_t(tm)}, {guests} guests\n"
                     f"Phone: {phone or '-'}\nRequests: {special_requests or '-'}")
        return (f"Reservation #{rid} confirmed for {name}: {guests} guests on "
                f"{d.strftime('%A %Y-%m-%d')} at {_fmt_t(tm)}. Tables are held for 15 minutes.")

    def place_order(name: str, items: str, order_type: str = "takeaway", address: str = "") -> str:
        """Place a food order. items like 'chicken burger x2, soft drink'. order_type: takeaway, delivery or dine-in."""
        order_type = order_type.lower().strip()
        if order_type not in ("takeaway", "delivery", "dine-in"):
            return "order_type must be takeaway, delivery or dine-in."
        if order_type == "delivery" and not address.strip():
            return "A delivery address is required for delivery orders."
        try:
            oid, total, summary = _place_order(db, biz, "order", "order", name, items, prices,
                                               f"Type: {order_type}. Address: {address or '-'}")
        except ValueError as e:
            return f"Could not place order: {e}"
        return (f"Order #{oid} placed ({order_type}): {summary}. Items total ${total:,.2f}. "
                "The restaurant has been notified.")

    return ([StructuredTool.from_function(book_table), StructuredTool.from_function(place_order)]
            + _booking_tools(db, biz, "reservation", "reservation", sch)
            + _order_tools(db, biz, "order", "order"))


def _healthcare(db, biz, prices):
    sch = SCHEDULES["healthcare"]

    def book_appointment(name: str, service: str, date: str, time: str, phone: str = "") -> str:
        """Book a clinic appointment. service must be one of the clinic services. date=YYYY-MM-DD, time=24h HH:MM."""
        key = _match_item(service, prices)
        if not key:
            return f"Unknown service. Available: {', '.join(prices)}."
        try:
            d, tm = _check_slot(db, "appointment", sch, date, time, 1)
            aid = _insert_booking(db, "appointment", name, key.title(), d, tm, 1, phone)
        except ValueError as e:
            return f"Could not book: {e}"
        notify_owner(biz, f"New appointment #{aid} from {name}",
                     f"{key.title()} on {d.strftime('%A %Y-%m-%d')} at {_fmt_t(tm)}\nPhone: {phone or '-'}")
        return (f"Appointment #{aid} confirmed: {key.title()} for {name} on "
                f"{d.strftime('%A %Y-%m-%d')} at {_fmt_t(tm)}. Fee (self-pay): {prices[key]}.")

    return ([StructuredTool.from_function(book_appointment)]
            + _booking_tools(db, biz, "appointment", "appointment", sch))


def _ecommerce(db, biz, prices):
    def place_order(name: str, items: str, shipping_address: str) -> str:
        """Place a store order. items like 'phone case x2, power bank'. A shipping address is required."""
        if not shipping_address.strip():
            return "A shipping address is required."
        try:
            oid, total, summary = _place_order(db, biz, "order", "order", name, items, prices,
                                               f"Ship to: {shipping_address}")
        except ValueError as e:
            return f"Could not place order: {e}"
        return (f"Order #{oid} placed: {summary}. Items total ${total:,.2f} "
                "(shipping per store policy). The store team has been notified.")

    def request_return(order_id: int, reason: str) -> str:
        """Open a return request for an existing order. The team checks eligibility against the return policy."""
        row = db.execute("SELECT name,status FROM orders WHERE id=? AND kind='order'", (order_id,)).fetchone()
        if not row:
            return f"No order #{order_id} found."
        if row[1] == "cancelled":
            return f"Order #{order_id} was cancelled, so no return is needed."
        cur = db.execute("INSERT INTO tickets (name,issue,contact,status,created_at) VALUES (?,?,?, 'open', ?)",
                         (row[0], f"RETURN request for order #{order_id}: {reason}", "", _stamp()))
        db.commit()
        notify_owner(biz, f"Return request (ticket #{cur.lastrowid}) for order #{order_id}",
                     f"Customer: {row[0]}\nReason: {reason}")
        return f"Return request logged as ticket #{cur.lastrowid}. The team will confirm eligibility and next steps."

    return ([StructuredTool.from_function(place_order), StructuredTool.from_function(request_return)]
            + _order_tools(db, biz, "order", "order"))


def _education(db, biz, prices):
    sch = SCHEDULES["education"]

    def enroll_in_course(name: str, course: str, payment_plan: str = "full") -> str:
        """Enroll a student in a course. payment_plan: 'full' or 'installments' (60% now, 40% at midpoint)."""
        key = _match_item(course, prices)
        if not key:
            return f"Unknown course. Available: {', '.join(prices)}."
        plan = payment_plan.lower().strip()
        if plan not in ("full", "installments"):
            return "payment_plan must be 'full' or 'installments'."
        dup = db.execute("SELECT id FROM orders WHERE kind='enrollment' AND lower(name)=lower(?) "
                         "AND items LIKE ? AND status!='cancelled'", (name, f'%"{key}"%')).fetchone()
        if dup:
            return f"{name} is already enrolled in {key.title()} (enrollment #{dup[0]})."
        fee = _price_num(prices[key])
        now_due = fee * 0.6 if plan == "installments" else fee
        notes = f"Plan: {plan}. Due now ${now_due:,.2f}; at midpoint ${fee - now_due:,.2f}."
        cur = db.execute("INSERT INTO orders (kind,name,items,total,notes,status,created_at) "
                         "VALUES ('enrollment',?,?,?,?, 'placed', ?)",
                         (name, json.dumps([(key, 1)]), fee, notes, _stamp()))
        db.commit()
        eid = cur.lastrowid
        notify_owner(biz, f"New enrollment #{eid}: {name}", f"{key.title()} - fee ${fee:,.2f}\n{notes}")
        return (f"Enrollment #{eid} created for {name} in {key.title()} (fee ${fee:,.2f}). {notes} "
                "A seat is confirmed once the first payment is received.")

    def book_demo_class(name: str, course: str, date: str, time: str, contact: str = "") -> str:
        """Book a free demo class. date=YYYY-MM-DD, time=24h HH:MM on the hour."""
        key = _match_item(course, prices)
        if not key:
            return f"Unknown course. Available: {', '.join(prices)}."
        try:
            d, tm = _check_slot(db, "demo_class", sch, date, time, 1)
            bid = _insert_booking(db, "demo_class", name, key.title(), d, tm, 1, contact)
        except ValueError as e:
            return f"Could not book: {e}"
        notify_owner(biz, f"New demo class booking #{bid}: {name}",
                     f"{key.title()} on {d.isoformat()} at {_fmt_t(tm)}\nContact: {contact or '-'}")
        return f"Demo class #{bid} booked for {name}: {key.title()} on {d.strftime('%A %Y-%m-%d')} at {_fmt_t(tm)}."

    return ([StructuredTool.from_function(enroll_in_course), StructuredTool.from_function(book_demo_class)]
            + _booking_tools(db, biz, "demo_class", "demo_class", sch)
            + _order_tools(db, biz, "enrollment", "enrollment"))


def _realestate(db, biz, prices):
    sch = SCHEDULES["realestate"]

    def schedule_viewing(name: str, property_type: str, date: str, time: str, phone: str) -> str:
        """Schedule a property viewing. date=YYYY-MM-DD, time=24h HH:MM on the hour. Phone is required."""
        if not phone.strip():
            return "A phone number is required to schedule a viewing."
        key = _match_item(property_type, prices) or property_type.strip()
        try:
            d, tm = _check_slot(db, "viewing", sch, date, time, 1)
            vid = _insert_booking(db, "viewing", name, key.title(), d, tm, 1, phone)
        except ValueError as e:
            return f"Could not schedule: {e}"
        notify_owner(biz, f"New viewing #{vid} from {name}",
                     f"{key.title()} on {d.isoformat()} at {_fmt_t(tm)}\nPhone: {phone}")
        return (f"Viewing #{vid} confirmed for {name}: {key.title()} on "
                f"{d.strftime('%A %Y-%m-%d')} at {_fmt_t(tm)}. Please bring a valid photo ID.")

    def register_inquiry(name: str, property_type: str, budget: str, phone: str) -> str:
        """Capture a buyer lead: property type, budget and phone. An agent will call back."""
        cur = db.execute("INSERT INTO leads (kind,name,detail,contact,created_at) VALUES ('inquiry',?,?,?,?)",
                         (name, f"{property_type}; budget {budget}", phone, _stamp()))
        db.commit()
        notify_owner(biz, f"New buyer inquiry #{cur.lastrowid} from {name}",
                     f"Looking for: {property_type}\nBudget: {budget}\nPhone: {phone}")
        return f"Inquiry #{cur.lastrowid} saved. An agent will contact {name} soon."

    return ([StructuredTool.from_function(schedule_viewing), StructuredTool.from_function(register_inquiry)]
            + _booking_tools(db, biz, "viewing", "viewing", sch))


def _marketing(db, biz, prices):
    sch = SCHEDULES["marketing"]

    def book_consultation(name: str, service: str, date: str, time: str, email: str) -> str:
        """Book a free 30-minute consultation. date=YYYY-MM-DD, time=24h HH:MM. Email is required."""
        if "@" not in email:
            return "A valid email address is required."
        key = _match_item(service, prices) or service.strip()
        try:
            d, tm = _check_slot(db, "consultation", sch, date, time, 1)
            cid = _insert_booking(db, "consultation", name, key.title(), d, tm, 1, email)
        except ValueError as e:
            return f"Could not book: {e}"
        notify_owner(biz, f"New consultation #{cid} from {name}",
                     f"Interested in: {key.title()}\n{d.isoformat()} at {_fmt_t(tm)}\nEmail: {email}")
        return (f"Consultation #{cid} booked for {name} on {d.strftime('%A %Y-%m-%d')} "
                f"at {_fmt_t(tm)}. A confirmation will be sent to {email}.")

    def request_quote(name: str, services: str, email: str) -> str:
        """Give an estimated monthly quote for one or more packages (comma-separated) and save it as a lead."""
        try:
            items = _parse_items(services, prices)
        except ValueError as e:
            return f"Could not quote: {e}"
        keys = sorted({k for k, _ in items})
        total = sum(_price_num(prices[k]) for k in keys)
        cur = db.execute("INSERT INTO leads (kind,name,detail,contact,created_at) VALUES ('quote',?,?,?,?)",
                         (name, f"{', '.join(keys)}; est ${total:,.0f}/month", email, _stamp()))
        db.commit()
        notify_owner(biz, f"New quote request #{cur.lastrowid} from {name}",
                     f"Services: {', '.join(keys)}\nEstimate: ${total:,.0f}/month\nEmail: {email}")
        return (f"Estimated quote for {', '.join(k.title() for k in keys)}: ${total:,.0f}/month "
                f"(ad spend not included; if 'full digital package' is chosen, other packages are already included). "
                f"Quote #{cur.lastrowid} saved and the team will follow up.")

    return ([StructuredTool.from_function(book_consultation), StructuredTool.from_function(request_quote)]
            + _booking_tools(db, biz, "consultation", "consultation", sch))


_BUILDERS = {"restaurant": _restaurant, "healthcare": _healthcare, "ecommerce": _ecommerce,
             "education": _education, "realestate": _realestate, "marketing": _marketing}


# ------------------------------------------------------------ common tools + entry point

@tool
def get_current_datetime() -> str:
    """Get today's date, weekday and current time. Call this to resolve words like 'today' or 'tomorrow'."""
    now = dt.datetime.now()
    return f"Today is {now.strftime('%A, %Y-%m-%d')}. Current time is {now.strftime('%H:%M')} (24-hour)."


def _ticket_tool(db, biz) -> StructuredTool:
    def create_support_ticket(name: str, issue: str, contact: str = "") -> str:
        """Hand over to a human: use when the user is upset, asks for a person, or the request cannot be done with the tools."""
        cur = db.execute("INSERT INTO tickets (name,issue,contact,status,created_at) VALUES (?,?,?, 'open', ?)",
                         (name, issue, contact, _stamp()))
        db.commit()
        notify_owner(biz, f"Support ticket #{cur.lastrowid} from {name}",
                     f"Issue: {issue}\nContact: {contact or 'not provided'}")
        return f"Support ticket #{cur.lastrowid} created. A team member will follow up."
    return StructuredTool.from_function(create_support_ticket)


def build_action_tools(domain: str, db, prices: Dict[str, str]) -> List[StructuredTool]:
    """Create tables (if needed) and return all action tools for this domain."""
    db.executescript(SCHEMA)
    try:  # older databases created before reminders existed
        db.execute("ALTER TABLE bookings ADD COLUMN reminder_sent INTEGER DEFAULT 0")
        db.commit()
    except sqlite3.OperationalError:
        pass
    biz = BUSINESS[domain]
    return [get_current_datetime, _ticket_tool(db, biz)] + _BUILDERS[domain](db, biz, prices)


def describe_tools(tools) -> str:
    """One prompt line per tool: '- name: first sentence of description'."""
    lines = []
    for t in tools:
        first = t.description.strip().splitlines()[0].strip()
        lines.append(f"- {t.name}: {first}")
    return "\n".join(lines)
