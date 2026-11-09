"""
Offline test for the action tools (bookings, orders, leads, tickets, notifications).
No API key or internet needed.   Run:  python test_actions_offline.py
"""
import datetime as dt
import sqlite3
import tempfile
from pathlib import Path

import actions

_tmp = Path(tempfile.mkdtemp())
actions.LOG_FILE = _tmp / "notifications.log"
actions.CUSTOMER_LOG = _tmp / "customer_messages.log"

PRICES = {
    "restaurant": {"chicken burger": "$8", "soft drink": "$2"},
    "healthcare": {"general checkup": "$50", "blood test": "$40"},
    "ecommerce": {"phone case": "$9", "power bank": "$25"},
    "education": {"python basics": "$99", "data science": "$199"},
    "realestate": {"2 bedroom apartment": "$220,000", "plot": "$90,000"},
    "marketing": {"seo": "$500/month", "google ads": "$400/month"},
}


def next_weekday(wd, weeks_ahead=1):
    d = dt.date.today() + dt.timedelta(days=1)
    while d.weekday() != wd:
        d += dt.timedelta(days=1)
    return (d + dt.timedelta(weeks=weeks_ahead - 1)).isoformat()


def tools_for(domain):
    db = sqlite3.connect(":memory:", check_same_thread=False)
    return {t.name: t for t in actions.build_action_tools(domain, db, PRICES[domain])}


def run(t, **kw):
    return t.invoke(kw)


def check(cond, msg):
    assert cond, msg


def test_restaurant():
    t = tools_for("restaurant")
    sat = next_weekday(5)
    r = run(t["book_table"], name="Ali", date=sat, time="19:00", guests=4, phone="555")
    check("Reservation #1 confirmed" in r, r)
    check("already has booking" in run(t["book_table"], name="Ali", date=sat, time="19:00", guests=2), "dup")
    for i in range(4):  # fill the 40-guest slot
        run(t["book_table"], name=f"G{i}", date=sat, time="19:00", guests=8)
    full = run(t["book_table"], name="Late", date=sat, time="19:00", guests=8)
    check("slot is full" in full and "Free slots" in full, full)
    check("call the restaurant" in run(t["book_table"], name="Big", date=sat, time="20:00", guests=9), "9+")
    check("in the past" in run(t["book_table"], name="Old", date="2020-01-01", time="19:00", guests=2), "past")
    check("bookings run" in run(t["book_table"], name="Early", date=sat, time="08:00", guests=2), "hours")
    check("Confirmation required" in run(t["cancel_reservation"], booking_id=1), "confirm gate")
    check("cancelled" in run(t["cancel_reservation"], booking_id=1, confirmed=True), "cancel")
    check("status cancelled" in run(t["check_reservation_status"], booking_id=1), "status")
    check("Free slots" in run(t["check_available_slots"], date=sat), "slots")
    o = run(t["place_order"], name="Ali", items="2 chicken burger, soft drink x1", order_type="takeaway")
    check("Order #1 placed" in o and "$18.00" in o, o)
    check("address is required" in run(t["place_order"], name="Ali", items="soft drink", order_type="delivery"), "addr")
    check("not available" in run(t["place_order"], name="Ali", items="sushi"), "unknown item")
    check("Confirmation required" in run(t["cancel_order"], order_id=1), "order confirm")
    check("cancelled" in run(t["cancel_order"], order_id=1, confirmed=True), "order cancel")


def test_healthcare():
    t = tools_for("healthcare")
    check("closed on Sundays" in run(t["book_appointment"], name="Sara", service="blood test",
          date=next_weekday(6), time="10:00"), "sunday")
    mon = next_weekday(0)
    r = run(t["book_appointment"], name="Sara", service="blood test", date=mon, time="10:00")
    check("Appointment #1 confirmed" in r and "$40" in r, r)
    run(t["book_appointment"], name="Omar", service="eye exam" if False else "general checkup", date=mon, time="10:00")
    check("slot is full" in run(t["book_appointment"], name="Zed", service="general checkup", date=mon, time="10:00"), "cap 2")
    check("moved to" in run(t["reschedule_appointment"], booking_id=1, new_date=mon, new_time="11:00"), "resched")
    check("Unknown service" in run(t["book_appointment"], name="X", service="surgery", date=mon, time="12:00"), "svc")


def test_ecommerce():
    t = tools_for("ecommerce")
    check("shipping address" in run(t["place_order"], name="Ali", items="phone case", shipping_address=" "), "addr")
    r = run(t["place_order"], name="Ali", items="phone case x2, power bank", shipping_address="12 Main St")
    check("Order #1 placed" in r and "$43.00" in r, r)
    check("Status: placed" in run(t["check_order_status"], order_id=1), "track")
    check("ticket #1" in run(t["request_return"], order_id=1, reason="wrong color"), "return")
    check("No order #9" in run(t["request_return"], order_id=9, reason="x"), "missing")


def test_education():
    t = tools_for("education")
    r = run(t["enroll_in_course"], name="Hina", course="python basics", payment_plan="installments")
    check("Enrollment #1" in r and "$59.40" in r, r)
    check("already enrolled" in run(t["enroll_in_course"], name="Hina", course="python basics"), "dup")
    d = run(t["book_demo_class"], name="Hina", course="data science", date=next_weekday(1), time="17:00")
    check("Demo class #1 booked" in d, d)
    check("Confirmation required" in run(t["cancel_enrollment"], order_id=1), "gate")
    check("cancelled" in run(t["cancel_enrollment"], order_id=1, confirmed=True), "cancel")


def test_realestate():
    t = tools_for("realestate")
    check("phone number is required" in run(t["schedule_viewing"], name="Bilal", property_type="plot",
          date=next_weekday(2), time="11:00", phone=""), "phone")
    r = run(t["schedule_viewing"], name="Bilal", property_type="plot", date=next_weekday(2), time="11:00", phone="0300")
    check("Viewing #1 confirmed" in r, r)
    check("Inquiry #1 saved" in run(t["register_inquiry"], name="Bilal", property_type="2 bedroom apartment",
          budget="$200k", phone="0300"), "lead")


def test_marketing():
    t = tools_for("marketing")
    q = run(t["request_quote"], name="Zoya", services="seo, google ads", email="z@x.com")
    check("$900/month" in q, q)
    check("valid email" in run(t["book_consultation"], name="Zoya", service="seo",
          date=next_weekday(1), time="10:00", email="nope"), "email")
    c = run(t["book_consultation"], name="Zoya", service="seo", date=next_weekday(1), time="10:00", email="z@x.com")
    check("Consultation #1 booked" in c, c)
    check("slot is full" in run(t["book_consultation"], name="Other", service="seo",
          date=next_weekday(1), time="10:00", email="o@x.com"), "cap 1")


def test_common():
    t = tools_for("restaurant")
    check("Today is" in run(t["get_current_datetime"]), "datetime")
    check("Support ticket #1" in run(t["create_support_ticket"], name="Ali", issue="Angry about cold food"), "ticket")
    log = actions.LOG_FILE.read_text(encoding="utf-8")
    check("Support ticket #1 from Ali" in log and "[Restaurant]" in log, "owner notification log")


def test_customer_messages_and_reminders():
    db = sqlite3.connect(":memory:", check_same_thread=False)
    t = {x.name: x for x in actions.build_action_tools("healthcare", db, PRICES["healthcare"])}
    mon = next_weekday(0, weeks_ahead=2)
    run(t["book_appointment"], name="Sara", service="blood test", date=mon, time="10:00", phone="+15551234567")
    log = actions.CUSTOMER_LOG.read_text(encoding="utf-8")
    check("Hi Sara, your appointment #1" in log and "10:00 AM" in log and "+15551234567" in log, log)
    # far in the future: no reminder yet
    start = dt.datetime.fromisoformat(f"{mon} 10:00")
    check(actions.send_due_reminders(db, now=start - dt.timedelta(days=5)) == 0, "too early")
    # 20 hours before: reminder goes out once
    check(actions.send_due_reminders(db, now=start - dt.timedelta(hours=20)) == 1, "due")
    check(actions.send_due_reminders(db, now=start - dt.timedelta(hours=19)) == 0, "only once")
    check("Reminder: Hi Sara" in actions.CUSTOMER_LOG.read_text(encoding="utf-8"), "reminder text")
    # reschedule resets the reminder and messages the customer
    tue = next_weekday(1, weeks_ahead=2)
    check("moved to" in run(t["reschedule_appointment"], booking_id=1, new_date=tue, new_time="09:30"), "resched")
    check("has been moved to" in actions.CUSTOMER_LOG.read_text(encoding="utf-8"), "resched msg")
    start2 = dt.datetime.fromisoformat(f"{tue} 09:30")
    check(actions.send_due_reminders(db, now=start2 - dt.timedelta(hours=2)) == 1, "re-reminded")
    run(t["cancel_appointment"], booking_id=1, confirmed=True)
    check("has been cancelled" in actions.CUSTOMER_LOG.read_text(encoding="utf-8"), "cancel msg")
    check(actions.send_due_reminders(db, now=start2 - dt.timedelta(hours=1)) == 0, "no reminder for cancelled")
    # no contact -> no message, no crash
    before = actions.CUSTOMER_LOG.read_text(encoding="utf-8")
    run(t["book_appointment"], name="NoContact", service="blood test", date=mon, time="11:00")
    check(actions.CUSTOMER_LOG.read_text(encoding="utf-8") == before, "no contact skipped")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS ", name)
    print("\nAll action tests passed.")
