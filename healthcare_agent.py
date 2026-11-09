from dotenv import load_dotenv
load_dotenv()

import os
import sqlite3
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from rag_core import build_knowledge_tool
from actions import build_action_tools


# Database

db = sqlite3.connect(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "healthcare.db"),
    check_same_thread=False
)

db.execute(
    """
    CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    reason_for_visit TEXT
    )
    """
)

db.commit()


# Add patient
@tool
def add_patient(name: str, reason_for_visit: str):
    """
    Add a new patient to the database.
    """

    try:
        db.execute(
            """
            INSERT INTO patients (name, reason_for_visit)
            VALUES (?, ?)
            """,
            (name, reason_for_visit)
        )
        db.commit()
        return f"Patient {name} added successfully."

    except sqlite3.IntegrityError:
        return f"Patient {name} already exists."


# Find patient

@tool
def find_patient(name: str):
    """
    Find a patient by name
    """
    row = db.execute(
        """
        SELECT id, name, reason_for_visit
        FROM patients
        WHERE name = ?
        """,
        (name,)
    ).fetchone()

    if row:
        return (
            f"ID: {row[0]}, "
            f"Name: {row[1]}, "
            f"Reason for Visit: {row[2]}"
        )

    return f"Patient {name} was not found."


# Show all patients

@tool
def show_patients():
    """
    Show all patients in the database
    """

    rows = db.execute(
        """
        SELECT id, name, reason_for_visit
        FROM patients
        ORDER BY id
        """
    ).fetchall()

    if not rows:
        return "There are no patients in the database."

    result = ""

    for row in rows:
        result += (
            f"ID: {row[0]} | "
            f"Name: {row[1]} | "
            f"Reason for Visit: {row[2]}\n"
        )

    return result


# Update patient

@tool
def update_patient(name: str, new_reason: str):
    """
    Change a patient's reason for visit
    """

    result = db.execute(
        """
        UPDATE patients
        SET reason_for_visit = ?
        WHERE name = ?
        """,
        (new_reason, name)
    )
    db.commit()

    if result.rowcount == 0:
        return f"Patient {name} was not found."

    return (
        f"{name}'s reason for visit "
        f"was changed to {new_reason}."
    )


# Delete patient

@tool
def delete_patient(name: str, confirmed: bool = False):
    """
    Delete a patient from the database. Only call with confirmed=True after the user explicitly confirms.
    """
    if not confirmed:
        return (
            f"Confirmation required: ask the user to confirm deleting {name}'s record, "
            "then call this tool again with confirmed=true."
        )

    result = db.execute(
        """
        DELETE FROM patients
        WHERE name = ?
        """,
        (name,)
    )

    db.commit()

    if result.rowcount == 0:
        return f"Patient {name} was not found."
    return f"Patient {name} was deleted successfully."


# Catalog (shared by the price tool and the action tools)
PRICES = {
    "general checkup": "$50",
    "dental cleaning": "$80",
    "eye exam": "$60",
    "blood test": "$40",
    "specialist consultation": "$120",
}


# Price

@tool
def get_consultation_fee(item: str):
    """
    Get the fee of a clinic service or consultation.
    """

    prices = PRICES

    item = item.lower().strip()

    if item in prices:
        return (
            f"{item.title()} costs "
            f"{prices[item]}."
        )

    return (
        "Sorry, that is not available. "
        "Available options are: "
        + ", ".join(prices.keys())
        + "."
    )


# Action tools: bookings, orders, leads, tickets, owner notifications
action_tools = build_action_tools("healthcare", db, PRICES)


# Knowledge base (Advanced RAG: hybrid search + re-ranking)
search_knowledge_base = build_knowledge_tool("healthcare")


system_prompt = """
You are a friendly and professional AI receptionist for a USA-based healthcare clinic. Your primary role is to assist patients with clinic services, consultation fees, and appointments, while strictly managing and storing all patient data in the database using the provided tools.

Available Tools & Actions:
- get_consultation_fee: Check fees for available clinic services (general checkup, dental cleaning, eye exam, etc.).
- add_patient: Add a new patient's name and reason for visit to the database.
- find_patient: Search for an existing patient in the database by name.
- show_patients: Retrieve and list all registered patients from the database.
- update_patient: Update a patient's reason for visit in the database.
- delete_patient: Remove a patient's record from the database upon request.
- search_knowledge_base: Look up clinic info (hours, appointments, insurance, billing, preparation, privacy).
- get_current_datetime: Get today's date, weekday and current time. Call this to resolve words like 'today' or 'tomorrow'.
- create_support_ticket: Hand over to a human: use when the user is upset, asks for a person, or the request cannot be done with the tools.
- book_appointment: Book a clinic appointment. service must be one of the clinic services. date=YYYY-MM-DD, time=24h HH:MM.
- check_available_slots: List free appointment slots on a date (YYYY-MM-DD). Optional size = party size.
- reschedule_appointment: Move an existing appointment to a new date (YYYY-MM-DD) and time (24h HH:MM).
- cancel_appointment: Cancel an appointment by id. Call with confirmed=true ONLY after the user explicitly agrees.
- check_appointment_status: Look up an appointment by its id.

Important Operational Rules:
1. Always use the respective tools for any database operations (adding, finding, updating, deleting) or pricing checks. Never guess or hardcode data.
2. Whenever a patient shares their name and reason for visit, automatically use the database tools to save or update their profile.
3. Keep your replies concise, warm, and professional.
4. Stay strictly focused on clinic services, appointments, and patient records. Do not wander off-topic or ask overly personal questions.
5. Never give medical advice, diagnosis, or treatment suggestions. For any medical question, tell the patient to speak with a doctor. For emergencies, tell them to call emergency services immediately.
6. For any question about policies, procedures, timings, terms or how the business works, call search_knowledge_base and answer ONLY from what it returns. If the answer is not in the results, say you don't have that information and offer to connect them with the team. Never invent policies, and mention the source name when helpful.
7. To resolve words like 'today', 'tomorrow' or 'next Friday', call get_current_datetime first. When calling tools use dates as YYYY-MM-DD and times as 24-hour HH:MM.
8. Before any booking, order, enrollment or quote: collect every required detail, read a short summary back to the user and wait for a clear 'yes'. Then call the tool. Never claim something is booked, ordered or cancelled unless the tool returned a confirmation, and always share the confirmation ID.
9. Before any cancellation or deletion: ask the user to confirm, then call the tool with confirmed=true. Never set confirmed=true without an explicit yes.
10. If the user is upset, asks for a human, or needs something the tools cannot do, call create_support_ticket and tell them a team member will follow up.
11. For every booking, ask for the customer's phone number or email if not given, so a confirmation and a reminder message can be sent to them."""


# llm connection
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

memory = MemorySaver()


agent = create_agent(
    model=llm,
    tools=[add_patient,
           find_patient,
           show_patients,
           update_patient,
           delete_patient,
           get_consultation_fee,
           search_knowledge_base,
           *action_tools
           ],

    system_prompt=system_prompt,

    checkpointer=memory
)

def main():
    config = {
        "configurable": {
            "thread_id": "healthcare_001"
        }
    }

    print("\n=======================")
    print("AI Healthcare Receptionist")
    print("=======================")

    print("Type 'exit' to stop.")

    while True:
        user_input = input("\nPatient: ")

        if user_input.lower().strip() == "exit":
            print("\nGoodbye!")
            break

        if not user_input.strip():
            continue

        try:

            t0 = time.time()

            response = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input
                        }
                    ]
                },
                config=config
            )

            answer = response["messages"][-1].text

            print("\nAI:", answer)

            print(f"({time.time() - t0:.1f}s)")

        except Exception as e:

            print("\nError:", e)


if __name__ == "__main__":
    main()
