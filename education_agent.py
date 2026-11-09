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
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "education.db"),
    check_same_thread=False
)

db.execute(
    """
    CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    interested_course TEXT
    )
    """
)

db.commit()


# Add student
@tool
def add_student(name: str, interested_course: str):
    """
    Add a new student to the database.
    """

    try:
        db.execute(
            """
            INSERT INTO students (name, interested_course)
            VALUES (?, ?)
            """,
            (name, interested_course)
        )
        db.commit()
        return f"Student {name} added successfully."

    except sqlite3.IntegrityError:
        return f"Student {name} already exists."


# Find student

@tool
def find_student(name: str):
    """
    Find a student by name
    """
    row = db.execute(
        """
        SELECT id, name, interested_course
        FROM students
        WHERE name = ?
        """,
        (name,)
    ).fetchone()

    if row:
        return (
            f"ID: {row[0]}, "
            f"Name: {row[1]}, "
            f"Interested Course: {row[2]}"
        )

    return f"Student {name} was not found."


# Show all students

@tool
def show_students():
    """
    Show all students in the database
    """

    rows = db.execute(
        """
        SELECT id, name, interested_course
        FROM students
        ORDER BY id
        """
    ).fetchall()

    if not rows:
        return "There are no students in the database."

    result = ""

    for row in rows:
        result += (
            f"ID: {row[0]} | "
            f"Name: {row[1]} | "
            f"Interested Course: {row[2]}\n"
        )

    return result


# Update student

@tool
def update_student(name: str, new_course: str):
    """
    Change a student's interested course
    """

    result = db.execute(
        """
        UPDATE students
        SET interested_course = ?
        WHERE name = ?
        """,
        (new_course, name)
    )
    db.commit()

    if result.rowcount == 0:
        return f"Student {name} was not found."

    return (
        f"{name}'s interested course "
        f"was changed to {new_course}."
    )


# Delete student

@tool
def delete_student(name: str, confirmed: bool = False):
    """
    Delete a student from the database. Only call with confirmed=True after the user explicitly confirms.
    """
    if not confirmed:
        return (
            f"Confirmation required: ask the user to confirm deleting {name}'s record, "
            "then call this tool again with confirmed=true."
        )

    result = db.execute(
        """
        DELETE FROM students
        WHERE name = ?
        """,
        (name,)
    )

    db.commit()

    if result.rowcount == 0:
        return f"Student {name} was not found."
    return f"Student {name} was deleted successfully."


# Catalog (shared by the price tool and the action tools)
PRICES = {
    "python basics": "$99",
    "web development": "$149",
    "data science": "$199",
    "ielts preparation": "$89",
    "graphic design": "$79",
}


# Price

@tool
def get_course_fee(item: str):
    """
    Get the fee of a course offered by the academy.
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
action_tools = build_action_tools("education", db, PRICES)


# Knowledge base (Advanced RAG: hybrid search + re-ranking)
search_knowledge_base = build_knowledge_tool("education")


system_prompt = """
You are a friendly and professional AI admissions assistant for an online academy. Your primary role is to assist students with courses, fees, and admissions, while strictly managing and storing all student data in the database using the provided tools.

Available Tools & Actions:
- get_course_fee: Check fees for available courses (python basics, web development, data science, etc.).
- add_student: Add a new student's name and interested course to the database.
- find_student: Search for an existing student in the database by name.
- show_students: Retrieve and list all registered students from the database.
- update_student: Update a student's interested course in the database.
- delete_student: Remove a student's record from the database upon request.
- search_knowledge_base: Look up academy info (admissions, schedule, installments, refunds, certificates).
- get_current_datetime: Get today's date, weekday and current time. Call this to resolve words like 'today' or 'tomorrow'.
- create_support_ticket: Hand over to a human: use when the user is upset, asks for a person, or the request cannot be done with the tools.
- enroll_in_course: Enroll a student in a course. payment_plan: 'full' or 'installments' (60% now, 40% at midpoint).
- book_demo_class: Book a free demo class. date=YYYY-MM-DD, time=24h HH:MM on the hour.
- check_available_slots: List free demo_class slots on a date (YYYY-MM-DD). Optional size = party size.
- reschedule_demo_class: Move an existing demo_class to a new date (YYYY-MM-DD) and time (24h HH:MM).
- cancel_demo_class: Cancel a demo_class by id. Call with confirmed=true ONLY after the user explicitly agrees.
- check_demo_class_status: Look up a demo_class by its id.
- check_enrollment_status: Look up the status of an enrollment by its id.
- cancel_enrollment: Cancel an enrollment that is still 'placed'. Use confirmed=true ONLY after the user agrees.

Important Operational Rules:
1. Always use the respective tools for any database operations (adding, finding, updating, deleting) or pricing checks. Never guess or hardcode data.
2. Whenever a student shares their name and interested course, automatically use the database tools to save or update their profile.
3. Keep your replies concise, warm, and professional.
4. Stay strictly focused on courses, fees, admissions, and student records. Do not wander off-topic or ask overly personal questions.
5. For any question about policies, procedures, timings, terms or how the business works, call search_knowledge_base and answer ONLY from what it returns. If the answer is not in the results, say you don't have that information and offer to connect them with the team. Never invent policies, and mention the source name when helpful.
6. To resolve words like 'today', 'tomorrow' or 'next Friday', call get_current_datetime first. When calling tools use dates as YYYY-MM-DD and times as 24-hour HH:MM.
7. Before any booking, order, enrollment or quote: collect every required detail, read a short summary back to the user and wait for a clear 'yes'. Then call the tool. Never claim something is booked, ordered or cancelled unless the tool returned a confirmation, and always share the confirmation ID.
8. Before any cancellation or deletion: ask the user to confirm, then call the tool with confirmed=true. Never set confirmed=true without an explicit yes.
9. If the user is upset, asks for a human, or needs something the tools cannot do, call create_support_ticket and tell them a team member will follow up.
10. For every booking, ask for the customer's phone number or email if not given, so a confirmation and a reminder message can be sent to them."""


# llm connection
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

memory = MemorySaver()


agent = create_agent(
    model=llm,
    tools=[add_student,
           find_student,
           show_students,
           update_student,
           delete_student,
           get_course_fee,
           search_knowledge_base,
           *action_tools
           ],

    system_prompt=system_prompt,

    checkpointer=memory
)

def main():
    config = {
        "configurable": {
            "thread_id": "education_001"
        }
    }

    print("\n=======================")
    print("AI Admissions Assistant")
    print("=======================")

    print("Type 'exit' to stop.")

    while True:
        user_input = input("\nStudent: ")

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
