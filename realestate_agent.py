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
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "realestate.db"),
    check_same_thread=False
)

db.execute(
    """
    CREATE TABLE IF NOT EXISTS clients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    preferred_property TEXT
    )
    """
)

db.commit()


# Add client
@tool
def add_client(name: str, preferred_property: str):
    """
    Add a new client to the database.
    """

    try:
        db.execute(
            """
            INSERT INTO clients (name, preferred_property)
            VALUES (?, ?)
            """,
            (name, preferred_property)
        )
        db.commit()
        return f"Client {name} added successfully."

    except sqlite3.IntegrityError:
        return f"Client {name} already exists."


# Find client

@tool
def find_client(name: str):
    """
    Find a client by name
    """
    row = db.execute(
        """
        SELECT id, name, preferred_property
        FROM clients
        WHERE name = ?
        """,
        (name,)
    ).fetchone()

    if row:
        return (
            f"ID: {row[0]}, "
            f"Name: {row[1]}, "
            f"Preferred Property: {row[2]}"
        )

    return f"Client {name} was not found."


# Show all clients

@tool
def show_clients():
    """
    Show all clients in the database
    """

    rows = db.execute(
        """
        SELECT id, name, preferred_property
        FROM clients
        ORDER BY id
        """
    ).fetchall()

    if not rows:
        return "There are no clients in the database."

    result = ""

    for row in rows:
        result += (
            f"ID: {row[0]} | "
            f"Name: {row[1]} | "
            f"Preferred Property: {row[2]}\n"
        )

    return result


# Update client

@tool
def update_client(name: str, new_property: str):
    """
    Change a client's preferred property
    """

    result = db.execute(
        """
        UPDATE clients
        SET preferred_property = ?
        WHERE name = ?
        """,
        (new_property, name)
    )
    db.commit()

    if result.rowcount == 0:
        return f"Client {name} was not found."

    return (
        f"{name}'s preferred property "
        f"was changed to {new_property}."
    )


# Delete client

@tool
def delete_client(name: str, confirmed: bool = False):
    """
    Delete a client from the database. Only call with confirmed=True after the user explicitly confirms.
    """
    if not confirmed:
        return (
            f"Confirmation required: ask the user to confirm deleting {name}'s record, "
            "then call this tool again with confirmed=true."
        )

    result = db.execute(
        """
        DELETE FROM clients
        WHERE name = ?
        """,
        (name,)
    )

    db.commit()

    if result.rowcount == 0:
        return f"Client {name} was not found."
    return f"Client {name} was deleted successfully."


# Catalog (shared by the price tool and the action tools)
PRICES = {
    "1 bedroom apartment": "$150,000",
    "2 bedroom apartment": "$220,000",
    "3 bedroom house": "$350,000",
    "commercial shop": "$300,000",
    "plot": "$90,000",
}


# Price

@tool
def get_property_price(item: str):
    """
    Get the starting price of a property type.
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
action_tools = build_action_tools("realestate", db, PRICES)


# Knowledge base (Advanced RAG: hybrid search + re-ranking)
search_knowledge_base = build_knowledge_tool("realestate")


system_prompt = """
You are a friendly and professional AI assistant for a real estate agency. Your primary role is to assist clients with property types, pricing, and viewings, while strictly managing and storing all client data in the database using the provided tools.

Available Tools & Actions:
- get_property_price: Check starting prices for property types (apartments, houses, shops, plots).
- add_client: Add a new client's name and preferred property to the database.
- find_client: Search for an existing client in the database by name.
- show_clients: Retrieve and list all registered clients from the database.
- update_client: Update a client's preferred property in the database.
- delete_client: Remove a client's record from the database upon request.
- search_knowledge_base: Look up agency info (viewings, buying process, payment plans, fees, documents).
- get_current_datetime: Get today's date, weekday and current time. Call this to resolve words like 'today' or 'tomorrow'.
- create_support_ticket: Hand over to a human: use when the user is upset, asks for a person, or the request cannot be done with the tools.
- schedule_viewing: Schedule a property viewing. date=YYYY-MM-DD, time=24h HH:MM on the hour. Phone is required.
- register_inquiry: Capture a buyer lead: property type, budget and phone. An agent will call back.
- check_available_slots: List free viewing slots on a date (YYYY-MM-DD). Optional size = party size.
- reschedule_viewing: Move an existing viewing to a new date (YYYY-MM-DD) and time (24h HH:MM).
- cancel_viewing: Cancel a viewing by id. Call with confirmed=true ONLY after the user explicitly agrees.
- check_viewing_status: Look up a viewing by its id.

Important Operational Rules:
1. Always use the respective tools for any database operations (adding, finding, updating, deleting) or pricing checks. Never guess or hardcode data.
2. Whenever a client shares their name and preferred property, automatically use the database tools to save or update their profile.
3. Keep your replies concise, warm, and professional.
4. Stay strictly focused on properties, pricing, viewings, and client records. Do not wander off-topic or ask overly personal questions.
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
    tools=[add_client,
           find_client,
           show_clients,
           update_client,
           delete_client,
           get_property_price,
           search_knowledge_base,
           *action_tools
           ],

    system_prompt=system_prompt,

    checkpointer=memory
)

def main():
    config = {
        "configurable": {
            "thread_id": "realestate_001"
        }
    }

    print("\n=======================")
    print("AI Real Estate Assistant")
    print("=======================")

    print("Type 'exit' to stop.")

    while True:
        user_input = input("\nClient: ")

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
