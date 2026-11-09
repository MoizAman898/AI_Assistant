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
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "marketing.db"),
    check_same_thread=False
)

db.execute(
    """
    CREATE TABLE IF NOT EXISTS clients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    interested_service TEXT
    )
    """
)

db.commit()


# Add client
@tool
def add_client(name: str, interested_service: str):
    """
    Add a new client to the database.
    """

    try:
        db.execute(
            """
            INSERT INTO clients (name, interested_service)
            VALUES (?, ?)
            """,
            (name, interested_service)
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
        SELECT id, name, interested_service
        FROM clients
        WHERE name = ?
        """,
        (name,)
    ).fetchone()

    if row:
        return (
            f"ID: {row[0]}, "
            f"Name: {row[1]}, "
            f"Interested Service: {row[2]}"
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
        SELECT id, name, interested_service
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
            f"Interested Service: {row[2]}\n"
        )

    return result


# Update client

@tool
def update_client(name: str, new_service: str):
    """
    Change a client's interested service
    """

    result = db.execute(
        """
        UPDATE clients
        SET interested_service = ?
        WHERE name = ?
        """,
        (new_service, name)
    )
    db.commit()

    if result.rowcount == 0:
        return f"Client {name} was not found."

    return (
        f"{name}'s interested service "
        f"was changed to {new_service}."
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
    "social media management": "$300/month",
    "seo": "$500/month",
    "google ads": "$400/month",
    "content marketing": "$350/month",
    "full digital package": "$1,200/month",
}


# Price

@tool
def get_package_price(item: str):
    """
    Get the monthly price of a digital marketing package.
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
action_tools = build_action_tools("marketing", db, PRICES)


# Knowledge base (Advanced RAG: hybrid search + re-ranking)
search_knowledge_base = build_knowledge_tool("marketing")


system_prompt = """
You are a friendly and professional AI assistant for a digital marketing agency. Your primary role is to assist clients with marketing services, package pricing, and consultations, while strictly managing and storing all client data in the database using the provided tools.

Available Tools & Actions:
- get_package_price: Check prices for available packages (SEO, Google Ads, social media, etc.).
- add_client: Add a new client's name and interested service to the database.
- find_client: Search for an existing client in the database by name.
- show_clients: Retrieve and list all registered clients from the database.
- update_client: Update a client's interested service in the database.
- delete_client: Remove a client's record from the database upon request.
- search_knowledge_base: Look up agency info (process, package contents, contracts, billing, reporting).
- get_current_datetime: Get today's date, weekday and current time. Call this to resolve words like 'today' or 'tomorrow'.
- create_support_ticket: Hand over to a human: use when the user is upset, asks for a person, or the request cannot be done with the tools.
- book_consultation: Book a free 30-minute consultation. date=YYYY-MM-DD, time=24h HH:MM. Email is required.
- request_quote: Give an estimated monthly quote for one or more packages (comma-separated) and save it as a lead.
- check_available_slots: List free consultation slots on a date (YYYY-MM-DD). Optional size = party size.
- reschedule_consultation: Move an existing consultation to a new date (YYYY-MM-DD) and time (24h HH:MM).
- cancel_consultation: Cancel a consultation by id. Call with confirmed=true ONLY after the user explicitly agrees.
- check_consultation_status: Look up a consultation by its id.

Important Operational Rules:
1. Always use the respective tools for any database operations (adding, finding, updating, deleting) or pricing checks. Never guess or hardcode data.
2. Whenever a client shares their name and interested service, automatically use the database tools to save or update their profile.
3. Keep your replies concise, warm, and professional.
4. Stay strictly focused on marketing services, packages, consultations, and client records. Do not wander off-topic or ask overly personal questions.
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
           get_package_price,
           search_knowledge_base,
           *action_tools
           ],

    system_prompt=system_prompt,

    checkpointer=memory
)

def main():
    config = {
        "configurable": {
            "thread_id": "marketing_001"
        }
    }

    print("\n=======================")
    print("AI Digital Marketing Assistant")
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
