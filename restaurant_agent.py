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
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "restaurant.db"),
    check_same_thread=False
)

db.execute(
    """
    CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    favorite_dish TEXT
    )
    """
)

db.commit()


# Add customer
@tool
def add_customer(name: str, favorite_dish: str):
    """
    Add a new customer to the database.
    """

    try:
        db.execute(
            """
            INSERT INTO customers (name, favorite_dish)
            VALUES (?, ?)
            """,
            (name, favorite_dish)
        )
        db.commit()
        return f"Customer {name} added successfully."

    except sqlite3.IntegrityError:
        return f"Customer {name} already exists."


# Find customer

@tool
def find_customer(name: str):
    """
    Find a customer by name
    """
    row = db.execute(
        """
        SELECT id, name, favorite_dish
        FROM customers
        WHERE name = ?
        """,
        (name,)
    ).fetchone()

    if row:
        return (
            f"ID: {row[0]}, "
            f"Name: {row[1]}, "
            f"Favorite Dish: {row[2]}"
        )

    return f"Customer {name} was not found."


# Show all customers

@tool
def show_customers():
    """
    Show all customers in the database
    """

    rows = db.execute(
        """
        SELECT id, name, favorite_dish
        FROM customers
        ORDER BY id
        """
    ).fetchall()

    if not rows:
        return "There are no customers in the database."

    result = ""

    for row in rows:
        result += (
            f"ID: {row[0]} | "
            f"Name: {row[1]} | "
            f"Favorite Dish: {row[2]}\n"
        )

    return result


# Update customer

@tool
def update_customer(name: str, new_dish: str):
    """
    Change a customer's favorite dish
    """

    result = db.execute(
        """
        UPDATE customers
        SET favorite_dish = ?
        WHERE name = ?
        """,
        (new_dish, name)
    )
    db.commit()

    if result.rowcount == 0:
        return f"Customer {name} was not found."

    return (
        f"{name}'s favorite dish "
        f"was changed to {new_dish}."
    )


# Delete customer

@tool
def delete_customer(name: str, confirmed: bool = False):
    """
    Delete a customer from the database. Only call with confirmed=True after the user explicitly confirms.
    """
    if not confirmed:
        return (
            f"Confirmation required: ask the user to confirm deleting {name}'s record, "
            "then call this tool again with confirmed=true."
        )

    result = db.execute(
        """
        DELETE FROM customers
        WHERE name = ?
        """,
        (name,)
    )

    db.commit()

    if result.rowcount == 0:
        return f"Customer {name} was not found."
    return f"Customer {name} was deleted successfully."


# Catalog (shared by the price tool and the action tools)
PRICES = {
    "chicken burger": "$8",
    "beef steak": "$22",
    "margherita pizza": "$12",
    "caesar salad": "$7",
    "pasta alfredo": "$11",
    "soft drink": "$2",
}


# Price

@tool
def get_menu_price(item: str):
    """
    Get the price of a menu item.
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
action_tools = build_action_tools("restaurant", db, PRICES)


# Knowledge base (Advanced RAG: hybrid search + re-ranking)
search_knowledge_base = build_knowledge_tool("restaurant")


system_prompt = """
You are a friendly and professional AI host for a USA-based restaurant. Your primary role is to assist customers with menu items, pricing, and table reservations, while strictly managing and storing all customer data in the database using the provided tools.

Available Tools & Actions:
- get_menu_price: Check prices for menu items (burger, steak, pizza, salad, etc.).
- add_customer: Add a new customer's name and favorite dish to the database.
- find_customer: Search for an existing customer in the database by name.
- show_customers: Retrieve and list all registered customers from the database.
- update_customer: Update a customer's favorite dish in the database.
- delete_customer: Remove a customer's record from the database upon request.
- search_knowledge_base: Look up restaurant info (hours, reservations, delivery, allergens, dietary options).
- get_current_datetime: Get today's date, weekday and current time. Call this to resolve words like 'today' or 'tomorrow'.
- create_support_ticket: Hand over to a human: use when the user is upset, asks for a person, or the request cannot be done with the tools.
- book_table: Reserve a table. date=YYYY-MM-DD, time=24h HH:MM. Parties of 9+ must phone the restaurant.
- place_order: Place a food order. items like 'chicken burger x2, soft drink'. order_type: takeaway, delivery or dine-in.
- check_available_slots: List free reservation slots on a date (YYYY-MM-DD). Optional size = party size.
- reschedule_reservation: Move an existing reservation to a new date (YYYY-MM-DD) and time (24h HH:MM).
- cancel_reservation: Cancel a reservation by id. Call with confirmed=true ONLY after the user explicitly agrees.
- check_reservation_status: Look up a reservation by its id.
- check_order_status: Look up the status of an order by its id.
- cancel_order: Cancel an order that is still 'placed'. Use confirmed=true ONLY after the user agrees.

Important Operational Rules:
1. Always use the respective tools for any database operations (adding, finding, updating, deleting) or pricing checks. Never guess or hardcode data.
2. Whenever a customer shares their name and favorite dish, automatically use the database tools to save or update their profile.
3. Keep your replies concise, warm, and professional.
4. Stay strictly focused on menu, pricing, reservations, and customer records. Do not wander off-topic or ask overly personal questions.
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
    tools=[add_customer,
           find_customer,
           show_customers,
           update_customer,
           delete_customer,
           get_menu_price,
           search_knowledge_base,
           *action_tools
           ],

    system_prompt=system_prompt,

    checkpointer=memory
)

def main():
    config = {
        "configurable": {
            "thread_id": "restaurant_001"
        }
    }

    print("\n=======================")
    print("AI Restaurant Host")
    print("=======================")

    print("Type 'exit' to stop.")

    while True:
        user_input = input("\nCustomer: ")

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
