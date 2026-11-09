# Zuvayra Systems - AI Business Agents with Advanced RAG

Six ready-to-demo conversational agents built with **LangGraph + Gemini + SQLite + FAISS**.
Each agent manages customer records with database tools **and** answers policy/FAQ
questions from its own knowledge base using an advanced RAG pipeline.

| Agent | File | Business | Records | Price tool |
|---|---|---|---|---|
| E-commerce | `ecommerce_agent.py` | Online store | customers | `get_product_price` |
| Education | `education_agent.py` | Online academy | students | `get_course_fee` |
| Healthcare | `healthcare_agent.py` | Clinic reception | patients | `get_consultation_fee` |
| Marketing | `marketing_agent.py` | Digital agency | clients | `get_package_price` |
| Real estate | `realestate_agent.py` | Property agency | clients | `get_property_price` |
| Restaurant | `restaurant_agent.py` | Restaurant host | customers | `get_menu_price` |

## What the agents can DO (`actions.py`)

Beyond answering questions, every agent performs real tasks and notifies the business owner.

| Agent | Tasks |
|---|---|
| Restaurant | book / reschedule / cancel table, place and track orders (takeaway, delivery, dine-in) |
| Healthcare | check free slots, book / reschedule / cancel appointments (double-booking protection) |
| E-commerce | place orders, track order, cancel order, return requests |
| Education | enroll (full or installments), book free demo class, cancel enrollment |
| Real estate | schedule viewings, capture buyer inquiries (leads) |
| Marketing | book consultation, instant package quote saved as a lead |
| All | human handoff ticket, current date/time awareness |

- **Rules enforced in code:** opening hours, slot length, capacity per slot, past dates, duplicate bookings, valid menu/catalog items.
- **Confirmation gate:** cancel/delete tools refuse to run unless `confirmed=true`; the agent is instructed to ask the user first.
- **Owner notifications:** every new booking, order, lead, cancellation or ticket is printed, appended to `notifications.log`, and emailed if SMTP is set in `.env`.
- **Owner panel:** `python owner_panel.py restaurant list orders` and `python owner_panel.py restaurant status orders 3 preparing` (customers see the new status when they track their order).

Example chat (restaurant): "Book a table for 4 tomorrow at 7 pm, name Ali" -> the agent checks the date, confirms the summary, books it, returns reservation #ID, and the owner gets an email.

## Customer messages and reminders

When a customer books, reschedules or cancels (restaurant, clinic, academy demo class, property viewing, consultation), the agent
sends them a message with the day and time, using the phone number or email they gave in chat:

- email address -> email (uses the same `SMTP_*` settings)
- phone number -> SMS or WhatsApp through Twilio (`TWILIO_*` settings, number in `+15551234567` format)
- always also written to `customer_messages.log`, so it works in demos without any account

Reminders go out `REMINDER_HOURS` (default 24) before the booking:

```bash
python reminders.py              # check once (use cron / Task Scheduler every 15-30 min)
python reminders.py --loop       # or leave this running
```

Each booking is reminded once; rescheduling resets it and cancelled bookings are skipped.

## How the RAG works (`rag_core.py`)

```
knowledge/<domain>/*.md
   -> header-aware chunking (each chunk keeps its section path)
   -> FAISS (semantic)  +  BM25 (keyword)
user question
   -> (optional) LLM query expansion
   -> semantic + keyword search for every query
   -> Reciprocal Rank Fusion (hybrid merge)
   -> (optional) LLM re-ranking
   -> top passages -> create_retrieval_chain + create_stuff_documents_chain
      (langchain_classic) -> grounded answer with sources -> agent replies
```

- **Hybrid search**: embeddings catch meaning, BM25 catches exact terms (prices, "24 hours", names).
- **Persistent index**: the FAISS index is cached in `.rag_index/` and rebuilt automatically when knowledge files change, so restarts cost no embedding calls.
- **Retrieval chain**: `HybridRetriever` plugs the hybrid search into `create_retrieval_chain` and `create_stuff_documents_chain`; the chain answers strictly from the retrieved context and cites the source file. Set `RAG_USE_CHAIN=0` to skip this extra LLM call.
- **Grounded answers**: the agent is instructed to answer policy questions only from retrieved text and to say "I don't know" otherwise.
- **Fail-safe**: if query expansion or re-ranking fails (rate limit, etc.), retrieval silently falls back to plain hybrid search.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # then add your GOOGLE_API_KEY
python restaurant_agent.py      # or any other agent
```

On the first run the knowledge base is embedded (a few seconds); later runs load from cache.

## Try it

- Restaurant: "Do you have vegetarian options?", "How much is delivery?", "I'm Ali, my favorite dish is pasta alfredo"
- Healthcare: "Do I need to fast for a blood test?", "What is your cancellation policy?"
- E-commerce: "What is the warranty on smart watches?", "How much are wireless earbuds?"

## Use your own client data

Replace the files in `knowledge/<domain>/` with the client's real documents
(`.md` or `.txt`, UTF-8). Use `#` / `##` headings per topic for best retrieval.
No code change needed; the index rebuilds on the next start.

## Configuration

All optional, set in `.env`: `GEMINI_MODEL`, `EMBEDDING_MODEL`, `RAG_CHUNK_SIZE`,
`RAG_CHUNK_OVERLAP`, `RAG_TOP_K`, `RAG_FETCH_K`, `RAG_MULTI_QUERY` and `RAG_RERANK` (both off by default for speed; set 1 for higher accuracy at the cost of one extra LLM call each), `RAG_DEBUG=1` to print retrieval timings.

## Tests

```bash
python test_rag_offline.py       # RAG engine, no API key needed
python test_actions_offline.py   # booking/order/messages/reminders, no API key needed
```

## Notes

- Demo data only. Pricing tools use sample prices; knowledge files are sample content.
- The healthcare agent never gives medical advice. Before using it with real patient data you need proper security, consent and HIPAA review (the SQLite storage here is not production-grade).
- Bookings and orders are stored in local SQLite and notifications go by email/log; connecting a real POS, calendar or payment system is the next integration step.
- Notification emails for the healthcare agent contain patient name and service; review privacy requirements before using real patient data.
- Conversation memory is in-process (`MemorySaver`) and resets on restart; customer records persist in SQLite.
