# TradeFlow AI Architecture

## 1. Overview

TradeFlow AI is an AI-powered negotiation platform that simulates automated bargaining between a buyer and a seller agent for listed products. The system uses a large language model (Gemini) to power intelligent negotiation agents, while enforcing business policies and monitoring every step for transparency.

The application consists of:

- **Frontend** – A mobile-first Streamlit dashboard (`app.py`) mimicking a smartphone UI.
- **Backend** – A production‑grade event‑driven negotiation engine (`agent_system.py`) with observability, vector memory, policy validation, and resilience patterns.

## 2. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       User (Browser)                        │
└───────────────────────────┬─────────────────────────────────┘
                            │ Streamlit UI (app.py)
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  Frontend (Streamlit)                                        │
│  - Product Selection, Budget Input                           │
│  - Real‑time Price Charts (CSV data)                         │
│  - Negotiation Chat UI (streaming agent messages)            │
│  - Result Cards & Debug Panel                                │
│  - Mock Fallback Engine                                      │
└───────────────────────────┬──────────────────────────────────┘
                            │ imports & invokes
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  Backend (agent_system.py)                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           TradeFlowContainer (DI Container)            │ │
│  │  - Embedding Provider (Gemini)                         │ │
│  │  - Event Bus (EnhancedEventBus)                        │ │
│  │  - Observability Engine                                │ │
│  │  - Retry Handler (Exponential Backoff)                 │ │
│  │  - Policy Engine                                       │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │            Market Intelligence Layer                   │ │
│  │  - EnhancedMarketIntelligence (from CSV)               │ │
│  │  - EnhancedMonteCarloPredictor                         │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │            Enhanced Negotiation Session                │ │
│  │  - Alıcı (Buyer) & Satıcı (Seller) Agents             │ │
│  │  - Vector Memory (FAISS)                              │ │
│  │  - Policy Validation                                   │ │
│  │  - State Machine (INIT → NEGOTIATING → CLOSED/FAILED) │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  External Services                                     │ │
│  │  - Gemini API (LLM + Embedding, optional)              │ │
│  │  - CSV Data File (market_listings_big.csv)             │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## 3. Frontend (`app.py`)

### 3.1 UI Pages

- **Ana Sayfa (Home)** – Greeting, summary cards (active agents, savings, deals), recent transactions.
- **Ajan Sayfası (Agent Page)** – Product selection from CSV, budget/slider input, live price trend chart, negotiation start button, chat interface, and result box.
- **Profil (Profile)** – User information and settings (mock data).

### 3.2 Key Features

- **Phone‑like container** – Custom CSS emulates a smartphone notch, status bar, and bottom navigation.
- **Live charts** – `st.line_chart` renders the last 30 days of prices from the CSV.
- **Progress simulation** – Before negotiation, a three‑step progress animation is shown.
- **Streaming chat** – Agent messages are displayed word‑by‑word with a typewriter effect.
- **Debug panel** – After a successful deal, an expander shows the decision timeline, policy rules, and full negotiation history.

### 3.3 Backend Integration

The frontend tries to import the backend classes (`EventBus`, `ObservabilityEngine`, `NegotiationSession`, etc.). If the import fails (e.g., missing dependencies), it switches to a fully functional **mock mode** that simulates the negotiation without any API call, ensuring the app remains usable even without backend connectivity.

## 4. Backend (`agent_system.py`)

### 4.1 Event‑Driven Architecture

All communication is routed through `EnhancedEventBus`. Components subscribe to event types such as `AGENT_THOUGHT`, `AGENT_MESSAGE`, `ACTION_TRIGGERED`, `TRANSACTION_COMPLETED`, and `POLICY_BLOCK`. This enables loose coupling and easy addition of new listeners (e.g., logging, UI updates).

### 4.2 Core Components

| Component | Responsibility |
|-----------|---------------|
| **EnhancedEventBus** | Publish/subscribe event broker with priority support and statistics. |
| **EnhancedObservabilityEngine** | Records every event into a timeline, computes metrics, and generates a session replay for debugging. |
| **EnhancedPolicyEngine** | Validates every proposed action against limits (budget, floor price, market anomalies). Returns `PolicyResult`. |
| **EnhancedNegotiationSession** | Orchestrates the negotiation loop. Holds the buyer/seller agents, manages state transitions, and applies policy checks. |
| **EnhancedAgentNode** | Individual agent that calls the Gemini LLM to generate a response and a possible price proposal. |
| **EnhancedSemanticAgentMemory** | FAISS‑based vector store that keeps past dialogue embeddings and retrieves relevant context for the next LLM call. |
| **EnhancedProductionRAG** | (Optional) Uses FAISS to retrieve similar products from CSV; **currently bypassed** due to embedding API deprecation – product data is directly read from CSV. |
| **GeminiEmbeddingProvider** | Wraps Gemini’s embedding API with caching and a circuit breaker. (May be deactivated when unused.) |
| **CircuitBreaker** | Prevents cascading failures by opening after repeated API errors and recovering after a timeout. |
| **RetryHandler** | Implements exponential backoff with jitter for resilient API calls. |
| **TradeFlowContainer** | Simple Dependency Injection container that wires all services together. |
| **TradeFlowEngineV3** | High‑level orchestrator used in standalone runs. Not directly called by the Streamlit frontend; the frontend uses the session directly. |

### 4.3 Negotiation Flow

1. **Initiation** – Frontend selects a product and budget, creates an `EnhancedNegotiationSession` with the product info (pulled from CSV) and user limits.
2. **State = NEGOTIATING** – The session begins a turn‑based loop (max 5 rounds).
   - **Buyer/Seller turn** – The agent fetches relevant context from vector memory, constructs a prompt, and calls the Gemini LLM via a retry handler.
   - **Response** – The LLM returns a JSON object containing a natural‑language message, a price proposal, and a flag `anlasma_saglandi` (deal agreed).
   - **Policy check** – If the agent proposes to close the deal (`PROPOSE_CLOSE`), the `PolicyEngine` validates the final price against limits (budget, floor, market anomalies).
   - **Event publishing** – Every step fires events (`AGENT_THOUGHT`, `AGENT_MESSAGE`, `ACTION_TRIGGERED`) that the observability engine records and the UI listens to for real‑time updates.
3. **State transitions**:
   - If policy validates → `CLOSED_SUCCESS`, final price stored, `TRANSACTION_COMPLETED` event published.
   - If maximum rounds reached without agreement → `FAILED`.
   - On any critical exception → `ERROR`.
4. **Result** – The frontend displays the final agreed price (if any) and the savings.

### 4.4 Resilience Patterns

- **Circuit Breaker** – Wraps the embedding/LLM API calls. If failures exceed a threshold, the circuit opens and fast‑fails subsequent calls for a cooldown period.
- **Retry with Exponential Backoff** – Every API call inside `EnhancedAgentNode` is retried up to 3 times with jitter.
- **Policy Block** – Invalid proposals are caught early, preventing nonsensical transactions (e.g., price below 50% of market).
- **Fail‑safe UI** – If the entire backend fails, the frontend seamlessly switches to mock mode, which generates a realistic (but random) negotiation purely in the browser without any API key.

## 5. Data Model

### 5.1 CSV Structure (`market_listings_big.csv`)

| Column | Example |
|--------|---------|
| `date` | `2025-06-01` |
| `product` | `MacBook Air M2` |
| `seller_id` | `S015` |
| `listing_price` | `7526` |
| `condition` | `new`, `used`, `like_new` |
| `days_on_market` | `22` |
| `location` | `Istanbul` |

The system auto‑detects the price column (`listing_price` or `fiyat`) and the product column (`product` or `baslik`).

### 5.2 Product Object (passed to negotiation)

```python
product = {
    "product": "MacBook Air M2",
    "price": 7510.0,        # average from CSV
    "min_price": 7487.0,
    "max_price": 7584.0,
    "sample_count": 8
}
```

## 6. Dependencies

- **Streamlit** – UI framework
- **Pandas, NumPy** – Data manipulation and random simulation
- **FAISS** – Vector similarity search (memory and optional RAG)
- **google‑generativeai** – Gemini LLM and embedding API
- **python‑dotenv** – Environment variable management

## 7. Error Handling Strategy

1. **Backend import error** – `BACKEND_READY` flag becomes `False`; the UI uses mock negotiation without any API.
2. **CSV file missing** – The app creates a demo DataFrame and still allows product selection.
3. **Gemini API unavailability** – The circuit breaker opens; subsequent requests fail fast. The UI’s `try/except` catches the exception and falls back to mock mode.
4. **Negotiation without deal** – The UI displays an informational message and does **not** show a 0‑price result (prevents misleading “100% discount” errors).

## 8. Deployment

- Run `streamlit run app.py`
- Ensure `market_listings_big.csv` exists in the same directory (or a fallback will be generated).
- Place a valid `GEMINI_API_KEY` inside a `.env` file if Gemini features are desired; otherwise the app works fully in mock mode.
- For production, consider replacing the in‑memory event bus with a persistent message queue and adding a database for transaction history.

---

*TradeFlow AI – Architecture Document – v1.0*