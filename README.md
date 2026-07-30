# RAG System API

Agentic RAG backend for the [RAG-SaaS-UI](../RAG-SaaS-UI) frontend.

**FastAPI** for HTTP · **LangGraph** for agent orchestration · **LangChain** for AI
components only · **Supabase** for auth/data/storage · **Qdrant** for vectors ·
**OpenRouter** for LLMs.

---

## Contents

- [Quick start](#quick-start)
- [Architecture](#architecture)
- [The agent graph](#the-agent-graph)
- [API](#api)
- [Wiring the frontend](#wiring-the-frontend)
- [Switching models](#switching-models)
- [Growing the system](#growing-the-system)
- [Testing](#testing)
- [Design notes](#design-notes)

---

## Quick start

### 1. Install

Python **3.11+** (developed and tested on 3.13).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Configure

`.env` is already created with your Supabase URL pre-filled. Three secrets are
left blank — fill them in:

| Variable | Where to get it |
| --- | --- |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase → Project Settings → API → `service_role` |
| `SUPABASE_JWT_SECRET` | Supabase → Project Settings → API → JWT Settings → JWT Secret |
| `OPENROUTER_API_KEY` | <https://openrouter.ai/keys> |

> The `service_role` key bypasses RLS. It must never reach the browser. The
> backend enforces ownership itself: every query filters on the `user_id` taken
> from the verified JWT's `sub` claim, never from a request body or path.

> If your project has migrated to Supabase's **asymmetric signing keys**, leave
> `SUPABASE_JWT_SECRET` blank — the backend then verifies against the project's
> published JWKS automatically. Your current anon key is HS256, so the shared
> secret is the right choice today.

### 3. Run the migrations

In the Supabase SQL editor, run in order:

1. `migrations/001_baseline.sql` — the tables your frontend already uses. Written
   `IF NOT EXISTS`, so it is a no-op on your existing project.
2. `migrations/002_agent_backend.sql` — adds per-assistant model config, ingestion
   bookkeeping, and the two memory tables. Fully additive.

### 4. Start Qdrant

```powershell
docker compose up -d qdrant
```

### 5. Start the API

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

- Docs: <http://localhost:8000/docs>
- Readiness (shows which pluggable implementations are live):
  <http://localhost:8000/ready>

First start downloads the ~130 MB embedding model. After that it is fully local.

---

## Architecture

```
                         HTTP  (app/api)
             routes validate input and call ONE service
                              │
                              ▼
                      app/services  ◄── all business logic
                              │
        ┌─────────────────────┼──────────────────────┐
        ▼                     ▼                      ▼
  app/repositories      app/graph              app/ai
  the only code that    LangGraph nodes;       the only code that
  touches the DB or     each calls one         imports LangChain
  storage               service                       │
        │                     │                       ▼
        ▼                     ▼               loaders · splitters
   Supabase            AgentState             embeddings · prompts
   (PostgREST +        (typed, JSON-          parsers · retrievers
    Storage)            serialisable)         chat models · rerankers
```

### Layer rules

These are the invariants that keep later features cheap:

| Rule | Why |
| --- | --- |
| Routes contain no logic | Adding a transport (WebSocket, gRPC) touches no logic |
| Services own all business logic | One place to change behaviour |
| **Only repositories touch the database** | No SQL in nodes, services or routes |
| **Only `app/ai/**` imports LangChain** | LangChain can be upgraded or replaced in isolation |
| Services depend on ports, not adapters | Implementations swap by configuration |
| Each graph node calls exactly one service | Nodes stay ~20 lines and testable |

### The extension seams

Every "add later" requirement is a protocol in `app/domain/ports.py` with a
registry and a config switch. Nothing in the request path needs to change:

| Port | Default today | Add later by |
| --- | --- | --- |
| `Embedder` | fastembed (local) | `EMBEDDING_PROVIDER=openai\|voyage` |
| `VectorStore` | Qdrant | new adapter implementing the port |
| `KnowledgeSource` | one (Qdrant) | register in `knowledge_sources/registry.py` |
| `Reranker` | pass-through | `RERANKER_PROVIDER=cohere` |
| `Cache` | no-op | `CACHE_PROVIDER=redis` |
| `Guardrail` | empty pipeline | `GUARDRAILS_ENABLED=true` + names |
| `ToolProvider` | none | `TOOLS_ENABLED=true` / `MCP_ENABLED=true` |
| `LLMProvider` | OpenRouter | new adapter implementing the port |
| Observability | off | `LANGSMITH_TRACING=true` |

The composition root — `app/container.py` — is the single place that knows which
implementation backs which port.

### Directory map

```
app/
├── main.py                    app factory + lifespan
├── container.py               composition root: ports → implementations
├── core/                      config, errors, logging, JWT, observability
├── domain/
│   ├── models.py              domain vocabulary (no framework imports)
│   └── ports.py               ⭐ the extension seams
├── repositories/              ⭐ the ONLY database/storage access
├── ai/                        ⭐ the ONLY LangChain imports
├── graph/
│   ├── state.py               AgentState
│   ├── builder.py             node wiring + numbered extension points
│   └── nodes/                 one file per node
├── services/                  all business logic
├── infrastructure/            port implementations (cache, guardrails, tools, sources)
├── schemas/                   request/response DTOs
└── api/v1/                    routes
```

---

## The agent graph

```
START
  → load_assistant       AssistantService     config + which model to use
  → load_user_memory     MemoryService        user summary, conversation summary, last 10 messages
  → retrieve_documents   RetrievalService     sources → merge → rerank
  → build_prompt         app/ai/prompts       LangChain templates
  → llm                  OpenRouterService    streaming
  → save_conversation    ConversationService  persist both messages
  → update_memory        MemoryService        refresh both summaries
END
```

### `AgentState`

Defined in `app/graph/state.py`. Every node reads what it needs and returns only
the keys it changed.

| Field | Written by |
| --- | --- |
| `user_id`, `assistant_id`, `conversation_id`, `question` | the caller |
| `assistant`, `system_prompt`, `model` | `load_assistant` |
| `user_summary`, `conversation_summary`, `recent_messages` | `load_user_memory` |
| `retrieved_docs`, `citations` | `retrieve_documents` |
| `prompt_messages` | `build_prompt` |
| `answer`, `usage` | `llm` |
| `user_message_id`, `assistant_message_id` | `save_conversation` |
| `tool_calls`, `guardrail_events`, `metadata`, `error` | extension slots |

Everything is JSON-serialisable, so a LangGraph checkpointer can be switched on
for durable/resumable runs without touching state or nodes.

### Prompt layering

`build_prompt` composes, in this order:

1. the assistant's **system prompt**
2. the long-term **user summary**
3. the **conversation summary**
4. the verbatim **recent messages**
5. the retrieved **document chunks**, numbered and attributed
6. the current **question**

### Ingestion pipeline

```
store metadata → load (LangChain) → split (LangChain)
              → embed → upsert to Qdrant → tagged with assistant_id
```

Runs in the background; the upload returns immediately with
`status='processing'`, which your frontend already polls.

Point IDs are `uuid5(file_id, chunk_index)`, so re-indexing a file overwrites its
chunks instead of duplicating them.

### Multi-tenancy

One Qdrant collection for everyone, following Qdrant's own recommendation. Every
point carries `user_id` and `assistant_id` in its payload, both keyword-indexed,
and **every** search and delete filters on them. Verified by
`scripts/test_retrieval.py`, which asserts one user cannot retrieve another's
chunks.

---

## API

All endpoints require `Authorization: Bearer <supabase access token>` except
`/health` and `/ready`.

### `/auth`

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/v1/auth/me` | Identity behind the presented token |
| POST | `/api/v1/auth/verify` | Verify a token from the body (returns `valid: false`, not 401) |

### `/assistants`

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/v1/assistants` | List |
| POST | `/api/v1/assistants` | Create |
| GET | `/api/v1/assistants/{id}` | Get one |
| PATCH | `/api/v1/assistants/{id}` | Update prompt, **model**, temperature, config |
| POST | `/api/v1/assistants/{id}/activate` | Make active (deactivates others) |
| DELETE | `/api/v1/assistants/{id}` | Delete + purge its vectors and blobs |

### `/knowledge`

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/v1/knowledge?assistant_id=` | List files |
| POST | `/api/v1/knowledge/upload?assistant_id=` | Upload PDF/TXT/MD, index in background |
| POST | `/api/v1/knowledge/index-pending?assistant_id=` | Index files uploaded straight to Supabase Storage |
| GET | `/api/v1/knowledge/stats?assistant_id=` | Counts, including live Qdrant vector count |
| GET | `/api/v1/knowledge/{id}` | Get one |
| GET | `/api/v1/knowledge/{id}/download-url` | Signed URL |
| POST | `/api/v1/knowledge/{id}/reindex` | Re-index synchronously (best way to debug a failure) |
| DELETE | `/api/v1/knowledge/{id}` | Delete row + blob + vectors |

### `/chat`

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/v1/chat` | Run the graph, **stream** the answer (SSE) |
| POST | `/api/v1/chat/sync` | Same graph, single JSON response |

Request:

```json
{ "question": "How do I reset my password?",
  "assistant_id": "uuid",
  "conversation_id": "uuid" }
```

`assistant_id` is optional — it falls back to the user's active assistant.
`conversation_id` is optional — omit it to start a new conversation, and read the
new id from the `start` event.

SSE frames are `data: <json>`, terminated by `data: [DONE]`:

| `type` | Payload |
| --- | --- |
| `start` | `conversation_id`, `assistant_id`, `model`, `is_new_conversation` |
| `token` | `content` — one delta |
| `revision` | `content` — a guardrail rewrote the answer; replace what you rendered |
| `done` | `answer`, `citations`, `usage`, message ids, optional `title`, optional `warning` |
| `error` | `code`, `message` |

### `/conversations`

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/v1/conversations?assistant_id=&limit=&offset=` | List |
| POST | `/api/v1/conversations` | Create empty (optional — `/chat` does it for you) |
| GET | `/api/v1/conversations/{id}` | Conversation + messages |
| GET | `/api/v1/conversations/{id}/messages` | Messages, oldest first |
| PATCH | `/api/v1/conversations/{id}` | Rename |
| DELETE | `/api/v1/conversations/{id}` | Delete conversation, messages and memory |

### `/memory`

| Method | Path | Purpose |
| --- | --- | --- |
| GET / PUT / DELETE | `/api/v1/memory/user` | The rolling user summary |
| GET / PUT / DELETE | `/api/v1/memory/conversations/{id}` | The rolling conversation summary |
| GET | `/api/v1/memory/snapshot?conversation_id=` | Exactly what the agent will load next turn |

### Errors

Every error uses one envelope:

```json
{ "error": { "code": "not_found", "message": "Assistant not found.",
             "details": { "assistant_id": "…" } } }
```

---

## Wiring the frontend

Your `chat.tsx` currently simulates the response
(`RAG-SaaS-UI/src/routes/_app/chat.tsx`, the `handleSend` function). Replace the
simulation with the SSE stream — the backend persists both messages itself, so
the frontend no longer inserts them:

```ts
// RAG-SaaS-UI/.env
// VITE_API_URL=http://localhost:8000

const { data: { session } } = await supabase.auth.getSession();

const res = await fetch(`${import.meta.env.VITE_API_URL}/api/v1/chat`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Authorization: `Bearer ${session!.access_token}`,
  },
  body: JSON.stringify({
    question: content,
    assistant_id: activeAssistantId,
    conversation_id: activeConversationId, // omit to start a new conversation
  }),
});

const reader = res.body!.getReader();
const decoder = new TextDecoder();
let buffer = "";
let streamed = "";

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });

  // Frames are separated by a blank line.
  const frames = buffer.split("\n\n");
  buffer = frames.pop() ?? "";

  for (const frame of frames) {
    const payload = frame.replace(/^data:\s*/, "").trim();
    if (!payload || payload === "[DONE]") continue;
    const event = JSON.parse(payload);

    switch (event.type) {
      case "start":
        // Adopt the id when the backend created the conversation for us.
        setActiveConversationId(event.conversation_id);
        break;
      case "token":
        streamed += event.content;
        setStreamingText(streamed); // render progressively
        break;
      case "revision":
        streamed = event.content;   // a guardrail rewrote the answer
        setStreamingText(streamed);
        break;
      case "done":
        setStreamingText("");
        // Messages are already persisted — just refetch.
        await queryClient.invalidateQueries({ queryKey: ["messages", event.conversation_id] });
        await queryClient.invalidateQueries({ queryKey: ["conversations", activeAssistantId] });
        break;
      case "error":
        console.error(event.code, event.message);
        break;
    }
  }
}
```

Two other frontend touch-ups worth making:

- **Uploads.** Either switch `settings.tsx` to `POST /api/v1/knowledge/upload`
  (multipart), or keep the direct-to-storage upload and call
  `POST /api/v1/knowledge/index-pending?assistant_id=…` afterwards. Without one of
  these, files are stored but never embedded and stay at `status: "processing"`.
- **`types.ts`.** `Assistant` is missing the new `model`, `temperature`,
  `max_tokens` and `config` fields, and `KnowledgeFile` is missing `chunk_count`
  and `error_message`. Nothing breaks without them, but you cannot build a model
  picker until they are added.

---

## Switching models

The model is read from `assistants.model` on every turn. No code names a model
except the `OPENROUTER_DEFAULT_MODEL` fallback.

```bash
curl -X PATCH http://localhost:8000/api/v1/assistants/$ID \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"model": "anthropic/claude-sonnet-4.5", "temperature": 0.3}'
```

Or straight in SQL:

```sql
update assistants set model = 'google/gemini-2.0-flash-001' where id = '…';
```

Takes effect on the next message. No restart, no deploy.

For provider routing, reasoning effort or fallbacks, use `config.llm.extra_body` —
passed through to OpenRouter verbatim:

```sql
update assistants
set config = '{"llm": {"extra_body": {"provider": {"sort": "throughput"}}}}'::jsonb
where id = '…';
```

`config` also carries per-assistant retrieval, tool and MCP overrides, so most
future per-assistant settings need no migration:

```json
{
  "llm":       { "temperature": 0.2, "extra_body": {} },
  "retrieval": { "top_k": 8, "candidate_k": 40, "reranker": "cohere" },
  "tools":     ["web_search"],
  "mcp_servers": ["filesystem"]
}
```

---

## Growing the system

Each of these is configuration plus, at most, one new adapter file.

### Redis cache

```powershell
pip install redis
docker compose --profile cache up -d redis
```

```ini
CACHE_PROVIDER=redis
```

`NullCache` and `RedisCache` satisfy the same port, and call sites already cache
unconditionally, so nothing else changes. Key builders live in
`app/infrastructure/cache/factory.py`.

### Reranking

```powershell
pip install cohere
```

```ini
RERANKER_PROVIDER=cohere
RERANKER_API_KEY=…
RETRIEVAL_CANDIDATE_K=40   # fetch more, let the reranker pick
```

`RetrievalService` already runs retrieve → merge → **rerank** → trim; today the
rerank step is a pass-through.

### More knowledge sources

1. Implement `KnowledgeSource` (one method: `retrieve`).
2. Register it in `app/infrastructure/knowledge_sources/registry.py`.
3. Enable per assistant: `config.retrieval.sources = ["vector", "my_source"]`.

`RetrievalService` fans out concurrently and merges by score. A failing source is
logged and skipped, never fatal.

For LangChain's retriever ecosystem (MultiQuery, ContextualCompression,
ParentDocument, Ensemble), `app/ai/vectorstore.py` exposes the same collection as
a LangChain `BaseRetriever` with the tenant filter already baked in.

### Tools

1. Write a LangChain tool.
2. `@register_tool("my_tool")` in `app/infrastructure/tools/registry.py`.
3. `TOOLS_ENABLED=true`, `ENABLED_TOOLS=my_tool`.
4. Uncomment extension point **(3)** in `app/graph/builder.py` to add the
   `llm → tools → llm` conditional edge. `route_after_llm` is already written.

The LLM node already binds whatever the resolver returns.

### MCP servers

```powershell
pip install langchain-mcp-adapters
```

```ini
MCP_ENABLED=true
MCP_SERVERS={"filesystem":{"transport":"stdio","command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","/data"]}}
```

Then per assistant: `config.mcp_servers = ["filesystem"]`. `MCPToolProvider` is
already registered with `ToolResolver`, so MCP tools arrive by the same path as
native ones.

### Guardrails

```ini
GUARDRAILS_ENABLED=true
GUARDRAILS=input_length,pii
```

Two reference guardrails ship in
`app/infrastructure/guardrails/registry.py`. Add your own with
`@register_guardrail("name")`. Input checks run in `ChatService` before anything
is billed; output checks run in the LLM node and may rewrite the answer, which
surfaces to the client as a `revision` event.

### Observability

```powershell
pip install langsmith
```

```ini
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=…
LANGSMITH_PROJECT=rag-system-api
```

Genuinely zero-code: LangChain reads those env vars itself. Every LLM and graph
call already passes `run_name`, `metadata` (user/assistant/conversation ids) and
`callbacks`, so traces are filterable immediately. To add another backend, append
a handler in `app/core/observability.py::tracing_callbacks()` — every call site
already asks for its callbacks there.

### Durable runs / human-in-the-loop

Pass a LangGraph checkpointer at `checkpointer=` in `app/container.py`.
`thread_id` is already set to the conversation id, and `AgentState` is already
JSON-serialisable.

---

## Testing

Three self-contained scripts. No credentials, no running services, no network.

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py      # imports, wiring, graph, prompts, auth
.\.venv\Scripts\python.exe scripts\test_retrieval.py  # real embeddings + embedded Qdrant
.\.venv\Scripts\python.exe scripts\test_api.py        # real ASGI cycle, auth, SSE, CORS
```

What they cover:

- **`smoke_test.py`** — all 38 modules import; 25 routes registered; container
  wires 10 services; the graph has the 7 specified nodes wired in order; prompt
  layering; a full graph run against fake services with token streaming; loaders,
  splitters, parsers; JWT accept/expired/forged/wrong-audience; model resolution
  from assistant config.
- **`test_retrieval.py`** — real fastembed embeddings and the real Qdrant client
  in embedded mode: ingest → embed → upsert → search → rerank, relevance,
  **tenant isolation**, idempotent re-index, delete.
- **`test_api.py`** — startup with Qdrant unreachable, health/readiness, four
  auth-rejection paths, valid-token flow, validation envelopes, CORS preflight,
  SSE frame shape and termination, OpenAPI schema.

All currently pass. What they do **not** cover: a real OpenRouter call, real
Supabase reads/writes, and a real Qdrant server — those need your credentials.
The end-to-end path is exercised with fakes at the Supabase and OpenRouter
boundaries only.

---

## Design notes

Decisions worth knowing about, and why.

**Embeddings are local by default.** OpenRouter has no embeddings endpoint — it
serves chat completions only. Rather than require a second paid API key,
`EMBEDDING_PROVIDER=fastembed` runs BAAI/bge-small-en-v1.5 as ONNX in-process:
384 dimensions, no key, no network after the first download. Swap to a hosted
provider with one env var. Changing dimensions needs a fresh Qdrant collection;
`QdrantService.ensure_ready` detects the mismatch at startup and says so instead
of failing cryptically later.

**Database access is PostgREST over httpx**, not `supabase-py`. Timeouts,
connection pooling, retries and error translation are explicit and shared with
the rest of the app, and `delete`/`update` refuse to run without a filter — a
guard against an accidental full-table write.

**Both messages are saved after generation**, not the question up-front. A failed
turn leaves no orphaned user message in the thread.

**Memory updates run last, after the answer has streamed.** The summarisation LLM
call costs the user no perceived latency. It is also failure-tolerant: losing a
summary must never lose a saved answer.

**The user profile refreshes every N turns** (`MEMORY_USER_SUMMARY_EVERY_N_TURNS`),
not every turn. Rebuilding it constantly is wasteful and makes it drift.

**Streaming does not depend on LangGraph internals.** LangGraph's stream modes
emit state updates rather than token deltas, and its token-event APIs vary by
version. Instead the LLM node receives an `on_token` callback through the run
config, and `GraphService` bridges it to an async generator. The transport
decision stays in one place and survives version bumps.

**A client disconnect does not cancel the run.** If the browser goes away
mid-stream the answer may already be generated, so the graph is left to finish —
`save_conversation` and `update_memory` still execute and the turn appears in
history rather than vanishing. Detached runs are tracked and drained on shutdown.

**Retrieval degrades rather than aborts.** If Qdrant is unreachable the assistant
still answers, just without grounding, and `metadata.retrieval_error` records
why. The API also starts with Qdrant down — `/ready` reports `degraded`.

**Guardrail output rewrites emit a `revision` event.** Raw tokens have already
reached the client by the time the full answer can be checked, so the client is
told to replace what it rendered. The persisted message is always the checked
version.

**Token usage is passed through a caller-owned `usage_sink`,** not stored on the
service. `OpenRouterService` is a process-wide singleton; per-request state on it
would be clobbered by concurrent chats.

**Loading, splitting and embedding run in worker threads.** fastembed and the PDF
parser are synchronous and CPU-bound; running them on the event loop would stall
every other request for the duration of a large upload.

### Known gaps

Deliberate, given the stated scope:

- **Semantic memory is not implemented** — only the two rolling summaries and the
  recent-message window, as specified. The seam is
  `MemoryService.load_context()`.
- **Background indexing is in-process** (FastAPI `BackgroundTasks`). An API
  restart mid-ingestion leaves a row at `processing`;
  `POST /knowledge/index-pending` is the recovery path. A real queue (Celery,
  Arq) is the next step if ingestion volume grows.
- **`AssistantRepository.deactivate_all` is two round trips** — clear all, then
  set one. Fine for a per-user handful of assistants; a stored procedure would
  make it atomic.
- **No rate limiting.** Add it at the edge, or as middleware, before exposing
  this publicly.
