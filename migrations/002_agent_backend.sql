-- ===========================================================================
-- 002_agent_backend.sql
--
-- Everything the agent backend adds on top of the frontend's schema:
--   * per-assistant model configuration (so switching models is data, not code)
--   * ingestion bookkeeping on knowledge_files
--   * long-term memory tables (user summary + conversation summary)
--
-- Fully additive and idempotent. Run after 001_baseline.sql.
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- assistants: model configuration
--
-- `model`, `temperature`, `max_tokens` cover today. `config` is the forward
-- compatibility slot: retrieval overrides, tool allow-lists, MCP server refs,
-- reranker choice, guardrail sets — all land here without a migration.
-- ---------------------------------------------------------------------------
alter table public.assistants
  add column if not exists model       text            not null default 'openai/gpt-4o-mini',
  add column if not exists temperature numeric(3, 2)   not null default 0.7,
  add column if not exists max_tokens  integer         not null default 1024,
  add column if not exists config      jsonb           not null default '{}'::jsonb;

comment on column public.assistants.model is
  'OpenRouter model slug, e.g. openai/gpt-4o-mini, anthropic/claude-sonnet-4.5, google/gemini-2.0-flash-001.';
comment on column public.assistants.config is
  'Free-form per-assistant overrides read by the backend: {"retrieval":{"top_k":8},"tools":["web_search"],"mcp_servers":["files"],"reranker":"cohere","guardrails":["pii"]}.';

-- ---------------------------------------------------------------------------
-- knowledge_files: ingestion bookkeeping
-- ---------------------------------------------------------------------------
alter table public.knowledge_files
  add column if not exists chunk_count   integer not null default 0,
  add column if not exists error_message text,
  add column if not exists indexed_at    timestamptz;

-- ---------------------------------------------------------------------------
-- user_memory — one rolling long-term summary per user.
-- ---------------------------------------------------------------------------
create table if not exists public.user_memory (
  user_id     uuid primary key references auth.users (id) on delete cascade,
  summary     text not null default '',
  turn_count  integer not null default 0,
  updated_at  timestamptz not null default now()
);

comment on table public.user_memory is
  'Long-term memory, phase 1: a single rolling natural-language summary per user. Semantic/episodic memory would be added as sibling tables, not by changing this one.';

-- ---------------------------------------------------------------------------
-- conversation_memory — rolling summary of one conversation.
--
-- Separate table rather than a column on `conversations` so memory can grow
-- (embeddings, extracted facts, entity lists) without touching a table the
-- frontend reads with `select *`.
-- ---------------------------------------------------------------------------
create table if not exists public.conversation_memory (
  conversation_id uuid primary key references public.conversations (id) on delete cascade,
  user_id         uuid not null references auth.users (id) on delete cascade,
  summary         text not null default '',
  message_count   integer not null default 0,
  updated_at      timestamptz not null default now()
);

create index if not exists conversation_memory_user_id_idx on public.conversation_memory (user_id);

-- ---------------------------------------------------------------------------
-- RLS for the memory tables (frontend may read them with the anon key).
-- ---------------------------------------------------------------------------
alter table public.user_memory         enable row level security;
alter table public.conversation_memory enable row level security;

do $$
begin
  if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = 'user_memory' and policyname = 'user_memory_owner_all') then
    create policy user_memory_owner_all on public.user_memory
      for all to authenticated using (user_id = auth.uid()) with check (user_id = auth.uid());
  end if;

  if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = 'conversation_memory' and policyname = 'conversation_memory_owner_all') then
    create policy conversation_memory_owner_all on public.conversation_memory
      for all to authenticated using (user_id = auth.uid()) with check (user_id = auth.uid());
  end if;
end $$;

-- ---------------------------------------------------------------------------
-- Keep updated_at honest. The backend writes it explicitly, but the frontend
-- writes rows directly too, so enforce it in the database.
-- ---------------------------------------------------------------------------
create or replace function public.touch_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

do $$
declare
  t text;
begin
  foreach t in array array['assistants', 'conversations', 'knowledge_files', 'user_memory', 'conversation_memory']
  loop
    if not exists (
      select 1 from pg_trigger
      where tgname = t || '_touch_updated_at'
        and tgrelid = ('public.' || t)::regclass
    ) then
      execute format(
        'create trigger %I before update on public.%I for each row execute function public.touch_updated_at()',
        t || '_touch_updated_at', t
      );
    end if;
  end loop;
end $$;

-- ---------------------------------------------------------------------------
-- Bump conversations.updated_at when a message lands, so the frontend's
-- conversation list (ordered by updated_at desc) reorders correctly.
-- ---------------------------------------------------------------------------
create or replace function public.touch_conversation_on_message()
returns trigger
language plpgsql
as $$
begin
  update public.conversations set updated_at = now() where id = new.conversation_id;
  return new;
end;
$$;

do $$
begin
  if not exists (
    select 1 from pg_trigger
    where tgname = 'messages_touch_conversation'
      and tgrelid = 'public.messages'::regclass
  ) then
    create trigger messages_touch_conversation
      after insert on public.messages
      for each row execute function public.touch_conversation_on_message();
  end if;
end $$;
