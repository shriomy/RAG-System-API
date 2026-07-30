-- ===========================================================================
-- 001_baseline.sql
--
-- The tables the frontend (RAG-SaaS-UI) already talks to. Written
-- IF NOT EXISTS so it is a no-op on your existing project and a full
-- bootstrap on a fresh one. Run in the Supabase SQL editor.
-- ===========================================================================

create extension if not exists "pgcrypto";

-- ---------------------------------------------------------------------------
-- assistants
-- ---------------------------------------------------------------------------
create table if not exists public.assistants (
  id            uuid primary key default gen_random_uuid(),
  user_id       uuid not null default auth.uid() references auth.users (id) on delete cascade,
  name          text not null default 'New assistant',
  system_prompt text not null default '',
  is_active     boolean not null default false,
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now()
);

create index if not exists assistants_user_id_idx on public.assistants (user_id);

-- ---------------------------------------------------------------------------
-- conversations
-- ---------------------------------------------------------------------------
create table if not exists public.conversations (
  id           uuid primary key default gen_random_uuid(),
  user_id      uuid not null default auth.uid() references auth.users (id) on delete cascade,
  assistant_id uuid references public.assistants (id) on delete set null,
  title        text not null default 'New conversation',
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now()
);

create index if not exists conversations_user_id_idx on public.conversations (user_id);
create index if not exists conversations_assistant_id_idx on public.conversations (assistant_id);
create index if not exists conversations_updated_at_idx on public.conversations (updated_at desc);

-- ---------------------------------------------------------------------------
-- messages
-- ---------------------------------------------------------------------------
create table if not exists public.messages (
  id              uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references public.conversations (id) on delete cascade,
  user_id         uuid not null default auth.uid() references auth.users (id) on delete cascade,
  role            text not null check (role in ('user', 'assistant', 'system')),
  content         text not null,
  created_at      timestamptz not null default now()
);

create index if not exists messages_conversation_id_idx on public.messages (conversation_id, created_at);
create index if not exists messages_user_id_idx on public.messages (user_id);

-- ---------------------------------------------------------------------------
-- knowledge_files
-- ---------------------------------------------------------------------------
create table if not exists public.knowledge_files (
  id           uuid primary key default gen_random_uuid(),
  user_id      uuid not null default auth.uid() references auth.users (id) on delete cascade,
  assistant_id uuid not null references public.assistants (id) on delete cascade,
  filename     text not null,
  file_type    text not null check (file_type in ('pdf', 'txt', 'md')),
  storage_path text not null,
  status       text not null default 'processing' check (status in ('processing', 'indexed', 'failed')),
  file_size    bigint,
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now()
);

create index if not exists knowledge_files_assistant_id_idx on public.knowledge_files (assistant_id, created_at desc);
create index if not exists knowledge_files_user_id_idx on public.knowledge_files (user_id);

-- ---------------------------------------------------------------------------
-- Row Level Security — the frontend uses the anon key, so every table must be
-- locked to the owning user. The backend uses the service_role key, which
-- bypasses RLS; it enforces ownership itself with explicit user_id filters.
-- ---------------------------------------------------------------------------
alter table public.assistants      enable row level security;
alter table public.conversations   enable row level security;
alter table public.messages        enable row level security;
alter table public.knowledge_files enable row level security;

do $$
begin
  if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = 'assistants' and policyname = 'assistants_owner_all') then
    create policy assistants_owner_all on public.assistants
      for all to authenticated using (user_id = auth.uid()) with check (user_id = auth.uid());
  end if;

  if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = 'conversations' and policyname = 'conversations_owner_all') then
    create policy conversations_owner_all on public.conversations
      for all to authenticated using (user_id = auth.uid()) with check (user_id = auth.uid());
  end if;

  if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = 'messages' and policyname = 'messages_owner_all') then
    create policy messages_owner_all on public.messages
      for all to authenticated using (user_id = auth.uid()) with check (user_id = auth.uid());
  end if;

  if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = 'knowledge_files' and policyname = 'knowledge_files_owner_all') then
    create policy knowledge_files_owner_all on public.knowledge_files
      for all to authenticated using (user_id = auth.uid()) with check (user_id = auth.uid());
  end if;
end $$;

-- ---------------------------------------------------------------------------
-- Storage bucket for uploaded knowledge files (private).
-- ---------------------------------------------------------------------------
insert into storage.buckets (id, name, public)
values ('knowledge_files', 'knowledge_files', false)
on conflict (id) do nothing;

do $$
begin
  -- Frontend uploads to `${user_id}/${assistant_id}/${ts}-${name}`, so the
  -- first path segment is the owner's uuid.
  if not exists (select 1 from pg_policies where schemaname = 'storage' and tablename = 'objects' and policyname = 'knowledge_files_owner_all') then
    create policy knowledge_files_owner_all on storage.objects
      for all to authenticated
      using (bucket_id = 'knowledge_files' and (storage.foldername(name))[1] = auth.uid()::text)
      with check (bucket_id = 'knowledge_files' and (storage.foldername(name))[1] = auth.uid()::text);
  end if;
end $$;
