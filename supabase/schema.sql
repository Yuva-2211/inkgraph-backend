-- ============================================================
-- InkGraph — Supabase schema
-- Run in the Supabase SQL editor, or via `supabase db push`
--
-- NOTE: user_id is stored as a plain uuid without a FK to auth.users
-- so the demo works without Supabase Auth configured.
-- For production, add: REFERENCES auth.users(id) ON DELETE CASCADE
-- ============================================================

create extension if not exists "uuid-ossp";

-- ---------- documents ----------
create table if not exists documents (
  id              uuid        primary key default uuid_generate_v4(),
  user_id         uuid        not null,          -- client-generated uuid; no auth FK for demo
  title           text        not null default 'Untitled Document',
  prompt          text        not null,
  status          text        not null default 'planning'
    check (status in ('planning','writing','reviewing','awaiting_human','revising','approved','archived')),
  current_content text,
  outline         jsonb,
  review_cycle    int         not null default 0,
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now()
);

-- ---------- revisions ----------
-- Append-only history of every stage transition / draft version.
create table if not exists revisions (
  id          uuid        primary key default uuid_generate_v4(),
  document_id uuid        not null references documents (id) on delete cascade,
  stage       text        not null check (stage in ('planner','writer','reviewer','human','system')),
  content     text,
  note        text,
  created_at  timestamptz not null default now()
);

-- ---------- agent_runs ----------
-- One row per LangGraph node execution — for observability / cost tracking.
create table if not exists agent_runs (
  id          uuid        primary key default uuid_generate_v4(),
  document_id uuid        not null references documents (id) on delete cascade,
  agent       text        not null,
  input       jsonb,
  output      jsonb,
  latency_ms  int,
  token_usage jsonb,
  status      text        not null default 'success' check (status in ('success','error')),
  created_at  timestamptz not null default now()
);

-- ---------- indexes ----------
create index if not exists idx_documents_user      on documents (user_id);
create index if not exists idx_documents_status    on documents (status);
create index if not exists idx_revisions_document  on revisions (document_id, created_at);
create index if not exists idx_agent_runs_document on agent_runs (document_id, created_at);

-- ---------- updated_at trigger ----------
create or replace function set_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists trg_documents_updated_at on documents;
create trigger trg_documents_updated_at
  before update on documents
  for each row execute function set_updated_at();

-- ---------- row level security ----------
-- Disabled for demo (service role bypasses RLS anyway).
-- Re-enable for production with proper auth.
-- alter table documents enable row level security;
-- alter table revisions  enable row level security;
-- alter table agent_runs enable row level security;

-- ---------- realtime ----------
-- Clients subscribe to document status changes + new revisions.
alter publication supabase_realtime add table documents;
alter publication supabase_realtime add table revisions;
