# MCP Server: Research Grants, with Guardrails

A [Model Context Protocol](https://modelcontextprotocol.io) server that lets
an AI assistant work with a multi-tenant research-administration database:
list and search grant proposals, read funding-opportunity instructions,
turn those instructions into validation rules, check proposals against
the rules, and move proposals through their workflow.

The tools themselves are simple. The point of the project is the
engineering around them, the part that decides whether an AI integration
is safe to put in front of customers:

| Concern | How it's handled |
|---|---|
| **Authentication** | API key → principal (user, institution, role). Keys are stored only as SHA-256 hashes. A bad key fails at server startup. |
| **Permissions** | Roles `viewer` / `editor` / `admin` map to permissions (`read`, `write`, `propose_rules`, `approve_rules`), checked on every call. |
| **Tenant isolation** | Every query is scoped to the tenant from the API key, never from tool arguments. Another institution's record returns "not found", the same as a missing one, so its existence isn't leaked. |
| **Input validation** | Typed parameters (JSON Schema generated from type hints), enums, bounded limits and text lengths, parameterized SQL, escaped `LIKE` wildcards. |
| **Prompt injection** | Free-text fields (titles, abstracts, instructions) are returned with an `untrusted_fields` label and a notice; server instructions tell the model not to follow text inside them. The seeded data includes an abstract with an injection payload, and tests confirm reading it changes nothing. Write tools are limited, role-gated, and bounded by a state machine. |
| **Human review** | Rule approval is **not an MCP tool** at all. A person approves rules with `src/review_rules.py`. |
| **Auditability** | Every call (successful, denied, or failed) writes an audit row: user, tenant, tool, arguments, outcome. |

## Hybrid AI validation

Funding opportunities come with prose instructions ("Requested budget must
not exceed $750,000"). An assistant can read them and **propose** rules,
but the model's output is never trusted directly:

1. **Closed schema** (`grants_rules.parse_rule`). Only `max_amount`,
   `max_pages`, and `submit_by` exist; extra fields, wrong types, and
   unknown rule types are rejected.
2. **Grounding** (`check_grounding`). Each rule must cite a quote that
   appears verbatim in the instructions, and the rule's value must be
   stated in that quote. A model that "reads" $750,000 as $900,000 gets
   its rule rejected.
3. **Human approval**. Accepted rules are stored as `proposed` and do not
   run until an admin approves them outside the MCP surface.
4. **Deterministic evaluation**. Approved rules run as plain comparisons,
   so the same proposal always gets the same explainable result, and a
   submission that fails an approved rule is blocked.

## Tools

| Tool | Permission | Description |
|---|---|---|
| `whoami` | read | Caller, role, institution, permissions |
| `list_proposals` | read | Filter by status/opportunity, paginated |
| `search_proposals` | read | Phrase search over titles and abstracts |
| `get_proposal` | read | One proposal including abstract |
| `list_opportunities` | read | Sponsor calls and deadlines |
| `get_opportunity` | read | Instructions plus this tenant's rules |
| `propose_rules` | propose_rules | Submit extracted rules for review |
| `validate_proposal` | read | Run approved rules, report pass/fail |
| `update_status` | write | Workflow transitions; submission gated by rules |

## Layout

| File | Role |
|---|---|
| `src/grants_store.py` | Schema and seeded synthetic data (3 institutions, 60 proposals, 3 opportunities) |
| `src/grants_auth.py` | API-key authentication, roles, permissions |
| `src/grants_rules.py` | Rule schema, grounding check, deterministic evaluation |
| `src/grants_service.py` | Tool logic plus all guardrails, transport-independent |
| `src/grants_mcp_server.py` | Thin MCP layer (official Python SDK, `FastMCP`, stdio) |
| `src/review_rules.py` | Human-only rule approval CLI |
| `src/smoke_client.py` | Launches the server as a subprocess and drives it with the official MCP client |

## Run it

```bash
pip install -r requirements.txt
python src/grants_store.py                      # builds data/grants.db, prints dev keys
GRANTS_API_KEY=dev-t1-editor python src/grants_mcp_server.py
```

To use it from an MCP-capable assistant, register a stdio server whose
command is `python /path/to/src/grants_mcp_server.py` with `GRANTS_DB`
and `GRANTS_API_KEY` set in its environment.

## Tests

```bash
python -m pytest -v tests        # service, auth, rules, and in-memory MCP protocol tests
python src/smoke_client.py       # end-to-end over real stdio
```

- **Service tests** cover tenant isolation, permission denials (and their
  audit rows), input validation, SQL-injection and wildcard inputs, the
  injection-payload abstract, the propose → approve → validate workflow,
  rule-blocked submission, and state-machine transitions.
- **Protocol tests** connect a real MCP client to the server in memory.
  They check the tool list (including that rule approval is absent), the
  generated input schemas, and that errors surface as MCP tool errors.
- **CI** runs all of the above plus the stdio smoke test.

Mutation check: removing the tenant filter from one query, or the
`LIKE` escaping, makes the suite fail.

*Data is synthetic. Institution names, people, and proposals are invented.*
