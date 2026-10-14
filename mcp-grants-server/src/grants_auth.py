"""Authentication and role-based permissions.

An API key resolves to a ``Principal``: who is calling, which tenant they
belong to, and their role. Every tool checks a permission against that
principal, and every query is scoped to ``principal.tenant_id`` on the
server side -- the tenant never comes from tool arguments, so a model
cannot ask its way into another institution's data.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from grants_store import hash_key

# Approving extracted rules is deliberately *not* granted through any MCP
# tool, even for admins: a person approves rules with review_rules.py.
ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "viewer": frozenset({"read"}),
    "editor": frozenset({"read", "write", "propose_rules"}),
    "admin": frozenset({"read", "write", "propose_rules", "approve_rules"}),
}


class AuthError(Exception):
    """Bad or missing credentials."""


class PermissionDenied(Exception):
    """Authenticated, but the role does not allow the action."""


@dataclass(frozen=True)
class Principal:
    user_id: int
    tenant_id: int
    role: str
    name: str

    def can(self, permission: str) -> bool:
        return permission in ROLE_PERMISSIONS.get(self.role, frozenset())


def authenticate(conn: sqlite3.Connection, api_key: str | None) -> Principal:
    if not api_key or not isinstance(api_key, str):
        raise AuthError("missing API key")
    row = conn.execute(
        "SELECT u.id, u.tenant_id, u.role, u.name FROM api_keys k "
        "JOIN users u ON u.id = k.user_id WHERE k.key_hash = ?",
        (hash_key(api_key),),
    ).fetchone()
    if row is None:
        raise AuthError("invalid API key")
    return Principal(user_id=row["id"], tenant_id=row["tenant_id"], role=row["role"], name=row["name"])


def require(principal: Principal, permission: str) -> None:
    if not principal.can(permission):
        raise PermissionDenied(f"role '{principal.role}' lacks permission '{permission}'")
