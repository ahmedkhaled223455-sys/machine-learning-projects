import pytest

from grants_auth import AuthError, Principal, authenticate, require, PermissionDenied


def test_valid_key_resolves_to_tenant_and_role(grants_env):
    p = authenticate(grants_env.conn, grants_env.keys["t2-editor"])
    assert (p.tenant_id, p.role) == (2, "editor")


@pytest.mark.parametrize("key", [None, "", "dev-t1-root", "dev-t1-admin "])
def test_bad_keys_are_rejected(grants_env, key):
    with pytest.raises(AuthError):
        authenticate(grants_env.conn, key)


def test_keys_are_stored_hashed(grants_env):
    stored = {r[0] for r in grants_env.conn.execute("SELECT key_hash FROM api_keys")}
    assert not stored & set(grants_env.keys.values())
    assert all(len(h) == 64 for h in stored)


def test_role_permissions():
    viewer = Principal(1, 1, "viewer", "v")
    editor = Principal(2, 1, "editor", "e")
    require(viewer, "read")
    with pytest.raises(PermissionDenied):
        require(viewer, "write")
    require(editor, "propose_rules")
    with pytest.raises(PermissionDenied):
        require(editor, "approve_rules")
    with pytest.raises(PermissionDenied):
        require(Principal(3, 1, "superuser", "x"), "read")  # unknown role gets nothing
