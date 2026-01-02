# app/memory/acl.py

ALLOWED_ROLES = {"customer", "admin", "support"}

def validate_role(role: str):
    if role not in ALLOWED_ROLES:
        raise PermissionError(f"Invalid role: {role}")
