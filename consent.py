import os
import json
import hmac
import hashlib
import base64
import uuid
import time

KEY_PATH = os.environ.get("CONSENT_HMAC_KEY_PATH", "consent_key.bin")


def _load_or_create_key():
    if os.path.exists(KEY_PATH):
        with open(KEY_PATH, "rb") as f:
            return f.read()
    key = os.urandom(32)
    with open(KEY_PATH, "wb") as f:
        f.write(key)
    try:
        os.chmod(KEY_PATH, 0o600)
    except Exception:
        pass
    return key


def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _b64u_decode(s: str) -> bytes:
    padding = "=" * ((4 - len(s) % 4) % 4)
    return base64.urlsafe_b64decode((s + padding).encode("utf-8"))


def create_consent(participant_id: str, scope: str = "face_capture", metadata: dict = None):
    """Create an HMAC-signed consent token and return (token, record)

    Token format: base64url(payload).base64url(hmac_sha256)
    Payload contains participant_id, scope, timestamp, metadata
    """
    key = _load_or_create_key()
    payload = {
        "participant_id": participant_id,
        "scope": scope,
        "timestamp": int(time.time()),
        "nonce": str(uuid.uuid4()),
    }
    if metadata:
        payload["metadata"] = metadata

    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sig = hmac.new(key, payload_bytes, hashlib.sha256).digest()
    token = _b64u(payload_bytes) + "." + _b64u(sig)

    return token, payload


def verify_consent_token(token: str) -> bool:
    try:
        key = _load_or_create_key()
        payload_b64, sig_b64 = token.split(".")
        payload_bytes = _b64u_decode(payload_b64)
        sig = _b64u_decode(sig_b64)
        expected = hmac.new(key, payload_bytes, hashlib.sha256).digest()
        return hmac.compare_digest(expected, sig)
    except Exception:
        return False


def persist_consent(token: str, record: dict, directory: str = "consents"):
    os.makedirs(directory, exist_ok=True)
    ts = int(time.time())
    fn = f"consent_{ts}_{record.get('nonce', str(uuid.uuid4()))}.json"
    path = os.path.join(directory, fn)
    out = {"token": token, "record": record}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path


if __name__ == "__main__":
    # simple CLI demo
    pid = input("Participant ID (leave blank to generate): ") or str(uuid.uuid4())
    token, rec = create_consent(pid)
    p = persist_consent(token, rec)
    print("Saved consent:", p)
