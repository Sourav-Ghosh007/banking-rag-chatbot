"""
Auth Tests
==========
Tests for JWT, password hashing, PKCE, and encryption.
Run with: pytest tests/test_auth.py -v
"""

import os
import pytest

# Set a test Fernet key before any imports
os.environ["FERNET_KEY"] = "ZmDfcTF7_60GrrY167zsiPd67pEvs0aGOv2oasOM1Pg="
os.environ["SECRET_KEY"] = "test-secret-key-for-unit-tests"


# ---------------------------------------------------------------------------
# Test: Password hashing
# ---------------------------------------------------------------------------
class TestPasswordHashing:

    def test_hash_is_not_plain_text(self):
        """Hashed password should not equal the original."""
        from backend.utils.encryption import hash_password
        hashed = hash_password("MyPassword123")
        assert hashed != "MyPassword123"
        assert len(hashed) > 20

    def test_correct_password_verifies(self):
        """Correct password should verify against its hash."""
        from backend.utils.encryption import hash_password, verify_password
        hashed = hash_password("BankingPass@99")
        assert verify_password("BankingPass@99", hashed) is True

    def test_wrong_password_fails(self):
        """Wrong password should NOT verify."""
        from backend.utils.encryption import hash_password, verify_password
        hashed = hash_password("CorrectPassword")
        assert verify_password("WrongPassword", hashed) is False

    def test_two_hashes_of_same_password_are_different(self):
        """bcrypt uses salt — same password gives different hashes each time."""
        from backend.utils.encryption import hash_password
        h1 = hash_password("SamePassword")
        h2 = hash_password("SamePassword")
        assert h1 != h2   # different salts


# ---------------------------------------------------------------------------
# Test: Fernet encryption
# ---------------------------------------------------------------------------
class TestEncryption:

    def test_encrypt_decrypt_string(self):
        """Encrypted string should decrypt back to original."""
        from backend.utils.encryption import encrypt_string, decrypt_string
        original  = "my-secret-token-12345"
        encrypted = encrypt_string(original)
        decrypted = decrypt_string(encrypted)
        assert decrypted == original
        assert encrypted != original

    def test_encrypt_decrypt_dict(self):
        """Token dict should survive encrypt → decrypt round trip."""
        from backend.utils.encryption import encrypt_dict, decrypt_dict
        token = {
            "access_token":  "ya29.abc123",
            "refresh_token": "1//xyz789",
            "expires_in":    3600,
        }
        encrypted = encrypt_dict(token)
        decrypted = decrypt_dict(encrypted)
        assert decrypted["access_token"]  == "ya29.abc123"
        assert decrypted["refresh_token"] == "1//xyz789"
        assert decrypted["expires_in"]    == 3600

    def test_encrypted_value_is_different_each_time(self):
        """Fernet adds random IV — same input gives different ciphertext."""
        from backend.utils.encryption import encrypt_string
        e1 = encrypt_string("hello")
        e2 = encrypt_string("hello")
        assert e1 != e2

    def test_wrong_data_raises_error(self):
        """Decrypting garbage should raise ValueError."""
        from backend.utils.encryption import decrypt_string
        with pytest.raises((ValueError, Exception)):
            decrypt_string("this-is-not-encrypted-data")


# ---------------------------------------------------------------------------
# Test: JWT token creation
# ---------------------------------------------------------------------------
class TestJWT:

    def test_create_and_decode_token(self):
        """JWT token should encode and decode user_id correctly."""
        from backend.routers.auth import create_jwt_token
        from jose import jwt
        token = create_jwt_token("user-123", "test@example.com")
        payload = jwt.decode(token, os.environ["SECRET_KEY"], algorithms=["HS256"])
        assert payload["sub"]   == "user-123"
        assert payload["email"] == "test@example.com"

    def test_token_is_string(self):
        """JWT token should be a non-empty string."""
        from backend.routers.auth import create_jwt_token
        token = create_jwt_token("user-456", "user@bank.com")
        assert isinstance(token, str)
        assert len(token) > 50


# ---------------------------------------------------------------------------
# Test: PKCE security
# ---------------------------------------------------------------------------
class TestPKCE:

    def test_code_verifier_length(self):
        """Code verifier should be 64 characters by default."""
        from backend.auth.pkce import generate_code_verifier
        verifier = generate_code_verifier()
        assert 43 <= len(verifier) <= 128

    def test_code_challenge_is_deterministic(self):
        """Same verifier should always give same challenge."""
        from backend.auth.pkce import generate_code_challenge
        verifier   = "test-verifier-string-abc123"
        challenge1 = generate_code_challenge(verifier)
        challenge2 = generate_code_challenge(verifier)
        assert challenge1 == challenge2

    def test_code_challenge_differs_from_verifier(self):
        """Challenge should be a hash of the verifier, not the verifier itself."""
        from backend.auth.pkce import generate_code_verifier, generate_code_challenge
        verifier  = generate_code_verifier()
        challenge = generate_code_challenge(verifier)
        assert verifier != challenge

    def test_state_token_is_unique(self):
        """Each state token should be unique."""
        from backend.auth.pkce import generate_state_token
        s1 = generate_state_token()
        s2 = generate_state_token()
        assert s1 != s2

    def test_pkce_store_and_retrieve(self):
        """Stored PKCE session should be retrievable by state."""
        from backend.auth.pkce import store_pkce_session, retrieve_pkce_session
        store_pkce_session("state-abc", "verifier-xyz", user_id="user-1")
        result = retrieve_pkce_session("state-abc")
        assert result is not None
        assert result["code_verifier"] == "verifier-xyz"

    def test_pkce_retrieve_removes_session(self):
        """retrieve_pkce_session should remove the session (one-time use)."""
        from backend.auth.pkce import store_pkce_session, retrieve_pkce_session
        store_pkce_session("state-once", "verifier-once")
        retrieve_pkce_session("state-once")           # first call — returns data
        result = retrieve_pkce_session("state-once")  # second call — should be None
        assert result is None

    def test_invalid_state_returns_none(self):
        """Retrieving a non-existent state should return None."""
        from backend.auth.pkce import retrieve_pkce_session
        result = retrieve_pkce_session("state-does-not-exist")
        assert result is None

    def test_create_pkce_pair_has_all_keys(self):
        """create_pkce_pair should return all 3 required keys."""
        from backend.auth.pkce import create_pkce_pair
        pair = create_pkce_pair()
        assert "code_verifier"  in pair
        assert "code_challenge" in pair
        assert "state"          in pair
