import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_SESSION_ID = str(uuid.uuid4())

_BENIGN_RESULT = {
    "verdict": "ALLOW",
    "label": 0,
    "confidence": 0.08,
    "attack_family": None,
    "family_scores": {},
    "cluster_id": None,
    "cluster_label": None,
    "spans": [],
    "decoded_text": "hello",
    "rule_boost_applied": False,
    "model_version": "v0.1.0",
}


@pytest.fixture()
def client():
    from api.main import app
    from api.db.session import get_db
    from sqlalchemy.ext.asyncio import AsyncSession

    async def _fake_refresh(obj):
        if not getattr(obj, "id", None):
            obj.id = uuid.uuid4()
        if not getattr(obj, "session_id", None):
            obj.session_id = uuid.UUID(_SESSION_ID)

    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock(side_effect=_fake_refresh)

    async def override_get_db():
        yield mock_session

    with patch("api.ml.pipeline.run", return_value=_BENIGN_RESULT) as mock_pipeline, \
         patch("api.model_loader.ensure_loaded"), \
         patch("api.model_loader.get_best_model_name", return_value="RF"):
        app.dependency_overrides[get_db] = override_get_db
        yield TestClient(app, raise_server_exceptions=False), mock_pipeline, mock_session
        app.dependency_overrides.clear()


class TestSQLiLiteralStorage:
    def test_sqli_in_history_q_returns_200(self, client):
        tc, _, session = client
        mock_total = MagicMock()
        mock_total.scalar_one.return_value = 0
        mock_rows = MagicMock()
        mock_rows.scalars.return_value.all.return_value = []
        session.execute.side_effect = [mock_total, mock_rows]

        sid = str(uuid.uuid4())
        r = tc.get(f"/api/sessions/{sid}/history", params={"q": "' OR '1'='1 --"})
        assert r.status_code == 200

    def test_sqli_does_not_leak_all_rows(self, client):
        tc, _, session = client
        mock_total = MagicMock()
        mock_total.scalar_one.return_value = 0
        mock_rows = MagicMock()
        mock_rows.scalars.return_value.all.return_value = []
        session.execute.side_effect = [mock_total, mock_rows]

        sid = str(uuid.uuid4())
        body = tc.get(f"/api/sessions/{sid}/history", params={"q": "' OR '1'='1 --"}).json()
        assert body["total"] == 0
        assert body["items"] == []

    def test_union_injection_in_history_q(self, client):
        tc, _, session = client
        mock_total = MagicMock()
        mock_total.scalar_one.return_value = 0
        mock_rows = MagicMock()
        mock_rows.scalars.return_value.all.return_value = []
        session.execute.side_effect = [mock_total, mock_rows]

        sid = str(uuid.uuid4())
        payload = "x' UNION SELECT null,null,null,null --"
        r = tc.get(f"/api/sessions/{sid}/history", params={"q": payload})
        assert r.status_code == 200


class TestDoSSemaphore:

    def test_503_when_semaphore_exhausted(self, client):
        import asyncio
        tc, _, _ = client

        with patch("api.routers.classify._get_semaphore") as mock_sem_fn:
            mock_sem = MagicMock(spec=asyncio.Semaphore)
            mock_sem._value = 0
            mock_sem_fn.return_value = mock_sem

            r = tc.post(
                "/api/classify",
                json={"text": "test input", "session_id": _SESSION_ID},
            )
        assert r.status_code == 503

    def test_503_error_message(self, client):
        import asyncio
        tc, _, _ = client

        with patch("api.routers.classify._get_semaphore") as mock_sem_fn:
            mock_sem = MagicMock(spec=asyncio.Semaphore)
            mock_sem._value = 0
            mock_sem_fn.return_value = mock_sem

            body = tc.post(
                "/api/classify",
                json={"text": "test input", "session_id": _SESSION_ID},
            ).json()
        assert body.get("detail") is not None or "error" in body


class TestAuthDefence:
    """§2.3 API key auth: when a key is configured, wrong/missing key returns 401."""

    def test_wrong_api_key_returns_401(self, client):
        tc, _, _ = client
        with patch("api.core.auth.settings") as mock_settings:
            mock_settings.api_key = "supersecret"
            r = tc.post(
                "/api/classify",
                json={"text": "hello", "session_id": _SESSION_ID},
                headers={"X-API-Key": "wrongkey"},
            )
        assert r.status_code == 401

    def test_missing_api_key_returns_401(self, client):
        tc, _, _ = client
        with patch("api.core.auth.settings") as mock_settings:
            mock_settings.api_key = "supersecret"
            r = tc.post(
                "/api/classify",
                json={"text": "hello", "session_id": _SESSION_ID},
            )
        assert r.status_code == 401
