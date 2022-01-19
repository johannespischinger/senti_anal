import pytest
from fastapi.testclient import TestClient

from opensentiment.api.fast.serve_api import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Model Serve API": "running"}


response_missing = "dummy"


@pytest.mark.skip  # no model downloaded
@pytest.mark.parametrize(
    "path,expected_status,expected_text",
    [
        ("/api/v1/serve_single", 422, response_missing),
        (
            "/api/v1/serve_single?query_text=thisreviewisbad",
            200,
            "thisreviewisbad",
        ),
    ],
)
def test_serve_single(path, expected_status, expected_text):
    response = client.get(path)
    assert response.status_code == expected_status
    r_json = response.json()

    if expected_status == 200:
        assert (
            r_json["query_text"] == expected_text
        ), f"got {response.json()}, expected {expected_text}"
