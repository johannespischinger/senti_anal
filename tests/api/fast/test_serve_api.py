import pytest
from fastapi.testclient import TestClient
import glob
import os
from opensentiment.utils import paths_to_file_ext
from opensentiment.api.fast.serve_api import app

client_not_loaded = TestClient(app)


def test_read_main():
    response = client_not_loaded.get("/")
    assert response.status_code == 200
    assert response.json() == {"Model Serve API": "running"}


class TestAPP:
    not_model_exists = not bool(
        paths_to_file_ext(folder="model_store", file_ext="ckpt")[0]
    )

    response_missing = "dummy"

    @pytest.mark.skipif(not_model_exists, reason="no model found")
    @pytest.mark.parametrize(
        "path,expected_status,expected_text,expected_sentiment",
        [
            (
                "/api/v1/serve_single",
                422,
                response_missing,
                response_missing,
            ),
            (
                "/api/v1/serve_single?query_text=a positive review. i like it",
                200,
                "a positive review. i like it",
                "Positive",
            ),
            (
                "/api/v1/serve_single?query_text=a negative review. i hate it",
                200,
                "a negative review. i hate it",
                "Negativ",
            ),
        ],
    )
    def test_serve_single(
        self, path, expected_status, expected_text, expected_sentiment
    ):
        with TestClient(app) as client_loaded:
            response = client_loaded.get(path)
            assert response.status_code == expected_status
            r_json = response.json()

            if expected_status == 200:
                assert (
                    r_json["query_text"] == expected_text
                ), f"got {r_json}, expected {expected_text}"
                assert (
                    r_json["prediction"][0][1] == expected_sentiment
                ), f"got {r_json}, expected_sentiment {expected_sentiment}"

    @pytest.mark.skipif(not_model_exists, reason="no model found")
    @pytest.mark.parametrize(
        "path,expected_status,expected_text",
        [
            ("/api/v1/serve_batch", 422, response_missing),
            ("/api/v1/serve_batch?q=foo&q=bar", 200, ["foo", "bar"]),
        ],
    )
    def test_serve_batch(self, path, expected_status, expected_text):
        with TestClient(app) as client_loaded:
            response = client_loaded.get(path)
            assert response.status_code == expected_status
            r_json = response.json()

            if expected_status == 200:
                assert (
                    r_json["query_list"] == expected_text
                ), f"got {response.json()}, expected {expected_text}"
