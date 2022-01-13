from opensentiment.api.fast.serve_api_helper import score_model
from fastapi.encoders import jsonable_encoder


def test_score_model():
    # TODO: more tests
    out1, out2 = score_model("test review of some sentence good")
    assert type(out1) == bool
    assert type(out2) == dict
    out_json = jsonable_encoder(out2)
    assert out_json
