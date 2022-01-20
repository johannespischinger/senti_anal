from opensentiment.models.predict_model_pl import Prediction

model = Prediction(
    path_to_checkpoint="gs://model_senti_anal/pretrained-distilbert-2022-01-19/BERT/epoch=1-step=14.ckpt",
)


def get_sentiment(request):
    request_json = request.get_json()
    if request_json and 'message' in request_json:
        input_data = request_json["message"]
        input_dict = {"content": input_data, "label": None}
        return f"{model.predict(input_dict)}"
    else:
        return f'No data received!'



