import requests
import re


def print_answers(r):
    text = re.sub(r"[(\[\])\']", "", r.text).split(",")
    num_answers = int(len(text) / 4)
    answers = [text[4 * i : (4 * i) + 4] for i in range(num_answers)]
    for answer in answers:
        print(
            f'The review "{answer[0]}" is {answer[1].upper()} with a probability of '
            f"{max(float(answer[3]), float(answer[2])) * 100:.4}%"
        )


if __name__ == "__main__":
    url = "https://us-central1-sensi-anal.cloudfunctions.net/senti_anal_2"
    payload = {
        "message": [
            "DTU MLOPs course is a great course",
            "The workload was a lot though",
            "I learned a lot of new things",
            "BERT is way too big",
            "Distilled BERT is amazing",
            "Scripting a transformer model is a mess",
        ]
    }
    r = requests.post(url, json=payload)
    print_answers(r)
