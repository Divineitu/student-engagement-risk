"""
Flood test for the /predict endpoint. Point this at the nginx port in front
of the api containers (see docker-compose.yml), not directly at a single
container, otherwise scaling doesn't do anything.

    locust -f locust/locustfile.py --host http://localhost:8000
"""
import os
import random

from locust import HttpUser, task, between

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "test")
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def _pick_sample_images(n_per_class=5):
    paths = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(SAMPLE_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        files = os.listdir(class_dir)[:n_per_class]
        paths.extend(os.path.join(class_dir, f) for f in files)
    return paths


SAMPLE_IMAGES = _pick_sample_images()


class PredictUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        path = random.choice(SAMPLE_IMAGES)
        with open(path, "rb") as f:
            self.client.post("/predict", files={"file": (os.path.basename(path), f, "image/png")})
