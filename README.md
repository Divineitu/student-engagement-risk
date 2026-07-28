# Student Engagement Risk Detector

MLOps summative - end to end ML pipeline, deployed and load tested.

## Video demo

[YouTube link - TODO]

Shows: a prediction on a single uploaded face image, a bulk upload of new images, and
triggering retraining from the UI.

## Live app

- UI: `[Render URL - TODO]`
- API docs (swagger): `[Render URL - TODO]/docs`

## Why this project

This is a follow-up to my earlier Introduction to ML summative, which was a tabular
classifier predicting student dropout (Dropout / Enrolled / Graduate) from a Portuguese
university dataset. Logistic regression won there (0.79 accuracy, 0.72 macro F1), and the
project's own conclusion was that the strongest predictors - semester grades - only show up
once the semester is basically over. That's too late to act on.

This assignment requires non-tabular data, so I used it to chase that exact gap: is there a
signal available on day one, before any grades exist? I built a CNN that reads facial
affect (angry / disgust / fear / happy / neutral / sad / surprise) from a face crop, using
FER-2013, as a stand-in for a real-time behavioral early-warning signal - a disengaged or
stressed-looking student is a candidate for follow-up long before a transcript would show
anything. See `notebook/student_engagement_risk.ipynb` for the actual modelling and
evaluation.

## Repo structure

```
student-engagement-risk/
├── notebook/student_engagement_risk.ipynb   preprocessing, 3 experiments, evaluation
├── src/
│   ├── download_data.py     pulls FER-2013 and lays it out as data/train, data/test
│   ├── preprocessing.py     shared preprocessing (notebook, API, retrain all use this)
│   ├── model.py             CNN architectures used in the notebook's experiments
│   ├── prediction.py        loads the saved model, predicts a single image
│   ├── database.py          sqlite: uploads, retrain runs, request latency log
│   └── retrain.py           fine-tunes the saved model on newly uploaded images
├── api/main.py               FastAPI - /predict /upload /retrain /stats /health
├── ui/app.py                 Streamlit - Predict / Visualizations / Upload & Retrain / Status
├── data/train, data/test     FER-2013, folder-per-class
├── models/                   engagement_model.keras + label_map.json
├── locust/                   flood test
├── reports/                  flood test results + notebook plots
├── Dockerfile.api, Dockerfile.ui, docker-compose.yml
└── requirements*.txt
```

## Setup

### 1. Clone and install

```
git clone [repo url]
cd student-engagement-risk
python -m venv .venv
.venv/Scripts/activate        # source .venv/bin/activate on mac/linux
pip install -r requirements.txt
```

### 2. Get the data (already included in this repo under data/, skip if present)

```
python src/download_data.py
```

Pulls FER-2013 from the HF mirror `Aaryan333/fer2013_train_publicTest_privateTest` and
writes `data/train/<class>/*.png` and `data/test/<class>/*.png`.

### 3. Run the notebook

```
jupyter notebook notebook/student_engagement_risk.ipynb
```

Runs top to bottom: EDA, three model experiments, evaluation on the sealed test set, and
saves `models/engagement_model.keras` + `models/label_map.json`.

### 4. Run the API

```
uvicorn api.main:app --reload
```

- `GET /health` - uptime, last retrain info
- `POST /predict` - form-data `file`, returns predicted label + confidence + full probabilities
- `POST /upload` - form-data `label` + `files` (multiple), saves images for later retraining
- `POST /retrain` - fine-tunes the current model on whatever's been uploaded since the last run
- `GET /stats` - class counts + recent request latency, used by the UI

### 5. Run the UI

```
streamlit run ui/app.py
```

Set `API_URL` if the API isn't on `localhost:8000`.

### 6. Run with Docker

```
docker compose up --build
```

Brings up `api` (FastAPI), `ui` (Streamlit), and `nginx` (fronting the api service so it
can be scaled). UI on `localhost:8501`, API through nginx on `localhost:8000`.

## Retraining flow

1. **Upload** - the Upload & Retrain tab in the UI sends new labeled images to `POST
   /upload`, which saves them under `data/uploads/<label>/` and logs each one in
   `data/app.db` (`uploads` table).
2. **Preprocessing** - `retrain.py` reads the same-shaped grayscale/48x48/normalized
   pipeline from `preprocessing.py` that the notebook and the API's prediction path use, so
   the new images go through identical preprocessing to everything else.
3. **Retraining** - loads `models/engagement_model.keras` (the model trained in the
   notebook, not a fresh one), fine-tunes it for a few epochs at a low learning rate on the
   new uploads plus a resampled slice of the original train set, evaluates macro F1 on the
   sealed test set before and after, and saves a new versioned model file.

Trigger it from the UI's "Retrain now" button, or directly: `curl -X POST
localhost:8000/retrain`.

## Flood request simulation (Locust)

```
docker compose up --build --scale api=1
locust -f locust/locustfile.py --host http://localhost:8000 --headless -u 50 -r 10 -t 2m --csv reports/flood_1container
```

Repeated for `--scale api=2` and `--scale api=4`, results below.

### Results

| API containers | RPS | Median latency (ms) | p95 latency (ms) | Failures |
|---|---|---|---|---|
| 1 | 8.91 | 4800 | 6200 | 3 / 1042 |
| 2 | 13.13 | 3100 | 6600 | 0 / 1578 |
| 4 | 13.59 | 2700 | 8600 | 0 / 1629 |

Scaling from 1 to 2 containers is the clear win here - throughput up ~47%, median latency down
a third, failures gone. 2 to 4 barely helps RPS and actually makes p95 worse, because this all
ran on one laptop and 4 TensorFlow processes start fighting each other for CPU cores. See
`reports/flood_test_results.md` for the full breakdown of why.

See `reports/flood_test_results.md` for the full writeup and chart.

## Deployment (Render)

1. Push this repo to GitHub.
2. Create a new Render Web Service from the repo, Docker runtime, Dockerfile path
   `Dockerfile.api`, and a second Web Service for `Dockerfile.ui` with an `API_URL`
   environment variable pointing at the first service's Render URL.
3. Both services need at least a "Starter" instance (TensorFlow's memory footprint doesn't
   fit the free tier).

## Model evaluation summary

See `notebook/student_engagement_risk.ipynb` section 5 for the full breakdown (accuracy,
macro precision/recall/F1, confusion matrix, per-class ROC-AUC). Headline numbers:

- Accuracy: 0.558
- Macro F1: 0.522
- Best performing classes: happy (F1 0.78), surprise (F1 0.69)
- Weakest class: fear (F1 0.31, confused mostly with sad/angry/surprise) - same shape of
  problem as the Enrolled class in the tabular project, see the notebook's discussion section

## Limitations

FER-2013 is posed/labeled expression data, not real classroom footage, so this is a proof
of concept for the pipeline and the "early signal" argument, not a validated intervention
tool. A real deployment would need footage from an actual classroom setting and outcome
labels (did the student actually drop out) rather than expression labels, plus a serious
conversation about consent and surveillance before pointing a camera at students at all.
