# sentiment_service (Day 1–2 scaffold)

## Create & run (local)

```bash
cd sentiment_service
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs and try `/predict` with a JSON body like:
```json
{"text": "I love this product!"}
```

## Configure backends

By default, backend is `custom`. You can switch to a Hugging Face model with env vars:

```bash
export MODEL_BACKEND=transformers
export TRANSFORMERS_MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
uvicorn app.main:app --reload
```

For your own model:

```
export MODEL_BACKEND=custom
export MODEL_PATH=artifacts/model.pt
# export TOKENIZER_PATH=artifacts/vectorizer.joblib   # if you have one
uvicorn app.main:app --reload
```

Replace the stub in `app/model.py::_predict_custom` with your preprocessing + forward pass.
