import json
import logging
import os
import urllib.error
import urllib.request

import azure.functions as func

# Read these from Azure environment variables / application settings —
# never hard-code the key here.
ENDPOINT = os.environ.get("LANGUAGE_ENDPOINT", "").rstrip("/")
KEY = os.environ.get("LANGUAGE_KEY", "")

API_VERSION = "2023-04-01"
TIMEOUT_SECONDS = 20
MAX_CHARS = 5000


def main(req: func.HttpRequest) -> func.HttpResponse:
    endpoint = os.environ.get("LANGUAGE_ENDPOINT", "").rstrip("/")
    key = os.environ.get("LANGUAGE_KEY", "")
    if not endpoint or not key :
        return _json_response(
            {"error": "Server is missing LANGUAGE_ENDPOINT / LANGUAGE_KEY environment variables."},
            500,
        )

    try:
        body = req.get_json()
    except ValueError:
        body = {}

    text = (body.get("text") or "").strip()
    if not text:
        return _json_response({"error": 'Request body must include non-empty "text".'}, 400)
    if len(text) > MAX_CHARS:
        return _json_response({"error": f"Text must be {MAX_CHARS} characters or fewer."}, 400)

    try:
        sentiment_doc = _call_language("SentimentAnalysis", text)["results"]["documents"][0]
        keyphrase_doc = _call_language("KeyPhraseExtraction", text)["results"]["documents"][0]
        entity_doc = _call_language("EntityRecognition", text)["results"]["documents"][0]
    except Exception:
        logging.exception("Azure AI Language call failed")
        return _json_response(
            {"error": "Azure AI Language request failed. Check key/endpoint/quota."}, 502
        )

    result = {
        "sentiment": sentiment_doc["sentiment"],  # positive | negative | neutral | mixed
        "confidenceScores": sentiment_doc["confidenceScores"],
        "keyPhrases": keyphrase_doc.get("keyPhrases", []),
        "entities": [
            {"text": e["text"], "category": e["category"]}
            for e in entity_doc.get("entities", [])
        ],
    }
    return _json_response(result, 200)


def _call_language(kind: str, text: str) -> dict:
    url = f"{ENDPOINT}/language/:analyze-text?api-version={API_VERSION}"
    payload = {
        "kind": kind,
        "parameters": {"modelVersion": "latest"},
        "analysisInput": {"documents": [{"id": "1", "language": "en", "text": text}]},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": KEY,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"{kind} failed: {exc.code} {detail}") from exc


def _json_response(payload: dict, status: int) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(payload),
        status_code=status,
        mimetype="application/json",
    )
