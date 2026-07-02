import json
import logging
import os
import time
import urllib.error
import urllib.request

import azure.functions as func

# Read these from Azure environment variables
ENDPOINT = os.environ.get("LANGUAGE_ENDPOINT", "").rstrip("/")
KEY = os.environ.get("LANGUAGE_KEY", "")

API_VERSION = "2023-04-01"
TIMEOUT_SECONDS = 25
MAX_CHARS = 5000


def main(req: func.HttpRequest) -> func.HttpResponse:
    endpoint = os.environ.get("LANGUAGE_ENDPOINT", "").rstrip("/")
    key = os.environ.get("LANGUAGE_KEY", "")
    if not endpoint or not key:
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
        # --- 1. RUN OLD SYNCHRONOUS FEATURES ---
        sentiment_doc = _call_language_sync("SentimentAnalysis", text)["results"]["documents"][0]
        keyphrase_doc = _call_language_sync("KeyPhraseExtraction", text)["results"]["documents"][0]
        entity_doc = _call_language_sync("EntityRecognition", text)["results"]["documents"][0]

        # --- 2. RUN NEW ASYNCHRONOUS FEATURES (Language, PII, Summarization) ---
        async_job_result = _run_async_language_job(text)
        items = async_job_result.get("tasks", {}).get("items", [])
        
        # Default structures for new features
        detected_language = "Unknown"
        redacted_text = text
        summary_text = "No summary generated."

        for item in items:
            kind = item.get("kind")
            results = item.get("results", {})
            documents = results.get("documents", [])
            doc = documents[0] if documents else {}

            if "LanguageDetection" in kind:
                detected_language = doc.get("detectedLanguage", {}).get("name", "Unknown")
            elif "PiiEntityRecognition" in kind:
                redacted_text = doc.get("redactedText", text)
            elif "AbstractiveSummarization" in kind:
                summaries = doc.get("summaries", [])
                if summaries:
                    summary_text = summaries[0].get("text", "")

    except Exception as e:
        logging.exception("Azure AI Language call failed")
        return _json_response(
            {"error": f"Azure AI Language request failed: {str(e)}"}, 502
        )

    # --- 3. COMBINE EVERYTHING INTO ONE RESULT ---
    result = {
        # Old features preserved perfectly
        "sentiment": sentiment_doc["sentiment"],
        "confidenceScores": sentiment_doc["confidenceScores"],
        "keyPhrases": keyphrase_doc.get("keyPhrases", []),
        "entities": [
            {"text": e["text"], "category": e["category"]}
            for e in entity_doc.get("entities", [])
        ],
        # New features fully appended
        "language": detected_language,
        "redacted_text": redacted_text,
        "summary": summary_text
    }
    
    return _json_response(result, 200)


def _call_language_sync(kind: str, text: str) -> dict:
    url = f"{ENDPOINT}/language/:analyze-text?api-version={API_VERSION}"
    payload = {
        "kind": kind,
        "parameters": {"modelVersion": "latest"},
        "analysisInput": {"documents": [{"id": "1", "language": "en", "text": text}]},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": KEY,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _run_async_language_job(text: str) -> dict:
    submit_url = f"{ENDPOINT}/language/analyze-text/jobs?api-version={API_VERSION}"
    payload = {
        "displayName": "Text Analysis Project Extended Tasks",
        "analysisInput": {
            "documents": [{"id": "1", "text": text}]
        },
        "tasks": [
            {"kind": "LanguageDetection", "taskName": "LangTask"},
            {"kind": "PiiEntityRecognition", "taskName": "PiiTask", "parameters": {"piiCategories": ["All"]}},
            {"kind": "AbstractiveSummarization", "taskName": "SummTask", "parameters": {"sentenceCount": 1}}
        ]
    }

    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": KEY,
    }

    req = urllib.request.Request(submit_url, data=data, headers=headers, method="POST")
    
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
        operation_url = resp.headers.get("operation-location")
        if not operation_url:
            raise RuntimeError("Failed to obtain job tracking status URL from Azure.")

    # Poll the tracking URL until the compilation finishes
    for _ in range(12):
        time.sleep(1.5)
        poll_req = urllib.request.Request(operation_url, headers=headers, method="GET")
        with urllib.request.urlopen(poll_req, timeout=TIMEOUT_SECONDS) as poll_resp:
            job_status = json.loads(poll_resp.read().decode("utf-8"))
            if job_status.get("status") == "succeeded":
                return job_status
            if job_status.get("status") in ["failed", "cancelled"]:
                raise RuntimeError("Azure text analysis job compilation failed.")

    raise RuntimeError("Azure text analysis job processing timed out.")


def _json_response(payload: dict, status: int) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(payload),
        status_code=status,
        mimetype="application/json",
    )