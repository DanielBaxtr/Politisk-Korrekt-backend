#!/usr/bin/env python3
"""
Re-score eksisterende artikler uten topic-felt.
Kjør: python retopic.py

Henter alle artikler der topic IS NULL og kaller Gemini for å tildele topic.
Oppdaterer kun topic-feltet, ikke bias/reliability/reason.
"""
import os, re, time, json, asyncio, logging
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DB_URL = os.getenv("DB_URL", "sqlite:///./news.db")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash-001")

engine = create_engine(DB_URL, echo=False, future=True)
Session = sessionmaker(bind=engine, expire_on_commit=False)

from google import genai
from google.genai import types

client = genai.Client(api_key=GEMINI_API_KEY)

TOPIC_PROMPT = """\
Du er en norsk redaktør som kategoriserer nyhetsartikler etter SAK (ikke tema).

Tittel: {title}
Ingress: {summary}

Tildel ett topic som identifiserer hvilken konkret sak artikkelen handler om.
Bruk 1-3 ord, norsk, bindestreker. Eksempler:
  "trump-toll", "iran-atomvåpen", "norsk-skolereform", "israel-gaza",
  "statsbudsjettet-2025", "støre-stortingsvalg", "nrk-finansiering",
  "ukraina-krigen", "klimameldingen", "oljefondet"

Skriv null (JSON null) for upolitiske saker (sport, underholdning, ulykker, personlige nyheter).

Returner KUN: {{"topic": "..."}} eller {{"topic": null}}"""

def get_topic(title: str, summary: str) -> str | None:
    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=TOPIC_PROMPT.format(title=title, summary=(summary or "")[:300]),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        text = (resp.text or "").strip()
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            raw = data.get("topic")
            if raw and isinstance(raw, str):
                return re.sub(r"[^a-zæøå0-9\-]", "", raw.lower().strip().replace(" ", "-"))
    except Exception as e:
        logging.warning(f"topic error: {e}")
    return None

def main():
    db = Session()
    rows = db.execute(text("SELECT id, title, summary FROM articles WHERE topic IS NULL")).fetchall()
    print(f"{len(rows)} artikler mangler topic")

    updated = 0
    for i, row in enumerate(rows):
        topic = get_topic(row.title, row.summary or "")
        db.execute(
            text("UPDATE articles SET topic = :topic WHERE id = :id"),
            {"topic": topic, "id": row.id},
        )
        db.commit()
        updated += 1
        label = topic or "(null)"
        print(f"[{i+1}/{len(rows)}] {label:30s}  {row.title[:55]}")
        time.sleep(2)

    print(f"\nFerdig. {updated} artikler oppdatert.")
    db.close()

if __name__ == "__main__":
    main()
