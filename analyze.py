#!/usr/bin/env python3
"""
Dyp politisk bias-analyse for norske nyhetsartikler.

Bruk: python analyze.py <url>
Eksempel: python analyze.py https://www.nrk.no/nyheter/...

Analysen bruker tre dimensjoner:
  - Inkludering  (30%): Hvem siteres? Hvilke perspektiver presenteres?
  - Ekskludering (40%): Hva mangler? (viktigst — bias vises i det som utelates)
  - Framing      (30%): Ordvalg, vinkling, implisitte antakelser

Full JSON-rapport lagres til analysis_report.json
"""

import sys
import json
import re
import os
import textwrap
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# gemini-2.0-flash supports Google Search grounding
DEEP_ANALYSIS_MODEL = "gemini-2.0-flash"


# ---------- Article fetching ----------

def fetch_article(url: str) -> dict:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; BiasAnalyzer/1.0)"}
    with httpx.Client(timeout=30, headers=headers, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "aside", "header"]):
        tag.decompose()

    # Title
    title = ""
    for sel in ["h1", "title"]:
        el = soup.find(sel)
        if el:
            title = el.get_text(strip=True)
            break

    # Publication from domain
    publication = urlparse(url).netloc.replace("www.", "")

    # Article body — try common containers first
    body = ""
    for selector in [
        "article",
        "[itemprop='articleBody']",
        ".article-body",
        ".article-content",
        ".entry-content",
        "main",
    ]:
        el = soup.select_one(selector)
        if el:
            candidate = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
            if len(candidate) > 300:
                body = candidate
                break

    if not body:
        body = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()

    # Cap at 4000 chars to stay within token limits
    body = body[:4000]

    return {"url": url, "title": title, "publication": publication, "body": body}


# ---------- Gemini deep analysis ----------

ANALYSIS_PROMPT = """\
Du er en ekspert norsk medieanalytiker med dyp forståelse av norsk politikk og journalistikk.

## ARTIKKEL TIL ANALYSE

Tittel: {title}
Kilde: {publication}
URL: {url}

Innhold:
{body}

---

## ANALYSEPROSESS — følg disse stegene nøye

### STEG 1 — FORSTÅ SAKEN
Hva handler artikkelen egentlig om? Hva er de underliggende politiske spørsmålene?

### STEG 2 — RESEARCH (bruk Google Search)
Søk opp saken og finn:
- Hva er venstresidens typiske posisjon (Rødt/SV/MDG/Ap)?
- Hva er høyresidens posisjon (Frp/Høyre/KrF/Venstre)?
- Hva sier uavhengige eksperter/forskning om temaet?
- Hvordan dekker andre norske medier samme sak? Hva inkluderer de som denne artikkelen utelater?

### STEG 3 — INCLUSION-ANALYSE (vekt 30%)
- Hvem siteres og hvem får uttale seg? (navn, tilhørighet, politisk tilhørighet)
- Hvilke perspektiver presenteres eksplisitt?
- Hvilke fakta og tall fremheves?

### STEG 4 — EXCLUSION-ANALYSE (vekt 40% — VIKTIGST)
Bias vises oftest i det som MANGLER. Vær grundig her:
- Hvilken historisk kontekst eller bakgrunn mangler?
- Hvilke relevante kilder er ikke inkludert?
- Hvilke motargumenter eller motstridende fakta er utelatt?
- Hvilke strukturelle perspektiver ignoreres (økonomiske konsekvenser, miljø, sosiale effekter)?

### STEG 5 — FRAMING-ANALYSE (vekt 30%)
- Brukes ladede ord? (f.eks. "innvandrere" vs "flyktninger", "kutt" vs "effektivisering", "krig" vs "militæroperasjon")
- Hva er vinklingen i overskriften?
- Hvilke implisitte antakelser tas for gitt uten diskusjon?
- Hva eller hvem presenteres som "normalt" eller "selvsagt"?

### STEG 6 — BIAS-SCORE
Beregn score for hvert ledd, deretter endelig score:

  final_bias = (inclusion_score × 0.30) + (exclusion_score × 0.40) + (framing_score × 0.30)

Skala:
  -1.0 til -0.7 → Sterk venstrebias
  -0.6 til -0.4 → Moderat venstrebias
  -0.3 til -0.1 → Svak venstrelean
   0.0          → Nøytral/balansert
  +0.1 til +0.3 → Svak høyrelean
  +0.4 til +0.6 → Moderat høyrebias
  +0.7 til +1.0 → Sterk høyrebias

Norsk politisk kontekst:
  Venstre: omfordeling, velferd, fagforeninger, klima, offentlig sektor, kritisk til næringsliv/olje
  Høyre: skattekutt, privatisering, næringsliv, individuell frihet, innvandringskritikk, forsvar

---

Returner KUN gyldig JSON med denne eksakte strukturen (ingen annen tekst):

{{
  "article": {{
    "url": "{url}",
    "title": "{title}",
    "publication": "{publication}"
  }},
  "topic_analysis": {{
    "main_issue": "Hva saken egentlig handler om",
    "political_landscape": "Venstresidens og høyresidens typiske posisjoner i denne saken",
    "expert_consensus": "Hva sier forskning/eksperter om temaet"
  }},
  "inclusion_analysis": {{
    "sources_quoted": [
      {{"name": "...", "affiliation": "...", "political_lean": "venstre/høyre/nøytral"}}
    ],
    "perspectives_presented": ["..."],
    "facts_highlighted": ["..."],
    "score": 0.0
  }},
  "exclusion_analysis": {{
    "missing_context": ["..."],
    "omitted_sources": ["..."],
    "absent_counterarguments": ["..."],
    "structural_omissions": ["..."],
    "score": 0.0
  }},
  "framing_analysis": {{
    "word_choice_examples": [
      {{"word": "...", "alternative": "...", "lean": "venstre/høyre/nøytral"}}
    ],
    "headline_framing": "...",
    "structural_bias": "...",
    "implicit_assumptions": ["..."],
    "score": 0.0
  }},
  "final_bias_score": 0.0,
  "confidence_level": "høy/medium/lav",
  "reasoning": "Detaljert forklaring av scoren, med referanse til konkrete funn",
  "key_findings": [
    "De viktigste observasjonene som påvirket scoren"
  ]
}}
"""


def run_deep_analysis(article: dict) -> dict:
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = ANALYSIS_PROMPT.format(
        title=article["title"],
        publication=article["publication"],
        url=article["url"],
        body=article["body"],
    )

    resp = client.models.generate_content(
        model=DEEP_ANALYSIS_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.1,
        ),
    )

    text = (resp.text or "").strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        text = m.group(0)

    return json.loads(text)


# ---------- Pretty-print report ----------

def wrap(text: str, width: int = 72, indent: str = "   ") -> str:
    return textwrap.fill(str(text), width=width, initial_indent=indent, subsequent_indent=indent)


def print_report(r: dict):
    SEP = "─" * 64

    a = r.get("article", {})
    topic = r.get("topic_analysis", {})
    incl = r.get("inclusion_analysis", {})
    excl = r.get("exclusion_analysis", {})
    fram = r.get("framing_analysis", {})
    score = r.get("final_bias_score", 0.0)
    confidence = r.get("confidence_level", "")

    if score <= -0.7:   label = "STERK VENSTREBIAS"
    elif score <= -0.4: label = "MODERAT VENSTREBIAS"
    elif score <= -0.1: label = "SVAK VENSTRELEAN"
    elif score < 0.1:   label = "NØYTRAL / BALANSERT"
    elif score < 0.4:   label = "SVAK HØYRELEAN"
    elif score < 0.7:   label = "MODERAT HØYREBIAS"
    else:               label = "STERK HØYREBIAS"

    print(f"\n{SEP}")
    print("  DYP POLITISK BIAS-ANALYSE")
    print(SEP)
    print(f"\n  {a.get('title', '')}")
    print(f"  {a.get('publication', '')}  |  {a.get('url', '')[:60]}")

    print(f"\n{SEP}")
    print("  SAKSFORSTÅELSE")
    print(SEP)
    print(wrap(topic.get("main_issue", "")))
    print(f"\n  Politisk landskap:")
    print(wrap(topic.get("political_landscape", "")))
    print(f"\n  Ekspertkonsensus:")
    print(wrap(topic.get("expert_consensus", "")))

    print(f"\n{SEP}")
    print("  BIAS-SCORE")
    print(SEP)
    print(f"\n  Endelig score:   {score:+.2f}  →  {label}")
    print(f"  Konfidensgrad:   {confidence}")
    print(f"\n  Inkludering:     {incl.get('score', 0):+.2f}  (vekt 30%)")
    print(f"  Ekskludering:    {excl.get('score', 0):+.2f}  (vekt 40%)")
    print(f"  Framing:         {fram.get('score', 0):+.2f}  (vekt 30%)")

    print(f"\n{SEP}")
    print("  VIKTIGSTE FUNN")
    print(SEP)
    for finding in r.get("key_findings", []):
        print(wrap(f"• {finding}"))

    print(f"\n  Begrunnelse:")
    print(wrap(r.get("reasoning", "")))

    print(f"\n{SEP}")
    print("  INKLUDERING  (hvem og hva er med)")
    print(SEP)
    sources = incl.get("sources_quoted", [])
    if sources:
        print("  Siterte kilder:")
        for s in sources:
            lean = s.get("political_lean", "")
            print(f"   • {s.get('name', '?')} ({s.get('affiliation', '')}) — {lean}")
    perspectives = incl.get("perspectives_presented", [])
    if perspectives:
        print("\n  Perspektiver presentert:")
        for p in perspectives:
            print(wrap(f"• {p}"))

    print(f"\n{SEP}")
    print("  EKSKLUDERING  (hva mangler — viktigst)")
    print(SEP)
    for label_key, items in [
        ("Manglende kontekst", excl.get("missing_context", [])),
        ("Utelatte kilder", excl.get("omitted_sources", [])),
        ("Fraværende motargumenter", excl.get("absent_counterarguments", [])),
        ("Strukturelle utelatelser", excl.get("structural_omissions", [])),
    ]:
        if items:
            print(f"\n  {label_key}:")
            for item in items:
                print(wrap(f"• {item}"))

    print(f"\n{SEP}")
    print("  FRAMING  (språk og vinkling)")
    print(SEP)
    words = fram.get("word_choice_examples", [])
    if words:
        print("  Ordvalg:")
        for w in words:
            alt = w.get("alternative", "")
            lean = w.get("lean", "")
            print(f"   • «{w.get('word', '')}»  →  alternativ: «{alt}»  ({lean})")
    print(f"\n  Overskriftsvinkling:")
    print(wrap(fram.get("headline_framing", "")))
    assumptions = fram.get("implicit_assumptions", [])
    if assumptions:
        print(f"\n  Implisitte antakelser:")
        for a in assumptions:
            print(wrap(f"• {a}"))

    print(f"\n{SEP}\n")


# ---------- Entry point ----------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if not GEMINI_API_KEY:
        print("Feil: GEMINI_API_KEY mangler i .env-filen.")
        sys.exit(1)

    url = sys.argv[1]

    print(f"\nHenter artikkel: {url}", file=sys.stderr)
    try:
        article = fetch_article(url)
    except Exception as e:
        print(f"Feil ved henting av artikkel: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Tittel:  {article['title'][:80]}", file=sys.stderr)
    print(f"Kilde:   {article['publication']}", file=sys.stderr)
    print(f"Tekst:   {len(article['body'])} tegn hentet", file=sys.stderr)
    print("Kjører dyp Gemini-analyse med Google Search grounding...\n", file=sys.stderr)

    try:
        report = run_deep_analysis(article)
    except Exception as e:
        print(f"Feil under analyse: {e}", file=sys.stderr)
        sys.exit(1)

    print_report(report)

    output_file = "analysis_report.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Full JSON-rapport lagret til: {output_file}\n")
