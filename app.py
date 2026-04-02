import os, re, hashlib, datetime as dt, json
from typing import List, Optional
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import feedparser, httpx
from bs4 import BeautifulSoup

from sqlalchemy import create_engine, String, Float, DateTime, Text, Boolean
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Mapped, mapped_column

import asyncio
import logging
from google import genai
from google.genai import types

# ---------- Config ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash-001")
ADMIN_TOKEN       = os.getenv("ADMIN_TOKEN", "")
DB_URL            = os.getenv("DB_URL", "sqlite:///./news.db")
FRONTEND_URL      = os.getenv("FRONTEND_URL", "")

FEEDS = {
    "NRK":          "https://www.nrk.no/toppsaker.rss",
    "VG":           "https://www.vg.no/rss/feed/?limit=50",
    "Dagbladet":    "https://www.dagbladet.no/rss/",
    "Aftenposten":  "https://www.aftenposten.no/rss",
    "Document":     "https://www.document.no/feed/",
    "Resett":       "https://resett.no/feed/",
    "FilterNyheter":"https://filternyheter.no/feed/",
    "Subjekt":      "https://subjekt.no/feed/",
}

# ---------- DB ----------
class Base(DeclarativeBase): ...
engine = create_engine(DB_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

class Article(Base):
    __tablename__ = "articles"
    id:           Mapped[str]                  = mapped_column(String(64), primary_key=True)
    outlet:       Mapped[str]                  = mapped_column(String(64))
    url:          Mapped[str]                  = mapped_column(String(1024), unique=True)
    title:        Mapped[str]                  = mapped_column(String(1024))
    summary:      Mapped[str]                  = mapped_column(Text)
    published_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    bias:         Mapped[Optional[float]]       = mapped_column(Float, nullable=True)
    reliability:  Mapped[Optional[float]]       = mapped_column(Float, nullable=True)
    reason:       Mapped[Optional[str]]         = mapped_column(Text, nullable=True)
    topic:        Mapped[Optional[str]]         = mapped_column(String(128), nullable=True)
    faktisk_flag: Mapped[bool]                  = mapped_column(Boolean, default=False)
    created_at:   Mapped[dt.datetime]           = mapped_column(DateTime, default=dt.datetime.utcnow)

Base.metadata.create_all(engine)

# Add topic column to existing databases (safe no-op if already exists)
from sqlalchemy import text as _sql_text
try:
    with engine.begin() as _conn:
        _conn.execute(_sql_text("ALTER TABLE articles ADD COLUMN topic VARCHAR(128)"))
except Exception:
    pass  # column already exists

# ---------- Helpers ----------
def clean_html(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for t in soup(["script", "style", "noscript"]): t.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def parse_date(e):
    try:
        pp = e.get("published_parsed")
        if pp: return dt.datetime(*pp[:6])
    except Exception:
        pass
    return None

def safe_float(value, default: float) -> float:
    try:
        if value is None: return default
        if isinstance(value, str): value = value.strip().replace(",", ".")
        return float(value)
    except Exception:
        return default

# ---------- Lexicon fallback ----------
LEFT  = set("velferd likhet klima rettferdighet fagforening offentlig formuesskatt".split())
RIGHT = set("privatisering skattekutt næringsliv forsvar grensekontroll olje gass".split())

def lexicon_bias(text: str) -> float:
    words = re.findall(r"\w+", text.lower())
    l = sum(w in LEFT for w in words)
    r = sum(w in RIGHT for w in words)
    if l + r == 0: return 0.0
    return max(-1.0, min(1.0, (r - l) / (l + r)))

# ---------- Gemini Scoring ----------
_gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY.strip() else None

def _gemini_score_sync(title: str, summary: str) -> dict:
    prompt = f"""Du er en ekspert norsk medieanalytiker. Analyser artikkelen langs tre dimensjoner:

1. INKLUDERING (30%): Hvem siteres? Hvilke perspektiver og fakta fremheves?
2. EKSKLUDERING (40% — viktigst): Hva mangler? Hvilken kontekst er utelatt? Hvilke motargumenter mangler? Bias vises oftest i det som IKKE er der.
3. FRAMING (30%): Brukes ladede ord? ("kutt" vs "effektivisering", "flyktninger" vs "innvandrere"). Hvilken vinkling har overskriften?

Norsk politisk skala:
- Venstre (-1): SV/Rødt/MDG-perspektiv — omfordeling, velferd, fagforeninger, klima, offentlig sektor, kritisk til næringsliv og olje
- Senter (0): Ap/SP/Venstre-perspektiv — balansert, saklig, ingen tydelig slagside
- Høyre (+1): Høyre/FrP-perspektiv — skattekutt, privatisering, næringsliv, individuell frihet, innvandringskritikk, forsvar

Typisk medieprofil:
- NRK, Aftenposten, Filter Nyheter: Senter/senter-venstre
- VG, Dagbladet: Senter med tabloidvinkling
- Subjekt: Senter/senter-høyre
- Document, Resett: Høyre/nasjonalistisk

Tittel: {title}
Ingress: {summary}

Returner KUN gyldig JSON:
- bias: -1 til +1
- reliability: 0 til 1
- reason: 1-2 setninger på norsk. Forklar konkret hva som inkluderes, hva som mangler, og hvilken framing som brukes — ikke bare "artikkelen er nøytral".
- topic: 1-3 ord på norsk, bindestreker, som beskriver HVILKEN SAK artikkelen handler om (ikke tema, men konkret sak). Bruk samme topic for artikler om samme hendelse. Eksempler: "trump-toll", "iran-atomvåpen", "norsk-skolereform", "israel-gaza", "statsbudsjettet-2025", "støre-stortingsvalg". Skriv null (JSON null) for upolitiske saker som sport, underholdning, ulykker.

Bruk punktum som desimaltegn. Ingen annen tekst enn JSON."""

    resp = _gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    text = (resp.text or "").strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m: text = m.group(0)
    data = json.loads(text)
    data["bias"]        = max(-1.0, min(1.0, safe_float(data.get("bias"), 0.0)))
    data["reliability"] = max(0.0,  min(1.0, safe_float(data.get("reliability"), 0.5)))
    data["reason"]      = str(data.get("reason") or "").strip()
    raw_topic = data.get("topic")
    if raw_topic and isinstance(raw_topic, str):
        # normalize: lowercase, strip, replace spaces with hyphens
        data["topic"] = re.sub(r"[^a-zæøå0-9\-]", "", raw_topic.lower().strip().replace(" ", "-"))
    else:
        data["topic"] = None
    return data

async def score_article(title: str, summary: str) -> dict:
    if not GEMINI_API_KEY or _gemini_client is None:
        return {
            "bias":        lexicon_bias(f"{title}. {summary}"),
            "reliability": 0.5,
            "reason":      "Leksikon (ingen GEMINI_API_KEY).",
        }
    try:
        return await asyncio.to_thread(_gemini_score_sync, title, summary)
    except Exception as e:
        logging.warning(f"[GEMINI ERROR] {e}")
        return {
            "bias":        lexicon_bias(f"{title}. {summary}"),
            "reliability": 0.5,
            "reason":      "Leksikon (Gemini-feil).",
        }

# ---------- Story grouping ----------
STOP_WORDS = set(
    "og i er av på til med for fra som om den det de en et har ikke kan vil "
    "skal etter men også at ved sin sitt sine seg selv alle inn ut over under".split()
)

def title_keywords(title: str) -> set:
    words = re.findall(r"[a-zæøåA-ZÆØÅ]{5,}", title)
    return {w.lower() for w in words if w.lower() not in STOP_WORDS}

def topics_overlap(t1: Optional[str], t2: Optional[str]) -> bool:
    """True if topics share at least one word (e.g. 'trump-toll' and 'trump-handel')."""
    if not t1 or not t2:
        return False
    return bool(set(t1.split("-")) & set(t2.split("-")))

def group_into_stories(articles: list) -> list:
    n = len(articles)
    if n == 0: return []

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    keywords = [title_keywords(a.title) for a in articles]
    dates    = [a.published_at or dt.datetime.utcnow() for a in articles]
    topics   = [getattr(a, "topic", None) or "" for a in articles]

    for i in range(n):
        for j in range(i + 1, n):
            if abs((dates[i] - dates[j]).total_seconds()) > 7 * 86400:
                continue
            ti, tj = topics[i], topics[j]
            # 1. Exact topic match → always group
            if ti and tj and ti == tj:
                union(i, j)
                continue
            # 2. Overlapping topic words + at least 1 keyword match → group
            shared_kw = keywords[i] & keywords[j]
            if topics_overlap(ti, tj) and shared_kw:
                union(i, j)
                continue
            # 3. Pure keyword fallback (no topic info or no overlap)
            if len(shared_kw) >= 2 or any(len(w) >= 8 for w in shared_kw):
                union(i, j)

    groups: dict[int, list] = defaultdict(list)
    for i, a in enumerate(articles):
        groups[find(i)].append(a)

    stories = []
    for group_articles in groups.values():
        outlets = {a.outlet for a in group_articles}

        all_outlets = set(FEEDS.keys())
        # Blindsone if only 1 outlet covers the story, or fewer than half cover it
        blindspot = len(outlets) < len(all_outlets) / 2

        # Count unique outlets per bias category (not articles)
        # Pick the most extreme article per outlet to represent that outlet
        outlet_biases = {}
        for a in group_articles:
            existing = outlet_biases.get(a.outlet)
            if existing is None or abs(a.bias or 0.0) > abs(existing):
                outlet_biases[a.outlet] = a.bias or 0.0

        n = len(outlet_biases)
        left_count   = sum(1 for b in outlet_biases.values() if b < -0.33)
        right_count  = sum(1 for b in outlet_biases.values() if b > 0.33)
        center_count = n - left_count - right_count
        avg_left   = round(left_count / n, 2)
        avg_right  = round(right_count / n, 2)
        avg_center = round(center_count / n, 2)

        best = max(group_articles, key=lambda a: a.reliability or 0.0)
        pub  = max((a.published_at for a in group_articles if a.published_at), default=dt.datetime.utcnow())

        stories.append({
            "id":               sha1(best.title + str(len(group_articles))),
            "title":            best.title,
            "description":      (best.summary or "")[:250],
            "articles":         group_articles,
            "bias_distribution": {"left": avg_left, "center": avg_center, "right": avg_right},
            "blindspot":        blindspot,
            "source_count":     len(outlets),
            "published_at":     pub.isoformat(),
        })

    return sorted(stories, key=lambda s: s["published_at"], reverse=True)

# ---------- Ingestion ----------
async def fetch_articles() -> list[dict]:
    out = []
    headers = {"User-Agent": "Mozilla/5.0"}
    async with httpx.AsyncClient(timeout=20, headers=headers, follow_redirects=True) as client:
        for outlet, url in FEEDS.items():
            try:
                r = await client.get(url)
                r.raise_for_status()
                feed = feedparser.parse(r.text)
                for e in feed.entries:
                    link = (e.get("link") or "").strip()
                    if not link: continue
                    out.append({
                        "id":           sha1(f"{outlet}|{link}"),
                        "outlet":       outlet,
                        "url":          link,
                        "title":        (e.get("title") or "").strip(),
                        "summary":      clean_html(e.get("summary") or ""),
                        "published_at": parse_date(e) or dt.datetime.utcnow(),
                    })
            except Exception as ex:
                logging.warning(f"[INGEST][{outlet}] {ex}")
    seen, dedup = set(), []
    for r in out:
        if r["url"] not in seen:
            seen.add(r["url"]); dedup.append(r)
    return dedup

async def ingest_once(limit_per_outlet: int = 15):
    """Ingest up to limit_per_outlet new articles per outlet.
    Run multiple times to gradually fill the database across all outlets."""
    rows = await fetch_articles()
    db = SessionLocal()
    new_count = 0
    outlet_counts: dict[str, int] = defaultdict(int)
    try:
        for r in rows:
            if outlet_counts[r["outlet"]] >= limit_per_outlet:
                continue
            if db.get(Article, r["id"]): continue
            scores = await score_article(r["title"], r["summary"])
            db.add(Article(
                id=r["id"], outlet=r["outlet"], url=r["url"],
                title=r["title"], summary=r["summary"],
                published_at=r["published_at"],
                bias=scores["bias"], reliability=scores["reliability"],
                reason=scores.get("reason", ""),
                topic=scores.get("topic"),
                faktisk_flag=False,
            ))
            db.commit()
            new_count += 1
            outlet_counts[r["outlet"]] += 1
            await asyncio.sleep(2)
        return new_count
    finally:
        db.close()

# ---------- API ----------
app = FastAPI(title="Politisk Korrekt API")

allowed_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://politisk-korrekt.vercel.app",
    "https://politisk-korrekt-git-main-danielbaxtrs-projects.vercel.app",
    "https://politisk-korrekt-eyqxpfrn2-danielbaxtrs-projects.vercel.app",
]
if FRONTEND_URL:
    allowed_origins.append(FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Schemas ----------
class ArticleOut(BaseModel):
    id:           str
    outlet:       str
    url:          str
    title:        str
    summary:      str
    published_at: dt.datetime
    bias:         float
    reliability:  float
    reason:       str
    topic:        Optional[str] = None

class StoryOut(BaseModel):
    id:               str
    title:            str
    description:      str
    articles:         List[ArticleOut]
    bias_distribution: dict
    blindspot:        bool
    source_count:     int
    published_at:     str

def require_admin(x_admin_token: str = Header(None)):
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ---------- Endpoints ----------
@app.get("/api/articles", response_model=List[ArticleOut])
def list_articles(limit: int = 50, outlet: Optional[str] = None):
    db = SessionLocal()
    try:
        q = db.query(Article).order_by(Article.published_at.desc())
        if outlet: q = q.filter(Article.outlet == outlet)
        return [ArticleOut(
            id=r.id, outlet=r.outlet, url=r.url, title=r.title,
            summary=r.summary, published_at=r.published_at,
            bias=r.bias or 0.0, reliability=r.reliability or 0.5,
            reason=r.reason or "", topic=r.topic,
        ) for r in q.limit(limit).all()]
    finally:
        db.close()

@app.get("/api/stories", response_model=List[StoryOut])
def list_stories(limit: int = 20):
    db = SessionLocal()
    try:
        articles = db.query(Article).order_by(Article.published_at.desc()).limit(500).all()
        stories  = group_into_stories(articles)
        return stories[:limit]
    finally:
        db.close()

@app.get("/api/stories/{story_id}")
def get_story(story_id: str):
    db = SessionLocal()
    try:
        articles = db.query(Article).order_by(Article.published_at.desc()).limit(500).all()
        stories  = group_into_stories(articles)
        story    = next((s for s in stories if s["id"] == story_id), None)
        if not story:
            raise HTTPException(status_code=404, detail="Story not found")
        return story
    finally:
        db.close()

@app.post("/jobs/ingest")
async def run_ingest(limit_per_outlet: int = 15, _: None = Depends(require_admin)):
    n = await ingest_once(limit_per_outlet=limit_per_outlet)
    return {"ingested": n}

@app.post("/jobs/retopic")
async def run_retopic(_: None = Depends(require_admin)):
    """Back-fill topic field on existing articles that have none."""
    db = SessionLocal()
    try:
        rows = db.query(Article).filter(Article.topic == None).all()  # noqa: E711
        updated = 0
        for article in rows:
            scores = await score_article(article.title, article.summary or "")
            if scores.get("topic") is not None:
                article.topic = scores["topic"]
                db.commit()
                updated += 1
            await asyncio.sleep(2)
        return {"updated": updated, "total_without_topic": len(rows)}
    finally:
        db.close()

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- Dev scheduler ----------
if os.getenv("ENV", "dev") == "dev":
    from apscheduler.schedulers.background import BackgroundScheduler
    sched = BackgroundScheduler()
    sched.add_job(lambda: asyncio.run(ingest_once()), "interval", hours=12)
    sched.start()
