import os, re, hashlib, datetime as dt, json
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, Header
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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash-001")
ADMIN_TOKEN     = os.getenv("ADMIN_TOKEN", "")
DB_URL = os.getenv("DB_URL", "sqlite:///./news.db")

FEEDS = {
    "NRK": "https://www.nrk.no/toppsaker.rss",
    "VG": "https://www.vg.no/rss/feed/?limit=50",
    "Dagbladet": "https://www.dagbladet.no/rss/",
    "Nettavisen": "https://www.nettavisen.no/rss",
    "Document": "https://www.document.no/feed/",
    "Resett": "https://resett.no/feed/",
    "FilterNyheter": "https://filternyheter.no/feed/",
    "Subjekt": "https://subjekt.no/feed/",
}

# ---------- DB ----------
class Base(DeclarativeBase): ...
engine = create_engine(DB_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

class Article(Base):
    __tablename__ = "articles"
    id:          Mapped[str]               = mapped_column(String(64), primary_key=True)
    outlet:      Mapped[str]               = mapped_column(String(64))
    url:         Mapped[str]               = mapped_column(String(1024), unique=True)
    title:       Mapped[str]               = mapped_column(String(1024))
    summary:     Mapped[str]               = mapped_column(Text)
    published_at:Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    bias:        Mapped[Optional[float]]   = mapped_column(Float, nullable=True)       # -1..+1
    reason:      Mapped[Optional[str]]     = mapped_column(Text, nullable=True)
    faktisk_flag:Mapped[bool]              = mapped_column(Boolean, default=False)
    created_at:  Mapped[dt.datetime]       = mapped_column(DateTime, default=dt.datetime.utcnow)

Base.metadata.create_all(engine)

# ---------- Helpers ----------
def clean_html(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for t in soup(["script","style","noscript"]): t.decompose()
    text = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def parse_date(e):
    try:
        pp = e.get("published_parsed")
        if pp: return dt.datetime(*pp[:6])
    except Exception:
        pass
    return None

# ---------- v0 Lexicon fallback ----------
LEFT  = set("velferd likhet klima rettferdighet fagforening offentlig formuesskatt".split())
RIGHT = set("privatisering skattekutt næringsliv forsvar grensekontroll olje gass".split())
SENSATIONAL = set("sjokk skandale avslørt raser knuser".split())

def lexicon_bias(text: str) -> float:
    words = re.findall(r"\w+", text.lower())
    l = sum(w in LEFT for w in words)
    r = sum(w in RIGHT for w in words)
    if l + r == 0: return 0.0
    raw = (r - l) / (l + r)
    return max(-1.0, min(1.0, raw))

def safe_float(value, default: float) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.strip().replace(",", ".")
        return float(value)
    except Exception:
        return default

# ---------- Gemini Scoring ----------
# Uses Google AI Studio / Gemini API key from env: GEMINI_API_KEY
# Model is controlled by GEMINI_MODEL (default: models/gemini-2.0-flash-001)

_gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY.strip() else None

def _gemini_score_sync(title: str, summary: str) -> dict:
    """Synchronous Gemini call (wrapped by an async function below)."""
    prompt = f"""
Du er en nøytral norsk medieanalytiker. Vurder artikkelen basert på tittel og ingress/utdrag.

Tittel: {title}
Ingress/utdrag: {summary}
Returner KUN gyldig JSON med feltene:
- bias: tall mellom -1 (klart venstre) og +1 (klart høyre), 0 = nøytral
- reason: 1–2 korte setninger som forklarer vurderingen

Bruk punktum som desimaltegn i JSON (f.eks. 0.25, ikke 0,25).
Ingen annen tekst enn JSON.
""".strip()

    print("[GEMINI MODEL]", GEMINI_MODEL)
    resp = _gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    print("[GEMINI RAW]", (resp.text or "")[:500])

    text = (resp.text or "").strip()

    # Most of the time response_mime_type makes this valid JSON.
    # Fallback: if any extra text sneaks in, extract the first JSON object.
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        text = m.group(0)

    data = json.loads(text)

    data["bias"] = safe_float(data.get("bias"), 0.0)
    data["reason"] = str(data.get("reason") or "").strip()
    if data["reason"]:
        data["reason"] = "Gemini: " + data["reason"]
    else:
        data["reason"] = "Gemini: (no reason returned)"
    return data


async def score_article(title: str, summary: str) -> dict:
    """Return {bias, reason}. Falls back to lexicon if Gemini key missing or parsing fails."""
    if not GEMINI_API_KEY or _gemini_client is None:
        return {
            "bias": lexicon_bias(f"{title}. {summary}"),
            "reason": "Lexicon (no GEMINI_API_KEY).",
        }

    try:
        # Run the blocking SDK call in a background thread
        return await asyncio.to_thread(_gemini_score_sync, title, summary)
    except Exception as e:
        print("[GEMINI ERROR]", repr(e))
        # Optional: also print raw response text by catching inside _gemini_score_sync (see below)
        return {
            "bias": lexicon_bias(f"{title}. {summary}"),
            "reason": "Lexicon (Gemini error or invalid JSON).",
        }

# ---------- Ingestion ----------
async def fetch_articles() -> list[dict]:
    out = []
    req_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    }
    async with httpx.AsyncClient(timeout=20, headers=req_headers, follow_redirects=True) as client:
        for outlet, url in FEEDS.items():
            try:
                r = await client.get(url)
                r.raise_for_status()
                feed = feedparser.parse(r.text)  # parse the XML we fetched
                for e in feed.entries:
                    link = (e.get("link") or "").strip()
                    if not link:
                        continue
                    title = (e.get("title") or "").strip()
                    summary = clean_html(e.get("summary") or "")
                    pid = sha1(f"{outlet}|{link}")
                    out.append({
                        "id": pid,
                        "outlet": outlet,
                        "url": link,
                        "title": title,
                        "summary": summary,
                        "published_at": parse_date(e) or dt.datetime.utcnow()
                    })
            except Exception as ex:
                print(f"[INGEST][{outlet}] {ex}")
    # dedupe
    seen=set(); dedup=[]
    for r in out:
        if r["url"] in seen: continue
        seen.add(r["url"]); dedup.append(r)
    print(f"[INGEST] collected {len(dedup)} items from {len(FEEDS)} feeds")
    return dedup

async def ingest_once():
    rows = await fetch_articles()
    db = SessionLocal()
    new_count = 0
    try:
        for r in rows:
            pk = sha1(f"{r['outlet']}|{r['url']}")
            if db.get(Article, pk): 
                continue
            scores = await score_article(r["title"], r["summary"])
            art = Article(
                id=pk,
                outlet=r["outlet"],
                url=r["url"],
                title=r["title"],
                summary=r["summary"],
                published_at=r["published_at"],
                bias=safe_float(scores.get("bias"), 0.0),
                reason=scores.get("reason",""),
                faktisk_flag=False
            )
            db.add(art)
            new_count += 1
        db.commit()
        return new_count
    finally:
        db.close()

# ---------- API ----------
app = FastAPI(title="Politisk Korrekt API")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://editor.wix.com",  #Wix editor
        "https://www.wix.com",     #Wix preview domains
        "https://*.wixsite.com",    #my free wix domain
        "https://yourdomain.no",  # replace this with your actual Wix domain when you have one
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ArticleOut(BaseModel):
    id: str
    outlet: str
    url: str
    title: str
    summary: str
    published_at: dt.datetime
    bias: float
    reason: str
    faktisk_flag: bool | None = False

class AnalyzeIn(BaseModel):
    title: str
    summary: str = ""
    url: str | None = None
    outlet: str | None = None

class AnalyzeOut(BaseModel):
    title: str
    summary: str
    url: str | None = None
    outlet: str | None = None
    bias: float
    reason: str

def require_admin(x_admin_token: str = Header(None)):
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

@app.get("/api/articles", response_model=List[ArticleOut])
def list_articles(limit: int = 30, outlet: Optional[str] = None, order: str = "desc"):
    db = SessionLocal()
    try:
        q = db.query(Article)
        if outlet:
            q = q.filter(Article.outlet == outlet)
        q = q.order_by(Article.published_at.asc() if order=="asc" else Article.published_at.desc())
        rows = q.limit(limit).all()
        return [ArticleOut(
            id=r.id, outlet=r.outlet, url=r.url, title=r.title,
            summary=r.summary, published_at=r.published_at,
            bias=r.bias or 0.0,
            reason=r.reason or "", faktisk_flag=r.faktisk_flag
        ) for r in rows]
    finally:
        db.close()

@app.post("/jobs/ingest")
async def run_ingest(admin_ok: bool = Depends(require_admin)):
    n = await ingest_once()
    return {"ingested": n}

@app.post("/api/analyze", response_model=AnalyzeOut)
async def analyze_article(payload: AnalyzeIn, admin_ok: bool = Depends(require_admin)):
    scores = await score_article(payload.title, payload.summary or "")
    return AnalyzeOut(
        title=payload.title,
        summary=payload.summary or "",
        url=payload.url,
        outlet=payload.outlet,
        bias=safe_float(scores.get("bias"), 0.0),
        reason=scores.get("reason", ""),
    )

# ---------- Dev scheduler (optional) ----------
if os.getenv("ENV","dev") == "dev":
    from apscheduler.schedulers.background import BackgroundScheduler
    import asyncio
    sched = BackgroundScheduler()
    def job(): asyncio.run(ingest_once())
    sched.add_job(job, "interval", hours=1, next_run_time=dt.datetime.utcnow())
    sched.start()
