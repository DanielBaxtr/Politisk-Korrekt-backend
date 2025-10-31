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

# ---------- Config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
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
    reliability: Mapped[Optional[float]]   = mapped_column(Float, nullable=True)       # 0..1
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

def heuristic_reliability(title: str, summary: str) -> float:
    base = 0.75
    low_markers = ["kommentar","kronikk","debatt","mening"]
    if any(k in (title.lower()+" "+summary.lower()) for k in low_markers):
        base = 0.45
    sens = sum(w in SENSATIONAL for w in re.findall(r"\w+", (title+" "+summary).lower()))
    if sens: base -= 0.1
    return max(0.0, min(1.0, base))

# ---------- GPT Scoring ----------
GPT_URL = "https://api.openai.com/v1/chat/completions"

async def gpt_score(title: str, summary: str) -> dict:
    if not OPENAI_API_KEY:
        return {
            "bias": lexicon_bias(f"{title}. {summary}"),
            "reliability": heuristic_reliability(title, summary),
            "reason": "Lexicon fallback (no OPENAI_API_KEY)."
        }

    prompt = f"""
Du er en nøytral norsk medieanalytiker. Vurder artikkelen:
Tittel: {title}
Ingress/utdrag: {summary}

Gi meg et JSON-objekt med:
- bias: tall mellom -1 (venstre) og +1 (høyre)
- reliability: tall mellom 0 og 1
- reason: 1–2 setninger (kort)

Svar kun med JSON.
""".strip()

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"}
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role":"system","content":"You output valid JSON only."},
            {"role":"user","content": prompt}
        ],
        "temperature": 0.2
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(GPT_URL, headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
        txt = data["choices"][0]["message"]["content"]

    try:
        return json.loads(txt)
    except Exception:
        return {
            "bias": lexicon_bias(f"{title}. {summary}"),
            "reliability": heuristic_reliability(title, summary),
            "reason": "Fallback to lexicon (JSON parse)."
        }

# ---------- Ingestion ----------
async def fetch_articles() -> list[dict]:
    out = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    }
    async with httpx.AsyncClient(timeout=20, headers=headers, follow_redirects=True) as client:
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
            scores = await gpt_score(r["title"], r["summary"])
            art = Article(
                id=pk,
                outlet=r["outlet"],
                url=r["url"],
                title=r["title"],
                summary=r["summary"],
                published_at=r["published_at"],
                bias=float(scores.get("bias", 0)),
                reliability=float(scores.get("reliability", 0.6)),
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
        "https://editor.wix.com",
        "https://www.wix.com",
        "https://*.wixsite.com",
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
    reliability: float
    reason: str
    faktisk_flag: bool | None = False

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
            bias=r.bias or 0.0, reliability=r.reliability or 0.6,
            reason=r.reason or "", faktisch_flag=r.faktisk_flag
        ) for r in rows]
    finally:
        db.close()

@app.post("/jobs/ingest")
async def run_ingest(admin_ok: bool = Depends(require_admin)):
    n = await ingest_once()
    return {"ingested": n}

# ---------- Dev scheduler (optional) ----------
if os.getenv("ENV","dev") == "dev":
    from apscheduler.schedulers.background import BackgroundScheduler
    import asyncio
    sched = BackgroundScheduler()
    def job(): asyncio.run(ingest_once())
    sched.add_job(job, "interval", hours=1, next_run_time=dt.datetime.utcnow())
    sched.start()