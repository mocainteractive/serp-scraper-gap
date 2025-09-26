import math
import time
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import httpx
import pandas as pd
from bs4 import BeautifulSoup
import trafilatura
import streamlit as st

# ---- Config ----
st.set_page_config(page_title="SERP Scraper & Topic Gap Analyzer — Lite", page_icon="🔎", layout="wide")
st.title("🔎 SERP Scraper & Topic Gap Analyzer — Lite")
st.caption("SERP via Serper.dev → scraping → temi comuni & topic gap (no sklearn)")
SERPER_ENDPOINT = "https://google.serper.dev/search"
DEFAULT_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
STOPWORDS = {
    "it": set("""a ad al alla alle agli all' allo anche ancora avere avevi aveva avevano abbiamo hanno che chi cui come con col dai dal dalla dalle dallo degli dei del della dello delle e ed era eri erano essere fui fummo furono gli ha hai il in io la le li lo loro lui ma mi mia mie mio miei ne nei nel nella nelle nello no non o per però più po poi qua quale quali quanto questa queste questo questi quella quelle quello quelli se sei sia siamo siete solo sua sue suo suoi sul sulla sulle sullo tra tu un una uno vi""".split()),
    "en": set("""a an and the is are was were be been being to of in on for with as by at from this that these those it its it's into over about not no yes you your we our they their i me my he she his her them us do does did have has had can could would should may might will""".split()),
    "fr": set("""un une des le la les de du au aux et est sont était étaient être à en pour avec comme par sur dans ceci cela ces ceux il elle ils elles nous vous que qui dont ne pas""".split()),
    "de": set("""der die das ein eine und ist sind war waren sein zu von in auf für mit als bei aus über nicht ja nein du ich er sie wir ihr es den dem des""".split()),
    "es": set("""un una unos unas el la los las de del y es son fue fueron ser a en por con como para sobre no sí tu tus su sus mi mis nuestro nuestra nuestros nuestras que quien quienes""".split()),
}

@st.cache_data(show_spinner=False, ttl=3600)
def serper_search(api_key: str, query: str, gl: str, hl: str) -> Dict:
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    body = {"q": query, "gl": gl, "hl": hl}
    resp = requests.post(SERPER_ENDPOINT, headers=headers, json=body, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Serper.dev error: {resp.status_code} - {resp.text}")
    return resp.json()

def extract_main_text(html: str) -> str:
    extracted = trafilatura.extract(html, include_comments=False, include_tables=False, favor_recall=True)
    if extracted and len(extracted.strip()) > 300:
        return extracted.strip()
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def fetch_url(url: str, hl: str, timeout: int = 20) -> Tuple[str, Optional[str]]:
    headers = DEFAULT_HEADERS.copy(); headers["Accept-Language"] = f"{hl},{hl};q=0.8,en;q=0.6"
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client:
            r = client.get(url)
            if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
                return (url, None)
            text = extract_main_text(r.text)
            if len(text) < 300:
                return (url, None)
            return (url, text)
    except Exception:
        return (url, None)

def parallel_fetch(urls: List[str], hl: str, max_workers: int = 5) -> Dict[str, Optional[str]]:
    results: Dict[str, Optional[str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_url, u, hl): u for u in urls}
        for fut in as_completed(futures):
            u = futures[fut]
            try:
                url, text = fut.result(); results[url] = text
            except Exception:
                results[u] = None
    return results

def to_dataframe(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({"position": r.get("position"), "title": r.get("title"), "url": r.get("link"), "snippet": r.get("snippet", "")})
    return pd.DataFrame(rows).sort_values("position").reset_index(drop=True)

# --- simple NLP (no sklearn) ---
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]{2,}", re.UNICODE)
def tokenize(text: str, lang: str) -> List[str]:
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]
    stop = STOPWORDS.get(lang, STOPWORDS["en"])
    return [t for t in tokens if t not in stop and not t.isdigit()]

def top_terms_doc(text: str, lang: str, k: int = 20) -> List[str]:
    toks = tokenize(text, lang)
    bigrams = [f"{toks[i]} {toks[i+1]}" for i in range(len(toks)-1)]
    counts = Counter(toks) + Counter(bigrams)
    total = sum(counts.values()) or 1
    scored = [(term, cnt/total) for term, cnt in counts.items()]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [t for t, _ in scored[:k]]

def doc_freq(terms_per_doc: List[List[str]]) -> Dict[str, int]:
    df = defaultdict(int)
    for terms in terms_per_doc:
        for t in set(terms):
            df[t] += 1
    return dict(df)

def compute_common_and_gaps(terms_per_doc: List[List[str]], urls: List[str], paa_terms: List[str]):
    N = len(terms_per_doc)
    dfreq = doc_freq(terms_per_doc)
    common_terms = sorted([t for t, c in dfreq.items() if c == N])
    thr = max(2, math.ceil(0.5 * N))
    candidate_gaps = [t for t, c in dfreq.items() if thr <= c < N]
    rows = []
    for t in candidate_gaps:
        missing_idxs = [i for i, terms in enumerate(terms_per_doc) if t not in terms]
        rows.append({"topic": t, "missing_in_pages": len(missing_idxs), "missing_urls": ", ".join(urls[i] for i in missing_idxs)})
    gaps_df = pd.DataFrame(rows).sort_values(["missing_in_pages", "topic"], ascending=[False, True]).reset_index(drop=True)
    normalized_docs = " ".join([" ".join(terms) for terms in terms_per_doc]).lower()
    paa_simplified = [re.sub(r"[?¿¡!]", "", q).strip().lower() for q in paa_terms]
    paa_not_covered = [q for q in paa_simplified if q and q not in normalized_docs]
    return common_terms, gaps_df, paa_not_covered, dfreq

def normalize_paa(people_also_ask: List[Dict]) -> List[str]:
    qs, seen = [], set()
    for item in people_also_ask or []:
        q = (item.get("question") or "").strip()
        if q and q not in seen:
            seen.add(q); qs.append(q)
    return qs

# --- UI ---
with st.sidebar:
    st.subheader("Impostazioni")
    if "serper_api_key" not in st.session_state:
        st.session_state.serper_api_key = ""
    st.session_state.serper_api_key = st.text_input("Serper.dev API Key", value=st.session_state.serper_api_key, type="password",
                                                    help="Inserisci la tua chiave. Non viene salvata su file.")
    query = st.text_input("Query di ricerca", placeholder="es. miglior macchina caffè espresso casa")
    gl = st.selectbox("Country (gl)", ["it", "us", "uk", "fr", "de", "es"], index=0)
    hl = st.selectbox("Lingua UI (hl)", ["it", "en", "fr", "de", "es"], index=0)
    n_sites = st.slider("Numero di siti da analizzare (n)", 1, 20, 10)
    max_workers = st.slider("Concorrenza scraping (thread)", 2, 10, 5)
    run = st.button("Esegui analisi 🚀", type="primary")

st.markdown("---")

if run:
    api_key = st.session_state.serper_api_key.strip()
    if not api_key:
        st.error("Inserisci la **Serper.dev API Key** nella sidebar."); st.stop()
    if not query.strip():
        st.error("Inserisci una **query di ricerca** nella sidebar."); st.stop()

    with st.spinner("Chiamo l'API di Serper.dev e preparo i dati..."):
        data = serper_search(api_key, query.strip(), gl, hl)
        organic = (data.get("organic") or [])[: max(n_sites, 1)]
        paa = normalize_paa(data.get("peopleAlsoAsk") or [])
        df_serp = to_dataframe(organic)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"Primi {len(df_serp)} risultati organici")
        st.dataframe(df_serp, use_container_width=True, hide_index=True)
        st.download_button("Scarica CSV SERP", df_serp.to_csv(index=False).encode("utf-8"),
                           file_name="serp_results.csv", mime="text/csv")
    with col2:
        st.subheader("People Also Ask")
        if paa:
            for q in paa: st.write(f"• {q}")
        else:
            st.info("Nessuna PAA trovata per questa query.")

    # Scraping
    st.subheader("Scraping contenuti dei risultati")
    urls = df_serp["url"].dropna().tolist()
    start = time.time()
    with st.spinner(f"Scarico e analizzo {len(urls)} pagine..."):
        scraped_map = parallel_fetch(urls, hl=hl, max_workers=max_workers)
    elapsed = time.time() - start

    stat_rows, contents, valid_urls = [], [], []
    for u in urls:
        text = scraped_map.get(u); length = len(text) if text else 0
        stat_rows.append({"url": u, "estratto": bool(text), "caratteri": length})
        if text and length >= 300:
            contents.append(text); valid_urls.append(u)

    df_scrape = pd.DataFrame(stat_rows)
    st.caption(f"Completato in {elapsed:.1f}s — {len(valid_urls)}/{len(urls)} pagine con testo utile.")
    st.dataframe(df_scrape, use_container_width=True, hide_index=True)

    if not contents:
        st.warning("Nessuna pagina con testo utile. Riprova con un'altra query o riduci n."); st.stop()

    # Analisi temi
    st.subheader("Analisi dei temi (n-gram)")
    with st.spinner("Estraggo keyphrase e calcolo coperture..."):
        terms_per_doc = [top_terms_doc(txt, hl, k=20) for txt in contents]
        common_terms, gaps_df, paa_not_covered, dfreq = compute_common_and_gaps(terms_per_doc, valid_urls, paa_terms=paa)

    st.markdown("### ✅ Temi comuni trattati da tutti")
    st.write(", ".join(common_terms[:30]) if common_terms else "—")

    st.markdown("### ⚠️ Topic su cui i competitor sono deboli / non coperti")
    if not gaps_df.empty:
        st.dataframe(gaps_df, use_container_width=True, hide_index=True)
        st.download_button("Scarica CSV Topic Gap", gaps_df.to_csv(index=False).encode("utf-8"),
                           file_name="topic_gaps.csv", mime="text/csv")
    else:
        st.info("Nessun gap ≥50% trovato.")

    st.markdown("### 🗺️ Mappa copertura temi per URL")
    top_topics = [t for t, _ in sorted(dfreq.items(), key=lambda x: (-x[1], x[0]))[:25]]
    coverage_rows = []
    for i, url in enumerate(valid_urls):
        row = {"url": url}; doc_terms = set(terms_per_doc[i])
        for t in top_topics: row[t] = "✅" if t in doc_terms else ""
        coverage_rows.append(row)
    st.dataframe(pd.DataFrame(coverage_rows), use_container_width=True)

    st.markdown("### ❓ PAA non coperte (spunti di contenuto)")
    if paa_not_covered: 
        for q in paa_not_covered: st.write(f"• {q}")
    else:
        st.info("Tutte le PAA principali appaiono (almeno parzialmente) nei temi estratti.")

    st.markdown("---")
    st.caption("Versione LITE: analisi n-gram senza sklearn (deploy veloce su Streamlit Cloud).")
