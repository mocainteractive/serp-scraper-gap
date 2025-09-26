import math
import time
import re
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import httpx
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import trafilatura
from sklearn.feature_extraction.text import TfidfVectorizer

import streamlit as st

# -----------------------------
# Config & helpers
# -----------------------------
st.set_page_config(
    page_title="SERP Scraper & Topic Gap Analyzer",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 SERP Scraper & Topic Gap Analyzer")
st.caption("Google SERP via Serper.dev → scraping contenuti concorrenti → temi in comune & topic gap")

# Mappa semplice hl → stopwords sklearn
HL_TO_STOPWORDS = {
    "it": "italian",
    "en": "english",
    "fr": "french",
    "de": "german",
    "es": "spanish",
}

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

SERPER_ENDPOINT = "https://google.serper.dev/search"


@st.cache_data(show_spinner=False, ttl=3600)
def serper_search(api_key: str, query: str, gl: str, hl: str) -> Dict:
    """Chiama Serper.dev e ritorna il JSON della SERP."""
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    body = {"q": query, "gl": gl, "hl": hl}
    resp = requests.post(SERPER_ENDPOINT, headers=headers, json=body, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Serper.dev error: {resp.status_code} - {resp.text}")
    return resp.json()


def extract_main_text(html: str) -> str:
    """Prova prima con trafilatura, poi fallback a BeautifulSoup.get_text()."""
    extracted = trafilatura.extract(
        html, include_comments=False, include_tables=False, favor_recall=True
    )
    if extracted and len(extracted.strip()) > 300:
        return extracted.strip()

    # fallback
    soup = BeautifulSoup(html, "html.parser")
    # Rimuovi script/style/noscript
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Normalizza spazi
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def fetch_url(url: str, hl: str, timeout: int = 20) -> Tuple[str, Optional[str]]:
    """Scarica HTML e restituisce (url, testo). In caso di errore, testo=None."""
    headers = DEFAULT_HEADERS.copy()
    headers["Accept-Language"] = f"{hl},{hl};q=0.8,en;q=0.6"
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
    """Scarica in parallelo i contenuti delle pagine."""
    results: Dict[str, Optional[str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(fetch_url, u, hl): u for u in urls}
        for fut in as_completed(future_map):
            u = future_map[fut]
            try:
                url, text = fut.result()
                results[url] = text
            except Exception:
                results[u] = None
    return results


def build_tfidf(docs: List[str], hl: str, max_features: int = 2000) -> Tuple[TfidfVectorizer, np.ndarray, List[str]]:
    """Crea TF-IDF su unigrammi/bigrammi."""
    stop_lang = HL_TO_STOPWORDS.get(hl, "english")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=stop_lang,
        max_df=0.9,
        min_df=1,
        max_features=max_features,
        strip_accents="unicode",
    )
    matrix = vectorizer.fit_transform(docs)
    feats = vectorizer.get_feature_names_out().tolist()
    return vectorizer, matrix, feats


def top_terms_for_doc(vec_row, feature_names: List[str], k: int = 15) -> List[str]:
    """Estrae i top-k termini da una riga sparsa TF-IDF."""
    if vec_row.nnz == 0:
        return []
    idxs = vec_row.indices
    data = vec_row.data
    order = np.argsort(-data)
    top = [feature_names[idxs[i]] for i in order[:k]]
    return top


def doc_freq(terms_per_doc: List[List[str]]) -> Dict[str, int]:
    """Calcola in quanti documenti appare ciascun termine (doc frequency)."""
    df: Dict[str, int] = {}
    for terms in terms_per_doc:
        for t in set(terms):
            df[t] = df.get(t, 0) + 1
    return df


def compute_common_and_gaps(
    terms_per_doc: List[List[str]],
    urls: List[str],
    paa_terms: List[str]
) -> Tuple[List[str], pd.DataFrame, List[str]]:
    """
    Ritorna:
      - common_terms: termini presenti in TUTTI i doc
      - gaps_df: tabella con topic e in quante pagine mancano + elenco URL mancanti
      - paa_not_covered: domande PAA (semplificate a topic) non coperte in nessun doc
    """
    N = len(terms_per_doc)
    df = doc_freq(terms_per_doc)

    # Termini comuni a tutti i documenti
    common_terms = sorted([t for t, c in df.items() if c == N])

    # Gap = termini con presenza ampia ma non totale (≥50% e <100%)
    thr = max(2, math.ceil(0.5 * N))
    candidate_gaps = [t for t, c in df.items() if thr <= c < N]

    rows = []
    for t in candidate_gaps:
        missing_idxs = [i for i, terms in enumerate(terms_per_doc) if t not in terms]
        rows.append({
            "topic": t,
            "missing_in_pages": len(missing_idxs),
            "missing_urls": ", ".join(urls[i] for i in missing_idxs)
        })
    gaps_df = pd.DataFrame(rows).sort_values(
        ["missing_in_pages", "topic"], ascending=[False, True]
    ).reset_index(drop=True)

    # PAA non coperte: se nessun doc contiene la stringa della domanda semplificata
    normalized_docs = " ".join([" ".join(terms) for terms in terms_per_doc]).lower()
    paa_simplified = [re.sub(r"[?¿¡!]", "", q).strip().lower() for q in paa_terms]
    paa_not_covered = [q for q in paa_simplified if q and q not in normalized_docs]

    return common_terms, gaps_df, paa_not_covered


def normalize_paa(people_also_ask: List[Dict]) -> List[str]:
    qs = []
    for item in people_also_ask or []:
        q = (item.get("question") or "").strip()
        if q:
            qs.append(q)
    # de-dup preservando l'ordine
    seen = set()
    out = []
    for q in qs:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def to_dataframe(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "position": r.get("position"),
            "title": r.get("title"),
            "url": r.get("link"),
            "snippet": r.get("snippet", "")
        })
    df = pd.DataFrame(rows).sort_values("position").reset_index(drop=True)
    return df


# -----------------------------
# UI
# -----------------------------
with st.sidebar:
    st.subheader("Impostazioni")

    # API dal cliente: niente secrets/env — resta in sessione
    if "serper_api_key" not in st.session_state:
        st.session_state.serper_api_key = ""
    st.session_state.serper_api_key = st.text_input(
        "Serper.dev API Key",
        value=st.session_state.serper_api_key,
        type="password",
        help="Inserisci la tua chiave. Non viene salvata su file."
    )

    query = st.text_input("Query di ricerca", placeholder="es. miglior macchina caffè espresso casa")
    gl = st.selectbox("Country (gl)", ["it", "us", "uk", "fr", "de", "es"], index=0)
    hl = st.selectbox("Lingua UI (hl)", ["it", "en", "fr", "de", "es"], index=0)
    n_sites = st.slider("Numero di siti da analizzare (n)", min_value=1, max_value=20, value=10)
    max_workers = st.slider("Concorrenza scraping (thread)", min_value=2, max_value=10, value=5)
    run = st.button("Esegui analisi 🚀", type="primary")

st.markdown("---")

if run:
    api_key = st.session_state.serper_api_key.strip()
    if not api_key:
        st.error("Inserisci la **Serper.dev API Key** nella sidebar.")
        st.stop()
    if not query.strip():
        st.error("Inserisci una **query di ricerca** nella sidebar.")
        st.stop()

    with st.spinner("Chiamo l'API di Serper.dev e preparo i dati..."):
        data = serper_search(api_key, query.strip(), gl, hl)
        organic = (data.get("organic") or [])[: max(n_sites, 1)]
        paa = normalize_paa(data.get("peopleAlsoAsk") or [])
        df_serp = to_dataframe(organic)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"Primi {len(df_serp)} risultati organici")
        st.dataframe(df_serp, use_container_width=True, hide_index=True)
        csv_serp = df_serp.to_csv(index=False).encode("utf-8")
        st.download_button("Scarica CSV SERP", csv_serp, file_name="serp_results.csv", mime="text/csv")

    with col2:
        st.subheader("People Also Ask")
        if paa:
            for q in paa:
                st.write(f"• {q}")
        else:
            st.info("Nessuna PAA trovata per questa query.")

    # Scraping contenuti
    st.subheader("Scraping contenuti dei risultati")
    urls = df_serp["url"].dropna().tolist()
    start = time.time()
    with st.spinner(f"Scarico e analizzo {len(urls)} pagine..."):
        scraped_map = parallel_fetch(urls, hl=hl, max_workers=max_workers)
    elapsed = time.time() - start

    stat_rows = []
    contents = []
    valid_urls = []
    for u in urls:
        text = scraped_map.get(u)
        length = len(text) if text else 0
        stat_rows.append({"url": u, "estratto": bool(text), "caratteri": length})
        if text and length >= 300:
            contents.append(text)
            valid_urls.append(u)

    df_scrape = pd.DataFrame(stat_rows)
    st.caption(f"Completato in {elapsed:.1f}s — {len(valid_urls)}/{len(urls)} pagine con testo utile.")
    st.dataframe(df_scrape, use_container_width=True, hide_index=True)

    if not contents:
        st.warning("Non è stato possibile estrarre testo utile da nessuna pagina. Riprova con un'altra query o riduci n.")
        st.stop()

    # Topic extraction via TF-IDF
    st.subheader("Analisi dei temi (TF-IDF)")
    with st.spinner("Estraggo keyphrase e calcolo coperture..."):
        vectorizer, matrix, feats = build_tfidf(contents, hl=hl, max_features=2000)
        terms_per_doc: List[List[str]] = []
        for i in range(matrix.shape[0]):
            terms_per_doc.append(top_terms_for_doc(matrix.getrow(i), feats, k=20))

        common_terms, gaps_df, paa_not_covered = compute_common_and_gaps(
            terms_per_doc, valid_urls, paa_terms=paa
        )

    # Output: Temi comuni
    st.markdown("### ✅ Temi comuni trattati da tutti")
    if common_terms:
        st.write(", ".join(common_terms[:30]))
    else:
        st.info("Nessun tema compare in **tutti** i documenti. (Set di pagine molto eterogeneo)")

    # Output: Topic gap aggregati
    st.markdown("### ⚠️ Topic su cui i competitor sono deboli / non coperti")
    if not gaps_df.empty:
        st.dataframe(gaps_df, use_container_width=True, hide_index=True)
        csv_gaps = gaps_df.to_csv(index=False).encode("utf-8")
        st.download_button("Scarica CSV Topic Gap", csv_gaps, file_name="topic_gaps.csv", mime="text/csv")
    else:
        st.success("Non risultano topic con copertura ≥50% ma non totale. (Pochi gap visibili)")

    # Matrice coperture (temi più rilevanti)
    st.markdown("### 🗺️ Mappa copertura temi per URL")
    dfreq = doc_freq(terms_per_doc)
    top_topics = [t for t, _ in sorted(dfreq.items(), key=lambda x: (-x[1], x[0]))[:25]]
    coverage_rows = []
    for i, url in enumerate(valid_urls):
        row = {"url": url}
        doc_terms = set(terms_per_doc[i])
        for t in top_topics:
            row[t] = "✅" if t in doc_terms else ""
        coverage_rows.append(row)
    df_cover = pd.DataFrame(coverage_rows)
    st.dataframe(df_cover, use_container_width=True)

    # PAA non coperte
    st.markdown("### ❓ PAA non coperte (spunti di contenuto)")
    if paa_not_covered:
        for q in paa_not_covered:
            st.write(f"• {q}")
    else:
        st.info("Tutte le PAA principali appaiono (almeno parzialmente) nei temi estratti.")

    st.markdown("---")
    st.caption(
        "Nota: lo scraping va usato nel rispetto dei termini dei siti. "
        "La qualità dell’estrazione dipende dalla struttura HTML delle pagine."
    )
