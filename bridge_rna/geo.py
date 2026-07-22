"""GEO and PubMed enrichment for retrieved hits, over NCBI E-utilities.

A network fallback for the fields the local sources do not carry. With the
manifold cache built, `retrieval.py` fills most of these offline and this path
is only reached for what the cache does not have.
"""

from __future__ import annotations

import gzip
import time
from typing import Any

import pandas as pd
import requests

from .config import NCBI_API_KEY
from .util import _safe_str


def _enrich_hits_from_ncbi_eutils(hits_df: pd.DataFrame, entrez_email: str) -> pd.DataFrame:
    """Enrich hit rows with GEO metadata via NCBI E-utilities.

    This fallback works even without local ARCHS4 H5 metadata or Biopython.
    """
    email = _safe_str(entrez_email)
    if hits_df.empty or not email:
        return hits_df

    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    session = requests.Session()

    base_params = {
        "tool": "osdr_dash_app",
        "email": email,
    }
    if NCBI_API_KEY:
        base_params["api_key"] = NCBI_API_KEY

    def _first_pmid_string(pmids_val: Any) -> str:
        if pmids_val is None:
            return ""
        if isinstance(pmids_val, list):
            for p in pmids_val:
                s = _safe_str(p)
                if s:
                    return s
            return ""
        return _safe_str(pmids_val)

    def _pubmed_details(pmid: str) -> tuple[str, str, str, str]:
        if not pmid:
            return "", "", "", ""
        p = dict(base_params)
        p.update({"db": "pubmed", "id": pmid, "retmode": "json"})
        try:
            r = None
            for attempt in range(3):
                r = session.get(esummary_url, params=p, timeout=20)
                if r.status_code == 200:
                    break
                if r.status_code == 429:
                    time.sleep(0.7 * (attempt + 1))
            if r is None or r.status_code != 200:
                return "", "", "", ""
            j = r.json()
            uid = (j.get("result", {}).get("uids") or [None])[0]
            d = j.get("result", {}).get(uid, {}) if uid else {}
            title = _safe_str(d.get("title", ""))
            journal = _safe_str(d.get("fulljournalname", ""))
            pub_date = _safe_str(d.get("pubdate", ""))

            doi = ""
            for aid in d.get("articleids", []) if isinstance(d.get("articleids", []), list) else []:
                if isinstance(aid, dict) and _safe_str(aid.get("idtype", "")).lower() == "doi":
                    doi = _safe_str(aid.get("value", ""))
                    break
            return title, journal, pub_date, doi
        except Exception:
            return "", "", "", ""

    def _soft_summary_and_design(gse: str, ftp_link: str) -> tuple[str, str]:
        gse_u = _safe_str(gse).upper()
        if not gse_u:
            return "", ""
        if not gse_u.startswith("GSE"):
            gse_u = f"GSE{gse_u}"

        summary_parts: list[str] = []
        design_parts: list[str] = []

        candidates: list[str] = []
        ftp = _safe_str(ftp_link)
        if ftp.startswith("ftp://"):
            http_base = "https://" + ftp[len("ftp://") :].rstrip("/")
            candidates.append(f"{http_base}/soft/{gse_u}_family.soft.gz")

        prefix = f"{gse_u[:-3]}nnn" if len(gse_u) > 3 else gse_u
        candidates.append(
            f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse_u}/soft/{gse_u}_family.soft.gz"
        )

        for url in candidates:
            try:
                rr = session.get(url, timeout=25)
                if rr.status_code != 200:
                    continue
                text = gzip.decompress(rr.content).decode("utf-8", errors="replace")
                for line in text.splitlines():
                    if line.startswith("!Series_summary"):
                        val = line.split("=", 1)[1].strip() if "=" in line else ""
                        if val:
                            summary_parts.append(val)
                    elif line.startswith("!Series_overall_design"):
                        val = line.split("=", 1)[1].strip() if "=" in line else ""
                        if val:
                            design_parts.append(val)
                break
            except Exception:
                continue

        return " ".join(summary_parts).strip(), " ".join(design_parts).strip()

    rows = []
    for gsm in hits_df["gsm"].astype(str).dropna().unique().tolist():
        rec = {
            "gsm": gsm,
            "gse_ncbi": "",
            "title_ncbi": "",
            "summary_ncbi": "",
            "design_ncbi": "",
            "pubmed_ids_ncbi": "",
            "platform_ncbi": "",
            "taxon_ncbi": "",
            "entry_type_ncbi": "",
            "gds_type_ncbi": "",
            "pdat_ncbi": "",
            "n_samples_ncbi": "",
            "ftp_link_ncbi": "",
            "pubmed_title_ncbi": "",
            "pubmed_journal_ncbi": "",
            "pubmed_pub_date_ncbi": "",
            "pubmed_doi_ncbi": "",
        }
        try:
            p1 = dict(base_params)
            p1.update({"db": "gds", "term": f"{gsm}[ACCN]", "retmode": "json", "retmax": 1})
            r1 = None
            for attempt in range(3):
                r1 = session.get(esearch_url, params=p1, timeout=20)
                if r1.status_code == 200:
                    break
                if r1.status_code == 429:
                    time.sleep(0.7 * (attempt + 1))
            if r1 is None or r1.status_code != 200:
                rows.append(rec)
                continue
            j1 = r1.json()
            ids = j1.get("esearchresult", {}).get("idlist", [])
            if not ids:
                rows.append(rec)
                continue

            p2 = dict(base_params)
            p2.update({"db": "gds", "id": ids[0], "retmode": "json"})
            r2 = None
            for attempt in range(3):
                r2 = session.get(esummary_url, params=p2, timeout=20)
                if r2.status_code == 200:
                    break
                if r2.status_code == 429:
                    time.sleep(0.7 * (attempt + 1))
            if r2 is None or r2.status_code != 200:
                rows.append(rec)
                continue
            j2 = r2.json()
            uid = (j2.get("result", {}).get("uids") or [None])[0]
            doc = j2.get("result", {}).get(uid, {}) if uid else {}

            gse_raw = _safe_str(doc.get("gse", ""))
            rec["gse_ncbi"] = f"GSE{gse_raw}" if gse_raw and not gse_raw.upper().startswith("GSE") else gse_raw.upper()
            rec["title_ncbi"] = _safe_str(doc.get("title", ""))
            rec["summary_ncbi"] = _safe_str(doc.get("summary", ""))
            rec["platform_ncbi"] = _safe_str(doc.get("gpl", ""))
            rec["taxon_ncbi"] = _safe_str(doc.get("taxon", ""))
            rec["entry_type_ncbi"] = _safe_str(doc.get("entrytype", ""))
            rec["gds_type_ncbi"] = _safe_str(doc.get("gdstype", ""))
            rec["pdat_ncbi"] = _safe_str(doc.get("pdat", ""))
            rec["n_samples_ncbi"] = _safe_str(doc.get("n_samples", ""))
            rec["ftp_link_ncbi"] = _safe_str(doc.get("ftplink", ""))

            pmids = doc.get("pubmedids", [])
            if isinstance(pmids, list):
                rec["pubmed_ids_ncbi"] = ";".join([_safe_str(p) for p in pmids if _safe_str(p)])
            else:
                rec["pubmed_ids_ncbi"] = _safe_str(pmids)

            pmid_one = _first_pmid_string(pmids)
            t, jn, pub_date_val, doi = _pubmed_details(pmid_one)
            rec["pubmed_title_ncbi"] = t
            rec["pubmed_journal_ncbi"] = jn
            rec["pubmed_pub_date_ncbi"] = pub_date_val
            rec["pubmed_doi_ncbi"] = doi

            # Pull richer overall design/summary from GEO family SOFT when possible.
            soft_summary, soft_design = _soft_summary_and_design(rec["gse_ncbi"], rec["ftp_link_ncbi"])
            if soft_summary:
                rec["summary_ncbi"] = soft_summary
            rec["design_ncbi"] = soft_design
        except Exception:
            pass

        rows.append(rec)
        # Be polite to NCBI public API.
        time.sleep(0.12)

    if not rows:
        return hits_df

    enrich = pd.DataFrame(rows)
    out = hits_df.merge(enrich, on="gsm", how="left")

    # Fill only missing/blank values from NCBI fallback.
    out["gse"] = out.apply(lambda r: _safe_str(r.get("gse")) or _safe_str(r.get("gse_ncbi")), axis=1)
    out["title"] = out.apply(lambda r: _safe_str(r.get("title")) or _safe_str(r.get("title_ncbi")), axis=1)
    out["geo_summary"] = out.apply(lambda r: _safe_str(r.get("geo_summary")) or _safe_str(r.get("summary_ncbi")), axis=1)
    out["geo_design"] = out.apply(lambda r: _safe_str(r.get("geo_design")) or _safe_str(r.get("design_ncbi")), axis=1)
    out["pubmed_ids"] = out.apply(lambda r: _safe_str(r.get("pubmed_ids")) or _safe_str(r.get("pubmed_ids_ncbi")), axis=1)

    out["geo_platform_biopython"] = out.apply(lambda r: _safe_str(r.get("geo_platform_biopython")) or _safe_str(r.get("platform_ncbi")), axis=1)
    out["geo_taxon_biopython"] = out.apply(lambda r: _safe_str(r.get("geo_taxon_biopython")) or _safe_str(r.get("taxon_ncbi")), axis=1)
    out["geo_entry_type_biopython"] = out.apply(lambda r: _safe_str(r.get("geo_entry_type_biopython")) or _safe_str(r.get("entry_type_ncbi")), axis=1)
    out["geo_gds_type_biopython"] = out.apply(lambda r: _safe_str(r.get("geo_gds_type_biopython")) or _safe_str(r.get("gds_type_ncbi")), axis=1)
    out["geo_pdat_biopython"] = out.apply(lambda r: _safe_str(r.get("geo_pdat_biopython")) or _safe_str(r.get("pdat_ncbi")), axis=1)
    out["geo_n_samples_biopython"] = out.apply(lambda r: _safe_str(r.get("geo_n_samples_biopython")) or _safe_str(r.get("n_samples_ncbi")), axis=1)
    out["geo_ftp_link_biopython"] = out.apply(lambda r: _safe_str(r.get("geo_ftp_link_biopython")) or _safe_str(r.get("ftp_link_ncbi")), axis=1)

    out["pubmed_title_biopython"] = out.apply(lambda r: _safe_str(r.get("pubmed_title_biopython")) or _safe_str(r.get("pubmed_title_ncbi")), axis=1)
    out["pubmed_journal_biopython"] = out.apply(lambda r: _safe_str(r.get("pubmed_journal_biopython")) or _safe_str(r.get("pubmed_journal_ncbi")), axis=1)
    out["pubmed_pub_date_biopython"] = out.apply(lambda r: _safe_str(r.get("pubmed_pub_date_biopython")) or _safe_str(r.get("pubmed_pub_date_ncbi")), axis=1)
    out["pubmed_doi_biopython"] = out.apply(lambda r: _safe_str(r.get("pubmed_doi_biopython")) or _safe_str(r.get("pubmed_doi_ncbi")), axis=1)

    for c in [
        "gse_ncbi",
        "title_ncbi",
        "summary_ncbi",
        "design_ncbi",
        "pubmed_ids_ncbi",
        "platform_ncbi",
        "taxon_ncbi",
        "entry_type_ncbi",
        "gds_type_ncbi",
        "pdat_ncbi",
        "n_samples_ncbi",
        "ftp_link_ncbi",
        "pubmed_title_ncbi",
        "pubmed_journal_ncbi",
        "pubmed_pub_date_ncbi",
        "pubmed_doi_ncbi",
    ]:
        if c in out.columns:
            out = out.drop(columns=[c])
    return out
