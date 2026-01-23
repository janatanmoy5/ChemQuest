import streamlit as st
import requests
import re

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="ChemQuest",
    page_icon="🧪",
    layout="wide"
)

# ==============================
# CHEMICAL ONTOLOGY
# ==============================
CHEMICAL_CLASSES = {
    "salicylic acid derivatives": {
        "type": "Nonsteroidal Anti-inflammatory Drug (NSAID)",
        "mechanism": "Inhibits cyclooxygenase (COX) enzymes, reducing prostaglandin synthesis."
    },
    "quinazolines": {
        "type": "Kinase inhibitor",
        "mechanism": "Blocks ATP-binding site of receptor tyrosine kinases."
    },
    "beta-lactams": {
        "type": "Antibiotic",
        "mechanism": "Inhibits bacterial cell wall synthesis."
    },
    "statins": {
        "type": "Lipid-lowering agent",
        "mechanism": "Inhibits HMG-CoA reductase."
    }
}

# ==============================
# HELPERS
# ==============================
def fetch_pubchem_basic(name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/MolecularFormula,MolecularWeight,XLogP/JSON"
    r = requests.get(url)
    if r.status_code != 200: return None
    p = r.json()["PropertyTable"]["Properties"][0]
    cid = p.get("CID")
    return {
        "CID": cid,
        "Formula": p.get("MolecularFormula"),
        "Weight": p.get("MolecularWeight"),
        "XlogP": p.get("XLogP"),
        "Structure": f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/{cid}/PNG" if cid else None
    }

def fetch_pubchem_props(name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/HydrogenBondDonorCount,HydrogenBondAcceptorCount,TPSA,RotatableBondCount/JSON"
    r = requests.get(url)
    return r.json()["PropertyTable"]["Properties"][0] if r.status_code == 200 else {}

def fetch_pubchem_class(cid):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
    r = requests.get(url)
    classes = []
    if r.status_code != 200: return classes
    for sec in r.json()["Record"]["Section"]:
        if sec.get("TOCHeading") == "Chemical Classification":
            for s in sec.get("Section", []):
                try:
                    classes.append(s["Information"][0]["Value"]["StringWithMarkup"][0]["String"])
                except: pass
    return classes

def fetch_pubmed_refs(cid, limit=20):
    """Fetch PubMed references with abstracts, titles, and authors"""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/xrefs/PubMedID/JSON"
    r = requests.get(url)
    if r.status_code != 200: return []
    pmids = r.json()["InformationList"]["Information"][0].get("PubMedID", [])[:limit]
    references = []

    for idx, pmid in enumerate(pmids, start=1):
        try:
            efetch = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&rettype=abstract&retmode=text"
            ar = requests.get(efetch)
            if ar.status_code != 200:
                references.append({
                    "RefNo": idx, "PMID": pmid,
                    "Title": "Unable to fetch title",
                    "Authors": "N/A",
                    "Abstract": "Unable to fetch abstract"
                })
                continue

            text = ar.text.strip()
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            title = lines[0] if len(lines) > 0 else "No title"
            authors = lines[1] if len(lines) > 1 else "No authors"
            abstract_lines = lines[2:] if len(lines) > 2 else ["No abstract"]
            abstract_text = " ".join(abstract_lines)

            references.append({
                "RefNo": idx,
                "PMID": pmid,
                "Title": title,
                "Authors": authors,
                "Abstract": abstract_text
            })
        except:
            references.append({
                "RefNo": idx,
                "PMID": pmid,
                "Title": "Fetch error",
                "Authors": "Fetch error",
                "Abstract": "Fetch error"
            })

    return references

def infer_class(classes):
    for cls in classes:
        for k, v in CHEMICAL_CLASSES.items():
            if k.lower() in cls.lower():
                return v
    return {"type": "Unknown / experimental compound", "mechanism": "Mechanism not clearly established."}

def summarize_text_full(text):
    """AI-style full sentence summary per reference with ChemQuest signature"""
    sentences = re.split(r'\. |\n', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences: return "No abstract available."
    # Take top 3 sentences containing keywords
    keywords = ["inhibit","increase","decrease","effect","treatment","therapy",
                "induced","expression","observed","associated","caused","resulted"]
    scored = [(sum(s.lower().count(k) for k in keywords), s) for s in sentences]
    scored.sort(reverse=True)
    top_sentences = [s for score, s in scored[:3] if s.strip()]
    summary_text = ". ".join(top_sentences)
    if summary_text and not summary_text.endswith('.'):
        summary_text += "."
    return f"ChemQuest AI Summary: {summary_text}"

# ==============================
# UI
# ==============================
st.markdown("""
<h1 style="text-align:center;">🧪 ChemQuest</h1>
<p style="text-align:center; color:gray;">
Chemical Intelligence • Properties • Literature • AI Summary
</p>
<hr>
""", unsafe_allow_html=True)

compound = st.text_input("Enter chemical name (e.g., aspirin, imatinib)")

if compound:
    search_button = st.button("🔍 Search")
    if search_button:
        with st.spinner("Fetching chemical intelligence..."):
            basic = fetch_pubchem_basic(compound)

        if not basic:
            st.error("Compound not found.")
        else:
            props = fetch_pubchem_props(compound)
            classes = fetch_pubchem_class(basic["CID"])
            inferred = infer_class(classes)
            references = fetch_pubmed_refs(basic["CID"], limit=20)

            st.success(f"✅ Chemical data fetched successfully! {len(references)} references retrieved.")

            tab1, tab2, tab3, tab4 = st.tabs(
                ["🧪 Identity", "⚗️ Properties", "🧠 Interpretation", "📚 Literature"]
            )

            with tab1:
                st.write(f"**PubChem CID:** {basic['CID']}")
                st.write(f"**Molecular Formula:** {basic['Formula']}")
                if basic["Structure"]:
                    st.image(basic["Structure"], width=300)

            with tab2:
                st.write(f"**Molecular Weight:** {basic['Weight']}")
                st.write(f"**XlogP:** {basic['XlogP']}")
                for k, v in props.items():
                    st.write(f"**{k}:** {v}")

            with tab3:
                st.write(f"**Compound Type:** {inferred['type']}")
                st.write(f"**Mechanism of Action:** {inferred['mechanism']}")
                if classes:
                    st.write("**Chemical Classification:**")
                    for c in classes:
                        st.write(f"- {c}")

            with tab4:
                if references:
                    for r in references:
                        st.markdown(f"**Ref {r['RefNo']} (PMID: {r['PMID']}): {r['Title']}, {r['Authors']}**")
                        st.write(f"Abstract: {r['Abstract']}")
                        mini_summary = summarize_text_full(r['Abstract'])
                        st.markdown(f"🔹 **{mini_summary}**")
                        st.markdown("---")
                else:
                    st.write("No PubMed references found.")

st.markdown("""
<hr>
<p style="text-align:center; color:gray;">
ChemQuest • Streamlit Cloud Ready • GitHub Friendly
</p>
""", unsafe_allow_html=True)

