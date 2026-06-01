import streamlit as st
import requests
import re
import torch
from transformers import AutoTokenizer, AutoModel

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="ChemQuest AI",
    page_icon="🧪",
    layout="wide"
)

# ==============================
# AI CHEMICAL MODEL
# ==============================
CHEM_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"

@st.cache_resource
def load_chem_model():
    tokenizer = AutoTokenizer.from_pretrained(CHEM_MODEL_NAME)
    model = AutoModel.from_pretrained(CHEM_MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, chem_model = load_chem_model()

# ==============================
# CHEMICAL ONTOLOGY
# ==============================
CHEMICAL_CLASSES = {
    "salicylic acid derivatives": {
        "type": "Nonsteroidal Anti-inflammatory Drug (NSAID)",
        "mechanism": "Inhibits COX enzymes, reducing prostaglandin synthesis."
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
# SAFE CONVERSION
# ==============================
def safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None

# ==============================
# PUBCHEM FUNCTIONS
# ==============================
def fetch_pubchem_basic(name):
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/"
        f"MolecularFormula,MolecularWeight,XLogP,CanonicalSMILES/JSON"
    )

    try:
        r = requests.get(url, timeout=20)

        if r.status_code != 200:
            return None

        p = r.json()["PropertyTable"]["Properties"][0]
        cid = p.get("CID")

        return {
            "CID": cid,
            "Formula": p.get("MolecularFormula"),
            "Weight": safe_float(p.get("MolecularWeight")),
            "XlogP": safe_float(p.get("XLogP")),
            "SMILES": p.get("CanonicalSMILES"),
            "Structure": (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/{cid}/PNG"
                if cid else None
            )
        }

    except Exception:
        return None


def fetch_pubchem_props(name):
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/"
        f"HydrogenBondDonorCount,HydrogenBondAcceptorCount,TPSA,"
        f"RotatableBondCount,HeavyAtomCount,Complexity/JSON"
    )

    try:
        r = requests.get(url, timeout=20)

        if r.status_code == 200:
            props = r.json()["PropertyTable"]["Properties"][0]

            return {
                "HydrogenBondDonorCount": safe_float(props.get("HydrogenBondDonorCount")),
                "HydrogenBondAcceptorCount": safe_float(props.get("HydrogenBondAcceptorCount")),
                "TPSA": safe_float(props.get("TPSA")),
                "RotatableBondCount": safe_float(props.get("RotatableBondCount")),
                "HeavyAtomCount": safe_float(props.get("HeavyAtomCount")),
                "Complexity": safe_float(props.get("Complexity")),
            }

    except Exception:
        pass

    return {}


def fetch_pubchem_class(cid):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
    classes = []

    try:
        r = requests.get(url, timeout=20)

        if r.status_code != 200:
            return classes

        for sec in r.json()["Record"]["Section"]:
            if sec.get("TOCHeading") == "Chemical Classification":
                for s in sec.get("Section", []):
                    try:
                        classes.append(
                            s["Information"][0]["Value"]["StringWithMarkup"][0]["String"]
                        )
                    except Exception:
                        pass

    except Exception:
        pass

    return classes


def fetch_pubmed_refs(cid, limit=20):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/xrefs/PubMedID/JSON"

    try:
        r = requests.get(url, timeout=20)

        if r.status_code != 200:
            return []

        pmids = r.json()["InformationList"]["Information"][0].get("PubMedID", [])[:limit]

    except Exception:
        return []

    references = []

    for idx, pmid in enumerate(pmids, start=1):
        try:
            efetch = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                f"?db=pubmed&id={pmid}&rettype=abstract&retmode=text"
            )

            ar = requests.get(efetch, timeout=20)

            if ar.status_code != 200:
                continue

            text = ar.text.strip()
            lines = [l.strip() for l in text.split("\n") if l.strip()]

            title = lines[0] if len(lines) > 0 else "No title"
            authors = lines[1] if len(lines) > 1 else "No authors"
            abstract = " ".join(lines[2:]) if len(lines) > 2 else "No abstract available."

            references.append({
                "RefNo": idx,
                "PMID": pmid,
                "Title": title,
                "Authors": authors,
                "Abstract": abstract
            })

        except Exception:
            pass

    return references

# ==============================
# AI FUNCTIONS
# ==============================
def generate_chemberta_embedding(smiles):
    if not smiles:
        return None

    try:
        inputs = tokenizer(
            smiles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = chem_model(**inputs)

        embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]

        return embedding

    except Exception:
        return None


def simple_ai_summary(text):
    if not text or text == "No abstract available.":
        return "No abstract available for summary."

    sentences = re.split(r"\. |\n", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return "No abstract available."

    keywords = [
        "inhibit", "increase", "decrease", "effect", "treatment",
        "therapy", "drug", "compound", "activity", "toxicity",
        "expression", "associated", "significant", "clinical",
        "binding", "target", "mechanism"
    ]

    scored = [
        (sum(sentence.lower().count(k) for k in keywords), sentence)
        for sentence in sentences
    ]

    scored.sort(reverse=True)

    top_sentences = [sentence for score, sentence in scored[:3]]
    summary = ". ".join(top_sentences)

    if summary and not summary.endswith("."):
        summary += "."

    return summary


def infer_class(classes):
    for cls in classes:
        for key, value in CHEMICAL_CLASSES.items():
            if key.lower() in cls.lower():
                return value

    return {
        "type": "Unknown / experimental compound",
        "mechanism": "Mechanism not clearly established from ontology."
    }


def calculate_lipinski(data, props):
    mw = safe_float(data.get("Weight"))
    xlogp = safe_float(data.get("XlogP"))
    hbd = safe_float(props.get("HydrogenBondDonorCount"))
    hba = safe_float(props.get("HydrogenBondAcceptorCount"))

    rules = {
        "Molecular weight ≤ 500": mw is not None and mw <= 500,
        "XLogP ≤ 5": xlogp is not None and xlogp <= 5,
        "H-bond donors ≤ 5": hbd is not None and hbd <= 5,
        "H-bond acceptors ≤ 10": hba is not None and hba <= 10
    }

    passed = sum(1 for status in rules.values() if status)

    return rules, passed

# ==============================
# UI
# ==============================
st.markdown("""
<h1 style="text-align:center;">🧪 ChemQuest AI</h1>
<p style="text-align:center; color:gray;">
Chemical Intelligence • PubChem • PubMed • ChemBERTa Molecular AI
</p>
<hr>
""", unsafe_allow_html=True)

st.sidebar.title("ChemQuest AI")
st.sidebar.write("Chemical AI model:")
st.sidebar.code(CHEM_MODEL_NAME)

compound = st.text_input(
    "Enter chemical name",
    placeholder="Example: aspirin, imatinib, metformin"
)

if st.button("🔍 Search") and compound:

    with st.spinner("Fetching chemical intelligence..."):
        basic = fetch_pubchem_basic(compound)

    if not basic:
        st.error("Compound not found. Please check the spelling.")

    else:
        props = fetch_pubchem_props(compound)
        classes = fetch_pubchem_class(basic["CID"])
        inferred = infer_class(classes)
        references = fetch_pubmed_refs(basic["CID"], limit=20)
        embedding = generate_chemberta_embedding(basic["SMILES"])

        st.success(
            f"Chemical data fetched successfully. "
            f"{len(references)} PubMed references retrieved."
        )

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🧪 Identity",
            "⚗️ Properties",
            "🤖 ChemBERTa AI",
            "🧠 Interpretation",
            "📚 Literature"
        ])

        with tab1:
            st.subheader("Chemical Identity")

            col1, col2 = st.columns([1, 2])

            with col1:
                if basic["Structure"]:
                    st.image(basic["Structure"], width=300)

            with col2:
                st.write(f"**PubChem CID:** {basic['CID']}")
                st.write(f"**Molecular Formula:** {basic['Formula']}")
                st.write(f"**Molecular Weight:** {basic['Weight']}")
                st.write(f"**XLogP:** {basic['XlogP']}")
                st.write(f"**Canonical SMILES:** `{basic['SMILES']}`")

        with tab2:
            st.subheader("Molecular Properties")

            for k, v in props.items():
                st.write(f"**{k}:** {v}")

            st.markdown("### Lipinski Rule of Five")

            rules, passed = calculate_lipinski(basic, props)

            for rule, status in rules.items():
                if status:
                    st.success(f"PASS: {rule}")
                else:
                    st.warning(f"FAIL/UNKNOWN: {rule}")

            st.write(f"**Lipinski Score:** {passed}/4 rules passed")

        with tab3:
            st.subheader("ChemBERTa Chemical AI Representation")
            st.write(f"**Model used:** `{CHEM_MODEL_NAME}`")

            if embedding is not None:
                st.success("ChemBERTa molecular embedding generated successfully.")
                st.write(f"**Embedding dimension:** {embedding.shape[0]}")

                st.write("**First 20 embedding values:**")
                st.dataframe(
                    {
                        "Index": list(range(20)),
                        "Embedding Value": embedding[:20]
                    },
                    use_container_width=True
                )
            else:
                st.warning("ChemBERTa embedding could not be generated.")

        with tab4:
            st.subheader("Chemical Interpretation")

            st.write(f"**Predicted Compound Type:** {inferred['type']}")
            st.write(f"**Possible Mechanism of Action:** {inferred['mechanism']}")

            if classes:
                st.write("**Chemical Classification:**")
                for c in classes:
                    st.write(f"- {c}")
            else:
                st.write("No chemical classification found.")

        with tab5:
            st.subheader("PubMed Literature and AI Summary")

            if references:
                all_abstracts = " ".join([r["Abstract"] for r in references])

                st.markdown("### Overall Literature Summary")
                st.info(simple_ai_summary(all_abstracts))

                st.markdown("---")

                for r in references:
                    st.markdown(f"### Ref {r['RefNo']} | PMID: {r['PMID']}")
                    st.write(f"**Title:** {r['Title']}")
                    st.write(f"**Authors:** {r['Authors']}")

                    with st.expander("View Abstract"):
                        st.write(r["Abstract"])

                    st.markdown("**ChemQuest AI-style Summary:**")
                    st.info(simple_ai_summary(r["Abstract"]))

                    st.markdown("---")
            else:
                st.write("No PubMed references found.")

st.markdown("""
<hr>
<p style="text-align:center; color:gray;">
ChemQuest AI • Streamlit Cloud Ready • PubChem + PubMed + ChemBERTa
</p>
""", unsafe_allow_html=True)
