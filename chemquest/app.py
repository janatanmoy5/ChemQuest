import streamlit as st
import requests
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

st.set_page_config(
    page_title="ChemQuest AI",
    page_icon="🧪",
    layout="wide"
)

CHEM_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"

@st.cache_resource
def load_chem_model():
    tokenizer = AutoTokenizer.from_pretrained(CHEM_MODEL_NAME)
    model = AutoModel.from_pretrained(CHEM_MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, chem_model = load_chem_model()

def safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None

def fetch_pubchem_basic(name):
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/"
        f"MolecularFormula,MolecularWeight,XLogP,CanonicalSMILES/JSON"
    )
    try:
        r = requests.get(url, timeout=30)
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
            "Structure": f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/{cid}/PNG"
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
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            p = r.json()["PropertyTable"]["Properties"][0]
            return {
                "HydrogenBondDonorCount": safe_float(p.get("HydrogenBondDonorCount")),
                "HydrogenBondAcceptorCount": safe_float(p.get("HydrogenBondAcceptorCount")),
                "TPSA": safe_float(p.get("TPSA")),
                "RotatableBondCount": safe_float(p.get("RotatableBondCount")),
                "HeavyAtomCount": safe_float(p.get("HeavyAtomCount")),
                "Complexity": safe_float(p.get("Complexity")),
            }
    except Exception:
        pass
    return {}

def fetch_pubmed_refs(cid, limit=10):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/xrefs/PubMedID/JSON"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return []
        pmids = r.json()["InformationList"]["Information"][0].get("PubMedID", [])[:limit]
    except Exception:
        return []

    refs = []
    for i, pmid in enumerate(pmids, start=1):
        try:
            efetch = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                f"?db=pubmed&id={pmid}&rettype=abstract&retmode=text"
            )
            ar = requests.get(efetch, timeout=30)
            if ar.status_code != 200:
                continue
            text = ar.text.strip()
            lines = [x.strip() for x in text.split("\n") if x.strip()]
            refs.append({
                "RefNo": i,
                "PMID": pmid,
                "Title": lines[0] if len(lines) > 0 else "No title",
                "Authors": lines[1] if len(lines) > 1 else "No authors",
                "Abstract": " ".join(lines[2:]) if len(lines) > 2 else "No abstract available."
            })
        except Exception:
            pass
    return refs

def simple_ai_summary(text):
    if not text:
        return "No abstract available."

    sentences = re.split(r"\. |\n", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    keywords = [
        "inhibit", "increase", "decrease", "effect", "treatment",
        "therapy", "drug", "compound", "activity", "toxicity",
        "binding", "target", "mechanism", "clinical"
    ]

    scored = [
        (sum(s.lower().count(k) for k in keywords), s)
        for s in sentences
    ]
    scored.sort(reverse=True)

    summary = ". ".join([s for score, s in scored[:3]])
    return summary + "." if summary and not summary.endswith(".") else summary

def generate_chemberta_embedding(smiles):
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
        return outputs.last_hidden_state.mean(dim=1).numpy()[0]
    except Exception:
        return None

def mol_from_smiles(smiles):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None

def tanimoto_similarity(smiles1, smiles2):
    mol1 = mol_from_smiles(smiles1)
    mol2 = mol_from_smiles(smiles2)

    if mol1 is None or mol2 is None:
        return None

    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)

    return DataStructs.TanimotoSimilarity(fp1, fp2)

def fetch_pubchem_similar(cid, threshold=90, max_results=10):
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/cid/"
        f"{cid}/cids/JSON?Threshold={threshold}&MaxRecords={max_results}"
    )

    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            return []

        cids = r.json()["IdentifierList"]["CID"]

        if cid in cids:
            cids.remove(cid)

        if not cids:
            return []

        cid_text = ",".join(map(str, cids[:max_results]))

        prop_url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_text}/property/"
            f"Title,MolecularFormula,MolecularWeight,XLogP,CanonicalSMILES/JSON"
        )

        pr = requests.get(prop_url, timeout=60)

        if pr.status_code != 200:
            return []

        results = []

        for p in pr.json()["PropertyTable"]["Properties"]:
            results.append({
                "CID": p.get("CID"),
                "Name": p.get("Title"),
                "Formula": p.get("MolecularFormula"),
                "Weight": p.get("MolecularWeight"),
                "XLogP": p.get("XLogP"),
                "SMILES": p.get("CanonicalSMILES"),
            })

        return results

    except Exception:
        return []

def fetch_chembl_molecule_by_smiles(smiles):
    try:
        url = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
        params = {
            "molecule_structures__canonical_smiles__flexmatch": smiles,
            "limit": 1
        }
        r = requests.get(url, params=params, timeout=60)

        if r.status_code != 200:
            return None

        data = r.json().get("molecules", [])

        if not data:
            return None

        return data[0].get("molecule_chembl_id")

    except Exception:
        return None

def fetch_chembl_targets(chembl_id, limit=25):
    try:
        url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
        params = {
            "molecule_chembl_id": chembl_id,
            "limit": limit
        }

        r = requests.get(url, params=params, timeout=60)

        if r.status_code != 200:
            return []

        activities = r.json().get("activities", [])

        rows = []

        for a in activities:
            rows.append({
                "Target ChEMBL ID": a.get("target_chembl_id"),
                "Target Name": a.get("target_pref_name"),
                "Organism": a.get("target_organism"),
                "Activity Type": a.get("standard_type"),
                "Value": a.get("standard_value"),
                "Units": a.get("standard_units"),
                "Relation": a.get("standard_relation"),
                "Assay Description": a.get("assay_description")
            })

        return rows

    except Exception:
        return []

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

    passed = sum(1 for x in rules.values() if x)
    return rules, passed

st.markdown("""
<h1 style="text-align:center;">🧪 ChemQuest AI</h1>
<p style="text-align:center;color:gray;">
Chemical Intelligence • PubChem • PubMed • ChemBERTa • Tanimoto Similarity • ChEMBL Targets
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

similarity_threshold = st.sidebar.slider(
    "PubChem Similarity Threshold",
    min_value=70,
    max_value=99,
    value=90,
    step=1
)

max_similar = st.sidebar.slider(
    "Max Similar Compounds",
    min_value=5,
    max_value=25,
    value=10,
    step=5
)

if st.button("🔍 Search") and compound:

    with st.spinner("Fetching chemical data..."):
        basic = fetch_pubchem_basic(compound)

    if not basic:
        st.error("Compound not found.")
        st.stop()

    props = fetch_pubchem_props(compound)
    refs = fetch_pubmed_refs(basic["CID"], limit=10)
    embedding = generate_chemberta_embedding(basic["SMILES"])

    with st.spinner("Searching similar compounds in PubChem..."):
        similar = fetch_pubchem_similar(
            basic["CID"],
            threshold=similarity_threshold,
            max_results=max_similar
        )

    with st.spinner("Searching ChEMBL targets..."):
        chembl_id = fetch_chembl_molecule_by_smiles(basic["SMILES"])
        targets = fetch_chembl_targets(chembl_id) if chembl_id else []

    st.success("Chemical intelligence completed.")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧪 Identity",
        "⚗️ Properties",
        "🤖 ChemBERTa AI",
        "🔍 Similarity",
        "🎯 Targets",
        "📚 Literature"
    ])

    with tab1:
        st.subheader("Chemical Identity")

        col1, col2 = st.columns([1, 2])

        with col1:
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

        st.write(f"**Lipinski Score:** {passed}/4")

    with tab3:
        st.subheader("ChemBERTa Molecular AI")

        st.write(f"**Model used:** `{CHEM_MODEL_NAME}`")

        if embedding is not None:
            st.success("ChemBERTa embedding generated.")
            st.write(f"**Embedding dimension:** {embedding.shape[0]}")
            st.dataframe(
                pd.DataFrame({
                    "Index": list(range(min(20, len(embedding)))),
                    "Embedding Value": embedding[:20]
                }),
                use_container_width=True
            )
        else:
            st.warning("Embedding could not be generated.")

    with tab4:
        st.subheader("PubChem Similarity Search + RDKit Tanimoto Index")

        if similar:
            rows = []

            for item in similar:
                tanimoto = tanimoto_similarity(
                    basic["SMILES"],
                    item.get("SMILES")
                )

                rows.append({
                    "CID": item.get("CID"),
                    "Name": item.get("Name"),
                    "Formula": item.get("Formula"),
                    "MW": item.get("Weight"),
                    "XLogP": item.get("XLogP"),
                    "Tanimoto Index": round(tanimoto, 4) if tanimoto is not None else None,
                    "SMILES": item.get("SMILES")
                })

            sim_df = pd.DataFrame(rows)
            sim_df = sim_df.sort_values(
                by="Tanimoto Index",
                ascending=False
            )

            st.dataframe(sim_df, use_container_width=True)

        else:
            st.warning("No similar compounds found from PubChem.")

    with tab5:
        st.subheader("Target Finding from ChEMBL")

        if chembl_id:
            st.write(f"**ChEMBL Molecule ID:** `{chembl_id}`")

        if targets:
            target_df = pd.DataFrame(targets)
            st.dataframe(target_df, use_container_width=True)
        else:
            st.warning("No ChEMBL targets found for this compound.")

        st.info(
            "Target finding is retrieved from ChEMBL bioactivity records. "
            "For some PubChem compounds, no ChEMBL target may be available."
        )

    with tab6:
        st.subheader("PubMed Literature and AI-style Summary")

        if refs:
            all_abstracts = " ".join([r["Abstract"] for r in refs])
            st.markdown("### Overall Literature Summary")
            st.info(simple_ai_summary(all_abstracts))

            st.markdown("---")

            for r in refs:
                st.markdown(f"### Ref {r['RefNo']} | PMID: {r['PMID']}")
                st.write(f"**Title:** {r['Title']}")
                st.write(f"**Authors:** {r['Authors']}")

                with st.expander("View Abstract"):
                    st.write(r["Abstract"])

                st.markdown("**ChemQuest AI-style Summary:**")
                st.info(simple_ai_summary(r["Abstract"]))

                st.markdown("---")
        else:
            st.warning("No PubMed references found.")

st.markdown("""
<hr>
<p style="text-align:center;color:gray;">
ChemQuest AI • PubChem Similarity • RDKit Tanimoto • ChEMBL Target Finding
</p>
""", unsafe_allow_html=True)
