import streamlit as st
import requests
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False


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


try:
    tokenizer, chem_model = load_chem_model()
    CHEMBERTA_AVAILABLE = True
    CHEMBERTA_ERROR = None
except Exception as e:
    tokenizer, chem_model = None, None
    CHEMBERTA_AVAILABLE = False
    CHEMBERTA_ERROR = str(e)


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
        f"MolecularFormula,MolecularWeight,XLogP,CanonicalSMILES,IsomericSMILES/JSON"
    )

    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return None

        p = r.json()["PropertyTable"]["Properties"][0]
        cid = p.get("CID")

        return {
            "CID": cid,
            "Name": name,
            "Formula": p.get("MolecularFormula"),
            "Weight": safe_float(p.get("MolecularWeight")),
            "XlogP": safe_float(p.get("XLogP")),
            "SMILES": p.get("CanonicalSMILES") or p.get("IsomericSMILES"),
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
        cids = [x for x in cids if x != cid]

        if not cids:
            return []

        cid_text = ",".join(map(str, cids[:max_results]))

        prop_url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_text}/property/"
            f"Title,MolecularFormula,MolecularWeight,XLogP,CanonicalSMILES,IsomericSMILES/JSON"
        )

        pr = requests.get(prop_url, timeout=60)

        if pr.status_code != 200:
            return []

        results = []

        for p in pr.json()["PropertyTable"]["Properties"]:
            results.append({
                "Source": "PubChem",
                "ID": p.get("CID"),
                "Name": p.get("Title"),
                "Formula": p.get("MolecularFormula"),
                "MW": safe_float(p.get("MolecularWeight")),
                "XLogP": safe_float(p.get("XLogP")),
                "SMILES": p.get("CanonicalSMILES") or p.get("IsomericSMILES")
            })

        return results

    except Exception:
        return []


def fetch_chembl_molecule_by_name(compound_name):
    try:
        url = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"

        params = {
            "pref_name__iexact": compound_name,
            "limit": 1
        }

        r = requests.get(url, params=params, timeout=60)

        if r.status_code != 200:
            return None

        molecules = r.json().get("molecules", [])

        if not molecules:
            return None

        mol = molecules[0]

        return {
            "chembl_id": mol.get("molecule_chembl_id"),
            "pref_name": mol.get("pref_name"),
            "smiles": mol.get("molecule_structures", {}).get("canonical_smiles")
        }

    except Exception:
        return None


def fetch_chembl_similarity(smiles, similarity=85, max_results=10):
    if not smiles:
        return []

    try:
        url = f"https://www.ebi.ac.uk/chembl/api/data/similarity/{smiles}/{similarity}.json"

        r = requests.get(url, timeout=90)

        if r.status_code != 200:
            return []

        molecules = r.json().get("molecules", [])

        results = []

        for mol in molecules[:max_results]:
            structures = mol.get("molecule_structures") or {}

            results.append({
                "Source": "ChEMBL",
                "ID": mol.get("molecule_chembl_id"),
                "Name": mol.get("pref_name"),
                "Formula": mol.get("molecule_properties", {}).get("full_molformula"),
                "MW": safe_float(mol.get("molecule_properties", {}).get("full_mwt")),
                "XLogP": safe_float(mol.get("molecule_properties", {}).get("alogp")),
                "SMILES": structures.get("canonical_smiles"),
                "ChEMBL Similarity": mol.get("similarity")
            })

        return results

    except Exception:
        return []


def fetch_chembl_targets(chembl_id, limit=30):
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


def generate_chemberta_embedding(smiles):
    if not smiles:
        return None, "No SMILES found from PubChem."

    if not CHEMBERTA_AVAILABLE:
        return None, CHEMBERTA_ERROR

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

        embedding = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()[0]

        return embedding, None

    except Exception as e:
        return None, str(e)


def tanimoto_similarity(smiles1, smiles2):
    if not RDKIT_AVAILABLE:
        return None

    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return None

        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)

        return DataStructs.TanimotoSimilarity(fp1, fp2)

    except Exception:
        return None


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


def simple_ai_summary(text):
    if not text:
        return "No abstract available."

    sentences = re.split(r"\. |\n", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return "No abstract available."

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

    if summary and not summary.endswith("."):
        summary += "."

    return summary


st.markdown("""
<h1 style="text-align:center;">🧪 ChemQuest AI</h1>
<p style="text-align:center;color:gray;">
PubChem • ChEMBL • ChemBERTa • PubChem Similarity • ChEMBL Similarity • RDKit Tanimoto
</p>
<hr>
""", unsafe_allow_html=True)

st.sidebar.title("ChemQuest AI")

st.sidebar.write("Chemical AI model:")
st.sidebar.code(CHEM_MODEL_NAME)

if CHEMBERTA_AVAILABLE:
    st.sidebar.success("ChemBERTa loaded")
else:
    st.sidebar.warning("ChemBERTa not loaded")

if RDKIT_AVAILABLE:
    st.sidebar.success("RDKit available")
else:
    st.sidebar.warning("RDKit not available")

pubchem_threshold = st.sidebar.slider(
    "PubChem Similarity Threshold",
    min_value=70,
    max_value=99,
    value=90,
    step=1
)

chembl_threshold = st.sidebar.slider(
    "ChEMBL Similarity Threshold",
    min_value=70,
    max_value=100,
    value=85,
    step=1
)

max_similar = st.sidebar.slider(
    "Max Similar Compounds",
    min_value=5,
    max_value=25,
    value=10,
    step=5
)

compound = st.text_input(
    "Enter chemical name",
    placeholder="Example: aspirin, imatinib, metformin"
)

if st.button("🔍 Search") and compound:

    with st.spinner("Fetching PubChem data..."):
        basic = fetch_pubchem_basic(compound)

    if not basic:
        st.error("Compound not found in PubChem.")
        st.stop()

    props = fetch_pubchem_props(compound)

    with st.spinner("Generating ChemBERTa molecular representation..."):
        embedding, embedding_error = generate_chemberta_embedding(basic.get("SMILES"))

    with st.spinner("Searching similar chemicals from PubChem..."):
        pubchem_similar = fetch_pubchem_similar(
            basic["CID"],
            threshold=pubchem_threshold,
            max_results=max_similar
        )

    with st.spinner("Searching similar chemicals from ChEMBL..."):
        chembl_similar = fetch_chembl_similarity(
            basic.get("SMILES"),
            similarity=chembl_threshold,
            max_results=max_similar
        )

    with st.spinner("Finding ChEMBL target records..."):
        chembl_molecule = fetch_chembl_molecule_by_name(compound)
        chembl_id = chembl_molecule.get("chembl_id") if chembl_molecule else None
        targets = fetch_chembl_targets(chembl_id) if chembl_id else []

    with st.spinner("Fetching PubMed literature..."):
        refs = fetch_pubmed_refs(basic["CID"], limit=10)

    st.success("Chemical intelligence search completed.")

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
            if basic.get("Structure"):
                st.image(basic["Structure"], width=300)

        with col2:
            st.write(f"**PubChem CID:** {basic.get('CID')}")
            st.write(f"**Molecular Formula:** {basic.get('Formula')}")
            st.write(f"**Molecular Weight:** {basic.get('Weight')}")
            st.write(f"**XLogP:** {basic.get('XlogP')}")
            st.write(f"**Canonical SMILES:** `{basic.get('SMILES')}`")

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
        st.write(f"**Input SMILES:** `{basic.get('SMILES')}`")

        if embedding is not None:
            st.success("ChemBERTa embedding generated successfully.")
            st.write(f"**Embedding dimension:** {embedding.shape[0]}")

            emb_df = pd.DataFrame({
                "Index": list(range(min(20, len(embedding)))),
                "Embedding Value": embedding[:20]
            })

            st.dataframe(emb_df, use_container_width=True)

        else:
            st.error("Embedding could not be generated.")
            st.code(embedding_error)

    with tab4:
        st.subheader("Similarity Chemical Retrieval")

        st.markdown("### PubChem Similar Chemicals")

        if pubchem_similar:
            pubchem_rows = []

            for item in pubchem_similar:
                tanimoto = tanimoto_similarity(
                    basic.get("SMILES"),
                    item.get("SMILES")
                )

                pubchem_rows.append({
                    "Source": item.get("Source"),
                    "ID": item.get("ID"),
                    "Name": item.get("Name"),
                    "Formula": item.get("Formula"),
                    "MW": item.get("MW"),
                    "XLogP": item.get("XLogP"),
                    "RDKit Tanimoto": round(tanimoto, 4) if tanimoto is not None else "NA",
                    "SMILES": item.get("SMILES")
                })

            pubchem_df = pd.DataFrame(pubchem_rows)
            st.dataframe(pubchem_df, use_container_width=True)
        else:
            st.warning("No PubChem similar compounds found.")

        st.markdown("---")
        st.markdown("### ChEMBL Similar Chemicals")

        if chembl_similar:
            chembl_rows = []

            for item in chembl_similar:
                tanimoto = tanimoto_similarity(
                    basic.get("SMILES"),
                    item.get("SMILES")
                )

                chembl_rows.append({
                    "Source": item.get("Source"),
                    "ID": item.get("ID"),
                    "Name": item.get("Name"),
                    "Formula": item.get("Formula"),
                    "MW": item.get("MW"),
                    "XLogP": item.get("XLogP"),
                    "ChEMBL Similarity": item.get("ChEMBL Similarity"),
                    "RDKit Tanimoto": round(tanimoto, 4) if tanimoto is not None else "NA",
                    "SMILES": item.get("SMILES")
                })

            chembl_df = pd.DataFrame(chembl_rows)
            st.dataframe(chembl_df, use_container_width=True)
        else:
            st.warning("No ChEMBL similar compounds found.")

        st.markdown("---")
        st.markdown("### Combined Similar Chemical Retrieval")

        combined = []

        for item in pubchem_similar:
            tanimoto = tanimoto_similarity(
                basic.get("SMILES"),
                item.get("SMILES")
            )

            combined.append({
                "Source": "PubChem",
                "Database ID": item.get("ID"),
                "Name": item.get("Name"),
                "Formula": item.get("Formula"),
                "MW": item.get("MW"),
                "XLogP": item.get("XLogP"),
                "Database Similarity": f">= {pubchem_threshold}%",
                "RDKit Tanimoto": round(tanimoto, 4) if tanimoto is not None else "NA",
                "SMILES": item.get("SMILES")
            })

        for item in chembl_similar:
            tanimoto = tanimoto_similarity(
                basic.get("SMILES"),
                item.get("SMILES")
            )

            combined.append({
                "Source": "ChEMBL",
                "Database ID": item.get("ID"),
                "Name": item.get("Name"),
                "Formula": item.get("Formula"),
                "MW": item.get("MW"),
                "XLogP": item.get("XLogP"),
                "Database Similarity": item.get("ChEMBL Similarity"),
                "RDKit Tanimoto": round(tanimoto, 4) if tanimoto is not None else "NA",
                "SMILES": item.get("SMILES")
            })

        if combined:
            combined_df = pd.DataFrame(combined)
            st.dataframe(combined_df, use_container_width=True)

            csv = combined_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Combined Similarity CSV",
                data=csv,
                file_name=f"{compound}_PubChem_ChEMBL_similarity.csv",
                mime="text/csv"
            )
        else:
            st.warning("No combined similarity results available.")

    with tab5:
        st.subheader("Target Finding from ChEMBL")

        if chembl_id:
            st.write(f"**ChEMBL Molecule ID:** `{chembl_id}`")

        if targets:
            target_df = pd.DataFrame(targets)
            st.dataframe(target_df, use_container_width=True)

            target_csv = target_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download ChEMBL Targets CSV",
                data=target_csv,
                file_name=f"{compound}_ChEMBL_targets.csv",
                mime="text/csv"
            )
        else:
            st.warning("No ChEMBL target activity records found.")

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
ChemQuest AI • PubChem Similarity • ChEMBL Similarity • RDKit Tanimoto • Target Finding
</p>
""", unsafe_allow_html=True)
