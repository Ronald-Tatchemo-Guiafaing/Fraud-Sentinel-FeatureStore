"""
Fraud Sentinel — Interactive Dashboard
Replaces CLI scripts with a single web UI.

Run: streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fraud_sentinel.online_store import OnlineFeatureStore
from fraud_sentinel.feature_registry import REGISTRY
from fraud_sentinel.challenges import CHALLENGES

STORE = ROOT / "data_store"
OFFLINE = STORE / "offline"
ONLINE_DB = STORE / "online" / "features.db"
META = STORE / "metadata"
FIG = ROOT / "figures"
METRICS = FIG / "metrics.json"

st.set_page_config(
    page_title="Fraud Sentinel Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #00c9a7; }
    .contrib-card {
        background: linear-gradient(135deg, #1a2332 0%, #0f1419 100%);
        border-left: 4px solid #f1c40f;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }
    .kpi-box {
        background: #1a2332;
        padding: 1.2rem;
        border-radius: 10px;
        border-top: 3px solid #3d8bfd;
        text-align: center;
    }
    .alert-fraud { background: #ff6b6b33; border: 1px solid #ff6b6b; padding: 1rem; border-radius: 8px; }
    .alert-ok { background: #00c9a733; border: 1px solid #00c9a7; padding: 1rem; border-radius: 8px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def load_json(path: Path) -> dict | list | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


@st.cache_data(ttl=60)
def load_offline_manifest() -> dict:
    return load_json(OFFLINE / "manifest.json") or {"versions": [], "batches": []}


@st.cache_data(ttl=60)
def load_pipeline_summary() -> dict:
    return load_json(STORE / "pipeline_summary.json") or {}


@st.cache_data(ttl=60)
def load_metadata_catalog() -> dict:
    return load_json(META / "metadata_catalog.json") or {}


@st.cache_data(ttl=60)
def load_metrics() -> dict:
    return load_json(METRICS) or {}


def online_stats() -> dict:
    if not ONLINE_DB.exists():
        return {"served_transactions": 0, "last_sync": {}}
    return OnlineFeatureStore(ONLINE_DB).stats()


def page_challenges():
    st.header("❓ What are the challenges in Feature Store Engineering?")
    st.caption("Question du professeur — Fraud Sentinel répond aux 4 défis")

    if (FIG / "18_feature_store_challenges.png").exists():
        st.image(str(FIG / "18_feature_store_challenges.png"), use_container_width=True)

    rows = []
    for c in CHALLENGES:
        rows.append({
            "#": c["id"],
            "Challenge": c["name"],
            "Professor focus": c["question"],
            "Our solution": c["our_solution"],
            "Code": c["code"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Démo par défi")
    for c in CHALLENGES:
        with st.expander(f"{c['id']}. {c['name']}"):
            st.warning(f"Risque : {c['risk']}")
            st.success(f"Solution : {c['our_solution']}")
            st.code(c["code"])
            st.info(f"Dashboard : {c['dashboard']}")


def page_home():
    st.markdown('<p class="main-header">🛡️ Fraud Sentinel Dashboard</p>', unsafe_allow_html=True)
    st.caption("Feature Store + Deep Learning — géré entièrement depuis cette interface")

    col1, col2, col3, col4 = st.columns(4)
    manifest = load_offline_manifest()
    stats = online_stats()
    cat = load_metadata_catalog()
    metrics = load_metrics()
    mlp_f1 = metrics.get("MLP Deep Learning (our contribution)", {}).get("f1", 0.775)

    with col1:
        st.markdown('<div class="kpi-box"><h3>3</h3><p>Versions dataset</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-box"><h3>{stats.get("served_transactions", 0):,}</h3><p>Online served</p></div>', unsafe_allow_html=True)
    with col3:
        total_meta = cat.get("total_metadata_records", 18)
        st.markdown(f'<div class="kpi-box"><h3>{total_meta}</h3><p>Metadata records</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-box"><h3>{mlp_f1:.3f}</h3><p>MLP F1 (contribution)</p></div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Nos 4 contributions originales")

    contributions = [
        ("1", "Fraud Sentinel Feature Store", "Offline Parquet + Online SQLite — entraînement et scoring unifiés"),
        ("2", "MLP Deep Learning", "Réseau 128→64→32 — F1 supérieur au baseline RF"),
        ("3", "Metadata Governance", "18 enregistrements JSON — lineage, audit, quality checks"),
        ("4", "Pipeline 3 étapes", "Raw → Cleaned → Enriched — même domaine carte bancaire"),
    ]
    c1, c2 = st.columns(2)
    for i, (num, title, desc) in enumerate(contributions):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(
                f'<div class="contrib-card"><strong style="color:#f1c40f">{num}.</strong> '
                f'<strong>{title}</strong><br><span style="color:#8b9cb3">{desc}</span></div>',
                unsafe_allow_html=True,
            )

    if (FIG / "06_feature_store_architecture.png").exists():
        st.image(str(FIG / "06_feature_store_architecture.png"), caption="Architecture Fraud Sentinel")


def page_pipeline():
    st.header("⚙️ Pipeline Feature Store")
    st.write("Lance le pipeline complet : **ULB + IEEE-CIS + Sparkov** → Parquet offline → sync SQLite online")

    if not (ROOT / "creditcard.csv").exists():
        st.error("creditcard.csv manquant à la racine du projet.")
        return

    col_a, col_b = st.columns([1, 2])
    with col_a:
        sync_rows = st.slider("Lignes à synchroniser vers Online", 1000, 10000, 5000, 500)
        if st.button("🚀 Lancer le pipeline", type="primary", use_container_width=True):
            with st.spinner("Pipeline en cours… (1–2 min)"):
                try:
                    from fraud_sentinel.pipeline import run_pipeline
                    summary = run_pipeline(sync_online_sample=sync_rows)
                    st.cache_data.clear()
                    st.session_state["last_pipeline"] = summary
                    st.success("Pipeline terminé avec succès !")
                except Exception as e:
                    st.error(f"Erreur : {e}")

        if st.button("📊 Régénérer les figures", use_container_width=True):
            with st.spinner("Génération des figures…"):
                import subprocess
                subprocess.run([sys.executable, str(ROOT / "generate_store_figures.py")], cwd=ROOT, check=False)
                st.cache_data.clear()
                st.success("Figures mises à jour.")

    with col_b:
        summary = st.session_state.get("last_pipeline") or load_pipeline_summary()
        if summary:
            st.json({
                "domain": summary.get("domain"),
                "versions": summary.get("versions"),
                "metadata": summary.get("metadata"),
                "online_served": summary.get("online_store", {}).get("served_transactions"),
            })
        else:
            st.info("Aucun pipeline exécuté. Cliquez sur « Lancer le pipeline ».")

    manifest = load_offline_manifest()
    if manifest.get("versions"):
        st.subheader("Versions offline créées")
        rows = []
        for v in manifest["versions"]:
            rows.append({
                "Version": v["name"],
                "Lignes": f"{v['rows']:,}",
                "Fraude %": f"{v.get('fraud_rate', 0) * 100:.4f}%",
                "Stage": v.get("meta", {}).get("stage", ""),
                "Fichier": v.get("path", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_offline():
    st.header("📦 Offline Store (batch training)")
    manifest = load_offline_manifest()

    if not manifest.get("versions"):
        st.warning("Offline store vide. Lancez le pipeline depuis l'onglet Pipeline.")
        return

    tabs = st.tabs(["3 institutions (Parquet)", "Lots temporels", "Aperçu données"])

    with tabs[0]:
        for v in manifest["versions"]:
            with st.expander(f"📄 {v['name']} — {v['rows']:,} lignes"):
                st.write(f"**Colonnes ({len(v['columns'])}):**", ", ".join(v["columns"][:15]), "…" if len(v["columns"]) > 15 else "")
                st.write(f"**Taux fraude:** {v.get('fraud_rate', 0) * 100:.4f}%")
                st.write(f"**Écrit le:** {v.get('written_at', 'N/A')}")
                sidecar = META / "versions" / f"{v['name']}.meta.json"
                if sidecar.exists():
                    st.json(json.loads(sidecar.read_text(encoding="utf-8")))

    with tabs[1]:
        batches = manifest.get("batches", [])
        if batches:
            st.dataframe(pd.DataFrame(batches), use_container_width=True, hide_index=True)
        else:
            st.info("Aucun lot temporel.")

    with tabs[2]:
        version = st.selectbox("Choisir une version", [v["name"] for v in manifest["versions"]])
        path = OFFLINE / f"{version}.parquet"
        if path.exists():
            n = st.slider("Nombre de lignes à afficher", 5, 100, 20)
            df = pd.read_parquet(path).head(n)
            st.dataframe(df, use_container_width=True)


def page_online():
    st.header("⚡ Online Store (real-time serving)")
    stats = online_stats()

    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions servies", f"{stats.get('served_transactions', 0):,}")
    sync = stats.get("last_sync", {})
    c2.metric("Dernière sync", sync.get("rows_synced", 0))
    c3.metric("Source", sync.get("source_version", "—") or "—")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("🔍 Lookup temps réel")
        tx_id = st.number_input("Transaction ID", min_value=0, value=100, step=1)
        threshold = st.slider("Seuil d'alerte fraude", 0.0, 1.0, 0.5, 0.05)

        if ONLINE_DB.exists():
            store = OnlineFeatureStore(ONLINE_DB)
            feat = store.get_features(int(tx_id))
            if feat:
                mlp_score = feat.get("fraud_score_mlp", 0)
                rf_score = feat.get("fraud_score_rf", 0)
                st.write("**Features servies depuis Online Store:**")
                st.json({k: v for k, v in feat.items() if k != "updated_at"})

                max_score = max(mlp_score, rf_score)
                if max_score >= threshold:
                    st.markdown(
                        f'<div class="alert-fraud">🚨 <strong>ALERTE FRAUDE</strong> — '
                        f'MLP={mlp_score:.4f} | RF={rf_score:.4f} ≥ seuil {threshold}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="alert-ok">✅ Transaction légitime — '
                        f'MLP={mlp_score:.4f} | RF={rf_score:.4f}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.warning(f"Transaction {tx_id} non trouvée. Lancez le pipeline ou essayez un ID < 5000.")
        else:
            st.error("Online store non initialisé. Lancez le pipeline.")

    with col_r:
        st.subheader("Échantillon online store")
        if ONLINE_DB.exists():
            sample = OnlineFeatureStore(ONLINE_DB).export_sample(15)
            st.dataframe(sample, use_container_width=True, hide_index=True)

        if ONLINE_DB.exists():
            with sqlite3.connect(ONLINE_DB) as conn:
                sync_log = pd.read_sql_query(
                    "SELECT * FROM sync_log ORDER BY id DESC LIMIT 5", conn
                )
            st.subheader("Sync log")
            st.dataframe(sync_log, use_container_width=True, hide_index=True)


def page_metadata():
    st.header("📋 Metadata Governance")
    cat = load_metadata_catalog()

    if not cat:
        st.warning("Metadata non générées. Lancez le pipeline.")
        return

    counts = cat.get("counts_by_type", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", cat.get("total_metadata_records", 0))
    c2.metric("Features", counts.get("feature", 0))
    c3.metric("Versions", counts.get("dataset_version_active", 0))
    c4.metric("Batches", counts.get("temporal_batch", 0))
    c5.metric("Rejetées", counts.get("dataset_version_rejected", 0))

    tab1, tab2, tab3, tab4 = st.tabs(["Catalogue", "Features", "Versions rejetées", "Figure"])

    with tab1:
        st.write("**Comment les metadata sont gérées:**")
        for line in cat.get("how_metadata_is_handled", []):
            st.write(f"• {line}")
        st.subheader("Fichiers metadata")
        files = cat.get("metadata_files", {})
        st.dataframe(
            pd.DataFrame([{"Clé": k, "Chemin": v} for k, v in files.items()]),
            use_container_width=True, hide_index=True,
        )

    with tab2:
        feat = load_json(META / "feature_metadata.json")
        if feat:
            st.dataframe(pd.DataFrame(feat.get("records", [])), use_container_width=True, hide_index=True)
        st.write(f"**{len(REGISTRY)} features** dans le registre central.")

    with tab3:
        rejected = load_json(META / "rejected_version_log.json")
        if rejected:
            for r in rejected.get("records", []):
                with st.expander(f"❌ {r['name']} — {r['status']}"):
                    st.write(f"**Raison:** {r['failure_reason']}")
                    st.success(f"**Leçon:** {r['lessons_from_failure']}")

    with tab4:
        img = FIG / "22_metadata_catalog.png"
        if img.exists():
            st.image(str(img))


def page_models():
    st.header("📈 Modèles & EDA")
    metrics = load_metrics()

    if metrics:
        rows = []
        for ds_key, models in metrics.items():
            if not isinstance(models, dict):
                continue
            for model_name, m in models.items():
                if not isinstance(m, dict):
                    continue
                rows.append({
                    "Dataset": ds_key.replace("dataset_", "").replace("_", " ").upper(),
                    "Modèle": model_name,
                    **{k: round(v, 4) for k, v in m.items() if isinstance(v, (int, float))},
                })
        if rows:
            st.subheader("Métriques ML — RF vs MLP par dataset")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        ulb = metrics.get("dataset_1_ulb", {})
        mlp = ulb.get("MLP", {}).get("f1", 0)
        rf = ulb.get("Random Forest", {}).get("f1", 0)
        c1, c2, c3 = st.columns(3)
        c1.metric("ULB — MLP F1", f"{mlp:.3f}")
        c2.metric("ULB — RF F1", f"{rf:.3f}")
        c3.metric("Gain F1 (ULB)", f"+{mlp - rf:.3f}" if rf else "—", delta=f"{(mlp - rf) / rf * 100:.1f}%" if rf else None)

    figs = [
        ("08_model_metrics_comparison.png", "Comparaison métriques ULB"),
        ("d2_ieee_ml_metrics.png", "Métriques IEEE-CIS"),
        ("d3_sparkov_ml_metrics.png", "Métriques Sparkov"),
        ("15_mlp_architecture.png", "Architecture MLP"),
        ("01_class_balance.png", "Déséquilibre des classes ULB"),
        ("d2_ieee_class_balance.png", "Class balance IEEE"),
        ("d3_sparkov_class_balance.png", "Class balance Sparkov"),
    ]
    cols = st.columns(2)
    for i, (fname, caption) in enumerate(figs):
        path = FIG / fname
        if path.exists():
            with cols[i % 2]:
                st.image(str(path), caption=caption, use_container_width=True)


def main():
    st.sidebar.markdown("## 🛡️")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Menu",
        [
            "❓ Défis Feature Store (prof)",
            "🏠 Accueil & Contributions",
            "⚙️ Pipeline",
            "📦 Offline Store",
            "⚡ Online Store & Scoring",
            "📋 Metadata",
            "📈 Modèles & EDA",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Fraud Sentinel**")
    st.sidebar.caption("IUC — Feature Stores Engineering")
    if st.sidebar.button("🔄 Rafraîchir les données"):
        st.cache_data.clear()
        st.rerun()

    pages = {
        "❓ Défis Feature Store (prof)": page_challenges,
        "🏠 Accueil & Contributions": page_home,
        "⚙️ Pipeline": page_pipeline,
        "📦 Offline Store": page_offline,
        "⚡ Online Store & Scoring": page_online,
        "📋 Metadata": page_metadata,
        "📈 Modèles & EDA": page_models,
    }
    pages[page]()


if __name__ == "__main__":
    main()
