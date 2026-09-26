"""Packaged interpretation context for the offline mission."""

EVIDENCE_LIBRARY = (
    {
        "id": "pubchem-pug-rest",
        "topic": "Molecule resolution",
        "title": "PubChem PUG REST specification",
        "url": "https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest",
        "summary": (
            "PubChem accepts compound names and SMILES and can return identity "
            "and property records. Remote results still require local structure "
            "validation."
        ),
        "product_effect": (
            "Lookup is explicit, optional, cached by the app, and never required "
            "for the judged offline replay."
        ),
    },
    {
        "id": "haloperidol-pubchem",
        "topic": "Seed pharmacology",
        "title": "PubChem compound summary: Haloperidol",
        "url": "https://pubchem.ncbi.nlm.nih.gov/compound/Haloperidol",
        "summary": (
            "PubChem identifies haloperidol as a dopamine D2 receptor "
            "antagonist and provides the reference structure used for identity "
            "resolution."
        ),
        "product_effect": (
            "The judged Mission brief can support why Haloperidol is a "
            "DRD2-relevant reference while keeping all candidate scores tied "
            "to the installed models."
        ),
    },
    {
        "id": "ncats-gpcr",
        "topic": "DRD2 pharmacology",
        "title": "NCATS Assay Guidance Manual: GPCR pharmacology",
        "url": "https://www.ncbi.nlm.nih.gov/books/NBK549462/",
        "summary": (
            "GPCR lead optimization needs mechanistic pharmacology. Binding, "
            "Gi/cAMP signaling, and beta-arrestin recruitment answer different "
            "questions."
        ),
        "product_effect": (
            "The DRD2 model score is treated as a ranking signal. The validation "
            "runway requests binding and functional confirmation."
        ),
    },
    {
        "id": "ich-herg",
        "topic": "Cardiac safety",
        "title": "ICH E14/S7B implementation questions and answers",
        "url": (
            "https://database.ich.org/sites/default/files/"
            "E14-S7B_QAs_Step4_2022_0221.pdf"
        ),
        "summary": (
            "hERG current assessment belongs in an integrated proarrhythmia "
            "framework. Patch-clamp controls, protocol quality, and verified "
            "concentrations affect interpretation."
        ),
        "product_effect": (
            "The fixed gate applies to a model score, not a measured blockade "
            "probability or IC50."
        ),
    },
    {
        "id": "fda-cardiac-protocol",
        "topic": "Cardiac safety",
        "title": "FDA recommended cardiac ion-channel voltage protocols",
        "url": "https://www.fda.gov/media/151418/download",
        "summary": (
            "Voltage protocol and assay controls are material to interpreting "
            "drug effects on cardiac ion channels."
        ),
        "product_effect": (
            "A concentration-response hERG patch-clamp assay is the first safety "
            "confirmation step."
        ),
    },
    {
        "id": "kinetic-solubility",
        "topic": "Solubility",
        "title": "Exploratory analysis of kinetic solubility measurements",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3236531/",
        "summary": (
            "Kinetic and thermodynamic solubility answer different questions. "
            "pH, units, assay conditions, and solid state affect the result."
        ),
        "product_effect": (
            "Predicted LogS ranks candidates. Defined-pH kinetic solubility is "
            "measured first, followed by equilibrium work on promising material."
        ),
    },
    {
        "id": "bbb-assays",
        "topic": "Blood-brain barrier",
        "title": "PAMPA-BBB QSAR development and validation",
        "url": (
            "https://www.frontiersin.org/journals/pharmacology/articles/"
            "10.3389/fphar.2023.1291246/full"
        ),
        "summary": (
            "PAMPA-BBB screens passive permeability but cannot assess active "
            "efflux. Bidirectional MDCK-MDR1 adds an efflux-sensitive readout."
        ),
        "product_effect": (
            "The BBB score is a rank-ordering signal, not proof of brain exposure."
        ),
    },
    {
        "id": "sa-score",
        "topic": "Synthetic accessibility",
        "title": "Estimation of synthetic accessibility score",
        "url": "https://link.springer.com/article/10.1186/1758-2946-1-8",
        "summary": (
            "The score combines fragment contributions with molecular-complexity "
            "penalties."
        ),
        "product_effect": (
            "SA ranks review priority. It is not a route, yield, cost estimate, "
            "or proof of feasibility."
        ),
    },
    {
        "id": "oecd-qsar",
        "topic": "Model scope",
        "title": "OECD guidance on validation of QSAR models",
        "url": (
            "https://www.oecd.org/en/publications/"
            "guidance-document-on-the-validation-of-quantitative-structure-"
            "activity-relationship-q-sar-models_9789264085442-en.html"
        ),
        "summary": (
            "Scientifically grounded QSAR use requires defined applicability and "
            "validation appropriate to the intended use."
        ),
        "product_effect": (
            "Seed similarity is labeled structural locality. It is not a true "
            "model-specific applicability domain."
        ),
    },
    {
        "id": "rdkit-etkdg",
        "topic": "3D conformer",
        "title": "RDKit distance-geometry API and ETKDG method",
        "url": "https://www.rdkit.org/docs/source/rdkit.Chem.rdDistGeom.html",
        "summary": (
            "ETKDG augments distance geometry with experimental torsion and "
            "chemical-knowledge preferences. Embedding can fail."
        ),
        "product_effect": (
            "The interface records the method and falls back to 2D. The view is "
            "never called a docked pose or measured structure."
        ),
    },
    {
        "id": "nist-xai",
        "topic": "Explainability",
        "title": "NISTIR 8312: Four Principles of Explainable AI",
        "url": "https://www.nist.gov/publications/four-principles-explainable-artificial-intelligence",
        "summary": (
            "Explanations should provide meaningful evidence, accurately reflect "
            "the process, and identify knowledge limits."
        ),
        "product_effect": (
            "The ledger exposes rankings, formulas, exclusions, thresholds, and "
            "validation checks. It does not invent hidden chain-of-thought."
        ),
    },
)
