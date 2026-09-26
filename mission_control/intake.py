"""Offline-first compound intake with an explicitly enabled PubChem fallback."""

from __future__ import annotations

import json
import math
from types import MappingProxyType
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

from .chemistry import molecular_identity


TARGET = "DRD2"
INSTALLED_ACTIVITY_TARGETS = frozenset({TARGET})
DEFAULT_TIMEOUT = 5.0


class CompoundResolutionError(ValueError):
    """Raised when a compound query cannot be resolved."""


class PubChemLookupError(CompoundResolutionError):
    """Raised when an explicitly requested PubChem lookup fails."""


_CATALOG_DATA = {
    "Haloperidol": {
        "smiles": (
            "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1"
        ),
        "therapeutic_intent": "Antipsychotic DRD2 antagonism",
        "seed_rationale": (
            "Established DRD2 antagonist and a useful reference "
            "seed with visible CNS and safety trade-offs."
        ),
        "optimization_hypothesis": (
            "Retain modeled DRD2 activity while reducing lipophilicity and "
            "hERG liability and preserving brain exposure."
        ),
    },
    "Sulpiride": {
        "smiles": "CCN1CCCC1CNC(=O)c1cc(S(N)(=O)=O)ccc1OC",
        "therapeutic_intent": "Selective DRD2/DRD3 antagonism",
        "seed_rationale": (
            "A chemically distinct benzamide antipsychotic that broadens the "
            "starting scaffold set."
        ),
        "optimization_hypothesis": (
            "Balance the polar sulfonamide/amide motif with permeability while "
            "maintaining DRD2 recognition."
        ),
    },
    "Clozapine": {
        "smiles": "CN1CCN(C2=Nc3ccccc3Nc3ccc(Cl)cc32)CC1",
        "therapeutic_intent": "Atypical antipsychotic DRD2 modulation",
        "seed_rationale": (
            "Clinically important atypical antipsychotic with a differentiated "
            "polypharmacology and tricyclic scaffold."
        ),
        "optimization_hypothesis": (
            "Preserve central activity while reducing metabolic and cardiac "
            "safety liabilities associated with the lipophilic scaffold."
        ),
    },
    "Olanzapine": {
        "smiles": "Cc1cc2c(s1)Nc1ccccc1N=C2N1CCN(C)CC1",
        "therapeutic_intent": "Atypical antipsychotic DRD2 modulation",
        "seed_rationale": (
            "A CNS-active thienobenzodiazepine reference with established DRD2 "
            "pharmacology."
        ),
        "optimization_hypothesis": (
            "Tune heteroatom placement and basicity to retain CNS exposure and "
            "DRD2 activity with a cleaner ADMET profile."
        ),
    },
    "Metoclopramide": {
        "smiles": "CCN(CC)CCNC(=O)c1cc(Cl)c(N)cc1OC",
        "therapeutic_intent": "Antiemetic DRD2 antagonism",
        "seed_rationale": (
            "A compact substituted benzamide DRD2 antagonist that offers a "
            "non-antipsychotic reference scaffold."
        ),
        "optimization_hypothesis": (
            "Constrain the flexible basic side chain to improve selectivity and "
            "metabolic stability while retaining DRD2 antagonism."
        ),
    },
}

OFFLINE_CATALOG = MappingProxyType(
    {
        name: MappingProxyType(dict(entry))
        for name, entry in _CATALOG_DATA.items()
    }
)
_CATALOG_BY_NAME = {name.casefold(): name for name in OFFLINE_CATALOG}
_CATALOG_BY_SMILES = {
    molecular_identity(entry["smiles"])["canonical_smiles"]: name
    for name, entry in OFFLINE_CATALOG.items()
}


def catalog_names() -> tuple[str, ...]:
    """Return the built-in compounds in stable display order."""

    return tuple(OFFLINE_CATALOG)


def _result(
    *,
    query: str,
    source: str,
    name: str,
    canonical_smiles: str,
    formula: str,
    molecular_weight: float,
    target: str,
    therapeutic_intent: str,
    seed_rationale: str,
    optimization_hypothesis: str,
    target_rationale_status: str,
    target_rationale: str,
) -> dict[str, str | float]:
    return {
        "query": query,
        "source": source,
        "name": name,
        "canonical_smiles": canonical_smiles,
        "formula": formula,
        "molecular_weight": float(molecular_weight),
        "target": target,
        "therapeutic_intent": therapeutic_intent,
        "seed_rationale": seed_rationale,
        "optimization_hypothesis": optimization_hypothesis,
        "target_rationale_status": target_rationale_status,
        "target_rationale": target_rationale,
    }


def _target_id(value: str | None, *, fallback: str = TARGET) -> str:
    if value is None:
        return fallback
    cleaned = " ".join(value.split())
    if not cleaned:
        raise CompoundResolutionError("Molecular target must be a non-empty string")
    if len(cleaned) > 64:
        raise CompoundResolutionError("Molecular target must be 64 characters or fewer")
    if cleaned.casefold() in {"d2", "drd2", "dopamine d2"}:
        return TARGET
    return cleaned


def _catalog_result(
    query: str,
    name: str,
    *,
    source: str,
    target: str | None = None,
) -> dict[str, str | float]:
    entry = OFFLINE_CATALOG[name]
    identity = molecular_identity(entry["smiles"])
    selected_target = _target_id(target)
    target_supported = selected_target == TARGET
    return _result(
        query=query,
        source=source,
        name=name,
        canonical_smiles=str(identity["canonical_smiles"]),
        formula=str(identity["formula"]),
        molecular_weight=float(identity["molecular_weight"]),
        target=selected_target,
        therapeutic_intent=(
            entry["therapeutic_intent"]
            if target_supported
            else f"User-defined {selected_target} lead assessment"
        ),
        seed_rationale=entry["seed_rationale"],
        optimization_hypothesis=(
            entry["optimization_hypothesis"]
            if target_supported
            else (
                f"Evaluate structure-led edits while preserving evidence for "
                f"{selected_target} activity and improving the ADMET profile."
            )
        ),
        target_rationale_status=(
            "supported_reference" if target_supported else "unverified"
        ),
        target_rationale=(
            f"{name} is a known DRD2-active reference in the installed catalog. "
            "DRD2 is the installed activity model for this build."
            if target_supported
            else (
                f"{selected_target} was supplied by the user. This catalog "
                f"record validates the {name} structure, but it does not "
                "establish target engagement or provide an installed activity model."
            )
        ),
    )


def _read_response(response: Any) -> bytes | str:
    if hasattr(response, "__enter__"):
        with response as stream:
            return stream.read()
    try:
        return response.read()
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


def lookup_pubchem(
    query: str,
    *,
    target: str | None = None,
    opener: Callable[..., Any] | Any | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, str | float]:
    """Resolve a name with PubChem PUG REST using an injectable stdlib opener."""

    if not isinstance(query, str) or not query.strip():
        raise PubChemLookupError("PubChem query must be a non-empty string")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or timeout <= 0
    ):
        raise ValueError("timeout must be a finite positive number")

    cleaned_query = query.strip()
    properties = (
        "Title,CanonicalSMILES,ConnectivitySMILES,"
        "MolecularFormula,MolecularWeight"
    )
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{quote(cleaned_query, safe='')}/property/{properties}/JSON"
    )
    open_function = opener or urlopen
    if not callable(open_function):
        open_function = getattr(open_function, "open", None)
    if not callable(open_function):
        raise TypeError("opener must be callable or provide an open() method")

    try:
        raw_response = _read_response(
            open_function(url, timeout=float(timeout))
        )
    except HTTPError as error:
        raise PubChemLookupError(
            f"PubChem lookup failed for {cleaned_query!r}: HTTP {error.code}"
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        reason = getattr(error, "reason", error)
        raise PubChemLookupError(
            f"PubChem lookup failed for {cleaned_query!r}: {reason}"
        ) from error

    try:
        if isinstance(raw_response, bytes):
            raw_response = raw_response.decode("utf-8")
        payload = json.loads(raw_response)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise PubChemLookupError(
            f"PubChem returned invalid JSON for {cleaned_query!r}"
        ) from error

    try:
        properties_payload = payload["PropertyTable"]["Properties"][0]
    except (KeyError, IndexError, TypeError) as error:
        raise PubChemLookupError(
            f"PubChem returned no compound properties for {cleaned_query!r}"
        ) from error

    remote_smiles = next(
        (
            properties_payload.get(key)
            for key in (
                "CanonicalSMILES",
                "ConnectivitySMILES",
                "IsomericSMILES",
                "SMILES",
            )
            if properties_payload.get(key)
        ),
        None,
    )
    try:
        identity = molecular_identity(remote_smiles)
    except (TypeError, ValueError) as error:
        raise PubChemLookupError(
            f"PubChem returned no valid SMILES for {cleaned_query!r}"
        ) from error

    formula = properties_payload.get("MolecularFormula")
    weight = properties_payload.get("MolecularWeight")
    try:
        numeric_weight = float(weight)
    except (TypeError, ValueError) as error:
        raise PubChemLookupError(
            f"PubChem returned an invalid molecular weight for {cleaned_query!r}"
        ) from error
    if not isinstance(formula, str) or not formula or not math.isfinite(numeric_weight):
        raise PubChemLookupError(
            f"PubChem returned incomplete properties for {cleaned_query!r}"
        )

    title = properties_payload.get("Title")
    name = title.strip() if isinstance(title, str) and title.strip() else cleaned_query
    selected_target = _target_id(target)
    return _result(
        query=query,
        source="pubchem",
        name=name,
        canonical_smiles=str(identity["canonical_smiles"]),
        formula=formula,
        molecular_weight=numeric_weight,
        target=selected_target,
        therapeutic_intent=f"{selected_target} lead assessment",
        seed_rationale=(
            "Explicitly requested external compound used as a user-selected "
            "optimization seed."
        ),
        optimization_hypothesis=(
            "Evaluate structure-led edits that improve the multi-property "
            f"profile while preserving evidence for {selected_target} activity."
        ),
        target_rationale_status="unverified",
        target_rationale=(
            f"{selected_target} was supplied by the user, but a PubChem identity "
            "record does not establish that this molecule engages the target."
        ),
    )


def resolve_compound(
    query: str,
    *,
    target: str | None = None,
    use_pubchem: bool = False,
    opener: Callable[..., Any] | Any | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, str | float]:
    """Resolve catalog names or SMILES locally, with opt-in PubChem lookup."""

    if not isinstance(query, str) or not query.strip():
        raise CompoundResolutionError("Compound query must be a non-empty string")

    cleaned_query = query.strip()
    selected_target = _target_id(target)
    catalog_name = _CATALOG_BY_NAME.get(cleaned_query.casefold())
    if catalog_name is not None:
        return _catalog_result(
            query,
            catalog_name,
            source="catalog",
            target=selected_target,
        )

    try:
        identity = molecular_identity(cleaned_query)
    except ValueError:
        identity = None

    if identity is not None:
        canonical_smiles = str(identity["canonical_smiles"])
        matching_name = _CATALOG_BY_SMILES.get(canonical_smiles)
        if matching_name is not None:
            return _catalog_result(
                query,
                matching_name,
                source="smiles",
                target=selected_target,
            )
        return _result(
            query=query,
            source="smiles",
            name="Custom molecule",
            canonical_smiles=canonical_smiles,
            formula=str(identity["formula"]),
            molecular_weight=float(identity["molecular_weight"]),
            target=selected_target,
            therapeutic_intent=f"{selected_target} lead assessment",
            seed_rationale="User-supplied valid molecular structure.",
            optimization_hypothesis=(
                f"Optimize the supplied scaffold's predicted {selected_target} "
                "activity and ADMET profile through measured structural edits."
            ),
            target_rationale_status="unverified",
            target_rationale=(
                f"The structure is valid and {selected_target} was supplied by "
                "the user, but that does not establish target engagement or "
                "model-domain coverage."
            ),
        )

    if use_pubchem:
        return lookup_pubchem(
            query,
            target=selected_target,
            opener=opener,
            timeout=timeout,
        )

    raise CompoundResolutionError(
        f"Unknown compound {cleaned_query!r}. Enter a valid SMILES, choose an "
        "offline catalog name, or explicitly enable PubChem lookup."
    )


# Compatibility-friendly descriptive aliases.
resolve_query = resolve_compound
resolve_input = resolve_compound
intake_molecule = resolve_compound


__all__ = [
    "CompoundResolutionError",
    "DEFAULT_TIMEOUT",
    "INSTALLED_ACTIVITY_TARGETS",
    "OFFLINE_CATALOG",
    "PubChemLookupError",
    "TARGET",
    "catalog_names",
    "intake_molecule",
    "lookup_pubchem",
    "resolve_compound",
    "resolve_input",
    "resolve_query",
]
