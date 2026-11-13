"""Validation intrinsèque du pipeline OCA sur un cas d'application.

Contrôle indépendant des résultats. À la différence de
``validate_against_reference.py``, qui contrôle la fidélité octet aux références
historiques du CAS_TEST_00, ce module valide la **correction intrinsèque** des
résultats produits : chaque grandeur publiée est recalculée indépendamment, par
une implémentation naïve distincte de celle du pipeline, puis confrontée à la
sortie.

Familles de contrôles :

    1. Équivalence du comptage de dominance vectorisé et de la boucle naïve.
    2. Matrice de corrélation ordinale initiale : périmètre, dimensions, domaine
       de définition, antisymétrie et recalcul terme à terme.
    3. Graphe d'ordonnancement de l'étape 1 : exhaustivité des nœuds, équivalence
       stricte entre arêtes et franchissements du seuil, égalité des degrés et du
       dénombrement des cellules, cohérence des niveaux hiérarchiques.
    4. Régularisation : admissibilité et optimalité du coefficient κ.
    5. Indice de parcimonie β : recalcul et cohérence des statistiques publiées.
    6. Ressources de sortie : présence, non vacuité et pagination du rapport
       consolidé.
    7. Conventions rédactionnelles du rapport : absence de tiret cadratin ou de
       tiret employé comme marqueur de liste ou de rupture de paragraphe.
    8. Cohérence de l'étape 2 : le graphe final reflète exactement la matrice
       ordinale augmentée de la colonne enveloppe.
    9. Épuration des figures : absence de titre et de légende descriptive sur les
       matrices et les graphes de hiérarchie.
   10. Découpage adaptatif des tableaux LaTeX larges (V6) : propriétés de la
       partition des colonnes et restitution intégrale, dans l'ordre, des
       colonnes des tableaux 01 et 04.
   11. Arrondi dynamique et uniformité typographique (V6) : régimes de la règle
       de formatage, absence de troncature à zéro, et taille de caractères fixe
       et unique pour les trois tableaux de données de l'application.
   12. Composition sans dépassement de marge (V6) : chaque source LaTeX de
       données est recompilée et son journal dépouillé, la mise à l'échelle étant
       proscrite et la tenue dans la page reposant sur le seul découpage.
   13. Tri horizontal par dominance (V6) : les colonnes des tableaux 01, 04 et 07
       suivent la fréquence de dominance croissante, les colonnes de
       régularisation demeurent terminales, la teinte d'une zone est commune aux
       deux tableaux du livrable 04, et chaque cellule est confrontée aux données
       sources afin d'exclure tout décalage.
   14. Résolution des annotations de type (V6) : les annotations étant différées
       par ``from __future__ import annotations``, un nom de type absent des
       importations passe inaperçu à l'exécution. Le contrôle force leur
       évaluation sur chaque fonction publique des modules livrés.

Usage:
    python tests/validate_pipeline.py                 # exécute puis valide
    python tests/validate_pipeline.py --no-run        # valide l'existant
    python tests/validate_pipeline.py --config x.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

import latex_support  # noqa: E402
from post_processor import (  # noqa: E402
    DECIMAL_NUMBER_FORMAT,
    LATEX_PASS_TIMEOUT_S,
    SCIENTIFIC_NUMBER_FORMAT,
    STANDARD_NUMBER_FORMAT,
    column_number_format,
    first_latex_error,
    format_frame_for_latex,
    read_overfull_boxes,
    split_column_bounds,
)

TOLERANCE = 1e-12


# ===========================================================================
# Recalculs indépendants (implémentations naïves de référence)
# ===========================================================================
def naive_dominance_counts(df: pd.DataFrame) -> Dict[str, int]:
    """Compte les dominations par une boucle explicite ligne à ligne.

    Reproduit littéralement la formulation d'origine
    ``all(row[col] > row.drop(col))``, sans vectorisation, afin de servir de
    référence indépendante au comptage du pipeline.

    Args:
        df: DataFrame des variables comparées.

    Returns:
        Dictionnaire colonne vers nombre de dominations strictes.
    """
    counts = {col: 0 for col in df.columns}
    for _, row in df.iterrows():
        for col in df.columns:
            if all(row[col] > row[other] for other in df.columns if other != col):
                counts[col] += 1
    return counts


def naive_ordinal_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Recalcule la matrice ordinale par accumulation explicite.

    Args:
        df: DataFrame des variables comparées.

    Returns:
        Matrice des proportions, diagonale ``NaN``.
    """
    cols = list(df.columns)
    matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    total = len(df)
    for var_i in cols:
        for var_j in cols:
            if var_i == var_j:
                matrix.loc[var_i, var_j] = np.nan
            else:
                count = int(sum(1 for a, b in zip(df[var_j], df[var_i]) if a <= b))
                matrix.loc[var_i, var_j] = count / total
    return matrix.astype(float)


# ===========================================================================
# Contrôles
# ===========================================================================
def check_dominance_counting(df_merged: pd.DataFrame, zones: List[str],
                             random_state: int) -> List[str]:
    """Contrôle 1 : équivalence du comptage vectorisé et de la boucle naïve.

    Args:
        df_merged: DataFrame fusionné.
        zones: Zones critiques analysées.
        random_state: Graine de l'échantillonnage du pipeline.

    Returns:
        Liste des écarts constatés.
    """
    from oca_model import count_dominance

    df = df_merged[zones].sample(n=len(df_merged), random_state=random_state)
    fast = count_dominance(df)
    slow = naive_dominance_counts(df)
    issues = []
    for col in zones:
        if fast[col] != slow[col]:
            issues.append(
                f"comptage divergent pour '{col}' : vectorise={fast[col]}, naif={slow[col]}")
    if sum(slow.values()) > len(df):
        issues.append(
            f"somme des dominations ({sum(slow.values())}) superieure a l'effectif ({len(df)})")
    return issues


def check_initial_matrix(df_merged: pd.DataFrame, zones: List[str],
                         results: Dict[str, Any]) -> List[str]:
    """Contrôle 2 : matrice de corrélation ordinale initiale.

    Vérifie que la matrice publiée porte sur l'intégralité des zones critiques
    déclarées, qu'elle est antérieure à toute sélection, que ses valeurs sont des
    proportions, que la diagonale est indéfinie et que l'antisymétrie
    ``p(i,j) + p(j,i) >= 1`` est satisfaite.

    Args:
        df_merged: DataFrame fusionné.
        zones: Zones critiques déclarées.
        results: Bloc ``results`` de ``output.json``.

    Returns:
        Liste des écarts constatés.
    """
    issues: List[str] = []
    initial = results.get("initial_ordinal_state", {})
    published_zones = initial.get("critical_zones", [])
    if list(published_zones) != list(zones):
        issues.append(
            f"perimetre de la matrice initiale != zones critiques declarees : "
            f"{published_zones} vs {zones}")

    published = pd.DataFrame(initial.get("ordinal_correlation_matrix", {}))
    if published.empty:
        return issues + ["matrice ordinale initiale absente de output.json"]
    # output.json sérialise un dictionnaire de colonnes : on rétablit l'ordre.
    published = published.reindex(index=list(zones), columns=list(zones))

    expected = naive_ordinal_matrix(df_merged[list(zones)]).round(4)
    delta = (published - expected).abs()
    worst = float(np.nanmax(delta.values)) if delta.size else 0.0
    if worst > 1e-4 + TOLERANCE:
        issues.append(f"matrice initiale : ecart maximal au recalcul = {worst:.6f}")

    if not published.shape == (len(zones), len(zones)):
        issues.append(f"matrice initiale non carree : {published.shape}")
    if not bool(published.notna().values.sum() == len(zones) * (len(zones) - 1)):
        issues.append("diagonale de la matrice initiale non indefinie")
    values = published.values[~np.isnan(published.values)]
    if values.size and (values.min() < 0.0 or values.max() > 1.0):
        issues.append(
            f"proportions hors de [0, 1] : min={values.min()}, max={values.max()}")
    for i, var_i in enumerate(zones):
        for var_j in zones[i + 1:]:
            total = published.loc[var_i, var_j] + published.loc[var_j, var_i]
            if total < 1.0 - 1e-3:
                issues.append(
                    f"antisymetrie violee pour ({var_i}, {var_j}) : somme = {total:.4f}")
    return issues


def check_step1_graph(zones: List[str], results: Dict[str, Any],
                      threshold: float) -> List[str]:
    """Contrôle 3 : exhaustivité et correction du graphe de l'étape 1.

    Contrôle central de la révision V3 : toutes les zones critiques doivent
    figurer comme nœuds, y compris celles dépourvues d'arête franchissant le
    seuil. L'ensemble des arêtes doit coïncider exactement avec l'ensemble des
    couples dont la proportion atteint le seuil.

    Args:
        zones: Zones critiques déclarées.
        results: Bloc ``results`` de ``output.json``.
        threshold: Seuil du graphe de dominance.

    Returns:
        Liste des écarts constatés.
    """
    issues: List[str] = []
    initial = results.get("initial_ordinal_state", {})
    nodes = list(initial.get("nodes", []))

    missing = [z for z in zones if z not in nodes]
    if missing:
        issues.append(f"noeuds manquants au graphe de l'etape 1 : {missing}")
    extra = [n for n in nodes if n not in zones]
    if extra:
        issues.append(f"noeuds surnumeraires au graphe de l'etape 1 : {extra}")
    if initial.get("n_nodes") != len(zones):
        issues.append(
            f"n_nodes = {initial.get('n_nodes')} alors que {len(zones)} zones sont declarees")

    matrix = pd.DataFrame(initial.get("ordinal_correlation_matrix", {}))
    if matrix.empty:
        return issues + ["matrice initiale absente : arêtes non verifiables"]
    matrix = matrix.reindex(index=list(zones), columns=list(zones))

    sure_edges, possible_edges = _edge_bounds(matrix, zones, threshold)
    published_edges = {
        (rel["dominant"], rel["dominated"]) for rel in initial.get("relations", [])
    }
    if not (sure_edges <= published_edges <= possible_edges):
        issues.append(
            f"aretes incoherentes avec le seuil {threshold} : "
            f"attendues={sorted(sure_edges)} (a {ROUNDING_HALF_STEP:g} pres : "
            f"{sorted(possible_edges - sure_edges)}), publiees={sorted(published_edges)}")
    if initial.get("n_edges") != len(published_edges):
        issues.append(
            f"n_edges = {initial.get('n_edges')} pour {len(published_edges)} relation(s)")

    # Les variables isolées sont exactement celles n'apparaissant dans aucune arête.
    engaged = {node for edge in published_edges for node in edge}
    expected_isolated = [z for z in zones if z not in engaged]
    if list(initial.get("isolated_nodes", [])) != expected_isolated:
        issues.append(
            f"noeuds isoles : publies={initial.get('isolated_nodes')}, "
            f"attendus={expected_isolated}")

    # Cohérence des niveaux : toute arête doit progresser d'au moins un niveau.
    levels = initial.get("levels", {})
    for dominant, dominated in published_edges:
        if dominant in levels and dominated in levels:
            if levels[dominated] <= levels[dominant]:
                issues.append(
                    f"niveau non croissant sur l'arete {dominant} -> {dominated} : "
                    f"{levels[dominant]} -> {levels[dominated]}")

    # Egalite des degres et du denombrement des cellules au seuil : c'est le
    # controle que le lecteur exerce sur les figures, en comparant le nombre de
    # cellules encadrees d'une ligne au nombre de fleches emises par le noeud.
    issues += _degree_discrepancies(matrix, initial.get("degrees", {}), threshold, zones)
    return issues


#: Demi-pas d'arrondi des matrices publiées (4 décimales). Les arêtes sont décidées
#: sur les proportions non arrondies : une cellule publiée à moins d'un demi-pas du
#: seuil est ambiguë, l'une ou l'autre décision est admise.
ROUNDING_HALF_STEP = 5e-5


def _edge_bounds(matrix: pd.DataFrame, variables: List[str],
                 threshold: float) -> "tuple[set, set]":
    """Arêtes sûres et arêtes possibles d'après une matrice publiée arrondie."""
    sure, possible = set(), set()
    for var_i in variables:
        for var_j in variables:
            if var_i == var_j or pd.isna(matrix.loc[var_i, var_j]):
                continue
            value = float(matrix.loc[var_i, var_j])
            if value >= threshold + ROUNDING_HALF_STEP:
                sure.add((var_i, var_j))
            if value >= threshold - ROUNDING_HALF_STEP:
                possible.add((var_i, var_j))
    return sure, possible


def _degree_discrepancies(matrix: pd.DataFrame, degrees: Dict[str, Any],
                          threshold: float, variables: List[str]) -> List[str]:
    """Confronte les degrés publiés au dénombrement des cellules au seuil.

    Args:
        matrix: Matrice des proportions de dominance, réindexée sur ``variables``.
        degrees: Degrés publiés, par variable, sous les clés ``out`` et ``in``.
        threshold: Seuil de dominance.
        variables: Variables du graphe, dans l'ordre d'affichage.

    Returns:
        Liste des écarts constatés.
    """
    issues: List[str] = []
    if not degrees:
        return ["degres absents de output.json : coherence non verifiable"]
    for var in variables:
        row = matrix.loc[var].drop(labels=[var], errors="ignore").dropna()
        column = matrix[var].drop(labels=[var], errors="ignore").dropna()
        out_low = int((row >= threshold + ROUNDING_HALF_STEP).sum())
        out_high = int((row >= threshold - ROUNDING_HALF_STEP).sum())
        in_low = int((column >= threshold + ROUNDING_HALF_STEP).sum())
        in_high = int((column >= threshold - ROUNDING_HALF_STEP).sum())
        published = degrees.get(var, {})
        if not out_low <= int(published.get("out", -1)) <= out_high:
            issues.append(
                f"{var} : degre sortant publie {published.get('out')} pour "
                f"{out_low} cellule(s) de ligne au seuil {threshold}")
        if not in_low <= int(published.get("in", -1)) <= in_high:
            issues.append(
                f"{var} : degre entrant publie {published.get('in')} pour "
                f"{in_low} cellule(s) de colonne au seuil {threshold}")
    return issues


def check_step2_coherence(results: Dict[str, Any], threshold: float) -> List[str]:
    """Contrôle 8 : cohérence du graphe de l'étape 2 avec sa matrice.

    Le graphe final doit être l'image exacte de la matrice ordinale augmentée de la
    colonne enveloppe régularisée. Le contrôle est identique dans son principe à
    celui de l'étape 1, ce qui garantit que les deux étapes sont soumises à la même
    exigence de cohérence.

    Args:
        results: Bloc ``results`` de ``output.json``.
        threshold: Seuil du graphe de dominance.

    Returns:
        Liste des écarts constatés.
    """
    matrix = pd.DataFrame(results.get("ordinal_correlation_matrix", {}))
    if matrix.empty:
        return ["matrice ordinale de l'etape 2 absente de output.json"]
    variables = list(matrix.columns)
    matrix = matrix.reindex(index=variables, columns=variables)

    sure_edges, possible_edges = _edge_bounds(matrix, variables, threshold)
    published_edges = {
        (rel["dominant"], rel["dominated"]) for rel in results.get("ordinal_relations", [])
    }
    issues: List[str] = []
    if not (sure_edges <= published_edges <= possible_edges):
        issues.append(
            f"aretes de l'etape 2 incoherentes avec le seuil {threshold} : "
            f"attendues={sorted(sure_edges)} (a {ROUNDING_HALF_STEP:g} pres : "
            f"{sorted(possible_edges - sure_edges)}), publiees={sorted(published_edges)}")
    issues += _degree_discrepancies(
        matrix, results.get("ordinal_graph_degrees", {}), threshold, variables)
    return issues


def _normalise(text: str) -> str:
    """Réduit un texte à une forme ASCII minuscule, sans diacritiques.

    Permet de rechercher des locutions dans du texte extrait d'un PDF, où
    l'encodage des caractères accentués n'est pas garanti.

    Args:
        text: Texte brut.

    Returns:
        Texte replié en ASCII minuscule.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(stripped.lower().split())


# Locutions bannies des figures : titres et légendes descriptives retirés en V4.
_FORBIDDEN_FIGURE_TEXT = [
    "matrice de correlation",
    "graphe d'ordonnancement",
    "graphe de dominance",
    "ordre de loewner",
    "ordre de pareto",
    "variable engagee",
    "variable sans relation",
    "seuil de dominance",
    "variables representees",
    "zones critiques",
    "hierarchie brute",
    "role d'enveloppe",
]


def check_figures_free_of_titles(produced_dir: str, config: Any) -> List[str]:
    """Contrôle 9 : épuration des figures.

    Les matrices ordinales et les graphes de hiérarchie ne doivent porter ni titre
    ni légende descriptive : la légende du document appelant décrit la figure. Le
    contrôle extrait le texte de chaque PDF et y recherche les locutions retirées.
    Les étiquettes de nœuds, de graduations et d'axes ne sont pas concernées, aucune
    ne constituant une locution de prose.

    Args:
        produced_dir: Dossier de sortie.
        config: Configuration OCA.

    Returns:
        Liste des écarts constatés.
    """
    try:
        from PyPDF2 import PdfReader
    except Exception:  # pragma: no cover
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception:
            return ["PyPDF2 indisponible : epuration des figures non verifiable"]

    of = config.output_files
    figures = [of["corr_matrix"], of["step1_graph"],
               of["ordinal_corr_matrix"], of["ordinal_graph"]]
    issues: List[str] = []
    for name in figures:
        path = os.path.join(produced_dir, name)
        if not os.path.isfile(path):
            issues.append(f"figure absente : {name}")
            continue
        try:
            text = _normalise("\n".join(page.extract_text() or ""
                                        for page in PdfReader(path).pages))
        except Exception as exc:
            issues.append(f"{name} : extraction du texte impossible ({exc})")
            continue
        for needle in _FORBIDDEN_FIGURE_TEXT:
            if needle in text:
                issues.append(f"{name} : locution de titre ou de legende presente ('{needle}')")
    return issues


# Décoration d'en-tête du mode de coloration ``text`` : la teinte et la graisse
# habillent le nom de la variable. L'expression est réécrite ici plutôt
# qu'importée du module contrôlé : un harnais ne saurait emprunter à ce qu'il
# vérifie l'outil même de sa vérification, faute de quoi une décoration erronée
# et son analyse partageraient la même faute et se valideraient mutuellement.
_HEADER_DECORATION = re.compile(
    r"^\\textcolor\{[^{}]*\}\{\s*"
    r"(?:\\(?:sffamily|bfseries|rmfamily|ttfamily|itshape)\s*)*(.*)\}$",
    flags=re.S)


def _undecorate(label: str) -> str:
    """Retire d'un en-tête la teinte et la graisse du mode ``text``.

    Args:
        label: Cellule d'en-tête telle qu'elle figure dans le ``.tex``.

    Returns:
        Nom de la variable, décoration ôtée.
    """
    match = _HEADER_DECORATION.match(str(label).strip())
    return (match.group(1) if match else str(label)).strip()


def _tex_blocks(text: str) -> List[List[str]]:
    """Extrait les en-têtes de colonnes de chaque bloc tabulaire d'un ``.tex``.

    Les noms sont rendus à leur forme nue, la décoration de couleur du mode
    ``text`` étant retirée : l'ordre des colonnes doit se contrôler à l'identique
    quel que soit le mode de coloration retenu.

    Args:
        text: Contenu du fichier LaTeX.

    Returns:
        Liste, par bloc, des noms de colonnes de son en-tête (colonne d'index
        exclue).
    """
    blocks: List[List[str]] = []
    for body in re.findall(r"\\toprule\s*\n(.*?)\\\\", text, flags=re.S):
        cells = [_undecorate(cell) for cell in body.split("&")]
        # La première cellule est l'en-tête vide de la colonne d'index.
        blocks.append([cell for cell in cells[1:] if cell])
    return blocks


def _tex_tables(text: str) -> List[str]:
    """Découpe un document LaTeX en environnements ``table`` successifs.

    Un livrable peut porter plusieurs tableaux, chacun éventuellement scindé en
    blocs. Cette séparation permet de raisonner tableau par tableau plutôt que
    bloc par bloc.

    Args:
        text: Contenu du fichier LaTeX.

    Returns:
        Contenu de chaque environnement ``table``, dans l'ordre d'apparition.
    """
    return re.findall(r"\\begin\{table\}.*?\\end\{table\}", text, flags=re.S)


def _tex_table_columns(table_text: str) -> List[str]:
    """Colonnes d'un tableau, blocs concaténés dans l'ordre.

    Args:
        table_text: Contenu d'un environnement ``table``.

    Returns:
        Noms de colonnes, colonne d'index exclue.
    """
    return [column for block in _tex_blocks(table_text) for column in block]


def check_table_splitting(produced_dir: str, zones: List[str],
                          config: Any) -> List[str]:
    """Contrôle 10 : découpage adaptatif des tableaux LaTeX larges (V6).

    Trois propriétés sont vérifiées, sur la routine de découpage elle même puis
    sur les documents effectivement produits :

        1. la partition des colonnes est exhaustive, disjointe, ordonnée, et
           aucun bloc n'excède le seuil de lisibilité ;
        2. le tableau d'échantillonnage des zones critiques de l'étape 1 restitue,
           par concaténation de ses blocs, l'intégralité des zones critiques dans
           leur ordre de déclaration, et **aucune** variable de conception ;
        3. le tableau de hiérarchisation de l'étape 1 obéit à la même règle sur
           les zones critiques.

    Args:
        produced_dir: Dossier de sortie.
        zones: Zones critiques de l'étape 1, dans leur ordre de déclaration.
        config: Configuration OCA.

    Returns:
        Liste des écarts constatés.
    """
    issues: List[str] = []
    of = config.output_files
    threshold = int(config.latex_output.get("max_columns_per_block", 6))

    # a. Propriétés de la partition, sur un balayage large de configurations.
    for total in list(range(1, 40)) + [64, 100]:
        for maximum in (1, 2, 3, 5, 8, 13):
            bounds = split_column_bounds(total, maximum)
            widths = [high - low for low, high in bounds]
            if sum(widths) != total:
                issues.append(f"partition non exhaustive : {total} colonnes, seuil {maximum}")
            if any(width > maximum for width in widths):
                issues.append(f"bloc trop large : {total} colonnes, seuil {maximum} -> {widths}")
            if any(width < 1 for width in widths):
                issues.append(f"bloc vide : {total} colonnes, seuil {maximum} -> {widths}")
            if max(widths) - min(widths) > 1:
                issues.append(f"partition desequilibree : {total}, seuil {maximum} -> {widths}")
            reconstructed = [index for low, high in bounds for index in range(low, high)]
            if reconstructed != list(range(total)):
                issues.append(f"ordre des colonnes altere : {total} colonnes, seuil {maximum}")

    # b. Livrable 04 : deux tableaux portant les seules zones critiques.
    full_path = os.path.join(produced_dir, f"{of['step1_zones_sample']}.tex")
    if not os.path.isfile(full_path):
        issues.append(f"tableau d'echantillonnage des zones critiques absent : {full_path}")
    else:
        with open(full_path, "r", encoding="utf-8") as stream:
            text = stream.read()
        tables = _tex_tables(text)
        if len(tables) != 2:
            issues.append(
                f"livrable 04 : {len(tables)} tableau(x) au lieu de 2 "
                "(avant et apres hierarchisation)")
        expected_columns = [str(zone) for zone in zones]
        expected_blocks = len(split_column_bounds(len(expected_columns), threshold))
        features = [str(name) for name in config.feature_cols_init]
        for position, table in enumerate(tables, start=1):
            blocks = _tex_blocks(table)
            flattened = [column for block in blocks for column in block]
            # ``underscore`` rend les soulignés littéraux : la comparaison porte
            # sur les libellés tels qu'ils figurent dans le document. L'ordre
            # relève du contrôle 13, dédié au tri par dominance ; ici seule
            # l'exhaustivité du périmètre est en cause.
            if sorted(flattened) != sorted(expected_columns):
                issues.append(
                    f"livrable 04, tableau {position} : colonnes restituees "
                    f"{sorted(flattened)} au lieu de {sorted(expected_columns)}")
            intruders = [column for column in flattened if column in features]
            if intruders:
                issues.append(
                    f"livrable 04, tableau {position} : variables de conception "
                    f"presentes alors qu'elles doivent etre exclues -> {intruders}")
            if any(len(block) > threshold for block in blocks):
                issues.append(
                    f"livrable 04, tableau {position} : bloc excedant le seuil "
                    f"{threshold} -> {[len(block) for block in blocks]}")
            if len(blocks) != expected_blocks:
                issues.append(
                    f"livrable 04, tableau {position} : {len(blocks)} bloc(s) au "
                    f"lieu de {expected_blocks}")

    # c. Tableau 01 : même règle appliquée aux zones critiques classées.
    result_path = os.path.join(produced_dir, f"{of['step1_result']}.tex")
    if not os.path.isfile(result_path):
        issues.append(f"tableau de hierarchisation absent : {result_path}")
    else:
        with open(result_path, "r", encoding="utf-8") as stream:
            blocks = _tex_blocks(stream.read())
        if any(len(block) > threshold for block in blocks):
            issues.append(
                f"tableau 01 : bloc excedant le seuil {threshold} -> "
                f"{[len(block) for block in blocks]}")
        n_zones = len(blocks[0]) if len(blocks) == 1 else sum(len(b) for b in blocks)
        expected_blocks = len(split_column_bounds(n_zones, threshold))
        if len(blocks) != expected_blocks:
            issues.append(
                f"tableau 01 : {len(blocks)} bloc(s) au lieu de {expected_blocks}")
    return issues


def check_number_formatting(produced_dir: str, df_merged: pd.DataFrame,
                            zones: List[str], config: Any) -> List[str]:
    """Contrôle 11 : arrondi dynamique et uniformité typographique (V6).

    Deux propriétés sont vérifiées :

        1. la règle de choix du format réagit correctement à la distribution
           d'une colonne, sur des distributions synthétiques couvrant les trois
           régimes, et une colonne de faibles amplitudes n'est jamais tronquée à
           zéro ;
        2. les blocs découpés d'un tableau partagent une taille de caractères
           fixe et unique, sans mise à l'échelle proportionnelle résiduelle.

    Args:
        produced_dir: Dossier de sortie.
        df_merged: DataFrame fusionné de l'étape 1.
        zones: Zones critiques de l'étape 1.
        config: Configuration OCA.

    Returns:
        Liste des écarts constatés.
    """
    issues: List[str] = []

    # a. Régimes de la règle de décision, sur des distributions synthétiques.
    regimes = [
        ("grandes amplitudes", np.array([120.0, 350.0, 78.5, 1e3]), STANDARD_NUMBER_FORMAT),
        ("amplitudes unitaires", np.array([1.0, 2.5, 4.0]), STANDARD_NUMBER_FORMAT),
        ("faibles amplitudes", np.array([0.004, 0.012, 0.03, 0.08]), DECIMAL_NUMBER_FORMAT),
        ("tres faibles amplitudes", np.array([1e-5, 4e-6, 2e-5]), SCIENTIFIC_NUMBER_FORMAT),
        ("colonne nulle", np.array([0.0, 0.0, 0.0]), STANDARD_NUMBER_FORMAT),
        ("colonne non finie", np.array([np.nan, np.inf]), STANDARD_NUMBER_FORMAT),
    ]
    for label, sample, expected in regimes:
        obtained = column_number_format(sample)
        if obtained != expected:
            issues.append(
                f"regime '{label}' : format {obtained!r} au lieu de {expected!r}")

    # b. Aucune valeur non nulle ne doit être rendue par un zéro.
    small = pd.DataFrame({"faible": [3.2e-4, 7.5e-4, 1.1e-4],
                          "tres_faible": [4.0e-9, 9.0e-9, 2.0e-9],
                          "standard": [150.0, 220.0, 310.0]})
    rendered = format_frame_for_latex(small)
    for column in small.columns:
        for original, text in zip(small[column], rendered[column]):
            if original != 0.0 and float(text) == 0.0:
                issues.append(
                    f"colonne '{column}' : la valeur {original!r} est rendue par "
                    f"'{text}', soit une troncature a zero")

    # c. Le format retenu pour chaque zone critique correspond bien à sa
    #    distribution réelle dans le jeu de données traité.
    for zone in zones:
        if zone not in df_merged.columns:
            continue
        values = df_merged[zone].to_numpy(dtype=float)
        expected = column_number_format(values)
        rendered_zone = format_frame_for_latex(df_merged[[zone]].head(5))[zone]
        for original, text in zip(df_merged[zone].head(5), rendered_zone):
            if text != expected.format(original):
                issues.append(
                    f"zone '{zone}' : rendu '{text}' au lieu de "
                    f"'{expected.format(original)}'")
                break

    # d. Uniformité typographique des trois tableaux de données de l'application.
    #    Le contrôle porte sur les tableaux d'un seul tenant comme sur les tableaux
    #    découpés : la version 6 proscrit toute mise à l'échelle, quel que soit le
    #    nombre de colonnes.
    of = config.output_files
    observed_sizes = set()
    for key in ("step1_result", "step1_zones_sample", "criticalzones"):
        path = os.path.join(produced_dir, f"{of[key]}.tex")
        if not os.path.isfile(path):
            issues.append(f"tableau absent : {of[key]}.tex")
            continue
        with open(path, "r", encoding="utf-8") as stream:
            text = stream.read()
        widths = re.findall(r"\\resizebox\{([0-9.]+)\\textwidth\}", text)
        if widths:
            issues.append(
                f"{of[key]} : mise a l'echelle residuelle -> {widths}")
        if "\\Huge" in text or "\\Large" in text:
            issues.append(
                f"{of[key]} : commande d'agrandissement residuelle dans le tabulaire")
        sizes = set(re.findall(r"\\(footnotesize|small|scriptsize|tiny)\b", text))
        # ``\small`` figure dans la legende de tous les tableaux : il est neutralise
        # ici, seule la taille du corps tabulaire etant en cause.
        sizes.discard("small")
        if len(sizes) > 1:
            issues.append(
                f"{of[key]} : tailles de caracteres heterogenes -> {sorted(sizes)}")
        observed_sizes |= sizes
        if not re.search(r"\\begingroup\\(footnotesize|scriptsize|tiny)", text):
            issues.append(
                f"{of[key]} : aucune taille de caracteres fixe declaree en tete de groupe")
    if len(observed_sizes) > 1:
        issues.append(
            f"tailles de caracteres heterogenes entre tableaux -> {sorted(observed_sizes)}")
    return issues


def check_column_ordering(produced_dir: str, df_merged: pd.DataFrame,
                          results: Dict[str, Any], config: Any) -> List[str]:
    """Contrôle 13 : tri horizontal par dominance et absence de décalage (V6).

    Trois propriétés sont vérifiées :

        1. les colonnes des tableaux 01 et 07 sont disposées par fréquence de
           dominance croissante, telle qu'établie à l'étape 1 sur l'intégralité
           des observations ;
        2. le tableau de l'étape 2 maintient en fin de ligne ses deux colonnes de
           régularisation, l'enveloppe et l'indice de parcimonie ;
        3. **aucune valeur n'a été décalée** : chaque cellule du tableau 07 est
           confrontée à la valeur que porte réellement l'observation
           correspondante dans le jeu de données, colonne par colonne.

    Args:
        produced_dir: Dossier de sortie.
        df_merged: DataFrame fusionné de l'étape 1.
        results: Bloc ``results`` de ``output.json``.
        config: Configuration OCA.

    Returns:
        Liste des écarts constatés.
    """
    issues: List[str] = []
    of = config.output_files

    table = results.get("gpo_selection", {}).get("dominance_table", {})
    if not table:
        return ["table de dominance absente de output.json"]
    frequencies = {
        zone: float(str(entry.get("p", "0")).rstrip("%"))
        for zone, entry in table.items()
    }

    def matrix_scores(matrix: Dict[str, Any]) -> Dict[str, float]:
        """Score moyen de majoration issu d'une matrice ordinale serialisee.

        La serialisation pandas produit un dictionnaire de colonnes :
        ``matrix[j][i]`` vaut la proportion d'observations ou la variable ``i``
        majore la variable ``j``. Le score de ``i`` est donc la moyenne, sur les
        autres variables ``j``, de ``matrix[j][i]``.
        """
        variables = list(matrix)
        scores: Dict[str, float] = {}
        for variable in variables:
            values = [matrix[other][variable] for other in variables
                      if other != variable and matrix[other].get(variable) is not None]
            if values:
                scores[variable] = sum(values) / len(values)
        return scores

    step1_scores = matrix_scores(
        results.get("initial_ordinal_state", {}).get("ordinal_correlation_matrix", {}))
    step2_scores = matrix_scores(results.get("ordinal_correlation_matrix", {}))

    def ranked_prefix(columns: List[str]) -> List[float]:
        return [frequencies[column] for column in columns if column in frequencies]

    def scored_prefix(columns: List[str], scores: Dict[str, float]) -> List[float]:
        return [scores[column] for column in columns if column in scores]

    # a. Tableau 01 : la totalité des colonnes est classée.
    path = os.path.join(produced_dir, f"{of['step1_result']}.tex")
    if not os.path.isfile(path):
        issues.append(f"tableau 01 absent : {path}")
    else:
        with open(path, "r", encoding="utf-8") as stream:
            columns = _tex_table_columns(stream.read())
        observed = ranked_prefix(columns)
        if observed != sorted(observed):
            issues.append(
                f"tableau 01 : colonnes non triees par dominance croissante -> {observed}")
        if len(observed) != len(frequencies):
            issues.append(
                f"tableau 01 : {len(observed)} colonne(s) classee(s) pour "
                f"{len(frequencies)} zone(s) au tableau de dominance")

    # a bis. Livrable 04 : le premier tableau suit l'ordre de déclaration, le
    #        second l'ordre de dominance croissante, et la teinte d'une zone est
    #        la même dans les deux.
    path = os.path.join(produced_dir, f"{of['step1_zones_sample']}.tex")
    if not os.path.isfile(path):
        issues.append(f"livrable 04 absent : {path}")
    else:
        with open(path, "r", encoding="utf-8") as stream:
            text = stream.read()
        tables = _tex_tables(text)
        if len(tables) < 2:
            issues.append(f"livrable 04 : {len(tables)} tableau(x) au lieu de 2")
        else:
            # L'ordre de déclaration est celui des zones critiques **publiées**,
            # variables synthétiques déclarées comprises. Le livrable 04 les porte
            # explicitement, et ``config.critical_zones`` n'énumère que les zones
            # brutes : s'y référer faisait échouer le contrôle dès qu'une variable
            # synthétique était déclarée, alors que le livrable était conforme.
            declared = [str(zone) for zone in
                        results.get("initial_ordinal_state", {}).get(
                            "critical_zones", config.critical_zones)]
            before = _tex_table_columns(tables[0])
            after = _tex_table_columns(tables[1])
            if before != [zone for zone in declared if zone in before]:
                issues.append(
                    f"livrable 04, tableau 1 : ordre {before} au lieu de l'ordre "
                    f"de declaration {declared}")
            observed = scored_prefix(after, step1_scores)
            if observed != sorted(observed):
                issues.append(
                    f"livrable 04, tableau 2 : colonnes non triees par score de "
                    f"majoration croissant -> {observed}")
            if len(observed) != len(after):
                issues.append(
                    f"livrable 04, tableau 2 : {len(observed)} colonne(s) notee(s) "
                    f"pour {len(after)} colonne(s) affichee(s)")
            if sorted(before) != sorted(after):
                issues.append(
                    "livrable 04 : les deux tableaux ne portent pas les memes colonnes")
            # Cohérence des teintes : une zone conserve sa couleur d'un tableau à
            # l'autre, faute de quoi le repérage visuel serait trompeur. La teinte
            # est portée par le fond de colonne en mode ``background`` et par
            # l'en-tête en mode ``text`` ; le contrôle relève l'une ou l'autre
            # selon le mode déclaré, la propriété vérifiée étant la même.
            palette = dict(re.findall(
                r"\\definecolor\{(\w+)\}\{rgb\}\{([0-9.,]+)\}", text))
            tint_pattern = (
                r"\\textcolor\{(\w+)\}\{\s*\\sffamily\\bfseries"
                if str(config.color_style.get("apply_to", "background")).strip().lower()
                == "text" else r"\\columncolor\{(\w+)\}")
            for index, table in enumerate(tables[:2], start=1):
                columns = _tex_table_columns(table)
                assigned = re.findall(tint_pattern, table)
                if len(assigned) != len(columns):
                    issues.append(
                        f"livrable 04, tableau {index} : {len(assigned)} teinte(s) "
                        f"pour {len(columns)} colonne(s)")
                    continue
                mapping = dict(zip(columns, assigned))
                if index == 1:
                    reference_palette = mapping
                elif mapping != {k: v for k, v in reference_palette.items()
                                 if k in mapping}:
                    divergent = [z for z in mapping
                                 if reference_palette.get(z) != mapping[z]]
                    issues.append(
                        f"livrable 04 : teintes divergentes entre les deux tableaux "
                        f"-> {divergent}")
            unknown = [name for name in set(re.findall(tint_pattern, text))
                       if name not in palette]
            if unknown:
                issues.append(f"livrable 04 : teintes non definies -> {sorted(unknown)}")

    # b. Tableau 07 : zones classées, colonnes de régularisation en fin.
    path = os.path.join(produced_dir, f"{of['criticalzones']}.tex")
    if not os.path.isfile(path):
        return issues + [f"tableau 07 absent : {path}"]
    with open(path, "r", encoding="utf-8") as stream:
        text = stream.read()
    columns = [c for block in _tex_blocks(text) for c in block]
    if len(columns) < 3:
        return issues + [f"tableau 07 : {len(columns)} colonne(s), structure inattendue"]

    zone_columns, trailing = columns[:-2], columns[-2:]
    observed = scored_prefix(zone_columns, step2_scores)
    if observed != sorted(observed):
        issues.append(
            f"tableau 07 : zones non triees par score de majoration croissant, "
            f"selon la matrice ordinale de l'etape 2 -> {observed}")
    if not any("beta" in name.lower() for name in trailing):
        issues.append(
            f"tableau 07 : l'indice de parcimonie n'est pas en derniere colonne -> {trailing}")

    # c. Absence de décalage : chaque cellule est confrontée aux données sources.
    _, row_order, values = _parse_tex_values(text)
    sample_size = int(config.latex_output["sample_size"])
    effective = min(sample_size, len(df_merged))
    expected_rows = (df_merged.sample(n=effective, random_state=config.sample_random_state)
                     .reset_index(drop=True))
    checked = 0
    for position, label in enumerate(row_order):
        cells = values.get(label, [])
        for column, cell in zip(columns, cells):
            if column not in df_merged.columns or position >= len(expected_rows):
                continue
            reference = float(expected_rows[column].iloc[position])
            rendered = column_number_format(df_merged[column].to_numpy(dtype=float))
            if cell != rendered.format(reference):
                issues.append(
                    f"tableau 07 : decalage detecte ligne {label}, colonne '{column}' -> "
                    f"'{cell}' au lieu de '{rendered.format(reference)}'")
                break
            checked += 1
    if checked == 0:
        issues.append("tableau 07 : aucune cellule n'a pu etre confrontee aux donnees")
    return issues


def _parse_tex_values(text: str) -> "tuple[List[str], List[str], Dict[str, List[str]]]":
    """Reconstruit le tableau logique porté par un document LaTeX.

    Args:
        text: Contenu du document LaTeX.

    Returns:
        Triplet ``(colonnes, ordre des lignes, valeurs par ligne)``.
    """
    columns: List[str] = []
    row_order: List[str] = []
    values: Dict[str, List[str]] = {}
    for block in re.findall(r"\\toprule\s*\n(.*?)\\bottomrule", text, flags=re.S):
        lines = [line.strip() for line in block.strip().splitlines()
                 if line.strip() and line.strip() != r"\midrule"]
        if not lines:
            continue
        strip_end = lambda item: re.sub(r"\\\\\s*$", "", item).strip()
        header = [_undecorate(cell) for cell in strip_end(lines[0]).split("&")]
        columns.extend(header[1:])
        for line in lines[1:]:
            cells = [cell.strip() for cell in strip_end(line).split("&")]
            label = cells[0]
            if label not in values:
                values[label] = []
                row_order.append(label)
            values[label].extend(cells[1:])
    return columns, row_order, values


def _contrast_on_white(color: Tuple[float, float, float]) -> float:
    """Rapport de contraste d'une couleur sur fond blanc.

    Implémentation indépendante de celle du module contrôlé : un harnais ne
    saurait emprunter à ce qu'il vérifie l'outil même de sa vérification.

    Args:
        color: Composantes rouge, vert et bleu dans [0, 1], espace sRGB.

    Returns:
        Rapport de contraste, compris entre 1 et 21.
    """
    linear = []
    for value in color:
        component = min(1.0, max(0.0, float(value)))
        linear.append(component / 12.92 if component <= 0.03928
                      else ((component + 0.055) / 1.055) ** 2.4)
    luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    return 1.05 / (luminance + 0.05)


def _contrast_with_black(color: Tuple[float, float, float]) -> float:
    """Rapport de contraste entre du texte noir et un fond de cette couleur.

    Args:
        color: Composantes rouge, vert et bleu dans [0, 1], espace sRGB.

    Returns:
        Rapport de contraste, compris entre 1 et 21.
    """
    linear = []
    for value in color:
        component = min(1.0, max(0.0, float(value)))
        linear.append(component / 12.92 if component <= 0.03928
                      else ((component + 0.055) / 1.055) ** 2.4)
    luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    return (luminance + 0.05) / 0.05


def check_color_mode(produced_dir: str, config: Any, results: Dict[str, Any]) -> List[str]:
    """Contrôle 15 : cohérence du mode de coloration retenu (V6).

    Le paramètre ``color_style.apply_to`` décide de ce que la couleur habille.
    Les deux modes s'excluent : un même livrable ne saurait porter à la fois un
    aplat de colonne et un en-tête teinté, la redondance nuisant à la lecture et
    trahissant une propagation incomplète du paramètre. Quatre propriétés sont
    donc vérifiées sur les documents effectivement produits.

        1. En mode ``text``, aucune commande ``\\columncolor`` ne subsiste dans le
           livrable 04, et chaque colonne y porte un en-tête décoré.
        2. En mode ``background``, aucun en-tête décoré ne s'y trouve, et chaque
           colonne porte un aplat.
        3. Les deux tableaux du livrable 04 attribuent à une même zone la même
           teinte, condition du repérage de son déplacement.
        4. Les colonnes terminales du livrable 07 reçoivent la mise en évidence
           du mode courant, aplat sombre ou graisse, à l'exclusion de l'autre.

    Args:
        produced_dir: Dossier de sortie.
        config: Configuration OCA.

    Returns:
        Liste des écarts constatés.
    """
    issues: List[str] = []
    mode = str(config.color_style.get("apply_to", "background")).strip().lower()

    step1_path = os.path.join(
        produced_dir, f"{config.output_files['step1_zones_sample']}.tex")
    if not os.path.exists(step1_path):
        return [f"livrable d'echantillonnage absent : {step1_path}"]
    with open(step1_path, "r", encoding="utf-8") as handle:
        step1_text = handle.read()

    tables = _tex_tables(step1_text)
    if len(tables) != 2:
        issues.append(f"04 : {len(tables)} tableau(x) au lieu de 2")

    decorated = re.findall(r"\\textcolor\{(\w+)\}\{\s*\\sffamily\\bfseries\s*([^{}]*)\}",
                           step1_text)
    aplats = re.findall(r"\\columncolor\{(ocaZone\d+)\}", step1_text)
    # Le livrable porte deux tableaux décrivant les mêmes zones : chacune doit
    # donc être habillée deux fois, une par tableau.
    # Les zones critiques publiées, variables synthétiques déclarées comprises :
    # ce sont elles que le livrable 04 habille, et non les seules zones brutes de
    # la configuration.
    published = results.get("initial_ordinal_state", {}).get(
        "critical_zones", config.critical_zones)
    expected = 2 * len(published)

    if mode == "text":
        if aplats:
            issues.append(
                f"04 : {len(aplats)} aplat(s) de colonne en mode 'text'")
        if len(decorated) != expected:
            issues.append(
                f"04 : {len(decorated)} en-tete(s) decore(s) pour {expected} "
                "attendu(s) en mode 'text'")
    else:
        if decorated:
            issues.append(
                f"04 : {len(decorated)} en-tete(s) decore(s) en mode 'background'")
        if len(aplats) != expected:
            issues.append(
                f"04 : {len(aplats)} aplat(s) de colonne pour {expected} "
                "attendu(s) en mode 'background'")

    # Lisibilité effective des teintes déclarées. En mode ``text``, la teinte
    # n'orne pas le libellé : elle l'identifie. Une teinte trop claire sur une
    # page blanche prive donc le tableau de son repérage. Le calcul du contraste
    # est réécrit ici plutôt qu'importé, un harnais ne pouvant emprunter à ce
    # qu'il vérifie l'outil même de sa vérification.
    declared_colors = re.findall(
        r"\\definecolor\{(\w+)\}\{rgb\}\{([0-9.]+),([0-9.]+),([0-9.]+)\}", step1_text)
    for name, red, green, blue in declared_colors:
        tint = (float(red), float(green), float(blue))
        if mode == "text":
            # La teinte porte le libellé : elle doit se détacher du blanc.
            ratio, support = _contrast_on_white(tint), "sur blanc"
        else:
            # La teinte est un fond : c'est le texte noir qu'elle porte qui doit
            # s'en détacher. L'exigence est symétrique, la correction inverse.
            ratio, support = _contrast_with_black(tint), "avec le texte noir"
        if ratio < 4.5:
            issues.append(
                f"04 : teinte '{name}' de contraste {ratio:.2f} {support}, "
                "inferieur au seuil de lisibilite 4.5")
    if not declared_colors:
        issues.append("04 : aucune teinte declaree")

    # Cohérence entre la teinte des tableaux et celle des figures. Les deux
    # procèdent désormais d'une association unique, indexée sur le classement de
    # dominance de l'étape 1 : la teinte de rang k du livrable 04 doit donc être
    # celle de la zone de rang k du classement publié.
    matrix = results.get("initial_ordinal_state", {}).get(
        "ordinal_correlation_matrix", {})
    scores: Dict[str, float] = {}
    for variable in matrix:
        values = [matrix[other][variable] for other in matrix
                  if other != variable and matrix[other].get(variable) is not None]
        if values:
            scores[variable] = sum(values) / len(values)
    ranking = [zone for zone, _ in sorted(scores.items(), key=lambda item: item[1])]
    if ranking and len(tables) == 2:
        ordered = _tex_table_columns(tables[1])
        if ordered and ordered != [zone for zone in ranking if zone in ordered]:
            issues.append(
                f"04 : ordre des colonnes {ordered} distinct du classement de "
                f"dominance {ranking}")
        # La teinte de rang k doit revenir à la zone de rang k : c'est ce qui
        # garantit qu'une zone porte dans les tableaux la teinte même qu'elle
        # porte dans les figures, toutes deux indexées sur ce classement.
        if mode == "text":
            assigned = re.findall(
                r"\\textcolor\{(\w+)\}\{\s*\\sffamily\\bfseries\s*([^{}]*)\}", tables[1])
            observed = [(name.strip(), color) for color, name in assigned]
        else:
            observed = list(zip(_tex_table_columns(tables[1]),
                                re.findall(r"\\columncolor\{(ocaZone\d+)\}", tables[1])))
        for zone, tint in observed:
            if zone in ranking:
                expected_tint = f"ocaZone{ranking.index(zone)}"
                if tint != expected_tint:
                    issues.append(
                        f"04 : zone '{zone}' de rang {ranking.index(zone)} porte "
                        f"la teinte '{tint}' au lieu de '{expected_tint}'")

    # Propriété 3 : l'association zone vers teinte est commune aux deux tableaux.
    if len(tables) == 2:
        assignments: List[Dict[str, str]] = []
        for table in tables:
            if mode == "text":
                pairs = re.findall(
                    r"\\textcolor\{(\w+)\}\{\s*\\sffamily\\bfseries\s*([^{}]*)\}", table)
                assignments.append({name.strip(): color for color, name in pairs})
            else:
                columns = _tex_table_columns(table)
                colors = re.findall(r"\\columncolor\{(ocaZone\d+)\}", table)
                assignments.append(dict(zip(columns, colors)))
        if assignments[0] and assignments[0] != assignments[1]:
            divergent = sorted(
                zone for zone in set(assignments[0]) | set(assignments[1])
                if assignments[0].get(zone) != assignments[1].get(zone))
            issues.append(
                "04 : teintes divergentes entre les deux tableaux pour "
                f"{divergent}")

    # Propriété 4 : mise en évidence des colonnes terminales du livrable 07.
    cz_path = os.path.join(
        produced_dir, f"{config.output_files['criticalzones']}.tex")
    if os.path.exists(cz_path):
        with open(cz_path, "r", encoding="utf-8") as handle:
            cz_text = handle.read()
        dark = cz_text.count(r"\columncolor{black!50!white}")
        bold = cz_text.count(r">{\bfseries}")
        if mode == "text" and (dark or not bold):
            issues.append(
                f"07 : mise en evidence incoherente en mode 'text' "
                f"({dark} aplat(s), {bold} graisse(s))")
        if mode == "background" and (bold or not dark):
            issues.append(
                f"07 : mise en evidence incoherente en mode 'background' "
                f"({dark} aplat(s), {bold} graisse(s))")
    else:
        issues.append(f"livrable des zones critiques absent : {cz_path}")

    return issues


def check_configuration_schema(config_path: str, config: Any, produced_dir: str,
                               results: Dict[str, Any]) -> List[str]:
    """Contrôle 16 : schéma de configuration de la version 7.

    Trois propriétés sont vérifiées.

        1. **Filtrage des clés d'auto-documentation.** Aucune clé suffixée
           ``__options`` ne doit subsister dans la configuration transmise aux
           modules de calcul, ni dans les hyperparamètres consignés au fichier de
           résultats. Ces clés énumèrent les valeurs admises d'un paramètre à
           l'intention du lecteur du fichier ; les laisser passer les exposerait à
           être confondues avec des paramètres.
        2. **Dissociation des cartes de couleurs, et exactitude de sa
           traçabilité.** Le style normalisé doit porter les deux clés
           ``color_map_matrix`` et ``color_map_critical_zones``. La carte
           **effectivement employée** au fond des matrices doit être une rampe
           continue, le fichier de résultats doit la consigner, et le journal doit
           porter l'avertissement de repli lorsqu'elle diffère de la carte
           déclarée. Le choix d'une carte catégorielle au fond d'une matrice n'est
           pas contrôlé ici : ce n'est pas une incorrection du calcul, mais un
           choix de configuration que le pipeline rattrape et signale au journal.
        3. **Rétrocompatibilité.** Une configuration écrite selon le schéma des
           versions antérieures, à carte unique, doit être lue sans erreur, la
           carte alimentant alors les deux usages.

    Args:
        config_path: Chemin du fichier de configuration lu.
        config: Configuration chargée.
        produced_dir: Dossier de sortie.
        results: Résultats publiés du sous-dossier contrôlé, dont les
            hyperparamètres consignés et le journal d'exécution.

    Returns:
        Liste des écarts constatés.
    """
    import json as _json

    from config_manager import (OPTION_METADATA_SUFFIX, _repair_json_text,
                                strip_json_comments, strip_option_metadata)
    from post_processor import (color_style, is_categorical_color_map,
                                matrix_color_map)

    issues: List[str] = []

    # Le contrôle porte partout sur les CLÉS et non sur le texte sérialisé. Une
    # recherche de sous-chaîne rendait un faux positif dès qu'une valeur portait le
    # suffixe, ainsi d'un nom de fichier de sortie, et n'établissait rien du
    # filtrage. La comparaison est insensible à la casse, comme celle du filtre.
    def stray_keys(node: Any, path: str = "") -> List[str]:
        found: List[str] = []
        if isinstance(node, dict):
            for key, value in node.items():
                location = f"{path}.{key}" if path else str(key)
                if str(key).lower().endswith(OPTION_METADATA_SUFFIX):
                    found.append(location)
                found.extend(stray_keys(value, location))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                found.extend(stray_keys(value, f"{path}[{index}]"))
        return found

    # Propriété 1 : aucune clé d'aide dans la configuration transmise aux modules.
    stray = stray_keys(config.raw)
    if stray:
        issues.append(
            f"cle(s) d'aide non filtree(s) dans la configuration transmise aux "
            f"modules de calcul : {stray}")

    # Le fichier de résultats est examiné dans sa **structure**. Le journal
    # d'exécution y est reproduit et nomme les clés d'aide écartées : c'est une
    # traçabilité voulue, et non une fuite. Ce qui serait fautif, c'est qu'une de
    # ces clés y figure comme clé d'un objet, où elle passerait pour un paramètre.
    results_path = os.path.join(produced_dir, "output.json")
    if os.path.isfile(results_path):
        with open(results_path, "r", encoding="utf-8") as handle:
            document = _json.load(handle)
        leaked = [key for key in stray_keys(document)
                  if not key.startswith("execution_log")]
        if leaked:
            issues.append(f"cle(s) d'aide reportee(s) dans output.json : {leaked}")

    # Dénombrement croisé : autant de clés relevées à la lecture que le fichier n'en
    # porte. La présence de telles clés n'est PAS exigée : elles sont une faculté
    # offerte à l'utilisateur, non un élément obligatoire du schéma. L'exiger
    # faisait échouer le harnais sur toute configuration antérieure à la version 7,
    # et sur toute configuration dépourvue de bloc 'color_style'.
    #
    # La lecture passe par la réparation des écarts d'écriture, comme le fait
    # 'load_config'. Analyser le texte brut levait une exception sur un fichier
    # portant un commentaire, que le produit revendique pourtant d'accepter, et
    # cette exception privait les quinze autres familles de tout verdict.
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8-sig") as handle:
            source = handle.read()
        # Le fichier est relu par la même chaîne que ``load_config`` : retrait des
        # commentaires d'abord, réparation des anomalies de ponctuation ensuite.
        # Analyser le texte brut échouait sur un fichier commenté, écriture que
        # l'application admet délibérément depuis la version 7.
        document = None
        cleaned, _, _ = strip_json_comments(source)
        try:
            document = _json.loads(cleaned)
        except ValueError:
            repaired, applied = _repair_json_text(cleaned)
            if applied:
                try:
                    document = _json.loads(repaired)
                except ValueError:
                    document = None
        if document is None:
            issues.append(
                f"le fichier '{config_path}' n'a pu etre relu pour denombrer ses "
                "cles d'aide")
        else:
            _, removed = strip_option_metadata(document)
            if len(removed) != len(config.option_keys):
                issues.append(
                    f"{len(config.option_keys)} cle(s) d'aide relevee(s) a la "
                    f"lecture pour {len(removed)} presente(s) dans le fichier")

    # Propriété 2 : dissociation effective, et exactitude de sa traçabilité.
    #
    # Le contrôle a d'abord porté sur la carte retournée par le résolveur, ce qui
    # était tautologique puisque celui ci applique lui même le repli. Il a ensuite
    # porté sur la carte déclarée, ce qui faisait échouer le harnais sur toute
    # configuration de la version 6 déclarant ``tab10``, laquelle était le défaut
    # même de cette version, alors que le rendu produit est identique et que le
    # repli était déjà silencieux à l'époque. Faire échouer une livraison correcte
    # est le mode de défaillance le plus coûteux pour un outil de validation.
    #
    # Le choix d'une carte catégorielle au fond d'une matrice n'est pas une
    # incorrection du calcul : c'est un choix de configuration que le pipeline
    # rattrape et **signale au journal**, lequel est le canal qui convient. Le
    # harnais vérifie donc ici ce qui relève de lui : que la carte effectivement
    # employée est bien une rampe continue, que le fichier de résultats la consigne,
    # et que le journal porte l'avertissement de repli lorsqu'elle diffère de la
    # carte déclarée. Cette confrontation entre le relevé de traçabilité et le
    # journal n'est pas tautologique : elle échouerait si l'un des deux mentait.
    style = color_style(config.color_style)
    for key in ("color_map_matrix", "color_map_critical_zones"):
        if key not in style:
            issues.append(f"style normalise depourvu de '{key}'")

    recorded = (results.get("hyperparameters", {}) or {}).get("color_style", {}) or {}
    effective = recorded.get("color_map_matrix_effective")
    if not effective:
        issues.append(
            "le fichier de resultats ne consigne pas 'color_map_matrix_effective'")
    else:
        if is_categorical_color_map(effective):
            issues.append(
                f"carte employee au fond des matrices '{effective}' categorielle "
                "alors qu'une rampe continue est requise")
        if effective != matrix_color_map(config.color_style):
            issues.append(
                f"carte consignee '{effective}' distincte de la carte resolue "
                f"'{matrix_color_map(config.color_style)}'")
        declared = style.get("color_map_matrix", "")
        if declared and effective != declared:
            journal = " ".join(
                str(record) for record in
                (results.get("execution_log", {}) or {}).get("records", []) or [])
            if declared not in journal:
                issues.append(
                    f"la carte declaree '{declared}' a ete remplacee par "
                    f"'{effective}' sans que le journal ne le signale")

    # Propriété 3 : rétrocompatibilité du schéma à carte unique.
    legacy = color_style({"color_map": "tab10", "saturation": 1.0, "alpha": 1.0,
                          "apply_to": "background"})
    if legacy["color_map_critical_zones"] != "tab10":
        issues.append(
            "schema a carte unique : la palette des zones ne reprend pas "
            f"'tab10' mais '{legacy['color_map_critical_zones']}'")
    if legacy["color_map_matrix"] != "tab10":
        issues.append(
            "schema a carte unique : la carte des matrices ne reprend pas "
            f"'tab10' mais '{legacy['color_map_matrix']}'")

    return issues


def check_graph_framing(produced_dir: str, config: Any, results: Dict[str, Any],
                        df_merged: "pd.DataFrame") -> List[str]:
    """Contrôle 17 : cadrage des graphes orientés et intégrité des étiquettes (V7).

    Les deux graphes de dominance, celui de l'étape 1 et celui de l'étape 2, sont
    reconstruits à l'identique et l'emprise réellement rendue de chaque étiquette
    est confrontée aux limites des axes.

    Trois propriétés sont vérifiées.

        1. Aucune étiquette n'est soumise au découpage. ``draw_networkx_labels``
           l'active par défaut, et une étiquette dépassant les limites s'y trouvait
           **tronquée** : le libellé de la variable régularisée, le plus long de la
           figure, perdait son suffixe.
        2. Aucune étiquette n'excède les limites des axes, la marge de sécurité
           étant calculée sur l'emprise mesurée des textes et non sur la seule
           étendue des positions de nœuds.
        3. Le contenu est centré : les marges opposées sont égales à la tolérance
           d'arrondi près, sur les deux axes.

    Args:
        produced_dir: Dossier de sortie.
        config: Configuration OCA.
        results: Résultats publiés du sous-dossier contrôlé.
        df_merged: Jeu de données fusionné, colonnes synthétiques comprises.

    Returns:
        Liste des écarts constatés.
    """
    import matplotlib.pyplot as plt

    from oca_model import (build_dominance_graph, matrix_dominance_ranking,
                           ordinal_correlation_matrix,
                           ordinal_correlation_matrix_with_envelope)
    from post_processor import plot_ordinal_graph, zone_color_map

    issues: List[str] = []
    style = config.color_style
    threshold = float(config.ordinal_graph["threshold"])
    zones = list(results["initial_ordinal_state"]["critical_zones"])
    retained = list(results["updated_critical_zones"])
    target = results["target_zone"]["resolved"]
    kappa = float(results["metrics"]["kappa_optimal"])

    returned = ordinal_correlation_matrix(df_merged, zones)
    step1 = returned[0] if isinstance(returned, tuple) else returned
    colors = zone_color_map(matrix_dominance_ranking(step1), style)
    step2, variables, alpha_col = ordinal_correlation_matrix_with_envelope(
        df_merged, retained, target, kappa)
    if alpha_col not in colors and target in colors:
        colors[alpha_col] = colors[target]

    scratch = os.path.join(produced_dir, "_controle_cadrage.png")
    families = (("etape 1", build_dominance_graph(step1, zones, threshold)),
                ("etape 2", build_dominance_graph(step2, variables, threshold)))
    for label, graph_data in families:
        # La fermeture est neutralisée le temps du tracé : la figure est enregistrée
        # normalement, ce qui construit un rendeur véritable, puis demeure
        # disponible pour la mesure.
        original_close = plt.close
        plt.close = lambda *args, **kwargs: None
        try:
            plot_ordinal_graph(graph_data, scratch, style=style, colors=colors)
        finally:
            plt.close = original_close
        figure = plt.figure(plt.get_fignums()[-1])
        axes = figure.axes[0]
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        frame = axes.get_window_extent(renderer=renderer)
        texts = [item for item in axes.texts if item.get_text().strip()]
        for item in texts:
            if item.get_clip_on():
                issues.append(
                    f"graphe de l'{label} : etiquette '{item.get_text()[:24]}' "
                    "soumise au decoupage, elle serait tronquee")
            box = item.get_window_extent(renderer=renderer)
            if min(box.x0 - frame.x0, frame.x1 - box.x1,
                   box.y0 - frame.y0, frame.y1 - box.y1) < -0.5:
                issues.append(
                    f"graphe de l'{label} : etiquette '{item.get_text()[:24]}' "
                    "hors des limites des axes")

        # Centrage : il porte sur l'ensemble du contenu, étiquettes, nœuds et
        # arêtes confondus, et non sur les seules étiquettes. Mesurer ces dernières
        # isolément donnerait des marges dissymétriques alors que le contenu est
        # bien centré, les disques et les arcs s'étendant au delà des textes.
        boxes = []
        for artist in texts + list(axes.collections) + list(axes.patches):
            try:
                boxes.append(artist.get_window_extent(renderer=renderer))
            except Exception:  # pragma: no cover - artiste sans emprise mesurable
                continue
        if boxes:
            content_left = min(box.x0 for box in boxes) - frame.x0
            content_right = frame.x1 - max(box.x1 for box in boxes)
            content_bottom = min(box.y0 for box in boxes) - frame.y0
            content_top = frame.y1 - max(box.y1 for box in boxes)
            for axis_label, low, high in (("abscisse", content_left, content_right),
                                          ("ordonnee", content_bottom, content_top)):
                if abs(low - high) > 2.0:
                    issues.append(
                        f"graphe de l'{label} : contenu non centre en {axis_label}, "
                        f"marges de {low:.1f} et {high:.1f} pixels")
                if min(low, high) < 1.0:
                    issues.append(
                        f"graphe de l'{label} : marge de securite nulle en "
                        f"{axis_label}, {min(low, high):.1f} pixel")
        plt.close(figure)
    if os.path.isfile(scratch):
        os.remove(scratch)
    return issues


def check_type_annotations() -> List[str]:
    """Contrôle 14 : résolution des annotations de type des modules livrés.

    Les modules emploient ``from __future__ import annotations`` : les
    annotations y sont conservées sous forme de chaînes et ne sont donc pas
    évaluées à l'import. Un nom de type absent des importations, par exemple
    ``Sequence`` oublié dans la liste de ``typing``, passe alors inaperçu à
    l'exécution du pipeline, tout en cassant l'analyse statique et toute
    introspection recourant à ``typing.get_type_hints``.

    Ce contrôle force l'évaluation des annotations de chaque fonction publique et
    signale les noms non résolus.

    La cohérence de ``__all__`` est vérifiée dans le même mouvement. Un nom qui y
    subsiste après la suppression de la fonction correspondante ne se manifeste
    qu'à l'importation par astérisque, laquelle n'intervient nulle part dans le
    pipeline : la déclaration reste fautive sans que rien ne le signale, jusqu'à
    ce qu'un utilisateur écrive ``from post_processor import *`` et reçoive une
    ``AttributeError``.

    Returns:
        Liste des écarts constatés.
    """
    import importlib
    import inspect
    import typing

    issues: List[str] = []
    for module_name in ("post_processor", "oca_model", "config_manager", "main_OCA",
                        "latex_support"):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - module absent du périmètre
            issues.append(f"module '{module_name}' non importable : {exc}")
            continue
        for exported in getattr(module, "__all__", []):
            if not hasattr(module, exported):
                issues.append(
                    f"{module_name}.__all__ declare '{exported}', absent du module")
        for name, member in vars(module).items():
            if name.startswith("_") or not inspect.isfunction(member):
                continue
            if getattr(member, "__module__", None) != module_name:
                continue
            try:
                typing.get_type_hints(member)
            except Exception as exc:
                issues.append(
                    f"{module_name}.{name} : annotations non resolubles "
                    f"({type(exc).__name__}: {exc})")
    return issues


def check_latex_margins(produced_dir: str, config: Any) -> List[str]:
    """Contrôle 12 : composition des tableaux sans dépassement de marge (V6).

    La version 6 proscrivant toute mise à l'échelle, c'est le découpage en blocs
    qui doit garantir la tenue des tableaux dans la justification. Le contrôle
    recompile chaque source LaTeX de données et dépouille le journal de
    ``pdflatex`` à la recherche de débordements horizontaux.

    Le compilateur est celui que ``latex_output.compiler`` désigne, lancé avec le même
    délai borné et le même contrôle du code retour que le pipeline : une compilation
    en erreur est un écart, non un contrôle réussi. En l'absence de compilateur (ou
    avec ``"none"``), le contrôle est déclaré non applicable plutôt que défaillant :
    la compilation est une dépendance optionnelle.

    Args:
        produced_dir: Dossier de sortie.
        config: Configuration OCA.

    Returns:
        Liste des écarts constatés.
    """
    latex_support.configure(config.latex_compiler)
    compiler = latex_support.find_latex_compiler()
    if compiler is None:
        return []
    kind, compiler_path = compiler

    issues: List[str] = []
    of = config.output_files
    absolute_dir = os.path.abspath(produced_dir)
    for key in ("step1_result", "step1_zones_sample", "criticalzones"):
        name = of[key]
        source = os.path.join(absolute_dir, f"{name}.tex")
        if not os.path.isfile(source):
            issues.append(f"source LaTeX absente : {name}.tex")
            continue
        with tempfile.TemporaryDirectory() as workspace:
            shutil.copy(source, os.path.join(workspace, f"{name}.tex"))
            failure = ""
            for _ in range(2):
                try:
                    completed = subprocess.run(
                        latex_support.compile_command(kind, compiler_path, f"{name}.tex"),
                        check=False, cwd=workspace, timeout=LATEX_PASS_TIMEOUT_S,
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
                except subprocess.TimeoutExpired:
                    failure = f"delai de {LATEX_PASS_TIMEOUT_S} s depasse"
                    break
                except OSError as exc:
                    failure = f"compilateur non lance ({type(exc).__name__})"
                    break
                if completed.returncode != 0:
                    failure = f"code retour {completed.returncode}"
                    break
            log_path = os.path.join(workspace, f"{name}.log")
            if failure:
                detail = first_latex_error(log_path)
                issues.append(f"{name} : compilation en echec ({failure})"
                              + (f" : {detail}" if detail else ""))
                continue
            overfull = read_overfull_boxes(log_path)
        if overfull:
            issues.append(
                f"{name} : {len(overfull)} depassement(s) de marge horizontale "
                f"({', '.join(overfull)})")
    return issues


def check_kappa(df_merged: pd.DataFrame, results: Dict[str, Any],
                config: Any) -> List[str]:
    """Contrôle 4 : admissibilité et optimalité du coefficient κ.

    Args:
        df_merged: DataFrame fusionné.
        results: Bloc ``results`` de ``output.json``.
        config: Configuration OCA.

    Returns:
        Liste des écarts constatés.
    """
    issues: List[str] = []
    kappa = float(results["metrics"]["kappa_optimal"])
    found = bool(results["metrics"]["kappa_found"])
    cz = results["target_zone"]["resolved"]
    zones = list(results["updated_critical_zones"])
    reg = config.gpo_regularization
    threshold = float(reg["threshold"])

    if cz not in df_merged.columns:
        return [f"cible resolue '{cz}' absente des donnees"]
    if not (float(reg["kappa_min"]) <= kappa <= float(reg["kappa_max"])):
        issues.append(f"kappa = {kappa} hors de la plage exploree")

    cz_values = df_merged[cz].to_numpy(dtype=float)
    zone_values = df_merged[zones].to_numpy(dtype=float)

    def satisfies(value: float) -> bool:
        """Vérifie que value * cz majore chaque zone au seuil requis."""
        envelope = (value * cz_values)[:, None]
        counts = (envelope >= zone_values).sum(axis=0)
        return all(float(count) / len(cz_values) >= threshold for count in counts)

    kappa_min, step = float(reg["kappa_min"]), float(reg["kappa_step"])
    # Grille indépendante : le point précédent est celui de la grille par indice
    # ``kappa_min + i * pas``, non ``kappa - pas`` (qui peut différer d'un ulp).
    position = round((kappa - kappa_min) / step)
    if abs(kappa_min + position * step - kappa) > 1e-9 * max(1.0, abs(kappa)):
        issues.append(f"kappa = {kappa} n'appartient pas a la grille "
                      f"{kappa_min} + i * {step}")
    if found:
        if not satisfies(kappa):
            issues.append(f"kappa = {kappa} ne satisfait pas le seuil {threshold}")
        for index in range(position):
            earlier = kappa_min + index * step
            if satisfies(earlier):
                issues.append(
                    f"kappa non optimal : {earlier} satisfait deja le seuil {threshold}")
                break
    elif satisfies(kappa):
        issues.append("kappa_found = false alors que le seuil est satisfait")

    # Sélection automatique de la cible : fréquence de dominance maximale.
    if results["target_zone"]["auto_selected"]:
        table = results["gpo_selection"]["dominance_table"]
        freq = {col: float(str(vals["p"]).rstrip("%")) for col, vals in table.items()}
        best = max(freq, key=lambda k: freq[k])
        if cz != best:
            issues.append(
                f"cible automatique '{cz}' != colonne de p maximal '{best}' ({freq})")
    return issues


def check_parsimony(df_merged: pd.DataFrame, results: Dict[str, Any]) -> List[str]:
    """Contrôle 5 : recalcul de l'indice de parcimonie β.

    Args:
        df_merged: DataFrame fusionné.
        results: Bloc ``results`` de ``output.json``.

    Returns:
        Liste des écarts constatés.
    """
    issues: List[str] = []
    kappa = float(results["metrics"]["kappa_optimal"])
    cz = results["target_zone"]["resolved"]
    zones = list(results["updated_critical_zones"])

    envelope = kappa * df_merged[cz]
    ratios = pd.DataFrame(index=df_merged.index)
    for var in zones:
        denominator = df_merged[var]
        ratios[var] = np.where(denominator != 0, envelope / denominator, np.inf)
    beta = ratios.min(axis=1)

    metrics = results["metrics"]
    for label, expected in (("beta_max", float(beta.max())),
                            ("beta_mean", float(beta.mean()))):
        published = metrics.get(label)
        if not np.isfinite(expected):
            # Une valeur non finie est publiee null (JSON n'admet pas l'infini).
            if published is not None:
                issues.append(f"{label} publie = {published} alors que le recalcul "
                              "donne une valeur non finie (attendu : null)")
        elif published is None:
            issues.append(f"{label} publie null alors que le recalcul donne {expected:.4f}")
        elif abs(float(published) - round(expected, 2)) > 5e-3:
            issues.append(
                f"{label} publie = {published} alors que le recalcul donne {expected:.4f}")
    defined = bool(metrics.get("beta_defined", True))
    if defined and float(beta.min()) < 0:
        issues.append("indice de parcimonie negatif alors que beta est publie defini")
    if not defined and "beta_note" not in metrics:
        issues.append("beta_defined = false sans beta_note")
    if float(results["metrics"]["n_sample"]) != len(df_merged):
        issues.append(
            f"n_sample = {results['metrics']['n_sample']} pour {len(df_merged)} observations")
    return issues


def check_resources(produced_dir: str, results: Dict[str, Any],
                    config: Any) -> List[str]:
    """Contrôle 6 : présence, non vacuité et pagination des ressources.

    Args:
        produced_dir: Dossier de sortie.
        results: Bloc ``results`` de ``output.json``.
        config: Configuration OCA.

    Returns:
        Liste des écarts constatés.
    """
    issues: List[str] = []
    of = config.output_files
    report = results.get("report", {}) or {}
    # Sections facultatives : un PDF issu de LaTeX n'est exigé que si le rapport est
    # déclaré complet, c'est-à-dire si un compilateur était disponible.
    absent_ok = set(report.get("missing_sections", []))
    latex_pdfs = (f"{of['step1_result']}.pdf", f"{of['step1_zones_sample']}.pdf",
                  f"{of['criticalzones']}.pdf")
    expected = [
        of["consolidated_report"], f"{of['step1_result']}.pdf", of["corr_matrix"],
        of["step1_graph"], f"{of['step1_zones_sample']}.pdf",
        of["ordinal_corr_matrix"], of["ordinal_graph"],
        f"{of['criticalzones']}.pdf", of["boxplot"], of["hist"], of["scatter"],
        of["sort_txt"], of["step1_corr_matrix_txt"], of["step1_graph_txt"],
        of["ordinal_graph_txt"], of["log"],
    ]
    expected = [name for name in expected
                if not (name in latex_pdfs and name in absent_ok)]
    for name in expected:
        path = os.path.join(produced_dir, name)
        if not os.path.isfile(path):
            issues.append(f"ressource absente : {name}")
        elif os.path.getsize(path) == 0:
            issues.append(f"ressource vide : {name}")

    # Le rapport consolidé doit totaliser au moins la pagination de ses sections.
    try:
        from PyPDF2 import PdfReader
    except Exception:  # pragma: no cover
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception:
            return issues
    merged_path = os.path.join(produced_dir, of["consolidated_report"])
    if not os.path.isfile(merged_path):
        return issues
    merged_pages = len(PdfReader(merged_path).pages)
    sections = [
        f"{of['report_intro']}.pdf", f"{of['step1_result']}.pdf", of["corr_matrix"],
        of["step1_graph"], f"{of['step1_zones_sample']}.pdf",
        of["ordinal_corr_matrix"], of["ordinal_graph"],
        f"{of['criticalzones']}.pdf", of["boxplot"], of["hist"], of["scatter"],
    ]
    total = 0
    for name in sections:
        path = os.path.join(produced_dir, name)
        if os.path.isfile(path) and name not in absent_ok:
            total += len(PdfReader(path).pages)
    if report.get("status_page_included"):
        total += 1   # page de statut du rapport partiel, produite sans LaTeX
    if merged_pages != total:
        issues.append(
            f"rapport consolide : {merged_pages} page(s) pour {total} page(s) de sections")
    if not absent_ok and merged_pages < len(sections):
        issues.append(f"rapport consolide trop court : {merged_pages} page(s)")
    return issues


def check_report_style(produced_dir: str, config: Any) -> List[str]:
    """Contrôle 7 : conventions rédactionnelles du rapport consolidé.

    Proscrit le tiret cadratin, le tiret demi cadratin et le tiret employé comme
    marqueur de liste ou de rupture de paragraphe. Les tirets internes à un
    identifiant (nom de fichier numéroté, par exemple ``02-OCA_CorrMat``) sont
    licites et donc exclus du contrôle, de même que les options de paquets
    LaTeX.

    Args:
        produced_dir: Dossier de sortie.
        config: Configuration OCA.

    Returns:
        Liste des écarts constatés.
    """
    issues: List[str] = []
    path = os.path.join(produced_dir, f"{config.output_files['report_intro']}.tex")
    if not os.path.isfile(path):
        return [f"source LaTeX du rapport absente : {path}"]
    with open(path, "r", encoding="utf-8") as tex_stream:
        text = tex_stream.read()

    for label, pattern in (
        ("tiret cadratin", r"—"),
        ("tiret demi cadratin", r"–"),
        ("tiret cadratin LaTeX", r"---"),
        ("tiret demi cadratin LaTeX", r"(?<!-)--(?!-)"),
        ("tiret en marqueur de liste", r"(?m)^\s*[-*]\s"),
        ("tiret en rupture de paragraphe", r"\s-\s"),
    ):
        for match in re.finditer(pattern, text):
            line = text[:match.start()].count("\n") + 1
            extract = text[max(0, match.start() - 40):match.end() + 40].replace("\n", " ")
            issues.append(f"{label} ligne {line} : ...{extract}...")
    return issues[:10]


# ===========================================================================
# Programme principal
# ===========================================================================
def main() -> int:
    """Exécute le pipeline puis l'intégralité des contrôles V3.

    Returns:
        Code de sortie 0 si tous les contrôles réussissent, 1 sinon.
    """
    parser = argparse.ArgumentParser(
        description="Validation intrinseque du pipeline OCA sur un cas d'application.")
    parser.add_argument("--config", default="examples/CAS_TEST_00/input.json")
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    os.chdir(PROJECT_ROOT)
    from config_manager import load_config

    # Chargement initial protégé : une configuration illisible, un calcul en échec ou
    # des sorties absentes donnent un diagnostic d'une ligne, non une trace.
    try:
        if not args.no_run:
            print("[VALIDATION] Execution du pipeline OCA...")
            from main_OCA import run

            run(args.config)
        config = load_config(args.config)
        # En traitement par lot (V6), la validation porte sur le premier sous-dossier
        # déclaré : chaque exécution étant strictement isolée, contrôler l'une d'elles
        # de bout en bout suffit à valider la logique, les autres étant vérifiées par
        # la présence de leurs ressources et par le document agrégé.
        config = config.for_subdir(config.merged_subdirs[0])
        produced_dir = config.output_dir_main()

        with open(config.output_json, "r", encoding="utf-8") as handle:
            document = json.load(handle)
        if document.get("status") not in (None, "complete", "partial"):
            raise ValueError(f"le document de resultats est marque "
                             f"'{document.get('status')}' : {document.get('reason', '')}")
        if "results" in document:
            results = document["results"]
        else:
            # Document agrégé du traitement par lot : indexé par sous-dossier.
            results = document["runs"][config.merged_subdir]

        # Le classeur de traçabilité est toujours écrit en .xlsx, même si la source
        # déclarée dans data_files est un .csv.
        saved_name = config.data_files
        if os.path.splitext(saved_name)[1].lower() != ".xlsx":
            saved_name = os.path.splitext(saved_name)[0] + ".xlsx"
        merged_path = os.path.join(produced_dir, saved_name)
        df_merged = pd.read_excel(merged_path, index_col=0).reset_index(drop=True).astype(float)
        zones = list(results["initial_ordinal_state"]["critical_zones"])
        for expr in (config.user_defined_vars_cols if config.user_defined_vars else []):
            if expr not in df_merged.columns:
                df_merged[expr] = df_merged.eval(expr)
        target = results["target_zone"]["resolved"]
        if target not in df_merged.columns:
            df_merged[target] = df_merged.eval(target)
    except Exception as exc:  # noqa: BLE001 - diagnostic au lieu d'une trace
        print(f"[VALIDATION][ERREUR] chargement impossible ({type(exc).__name__}) : "
              f"{str(exc).splitlines()[0] if str(exc) else 'erreur inconnue'}\n"
              "Lancez d'abord le pipeline (sans --no-run) et verifiez la configuration.")
        return 2

    threshold = float(config.ordinal_graph["threshold"])

    def guarded(label: str, family: Any, *arguments: Any) -> List[str]:
        """Exécute une famille de contrôles en convertissant toute exception en écart.

        Une défaillance de l'outil de contrôle ne doit pas priver de verdict les
        familles restantes. Le cas s'est présenté : une famille analysait le
        fichier de configuration sans passer par la réparation des écarts
        d'écriture, et levait sur un fichier portant un commentaire, ce qui
        interrompait l'intégralité du harnais sans qu'aucun contrôle ne rende son
        verdict.

        Args:
            label: Nom de la famille, employé dans le message d'écart.
            family: Fonction de contrôle.
            *arguments: Arguments transmis à la fonction.

        Returns:
            Écarts relevés, ou l'exception convertie en écart unique.
        """
        try:
            return family(*arguments)
        except Exception as exc:  # noqa: BLE001 - un contrôle ne doit jamais rompre
            return [f"le controle a leve une exception : {type(exc).__name__}: {exc}"]

    checks = {
        "1. comptage de dominance (vectorise contre naif)":
            guarded("1", check_dominance_counting, df_merged, zones,
                    config.sample_random_state),
        "2. matrice de correlation ordinale initiale (pre selection)":
            guarded("2", check_initial_matrix, df_merged, zones, results),
        "3. graphe d'ordonnancement de l'etape 1 (exhaustivite des noeuds)":
            guarded("3", check_step1_graph, zones, results, threshold),
        "4. regularisation kappa (admissibilite et optimalite)":
            guarded("4", check_kappa, df_merged, results, config),
        "5. indice de parcimonie beta":
            guarded("5", check_parsimony, df_merged, results),
        "6. ressources de sortie et pagination du rapport":
            guarded("6", check_resources, produced_dir, results, config),
        "7. conventions redactionnelles du rapport":
            guarded("7", check_report_style, produced_dir, config),
        "8. coherence matrice et graphe de l'etape 2":
            guarded("8", check_step2_coherence, results, threshold),
        "9. epuration des figures (ni titre ni legende)":
            guarded("9", check_figures_free_of_titles, produced_dir, config),
        "10. decoupage adaptatif des tableaux LaTeX larges":
            guarded("10", check_table_splitting, produced_dir, zones, config),
        "11. arrondi dynamique et uniformite typographique":
            guarded("11", check_number_formatting, produced_dir, df_merged, zones,
                    config),
        "12. composition sans depassement de marge":
            guarded("12", check_latex_margins, produced_dir, config),
        "13. tri horizontal par dominance et absence de decalage":
            guarded("13", check_column_ordering, produced_dir, df_merged, results,
                    config),
        "14. resolution des annotations de type":
            guarded("14", check_type_annotations),
        "15. coherence du mode de coloration":
            guarded("15", check_color_mode, produced_dir, config, results),
        "16. schema de configuration V7 (cles d'aide et cartes dissociees)":
            guarded("16", check_configuration_schema, args.config, config,
                    produced_dir, results),
        "17. cadrage des graphes et integrite des etiquettes":
            guarded("17", check_graph_framing, produced_dir, config, results,
                    df_merged),
    }

    print("\n" + "=" * 78)
    print("VALIDATION OCA : correction intrinseque des resultats")
    print(f"Configuration  : {args.config}")
    print(f"Jeu de donnees : {config.merged_subdir}/{config.data_files}")
    print(f"Zones critiques : {len(zones)} ; observations : {len(df_merged)}")
    print("=" * 78)
    failures = 0
    for name, issues in checks.items():
        if issues:
            failures += 1
            print(f"[FAIL] {name} : {len(issues)} ecart(s)")
            for line in issues[:8]:
                print(f"       {line}")
        else:
            print(f"[ OK ] {name}")
    print("=" * 78)

    metrics = results["metrics"]
    initial = results["initial_ordinal_state"]
    places = max(2, int(metrics.get("kappa_decimals", 2)))
    print(f"kappa optimal          : {metrics['kappa_optimal']:.{places}f} "
          f"(seuil atteint : {metrics['kappa_found']})")

    def shown_beta(value: Any) -> str:
        """Beta publie ou « non defini » pour la valeur null (non finie)."""
        return "non defini" if value is None else f"{float(value):.2f}"

    print(f"beta moyen / maximal   : {shown_beta(metrics.get('beta_mean'))} / "
          f"{shown_beta(metrics.get('beta_max'))}"
          + ("" if metrics.get("beta_defined", True) else " (beta non interpretable)"))
    print(f"cible de regularisation: {results['target_zone']['resolved']} "
          f"(automatique : {results['target_zone']['auto_selected']})")
    print(f"graphe etape 1         : {initial['n_nodes']} noeud(s), "
          f"{initial['n_edges']} arete(s), "
          f"{len(initial['isolated_nodes'])} isole(s)")
    degrees = initial.get("degrees", {})
    dominants = ", ".join(f"{node} domine {info['out']}"
                          for node, info in degrees.items() if info.get("out"))
    print(f"dominances etape 1     : {dominants or 'aucune au seuil'}")
    near = initial.get("near_threshold_pairs", [])
    if near:
        print(f"couples quasi dominants : {len(near)} sous le seuil {threshold}")
        for pair in near:
            print(f"   {pair['dominant']} majore {pair['dominated']} dans "
                  f"{pair['proportion']:.6f} des observations, soit "
                  f"{threshold - pair['proportion']:.6f} sous le seuil ; sans arete")
    print("=" * 78)
    if failures == 0:
        print("SUCCES : les resultats sont corrects et coherents avec les donnees "
              "d'entree, et les figures sont conformes aux conventions.")
        return 0
    print(f"ECHEC : {failures} famille(s) de controles en ecart.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
