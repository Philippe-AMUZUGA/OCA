"""Cœur algorithmique de l'OCA (Ordinal Correlation Algorithm), version 9.0.0.

Ce module implémente, sans aucune modification des équations d'origine :

    1. **Sélection par le GPO** (One-vs-All) : fréquence de dominance relative de
       chaque zone critique ; le seuil bas écarte les zones peu dominantes, le seuil
       haut désigne seulement les zones dimensionnantes (aucun filtrage).
    2. **Régularisation par le GPO** : recherche du coefficient multiplicateur
       optimal κ tel que la variable enveloppe κ·(D_e/E) domine toutes les autres
       selon un seuil de proportion, puis indice de parcimonie β.
    3. **Matrice de corrélation ordinale (OCM)** : proportions de dominance
       relative pour chaque paire de variables. Deux usages sont distingués :
       :func:`ordinal_correlation_matrix` fournit la matrice ordinale brute sur un
       ensemble quelconque de colonnes (utilisée en V3 pour l'état des lieux
       exhaustif **avant** toute sélection), tandis que
       :func:`ordinal_correlation_matrix_with_envelope` adjoint la colonne
       régularisée κ·cz (étape 2).
    4. **Graphe orienté de dominance** : construction d'un graphe (NetworkX) des
       relations d'ordre au-dessus d'un seuil, attribution des niveaux
       hiérarchiques (plus long chemin par tri topologique), positions et chaînes.
       Depuis la V3, l'intégralité des variables soumises est garantie présente
       sous forme de nœuds, y compris celles dépourvues d'arête au seuil.

Le module renvoie des structures de données pures ; la visualisation, l'écriture
des fichiers texte et l'export LaTeX/PDF sont assurés par ``post_processor``.
"""

from __future__ import annotations

import logging
import math
from decimal import Decimal
from typing import Any, Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

__all__ = [
    "count_dominance",
    "gpo_selection",
    "update_critical_zones",
    "highest_dominance_column",
    "dominance_frequencies",
    "dominance_ranking",
    "matrix_dominance_scores",
    "matrix_dominance_ranking",
    "order_columns_by_dominance",
    "find_kappa",
    "ordinal_correlation_matrix",
    "ordinal_correlation_matrix_with_envelope",
    "build_dominance_graph",
    "check_matrix_graph_coherence",
    "compute_parsimony",
    "beta_domain_report",
    "alpha_column_name",
    "dominance_ties",
    "find_kappa_details",
    "kappa_grid",
    "kappa_point_count",
    "kappa_decimals",
    "MAX_KAPPA_POINTS",
]

#: Nombre maximal de valeurs de κ explorées : borne contre un pas dérisoire.
MAX_KAPPA_POINTS = 100_000

#: Tolérance relative d'arrondi sur le nombre de pas de la grille de κ.
_GRID_TOLERANCE = 1e-9


def _numeric_error(message: str) -> Exception:
    """Erreur de configuration ``ConfigError`` (import différé : pas de cycle)."""
    from config_manager import ConfigError
    return ConfigError(message)


def _require_finite_envelope(envelope: "np.ndarray", kappa: float, cz: str) -> None:
    """Refuse une enveloppe κ·cible qui déborde le domaine des flottants.

    Un produit infini serait tenu pour majorant de toutes les zones : le succès de
    la recherche de κ n'aurait alors aucun sens.

    Raises:
        ConfigError: si une valeur de l'enveloppe n'est pas finie.
    """
    if not np.all(np.isfinite(envelope)):
        count = int(np.count_nonzero(~np.isfinite(envelope)))
        raise _numeric_error(
            f"Débordement numérique : l'enveloppe {kappa:g} x '{cz}' n'est pas finie sur "
            f"{count} observation(s) (valeurs de '{cz}' trop grandes pour ce κ). Un "
            "produit infini majorerait toutes les zones : le calcul est refusé. "
            "Changez d'unité ou de cible, ou resserrez 'gpo_regularization.kappa_max'.")


def alpha_column_name(kappa: float, cz: str, decimals: int = 2) -> str:
    """Nom de la colonne enveloppe régularisée, ex. ``"2.06$\\times$De/E"``.

    Args:
        kappa: Coefficient multiplicateur optimal.
        cz: Nom de la variable synthétique cible.
        decimals: Décimales affichées : au moins 2, davantage si le pas de κ
            l'exige (voir :func:`kappa_decimals`).

    Returns:
        Le libellé de colonne formaté comme dans le notebook d'origine.
    """
    return f"{kappa:.{max(2, int(decimals))}f}" + r"$\times$" + cz


def count_dominance(df: pd.DataFrame) -> Dict[str, int]:
    """Compte, par colonne, le nombre de lignes où elle domine strictement.

    Une colonne « domine » sur une ligne si sa valeur est strictement supérieure
    à toutes les autres valeurs de la ligne. Implémentation vectorisée
    strictement équivalente à la boucle ``all(row[col] > row.drop(col))``
    d'origine (``col > max(autres colonnes)``).

    Args:
        df: DataFrame des variables critiques (colonnes comparées).

    Returns:
        Dictionnaire colonne -> nombre de dominations.
    """
    counts: Dict[str, int] = {}
    for col in df.columns:
        others_max = df.drop(columns=col).max(axis=1)
        counts[col] = int((df[col] > others_max).sum())
    return counts


def gpo_selection(
    df_merged: pd.DataFrame,
    critical_zones: List[str],
    threshold_low: float,
    threshold_high: float,
    random_state: int = 6,
) -> Dict[str, Any]:
    """Étape 1 : sélection GPO par comptage One-vs-All.

    Construit le tableau des dominations (``n``, ``p``, cumuls ``N``, ``P``) trié
    par ``n`` croissant, puis classe les variables selon les seuils de proportion
    ``p`` (par colonne, en %) : dimensionnantes (``p`` élevé), non-dimensionnantes
    (``p`` faible).

    Args:
        df_merged: DataFrame fusionné (contenant les zones critiques).
        critical_zones: Colonnes critiques à analyser (avec la variable
            synthétique, ex. ``De/E``).
        threshold_low: Seuil bas de ``p`` (%) — colonnes non-dimensionnantes.
        threshold_high: Seuil haut de ``p`` (%) — colonnes dimensionnantes.
        random_state: Graine de l'échantillonnage (sans effet sur les comptes ;
            conservée par fidélité au notebook).

    Returns:
        Dictionnaire ``{df_results, dico_results, columns_below, columns_above,
        n_sample}``.
    """
    df = df_merged[critical_zones].copy()
    n_sample = len(df)
    df = df.sample(n=n_sample, random_state=random_state)

    counts = count_dominance(df)
    total_length = len(df)

    # Construction en une seule passe. La concaténation itérative à partir d'un
    # ``pd.DataFrame()`` vide est dépréciée depuis pandas 2.1 et son inférence de
    # type a changé en pandas 3.0 (les colonnes vides ne sont plus exclues du
    # calcul des dtypes) : elle pouvait faire basculer ``n`` et ``p`` en dtype
    # objet. L'ordre des lignes reste celui de ``df.columns``, valeurs inchangées.
    rows = [{"n": counts[column], "p": (counts[column] / total_length) * 100}
            for column in df.columns]
    df_results = pd.DataFrame(rows, index=list(df.columns))

    df_results["n"] = df_results["n"].astype(int)
    # Tri *stable* : à nombre de dominations ``n`` égal, l'ordre de déclaration
    # des zones critiques est préservé. Le tri par défaut de pandas ("quicksort")
    # n'est pas stable et son résultat sur les ex aequo varie selon la version de
    # NumPy, ce qui déplacerait les cumuls ``N`` et ``P`` d'une colonne à l'autre.
    df_results = df_results.sort_values(by="n", kind="stable")
    df_results["p"] = df_results["p"].apply(lambda x: f"{x:.2f}%")
    df_results = df_results.transpose()

    cumulative_sum_n = df_results.loc["n"].cumsum()
    df_results.loc["N"] = cumulative_sum_n
    cumulative_sum = df_results.loc["p"].str.rstrip("%").astype(float).cumsum().map("{:.2f}%".format)
    df_results.loc["P"] = cumulative_sum

    values_p = df_results.loc["p"].str.rstrip("%").astype(float)
    columns_below = values_p[values_p < threshold_low].index.tolist()
    columns_above = values_p[values_p > threshold_high].index.tolist()

    dico_results = {
        f"Colonnes où p est inférieure à {threshold_low:.3g}%": columns_below,
        f"Colonnes où p est supérieure à {threshold_high:.3g}%": columns_above,
    }

    return {
        "df_results": df_results,
        "dico_results": dico_results,
        "columns_below": columns_below,
        "columns_above": columns_above,
        "n_sample": n_sample,
    }


def update_critical_zones(
    critical_zones: List[str],
    columns_below: List[str],
    synthetic_target: str,
) -> List[str]:
    """Met à jour la liste des zones critiques après sélection GPO.

    Retire les colonnes non-dimensionnantes (``columns_below``) et garantit la
    présence de la variable synthétique cible.

    Args:
        critical_zones: Liste courante des zones critiques.
        columns_below: Colonnes à retirer (``p`` < seuil bas).
        synthetic_target: Variable synthétique à conserver (ex. ``De/E``).

    Returns:
        Nouvelle liste des zones critiques.
    """
    updated = [zone for zone in critical_zones if zone not in columns_below]
    if synthetic_target not in updated:
        updated.append(synthetic_target)
    return updated


def highest_dominance_column(df_results: pd.DataFrame) -> str:
    """Retourne la colonne à la fréquence de dominance ``p`` la plus élevée.

    Utilisée pour la sélection automatique de la cible de régularisation
    (``target_zone = "none"``) : la variable qui domine le plus fréquemment les
    autres à l'étape 1 (GPO One-vs-All).

    Args:
        df_results: Tableau des dominations renvoyé par :func:`gpo_selection`
            (lignes ``n``, ``p``, ``N``, ``P`` ; colonnes = zones critiques).

    Returns:
        Nom de la colonne dont ``p`` (par colonne) est maximal.
    """
    p = df_results.loc["p"].str.rstrip("%").astype(float)
    # Le critere n'est pas modifie : c'est bien le premier maximum de p qui est
    # retenu. Mais la frequence p SATURE — des qu'une zone domine largement, toutes
    # les autres tombent a zero, quelles que soient leurs relations mutuelles. Le
    # maximum peut donc etre atteint par plusieurs colonnes, voire valoir zero
    # partout, et le departage se fait alors par ordre de declaration. Presenter
    # cela comme une « frequence maximale » serait trompeur : l'ambiguite est donc
    # signalee a l'appelant par l'attribut du releve.
    best, tied = dominance_ties(df_results)
    highest_dominance_column.last_tie = tuple(tied)
    highest_dominance_column.last_value = best
    return str(p.idxmax())


def dominance_ties(df_results: pd.DataFrame) -> Tuple[float, List[str]]:
    """Zones ex æquo au maximum de la fréquence ``p`` (valeurs arrondies à 0,01 %).

    La règle de choix de la cible automatique est déterministe : parmi les zones
    ex æquo, c'est la première selon l'ordre des colonnes du tableau de comptage
    (nombre de dominations croissant, puis ordre de déclaration des zones).

    Args:
        df_results: Tableau des dominations renvoyé par :func:`gpo_selection`.

    Returns:
        Couple ``(p maximal en %, zones atteignant ce maximum dans l'ordre des
        colonnes)``.
    """
    p = df_results.loc["p"].str.rstrip("%").astype(float)
    best = float(p.max())
    return best, [str(name) for name, value in p.items() if float(value) == best]


def dominance_frequencies(df_results: pd.DataFrame) -> Dict[str, float]:
    """Fréquences de dominance ``p`` de chaque zone critique, en pourcentage.

    Ces fréquences sont issues du comptage un contre tous de l'étape 1, établi
    sur **l'intégralité des observations** du jeu de données. Elles constituent
    le score global de dominance qui gouverne le classement, la hiérarchisation
    et le filtrage des zones critiques.

    Args:
        df_results: Tableau des dominations renvoyé par :func:`gpo_selection`
            (lignes ``n``, ``p``, ``N``, ``P`` ; colonnes = zones critiques).

    Returns:
        Association zone vers fréquence ``p``, exprimée en pourcentage.
    """
    frequencies = df_results.loc["p"].str.rstrip("%").astype(float)
    return {str(zone): float(value) for zone, value in frequencies.items()}


def matrix_dominance_scores(prop_matrix: pd.DataFrame) -> Dict[str, float]:
    """Score de majoration de chaque variable, issu de la matrice ordinale.

    Le score d'une variable est la moyenne, sur toutes les autres, de la
    proportion d'observations où elle les majore. Il vaut donc

    .. math:: s_i = \\frac{1}{m-1} \\sum_{j \\neq i} p(i, j)

    où ``p(i, j)`` désigne la proportion d'observations telles que
    ``var_j <= var_i``, soit le coefficient en ligne ``i`` et colonne ``j`` de la
    matrice de corrélation ordinale.

    Ce score mesure la dominance **par comparaisons deux à deux**, à la
    différence de la fréquence ``p`` du comptage un contre tous de l'étape 1,
    laquelle exige qu'une variable majore *simultanément toutes* les autres.
    Cette dernière sature dès qu'une variable domine largement : toutes les
    autres tombent alors à zéro, quelles que soient leurs relations mutuelles, et
    ne sont plus départagées. Le score matriciel n'a pas cette faiblesse.

    Args:
        prop_matrix: Matrice de corrélation ordinale, diagonale ``NaN``.

    Returns:
        Association variable vers score, dans l'intervalle [0, 1].
    """
    scores: Dict[str, float] = {}
    for variable in prop_matrix.index:
        row = prop_matrix.loc[variable].drop(labels=[variable], errors="ignore")
        finite = row.dropna()
        scores[str(variable)] = float(finite.mean()) if len(finite) else float("nan")
    return scores


def matrix_dominance_ranking(prop_matrix: pd.DataFrame) -> List[str]:
    """Classement des variables par score de majoration croissant.

    L'ordre retourné va de la variable la moins dominante à la plus dominante,
    d'après la matrice de corrélation ordinale. C'est le critère qui gouverne la
    disposition horizontale des colonnes dans les tableaux d'échantillonnage.

    Le tri est **stable** : deux variables de même score conservent leur ordre
    d'apparition dans la matrice.

    Args:
        prop_matrix: Matrice de corrélation ordinale, diagonale ``NaN``.

    Returns:
        Variables, de la moins dominante à la plus dominante.
    """
    scores = matrix_dominance_scores(prop_matrix)
    series = pd.Series(scores, index=list(scores))
    return [str(variable) for variable in series.sort_values(kind="stable").index]


def dominance_ranking(df_results: pd.DataFrame) -> List[str]:
    """Classement des zones critiques par fréquence de dominance croissante.

    L'ordre retourné va de la zone la moins dominante à la plus dominante. Il
    gouverne la disposition horizontale des colonnes dans les tableaux
    d'échantillonnage, de sorte que la lecture de gauche à droite suive la
    hiérarchie établie par l'étape 1.

    Le tri est **stable** : deux zones de même fréquence conservent leur ordre de
    déclaration dans ``input.json``, ce qui rend la disposition reproductible
    d'une exécution et d'une version de bibliothèque à l'autre.

    Args:
        df_results: Tableau des dominations renvoyé par :func:`gpo_selection`.

    Returns:
        Zones critiques, de la moins dominante à la plus dominante.
    """
    frequencies = df_results.loc["p"].str.rstrip("%").astype(float)
    return [str(zone) for zone in frequencies.sort_values(kind="stable").index]


def order_columns_by_dominance(
    frame: pd.DataFrame,
    ranking: Sequence[str],
    n_trailing: int = 0,
) -> pd.DataFrame:
    """Réordonne les colonnes d'un tableau selon un classement de dominance.

    La réorganisation s'opère par sélection de colonnes entières
    (``frame[nouvel_ordre]``), opération qui déplace chaque colonne avec
    l'intégralité de ses valeurs et son index. **Aucun décalage de données entre
    lignes n'est donc structurellement possible** : les valeurs restent
    solidaires de leur colonne et de leur observation. Le contrôle numéro 13 du
    harnais de validation le vérifie sur les documents effectivement produits.

    Les ``n_trailing`` dernières colonnes sont maintenues en fin de tableau, quel
    que soit leur rang de dominance. Elles portent, dans le tableau de l'étape 2,
    l'enveloppe κ·cz et l'indice de parcimonie β, dont la position terminale et
    la mise en évidence typographique font partie de la convention de lecture.

    Les colonnes absentes du classement sont conservées, à la suite de celles qui
    y figurent : une colonne écartée par la sélection ou ajoutée en cours de
    traitement n'est jamais perdue.

    Args:
        frame: Tableau à réordonner.
        ranking: Zones critiques, de la moins dominante à la plus dominante.
        n_trailing: Nombre de colonnes terminales à maintenir en place.

    Returns:
        Nouveau tableau, de mêmes valeurs, aux colonnes réordonnées.
    """
    columns = list(frame.columns)
    if n_trailing > 0:
        movable, trailing = columns[:-n_trailing], columns[-n_trailing:]
    else:
        movable, trailing = columns, []

    ranked = [zone for zone in ranking if zone in movable]
    unranked = [column for column in movable if column not in ranked]
    return frame[ranked + unranked + trailing]


def kappa_point_count(kappa_min: float, kappa_max: float, kappa_step: float) -> int:
    """Nombre de valeurs de κ de la grille ``kappa_min + i * kappa_step``.

    La borne supérieure est incluse, avec une tolérance d'arrondi de 1e-9 relatif
    sur le nombre de pas : ``[1, 1.2]`` au pas de 0,1 compte 3 valeurs bien que
    ``1 + 2 * 0.1`` dépasse 1,2 d'un ulp dans certaines accumulations.

    Args:
        kappa_min: Borne basse.
        kappa_max: Borne haute.
        kappa_step: Pas, strictement positif.

    Returns:
        Le nombre de points, ou une valeur supérieure à ``MAX_KAPPA_POINTS`` si le
        quotient n'est pas fini.
    """
    span = (float(kappa_max) - float(kappa_min)) / float(kappa_step)
    if not math.isfinite(span):
        return MAX_KAPPA_POINTS + 1
    return int(math.floor(span + _GRID_TOLERANCE * max(1.0, abs(span)))) + 1


def _decimals_of(value: float) -> int:
    """Nombre de décimales de l'écriture décimale la plus courte d'un flottant."""
    exponent = Decimal(repr(float(value))).as_tuple().exponent
    return max(0, -int(exponent)) if isinstance(exponent, int) else 0


def kappa_decimals(kappa_min: float, kappa_step: float) -> int:
    """Décimales d'écriture de κ : celles que le pas et la borne basse exigent.

    Au moins 2 (valeur historique), davantage pour un pas plus fin : un pas de
    0,001 publie κ avec 3 décimales, faute de quoi la valeur publiée ne serait plus
    celle qui a été éprouvée.

    Args:
        kappa_min: Borne basse de κ.
        kappa_step: Pas de κ.

    Returns:
        Le nombre de décimales, entre 2 et 15.
    """
    return int(min(15, max(2, _decimals_of(kappa_min), _decimals_of(kappa_step))))


def kappa_grid(kappa_min: float, kappa_max: float, kappa_step: float) -> List[float]:
    """Grille de κ construite par indice : ``kappa_min + i * kappa_step``.

    Aucune accumulation d'incréments flottants : chaque point est calculé
    directement, arrondi à :func:`kappa_decimals` décimales (pour qu'il soit
    écrit exactement tel qu'il a été éprouvé) et borné à ``[kappa_min, kappa_max]``.

    Args:
        kappa_min: Borne basse.
        kappa_max: Borne haute (incluse).
        kappa_step: Pas strictement positif.

    Returns:
        Les valeurs de κ, strictement croissantes (donc distinctes).

    Raises:
        ValueError: si le pas n'est pas strictement positif et fini, si la grille
            dépasse ``MAX_KAPPA_POINTS`` points, ou si deux points consécutifs ne
            sont pas distincts à la précision des flottants.
    """
    if not (math.isfinite(kappa_step) and kappa_step > 0):
        raise ValueError("Le pas de κ doit être un nombre fini strictement positif.")
    if not (math.isfinite(kappa_min) and math.isfinite(kappa_max) and kappa_min <= kappa_max):
        raise ValueError("Les bornes de κ doivent être finies, avec kappa_min <= kappa_max.")
    count = kappa_point_count(kappa_min, kappa_max, kappa_step)
    if count > MAX_KAPPA_POINTS:
        raise ValueError(
            f"La grille de κ compterait plus de {MAX_KAPPA_POINTS} points : augmentez "
            "le pas ou resserrez la plage.")
    decimals = kappa_decimals(kappa_min, kappa_step)
    grid = []
    for index in range(count):
        value = round(kappa_min + index * kappa_step, decimals)
        grid.append(min(max(value, kappa_min), kappa_max))
    for previous, current in zip(grid, grid[1:]):
        if not current > previous:
            raise ValueError(
                f"Les points de la grille de κ ne sont pas distincts (pas {kappa_step:g} "
                f"trop fin devant la précision des flottants à κ = {previous:g}) : "
                "augmentez le pas ou resserrez la plage.")
    return grid


def find_kappa_details(
    df_merged: pd.DataFrame,
    critical_zones: List[str],
    cz: str,
    threshold: float,
    kappa_min: float,
    kappa_max: float,
    kappa_step: float,
) -> Dict[str, Any]:
    """Étape 2 : recherche du coefficient multiplicateur optimal κ, avec son bilan.

    Balaye la grille de κ (voir :func:`kappa_grid`) et retient la première valeur
    telle que, pour **toutes** les zones de ``critical_zones`` (zones conservées par
    la sélection, cible comprise), la proportion de lignes où ``κ·cz >= zone``
    atteigne le seuil requis. Le critère et son seuil inclusif sont inchangés.

    Args:
        df_merged: DataFrame fusionné.
        critical_zones: Zones critiques conservées (incluant ``cz``).
        cz: Variable synthétique enveloppe.
        threshold: Seuil de proportion de dominance requis.
        kappa_min: Borne basse de κ.
        kappa_max: Borne haute de κ.
        kappa_step: Pas de κ.

    Returns:
        Dictionnaire ``{kappa, found, last_tested, n_points, n_tested,
        min_coverage, decimals}`` : ``kappa`` est la première valeur satisfaisante
        si ``found``, sinon le dernier point éprouvé ; ``min_coverage`` est la plus
        faible proportion de majoration, sur les zones, à la valeur retournée.
    """
    variables = df_merged[critical_zones].columns.tolist()
    total_obs = len(df_merged)
    grid = kappa_grid(kappa_min, kappa_max, kappa_step)
    cz_values = df_merged[cz].to_numpy(dtype=float)
    columns = {var: df_merged[var].to_numpy(dtype=float) for var in variables}

    def coverage(alpha: float) -> float:
        envelope = alpha * cz_values
        return min(
            (float(np.count_nonzero(envelope >= columns[var])) / total_obs
             for var in variables), default=1.0)

    found = False
    tested = 0
    kappa = grid[0]
    for alpha in grid:
        tested += 1
        kappa = alpha
        with np.errstate(over="ignore", invalid="ignore"):
            envelope = alpha * cz_values
        _require_finite_envelope(envelope, alpha, cz)
        all_above = True
        for var_i in variables:
            proportion = np.count_nonzero(envelope >= columns[var_i]) / total_obs
            if proportion < threshold:
                all_above = False
                break
        if all_above:
            found = True
            break

    return {
        "kappa": float(kappa),
        "found": found,
        "last_tested": float(kappa),
        "n_points": len(grid),
        "n_tested": tested,
        "min_coverage": coverage(kappa),
        "decimals": kappa_decimals(kappa_min, kappa_step),
    }


def find_kappa(
    df_merged: pd.DataFrame,
    critical_zones: List[str],
    cz: str,
    threshold: float,
    kappa_min: float,
    kappa_max: float,
    kappa_step: float,
) -> Tuple[float, bool]:
    """Étape 2 : recherche du coefficient multiplicateur optimal κ.

    Voir :func:`find_kappa_details`. Les zones examinées sont celles que la
    sélection a conservées : l'enveloppe majore ces zones, non celles qui ont été
    écartées par le seuil bas.

    Args:
        df_merged: DataFrame fusionné.
        critical_zones: Zones critiques conservées (incluant ``cz``).
        cz: Variable synthétique enveloppe (ex. ``De/E``).
        threshold: Seuil de proportion de dominance requis (ex. 0.99).
        kappa_min: Borne basse de κ.
        kappa_max: Borne haute de κ.
        kappa_step: Pas d'incrément de κ.

    Returns:
        Tuple ``(kappa, found)`` ; ``found`` est ``False`` si aucune valeur ne
        satisfait le seuil ; κ vaut alors le dernier point EFFECTIVEMENT éprouvé,
        qui appartient toujours à ``[kappa_min, kappa_max]``. Le succès se lit sur
        ``found`` et non sur l'égalité de κ à ``kappa_max`` : un succès à la borne
        haute est possible.
    """
    details = find_kappa_details(
        df_merged, critical_zones, cz, threshold, kappa_min, kappa_max, kappa_step)
    return details["kappa"], details["found"]


def _ordinal_proportions(df: pd.DataFrame) -> pd.DataFrame:
    """Matrice carrée des proportions de dominance relative d'un DataFrame.

    Pour chaque paire ordonnée de colonnes ``(var_i, var_j)``, calcule la
    proportion d'observations telles que ``var_j <= var_i``. La diagonale est
    mise à ``NaN`` (une variable n'est pas comparée à elle-même).

    Note:
        La diagonale est neutralisée par ``DataFrame.mask`` et non par une
        écriture directe dans le tampon ``.values``. Depuis pandas 3.0, le
        tableau NumPy exposé par ``.values`` est en lecture seule (Copy-on-Write
        généralisé) : ``np.fill_diagonal(df.values, np.nan)`` y lève
        ``ValueError: underlying array is read-only``. Le masquage produit
        exactement la même matrice tout en respectant les mécanismes
        d'indexation de pandas.

    Args:
        df: DataFrame dont les colonnes sont les variables à comparer.

    Returns:
        Matrice ``len(df.columns) x len(df.columns)`` de proportions.
    """
    variables = df.columns.tolist()
    total_obs = len(df)
    prop_matrix = pd.DataFrame(index=variables, columns=variables, dtype=float)
    for var_i in variables:
        for var_j in variables:
            violations = (df[var_j] <= df[var_i]).sum()
            prop_matrix.loc[var_i, var_j] = violations / total_obs

    diagonal = pd.DataFrame(
        np.eye(len(variables), dtype=bool), index=variables, columns=variables
    )
    return prop_matrix.astype(float).mask(diagonal)


def ordinal_correlation_matrix(
    df_merged: pd.DataFrame,
    variables: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Matrice de corrélation ordinale (ordre de Pareto) d'un jeu de variables.

    Cette forme n'introduit aucune colonne dérivée : elle décrit l'ordre partiel
    de dominance des variables telles qu'elles se présentent dans les données.
    C'est la primitive utilisée en V3 pour établir l'état des lieux exhaustif des
    relations hiérarchiques ordinales **avant** la première sélection GPO, ainsi
    que pour le graphe d'ordonnancement de l'étape 1.

    Args:
        df_merged: DataFrame fusionné.
        variables: Colonnes à comparer, dans l'ordre d'affichage souhaité.

    Returns:
        Tuple ``(prop_matrix, variables)`` où ``prop_matrix`` est la matrice des
        proportions de dominance (diagonale ``NaN``).
    """
    df = df_merged[list(variables)].copy()
    return _ordinal_proportions(df), df.columns.tolist()


def ordinal_correlation_matrix_with_envelope(
    df_merged: pd.DataFrame,
    critical_zones: List[str],
    cz: str,
    kappa: float,
    decimals: int = 2,
) -> Tuple[pd.DataFrame, List[str], str]:
    """Matrice de corrélation ordinale augmentée de la colonne enveloppe κ·cz.

    Forme employée à l'étape 2, après régularisation : la colonne
    ``κ·cz`` est adjointe aux zones critiques retenues afin de quantifier son
    rôle d'enveloppe majorante vis-à-vis de chacune d'elles.

    Args:
        df_merged: DataFrame fusionné.
        critical_zones: Zones critiques retenues après sélection GPO.
        cz: Variable cible de la régularisation.
        kappa: Coefficient multiplicateur optimal.
        decimals: Décimales du libellé de la colonne enveloppe (voir
            :func:`kappa_decimals`).

    Returns:
        Tuple ``(prop_matrix, variables, alpha_col_name)``.
    """
    df = df_merged[list(critical_zones)].copy()
    alpha_col = alpha_column_name(kappa, cz, decimals)
    if alpha_col in df.columns:
        raise _numeric_error(
            f"Une zone porte déjà le nom '{alpha_col}', réservé à l'enveloppe κ·cible : "
            "renommez cette colonne.")
    with np.errstate(over="ignore", invalid="ignore"):
        envelope = kappa * df[cz].to_numpy(dtype=float)
    _require_finite_envelope(envelope, kappa, cz)
    df[alpha_col] = envelope
    return _ordinal_proportions(df), df.columns.tolist(), alpha_col


def _assign_levels_longest_path(graph: "nx.DiGraph") -> Dict[str, int]:
    """Attribue un niveau à chaque nœud par plus long chemin.

    Si le graphe est acyclique (cas nominal), les niveaux sont calculés
    directement par tri topologique (comportement du notebook d'origine). En
    présence de cycles, le graphe est d'abord **condensé en DAG** (composantes
    fortement connexes) pour rester robuste ; les nœuds d'un même cycle partagent
    alors le niveau de leur composante.

    Args:
        graph: Graphe orienté de dominance.

    Returns:
        Dictionnaire nœud -> niveau hiérarchique.
    """
    if nx.is_directed_acyclic_graph(graph):
        levels = {node: 0 for node in graph.nodes()}
        for node in nx.topological_sort(graph):
            for successor in graph.successors(node):
                if levels[successor] < levels[node] + 1:
                    levels[successor] = levels[node] + 1
        return levels

    # Repli robuste : condensation en DAG puis niveaux sur les composantes.
    # L'avertissement est porté au journal ``oca`` et non à la seule sortie
    # standard : un cycle dans le graphe de dominance est une circonstance dont la
    # trace doit figurer au rapport archivé, la condensation modifiant la
    # hiérarchie des niveaux effectivement représentée.
    logging.getLogger("oca").warning(
        "Cycle(s) détecté(s) dans le graphe de dominance : condensation en graphe "
        "acyclique orienté par composantes fortement connexes pour l'établissement "
        "des niveaux hiérarchiques.")
    cond = nx.condensation(graph)
    mapping = cond.graph["mapping"]
    scc_level = {scc: 0 for scc in cond.nodes()}
    for scc in nx.topological_sort(cond):
        for successor in cond.successors(scc):
            if scc_level[successor] < scc_level[scc] + 1:
                scc_level[successor] = scc_level[scc] + 1
    return {node: scc_level[mapping[node]] for node in graph.nodes()}


def _layered_positions(
    graph: "nx.DiGraph",
    variables: List[str],
    levels: Dict[str, int],
    isolated: List[str],
    max_level: int,
) -> Dict[str, Tuple[float, float]]:
    """Calcule les positions des nœuds par rangées hiérarchiques.

    Les variables reliées occupent les rangées de niveau 0, les plus dominantes,
    à ``max_level`` ; les variables isolées sont regroupées sur une rangée dédiée
    placée sous la hiérarchie. Chaque rangée est centrée sur l'axe des abscisses
    afin d'équilibrer la figure. L'ordonnée est inversée à l'affichage, si bien que
    le niveau 0 apparaît en haut.

    L'ordre des nœuds au sein d'une rangée est déterminé par une passe
    barycentrique descendante : chaque nœud est placé au barycentre des abscisses de
    ses prédécesseurs déjà positionnés, ce qui réduit le nombre de croisements
    d'arêtes sans recourir à aucune heuristique spécifique à une topologie. Les
    égalités sont tranchées par l'ordre de déclaration des variables, ce qui
    préserve le déterminisme du tracé.

    Args:
        graph: Graphe orienté de dominance.
        variables: Liste ordonnée des variables.
        levels: Niveau hiérarchique de chaque nœud.
        isolated: Variables dépourvues de relation au seuil.
        max_level: Niveau hiérarchique maximal des nœuds reliés.

    Returns:
        Dictionnaire nœud vers couple d'abscisse et d'ordonnée.
    """
    spacing_x = 2
    spacing_y = 3
    isolated_set = set(isolated)
    declaration_rank = {name: index for index, name in enumerate(variables)}

    level_nodes: Dict[int, List[str]] = {}
    for node in variables:
        if node not in isolated_set:
            level_nodes.setdefault(levels[node], []).append(node)

    pos: Dict[str, Tuple[float, float]] = {}

    def _place_row(row: List[str], y: float) -> None:
        """Répartit une rangée de nœuds symétriquement autour de l'abscisse nulle."""
        offset = (len(row) - 1) / 2.0
        for position, node in enumerate(row):
            pos[node] = ((position - offset) * spacing_x, y)

    for level in range(max_level + 1):
        row = level_nodes.get(level, [])
        if level > 0 and len(row) > 1:
            def _barycentre(node: str) -> Tuple[float, int]:
                """Barycentre des abscisses des prédécesseurs déjà positionnés."""
                abscissas = [pos[p][0] for p in graph.predecessors(node) if p in pos]
                if not abscissas:
                    return (0.0, declaration_rank[node])
                return (sum(abscissas) / len(abscissas), declaration_rank[node])

            row = sorted(row, key=_barycentre)
        _place_row(row, (level + 1) * spacing_y)

    if isolated:
        _place_row(isolated, (max_level + 2) * spacing_y)
    return pos


def build_dominance_graph(
    prop_matrix: pd.DataFrame,
    variables: List[str],
    threshold: float,
    near_margin: float = 0.01,
) -> Dict[str, Any]:
    """Construit le graphe orienté de dominance et ses niveaux hiérarchiques.

    Les arêtes ``var_i -> var_j`` sont ajoutées lorsque
    ``prop_matrix[i, j] >= threshold``. Les niveaux sont attribués par plus long
    chemin via tri topologique ; les positions placent les nœuds dominants aux
    niveaux supérieurs. Les chaînes de comparaison remontent des nœuds de niveau
    maximal vers les prédécesseurs.

    Exhaustivité des nœuds (V3) : l'intégralité des variables de ``variables`` est
    déclarée comme nœud **avant** l'insertion des arêtes. Une variable dépourvue de
    relation d'ordre franchissant le seuil demeure donc présente dans le graphe et
    reste systématiquement affichée. Elle est identifiée dans ``isolated`` et
    positionnée sur une rangée dédiée, distincte des niveaux hiérarchiques, afin de
    ne pas lui prêter un rang de dominance qu'elle ne possède pas.

    Cohérence avec la matrice (V4) : le graphe est, par construction, l'image exacte
    de ``prop_matrix`` au seuil retenu. Trois dispositifs rendent cette propriété
    vérifiable plutôt que simplement postulée.

        1. ``degrees`` publie, pour chaque nœud, le degré sortant et le degré
           entrant, à confronter au dénombrement des cellules de la ligne et de la
           colonne homologues atteignant le seuil.
        2. ``edge_colors`` distingue les arêtes par leur **nœud source** et non plus
           par le niveau de celle-ci. Plusieurs variables occupant le même niveau,
           un codage par niveau leur attribuait une teinte commune et interdisait
           d'imputer visuellement une arête à sa source, d'où des dénombrements
           erronés sur les figures denses.
        3. ``near_threshold`` recense les couples dont la proportion approche le
           seuil sans l'atteindre. Ces couples se rendent par une teinte
           indiscernable de celle des couples retenus sur une échelle saturée à
           l'unité : les expliciter lève l'incohérence apparente entre la matrice
           et le graphe, sans altérer le critère de dominance.

    Args:
        prop_matrix: Matrice des proportions de dominance (diagonale NaN).
        variables: Liste ordonnée des variables (labels des nœuds). Toutes sont
            garanties présentes dans le graphe retourné.
        threshold: Seuil de proportion pour créer une arête.
        near_margin: Largeur de la bande située sous le seuil dans laquelle un
            couple est signalé comme quasi dominant.

    Returns:
        Dictionnaire ``{pairs_above, graph, isolated, levels, max_level, pos,
        node_colors, edge_colors, color_cycle, chains, degrees, near_threshold}``.
        ``isolated`` est une liste ordonnée selon ``variables``.
    """
    pairs_above: List[Tuple[str, str, float]] = []
    near_threshold: List[Tuple[str, str, float]] = []
    for i in range(len(variables)):
        for j in range(len(variables)):
            if i != j:
                proportion = prop_matrix.iloc[i, j]
                if pd.isna(proportion):
                    continue
                if proportion >= threshold:
                    pairs_above.append((variables[i], variables[j], float(proportion)))
                elif proportion >= threshold - near_margin:
                    near_threshold.append((variables[i], variables[j], float(proportion)))

    # Déclaration exhaustive des nœuds AVANT les arêtes : garantit qu'aucune
    # variable ne disparaisse du graphe et fixe un ordre d'itération déterministe.
    graph = nx.DiGraph()
    graph.add_nodes_from(variables)
    for var_i, var_j, _ in pairs_above:
        graph.add_edge(var_i, var_j)

    nodes_in_edges = set()
    for var_i, var_j, _ in pairs_above:
        nodes_in_edges.add(var_i)
        nodes_in_edges.add(var_j)
    isolated = [var for var in variables if var not in nodes_in_edges]
    isolated_set = set(isolated)

    # Niveaux par plus long chemin (robuste aux cycles via condensation).
    levels = _assign_levels_longest_path(graph)
    connected_levels = [lvl for node, lvl in levels.items() if node not in isolated_set]
    max_level = max(connected_levels) if connected_levels else 0

    pos = _layered_positions(graph, variables, levels, isolated, max_level)

    node_colors = []
    for node in graph.nodes():
        node_colors.append("lightgreen" if node in isolated_set else "lightblue")

    # Teinte indexée sur le rang du nœud source dans l'ordre des variables : le
    # faisceau d'arêtes émis par une variable dominante devient identifiable, donc
    # dénombrable, ce qu'un codage par niveau ne permettait pas.
    color_cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    source_rank = {name: index for index, name in enumerate(variables)}
    edge_colors = [
        color_cycle[source_rank.get(source, 0) % len(color_cycle)]
        for source, _target in graph.edges()
    ]

    degrees = {
        node: {"out": int(graph.out_degree(node)), "in": int(graph.in_degree(node))}
        for node in variables
    }

    # Chaînes de comparaison (des nœuds de niveau maximal vers les prédécesseurs).
    # Les variables isolées sont exclues : dépourvues de relation, elles ne
    # forment aucune chaîne de comparaison.
    chains: List[List[str]] = []
    for node in graph.nodes():
        if node in isolated_set:
            continue
        if levels.get(node, 0) == max_level:
            chain = [node]
            current = node
            while True:
                preds = list(graph.predecessors(current))
                if preds:
                    pred = max(preds, key=lambda x: levels.get(x, 0))
                    if pred in chain:  # garde-fou anti-cycle
                        break
                    chain.append(pred)
                    current = pred
                else:
                    break
            chains.append(chain[::-1])

    return {
        "pairs_above": pairs_above,
        "graph": graph,
        "isolated": isolated,
        "levels": levels,
        "max_level": max_level,
        "pos": pos,
        "node_colors": node_colors,
        "edge_colors": edge_colors,
        "color_cycle": color_cycle,
        "chains": chains,
        "degrees": degrees,
        "near_threshold": near_threshold,
        "source_rank": source_rank,
        # Ordre des variables du graphe. Le post traitement s'en sert pour
        # attribuer à chaque nœud la teinte de son rang, selon le style configuré.
        "variables": list(variables),
    }


def check_matrix_graph_coherence(
    prop_matrix: pd.DataFrame,
    graph_data: Dict[str, Any],
    threshold: float,
) -> List[str]:
    """Vérifie que le graphe de dominance est l'image exacte de la matrice.

    Le contrôle est purement structurel et indépendant de toute topologie
    particulière. Quatre propriétés sont exigées :

        1. tout nœud de la matrice figure dans le graphe, et réciproquement ;
        2. l'ensemble des arêtes coïncide exactement avec l'ensemble des couples
           ``(i, j)`` vérifiant ``prop_matrix[i, j] >= threshold`` ;
        3. le degré sortant de chaque nœud égale le nombre de cellules de sa ligne
           atteignant le seuil, et son degré entrant le nombre de cellules de sa
           colonne l'atteignant ;
        4. tout nœud déclaré isolé est de degré total nul, et réciproquement.

    La propriété 3 est celle que le lecteur exerce sur les figures : dénombrer les
    cellules encadrées d'une ligne de la matrice, puis les arêtes émises par le nœud
    homologue du graphe. La vérifier programmatiquement interdit qu'un écart entre
    les deux représentations passe inaperçu.

    Args:
        prop_matrix: Matrice des proportions de dominance.
        graph_data: Structure renvoyée par :func:`build_dominance_graph`.
        threshold: Seuil de dominance appliqué.

    Returns:
        Liste des incohérences constatées, vide si le graphe reflète exactement la
        matrice.
    """
    issues: List[str] = []
    graph = graph_data["graph"]
    variables = list(prop_matrix.index)

    if set(variables) != set(graph.nodes()):
        issues.append(
            f"nœuds du graphe {sorted(graph.nodes())} distincts des variables "
            f"de la matrice {sorted(variables)}")
        return issues

    expected_edges = set()
    for var_i in variables:
        for var_j in variables:
            if var_i == var_j:
                continue
            value = prop_matrix.loc[var_i, var_j]
            if not pd.isna(value) and float(value) >= threshold:
                expected_edges.add((var_i, var_j))
    observed_edges = set(graph.edges())
    for missing in sorted(expected_edges - observed_edges):
        issues.append(f"arête absente du graphe : {missing[0]} -> {missing[1]}")
    for spurious in sorted(observed_edges - expected_edges):
        issues.append(f"arête surnuméraire : {spurious[0]} -> {spurious[1]}")

    for var in variables:
        row = prop_matrix.loc[var].drop(labels=[var], errors="ignore")
        column = prop_matrix[var].drop(labels=[var], errors="ignore")
        expected_out = int((row.dropna() >= threshold).sum())
        expected_in = int((column.dropna() >= threshold).sum())
        if int(graph.out_degree(var)) != expected_out:
            issues.append(
                f"{var} : degré sortant {graph.out_degree(var)} pour "
                f"{expected_out} cellule(s) de ligne au seuil")
        if int(graph.in_degree(var)) != expected_in:
            issues.append(
                f"{var} : degré entrant {graph.in_degree(var)} pour "
                f"{expected_in} cellule(s) de colonne au seuil")

    declared_isolated = set(graph_data.get("isolated", []))
    actual_isolated = {v for v in variables if graph.degree(v) == 0}
    if declared_isolated != actual_isolated:
        issues.append(
            f"nœuds isolés déclarés {sorted(declared_isolated)} distincts des nœuds "
            f"de degré nul {sorted(actual_isolated)}")
    return issues


def compute_parsimony(
    df_merged: pd.DataFrame,
    critical_zones: List[str],
    cz: str,
    kappa: float,
) -> Tuple[pd.DataFrame, float, float]:
    """Calcule l'indice de parcimonie β et ses statistiques.

    Pour chaque ligne, β est le minimum sur les colonnes critiques du rapport
    ``κ·cz / colonne`` (division protégée : ``+inf`` si dénominateur nul).

    Args:
        df_merged: DataFrame fusionné.
        critical_zones: Zones critiques mises à jour.
        cz: Variable synthétique enveloppe.
        kappa: Coefficient multiplicateur optimal.

    Returns:
        Tuple ``(df_sample, max_beta, mean_beta)`` où ``df_sample`` contient les
        zones critiques, la colonne ``$\\alpha$cz`` (= κ·cz) et ``$\\beta$``.
    """
    df_sample = df_merged[critical_zones].copy()
    alpha_col = r"$\alpha$" + cz
    if alpha_col in df_sample.columns or r"$\beta$" in df_sample.columns:
        raise _numeric_error(
            f"Une zone porte un nom réservé aux colonnes calculées ('{alpha_col}' ou "
            "'$\\beta$') : renommez cette colonne.")
    with np.errstate(over="ignore", invalid="ignore"):
        envelope = kappa * df_sample[cz].to_numpy(dtype=float)
    _require_finite_envelope(envelope, kappa, cz)
    df_sample[alpha_col] = envelope

    columns_to_compare = [col for col in df_sample.columns if col != alpha_col]

    ratios = pd.DataFrame(index=df_sample.index)
    for col in columns_to_compare:
        denom = df_sample[col]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            quotient = df_sample[alpha_col] / denom
        # Un quotient non fini avec un dénominateur non nul est un débordement, non
        # la convention « +inf si le dénominateur est nul » : il est refusé.
        if bool(((denom != 0) & ~np.isfinite(quotient)).any()):
            raise _numeric_error(
                f"Débordement numérique dans le rapport κ·cible / '{col}' (dénominateur "
                "non nul mais très petit devant l'enveloppe) : β ne peut pas être calculé.")
        ratios[col] = np.where(denom != 0, quotient, np.inf)
    beta = ratios.min(axis=1)

    df_sample[r"$\beta$"] = beta
    max_beta = float(beta.max())
    mean_beta = float(beta.mean())
    return df_sample, max_beta, mean_beta


def beta_domain_report(
    df_merged: pd.DataFrame,
    critical_zones: List[str],
    cz: str,
    beta: "pd.Series | None" = None,
) -> Dict[str, Any]:
    """Domaine de définition de β : le calcul n'est pas modifié, son statut est publié.

    β est le minimum des rapports ``κ·cz / zone``. Son interprétation comme marge
    de conservatisme de l'enveloppe suppose des valeurs strictement positives :
    une zone nulle donne un rapport posé à ``+inf`` par convention (division
    protégée), et une valeur négative renverse le sens des rapports et de
    l'enveloppe multiplicative. Ce relevé compte les valeurs concernées.

    Args:
        df_merged: DataFrame fusionné.
        critical_zones: Zones conservées (incluant la cible).
        cz: Variable cible de la régularisation.
        beta: Série β déjà calculée (pour compter les valeurs infinies).

    Returns:
        Dictionnaire ``{beta_defined, n_zero_values, n_negative_values,
        zero_columns, negative_columns, n_infinite_beta, note}`` ; ``beta_defined``
        est faux dès qu'une valeur des zones conservées ou de la cible est négative
        ou nulle.
    """
    columns = list(dict.fromkeys(list(critical_zones) + [cz]))
    values = df_merged[columns]
    negative = (values < 0).sum()
    zero = (values == 0).sum()
    negative_columns = [str(c) for c in columns if int(negative[c]) > 0]
    zero_columns = [str(c) for c in columns if int(zero[c]) > 0]
    n_negative = int(negative.sum())
    n_zero = int(zero.sum())
    n_infinite = 0 if beta is None else int(np.isinf(beta.to_numpy(dtype=float)).sum())
    defined = n_negative == 0 and n_zero == 0
    note = ""
    if not defined:
        parts = []
        if n_negative:
            parts.append(f"{n_negative} valeur(s) négative(s) (colonnes {negative_columns})")
        if n_zero:
            parts.append(f"{n_zero} valeur(s) nulle(s) (colonnes {zero_columns})")
        note = (
            "β non interprétable comme marge de conservatisme : " + " et ".join(parts)
            + ". Un dénominateur nul donne un rapport posé à +inf (division protégée) ; "
            "une valeur négative renverse le sens des rapports. κ, β et leurs "
            "statistiques sont calculés comme d'habitude mais ne se lisent pas comme "
            "une enveloppe multiplicative.")
    return {
        "beta_defined": defined,
        "n_zero_values": n_zero,
        "n_negative_values": n_negative,
        "zero_columns": zero_columns,
        "negative_columns": negative_columns,
        "n_infinite_beta": n_infinite,
        "note": note,
    }
