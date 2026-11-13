"""Visualisation, post-traitement et exportation du pipeline OCA (version 9.0.0).

Ce module centralise toutes les sorties : fichiers texte (``OCA_Sort_Targets.txt``,
``OCA_OrdinalGraph_Targets.txt``), graphiques PDF (matrices de corrélation
ORDINALE — la matrice linéaire de Pearson a été retirée en version 3 —, boîtes
à moustaches, histogrammes, nuages de points, matrice de corrélation ordinale,
graphe de dominance orienté), rapports LaTeX compilés (``Result.pdf``,
``CriticalZones_Targets.pdf``), PDF consolidé de tous les résultats, et
``output.json``. Les formats reproduisent fidèlement le notebook d'origine.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

import latex_support  # noqa: E402

# Fusion PDF (V5) : ``pypdf`` est désormais la dépendance de référence, PyPDF2
# étant archivé depuis 2023 et n'émettant plus qu'un ``DeprecationWarning`` à
# l'import. L'ordre de résolution couvre les trois états du parc installé :
#   1. ``pypdf.PdfWriter``  — API de fusion des pypdf récents (>= 3.9, dont 5.x/6.x
#      où ``PdfMerger`` a été retiré) ;
#   2. ``pypdf.PdfMerger``  — pypdf antérieurs à 5.0 ;
#   3. ``PyPDF2.PdfMerger`` — installations historiques conservées à l'identique.
# Les trois classes exposent ``append``/``write``/``close`` : la fusion est donc
# rigoureusement la même quelle que soit la branche retenue.
try:  # pragma: no cover - dépend de l'environnement
    from pypdf import PdfWriter as PdfMerger  # type: ignore
except Exception:
    try:
        from pypdf import PdfMerger  # type: ignore
    except Exception:
        try:
            from PyPDF2 import PdfMerger  # type: ignore
        except Exception:
            PdfMerger = None  # type: ignore

# Les fichiers .txt de référence ont été écrits par le notebook via open(path, 'w')
# sans encodage explicite, donc dans l'encodage ANSI par défaut de Windows (cp1252).
# On reproduit ce même encodage pour une fidélité octet des .txt ; les .tex restent
# en UTF-8 (comme dans le notebook, qui écrivait les .tex avec encoding='utf-8').
# Un nom de zone qu'il ne sait pas représenter (lettre grecque par exemple) est
# remplacé par « ? » dans ces fichiers (errors="replace") : l'écriture n'échoue
# jamais ; le nom exact figure dans output.json, le classeur et les tableaux.
TXT_ENCODING = "cp1252"

#: Lecture de la notation ``A < B`` des relevés et des chaînes de comparaison.
NOTATION_SENTENCE = (
    "Lecture : « A < B » signifie que A précède B dans le classement de dominance "
    "(A majore B dans la proportion indiquée).")

#: Préfixe mathématique engendré par OCA pour l'enveloppe, ``2.00$\times$``.
_GENERATED_LABEL = re.compile(r"^(\s*[0-9]+(?:\.[0-9]+)?\$\\times\$)(.*)$", re.S)


def mpl_text(name: Any) -> str:
    """Nom brut affiché littéralement par matplotlib.

    Matplotlib interprète une paire de dollars comme une formule (Mathtext) : un nom
    de zone ou de variable contenant ``$`` s'afficherait comme une formule, voire
    ferait échouer le rendu. Les dollars des noms utilisateur sont donc échappés. Le
    préfixe ``2.00$\\times$`` du libellé de l'enveloppe, engendré par le programme et
    voulu comme formule, est conservé ; seul le nom de la cible qui le suit est
    échappé.

    Args:
        name: Nom à afficher.

    Returns:
        Chaîne à transmettre à matplotlib.
    """
    text = str(name)
    match = _GENERATED_LABEL.match(text)
    if match:
        return match.group(1) + match.group(2).replace("$", r"\$")
    return text.replace("$", r"\$")


__all__ = [
    "to_native",
    "mpl_text",
    "write_partial_status_pdf",
    "write_sort_txt",
    "write_ordinal_matrix_txt",
    "write_ordinal_graph_txt",
    "plot_scatter",
    "plot_hist",
    "plot_boxplot",
    "plot_ordinal_corr_matrix",
    "plot_ordinal_graph",
    "generate_result_table",
    "generate_step1_zones_sample_table",
    "generate_criticalzones_table",
    "split_column_bounds",
    "latex_stacked_table",
    "plain_column_format",
    "highlighted_tail_column_format",
    "emphasized_tail_column_format",
    "tail_column_format_for",
    "colored_column_format",
    "colored_header_labels",
    "dominance_color_map",
    "color_style",
    "is_text_color_mode",
    "is_categorical_color_map",
    "apply_saturation_and_alpha",
    "palette_colors",
    "zone_color_map",
    "resolve_colors",
    "relative_luminance",
    "contrast_ratio",
    "enforce_text_contrast",
    "enforce_background_contrast",
    "reset_style_notices",
    "matrix_color_map",
    "column_number_format",
    "format_frame_for_latex",
    "DEFAULT_MAX_COLUMNS_PER_BLOCK",
    "SPLIT_BLOCK_FONT_SIZE",
    "compile_latex",
    "merge_pdfs",
    "merge_pdfs_report",
    "first_latex_error",
    "tex_escape_name",
    "LATEX_PASS_TIMEOUT_S",
    "NOTATION_SENTENCE",
    "generate_report_intro",
    "build_output_json",
]

_SCATTER_COLORS = [
    "blue", "red", "green", "orange", "black", "purple", "gray", "C0", "C1", "C3", "C4",
    "royalblue", "lightcoral", "peachpuff", "lightyellow", "palegreen", "lightcoral", "plum",
    "lightpink", "powderblue", "lightblue", "thistle", "mistyrose", "lavenderblush", "honeydew",
    "lemonchiffon", "mintcream", "lightgoldenrodyellow", "linen",
]


# ===========================================================================
# Sérialisation
# ===========================================================================
def to_native(obj: Any) -> Any:
    """Convertit récursivement les types numpy/pandas en types Python natifs.

    Args:
        obj: Objet arbitraire.

    Returns:
        Structure sérialisable en JSON.
    """
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_native(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pd.DataFrame):
        return to_native(obj.to_dict())
    # Les valeurs non finies sont normalisees en 'null'. La norme JSON ne connait ni
    # 'NaN' ni 'Infinity' : un litteral de ce genre fait rejeter le document entier
    # par tout analyseur strict, et la trace est precisement ce qu'un outil aval
    # consomme. 'null' est en outre exact pour la diagonale d'une matrice de
    # majoration, qui n'est pas un nombre mais une comparaison laissee indefinie.
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _tight_layout(fig: "plt.Figure") -> None:
    """Ajuste la mise en page d'une figure sans bruit si matplotlib y renonce.

    Sur des données dégénérées (série constante par exemple) matplotlib peut
    refuser l'ajustement et émet un ``UserWarning``. La figure reste lisible : le
    message, sans portée pour l'utilisateur, n'est pas relayé.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(pad=2.0)


# ===========================================================================
# Écritures texte (formats identiques au notebook)
# ===========================================================================
def write_sort_txt(
    df_results: pd.DataFrame,
    dico_results: Dict[str, List[str]],
    updated_zones: List[str],
    path: str,
) -> str:
    """Écrit ``OCA_Sort_Targets.txt`` (tableau GPO + sélection + zones mises à jour).

    Reproduit exactement la séquence d'écriture des cellules d'origine.

    Args:
        df_results: Tableau des dominations (n, p, N, P).
        dico_results: Dictionnaire de sélection (colonnes sous/au-dessus du seuil).
        updated_zones: Liste des zones critiques après mise à jour.
        path: Chemin du fichier.

    Returns:
        Le chemin écrit.
    """
    text_ = (
        "Compter combien de fois une valeur de la colonne est supérieure à toutes "
        "les autres valeurs des autres colonnes sur la même ligne\n"
    )
    with open(path, "w", encoding=TXT_ENCODING, errors="replace") as f:
        f.write(text_)
        f.write(str(df_results))
    with open(path, "a", encoding=TXT_ENCODING, errors="replace") as f:
        f.write("\n" + str(dico_results) + "\n")
        f.write("\n" + "Liste des zones critiques après mise à jour :" + str(updated_zones) + "\n")
    return path


def write_ordinal_matrix_txt(
    prop_matrix: pd.DataFrame,
    path: str,
    header: str,
    threshold: Optional[float] = None,
) -> str:
    """Écrit le relevé textuel d'une matrice de corrélation ordinale.

    Assure la traçabilité numérique de la matrice au-delà de sa représentation
    graphique : la valeur en ligne ``i`` et colonne ``j`` est la proportion
    d'observations pour lesquelles la variable ``j`` est majorée par la variable
    ``i``.

    Lorsque ``threshold`` est fourni, un décompte par variable des dominations
    atteignant le seuil est adjoint. Ce décompte est le degré sortant attendu du
    nœud correspondant dans le graphe de dominance : il fournit la vérification
    numérique de la cohérence entre la matrice et le graphe.

    Args:
        prop_matrix: Matrice des proportions de dominance (diagonale NaN).
        path: Chemin du fichier.
        header: Ligne d'en-tête décrivant le contexte de calcul.
        threshold: Seuil de dominance, pour le décompte par variable.

    Returns:
        Le chemin écrit.
    """
    with open(path, "w", encoding=TXT_ENCODING, errors="replace") as f:
        f.write(header.rstrip() + "\n")
        f.write(
            "Lecture : la valeur en ligne i et colonne j est la proportion "
            "d'observations telles que j est majoree par i.\n\n"
        )
        f.write(prop_matrix.round(4).to_string())
        f.write("\n")
        if threshold is not None:
            f.write(
                f"\nDominations atteignant le seuil {threshold} "
                "(degre sortant attendu au graphe) :\n")
            for var in prop_matrix.index:
                row = prop_matrix.loc[var].dropna()
                dominated = [str(other) for other, value in row.items()
                             if float(value) >= threshold]
                f.write(f"  {var} : {len(dominated)}")
                if dominated:
                    f.write(" -> " + ", ".join(dominated))
                f.write("\n")
    return path


def write_ordinal_graph_txt(
    graph_data: Dict[str, Any],
    threshold: float,
    path: str,
) -> str:
    """Écrit le relevé textuel d'un graphe de dominance (relations et chaînes).

    Reproduit la séquence d'écriture de la cellule d'origine. Les chaînes de
    comparaison sont désormais reprises telles que calculées par
    ``oca_model.build_dominance_graph`` (suppression de la duplication de la
    remontée hiérarchique, à sortie identique).

    Args:
        graph_data: Structure renvoyée par ``oca_model.build_dominance_graph``.
        threshold: Seuil de proportion utilisé.
        path: Chemin du fichier.

    Returns:
        Le chemin écrit.
    """
    pairs_above = graph_data["pairs_above"]
    isolated = graph_data["isolated"]

    with open(path, "w", encoding=TXT_ENCODING, errors="replace") as f:
        f.write("Graphe des relations d'ordre entre les variables\n")
        f.write(NOTATION_SENTENCE + "\n")
    with open(path, "a", encoding=TXT_ENCODING, errors="replace") as f:
        f.write(f"Paires avec une proportion de violations de supériorité >= {threshold} :\n")
        for var_i, var_j, proportion in pairs_above:
            f.write(f"{var_i} < {var_j} : proportion = {proportion:.2f}\n")
        f.write("\nVariables qui n'ont pas passé le seuil (pas de relations d'ordre) :\n")
        for var in isolated:
            f.write(f"{var}\n")
        f.write("\nChaînes de comparaison :\n")
        for chain in graph_data["chains"]:
            f.write(" < ".join(chain) + "\n")
    return path


# ===========================================================================
# Graphiques
# ===========================================================================
def plot_scatter(
    df_merged: pd.DataFrame, critical_zones: List[str], cz: str, path: str,
    random_state: int = 6, style: Optional[Dict[str, Any]] = None,
    colors: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> str:
    """Trace les nuages de points (cibles vs ``De/E``) avec coefficient de Pearson.

    Args:
        df_merged: DataFrame fusionné.
        critical_zones: Variables affichées.
        cz: Variable de l'axe des abscisses (ex. ``De/E``).
        path: Chemin du PDF.
        random_state: Graine de l'échantillonnage.

    Returns:
        Le chemin du PDF.
    """
    df_sample = df_merged.sample(frac=1.0, random_state=random_state)
    variables = critical_zones
    target_col = cz
    series_colors = resolve_colors(variables, colors, style)
    n = len(variables)
    ncols = 3
    nrows = n // ncols + (n % ncols > 0)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 24))
    axes = np.atleast_1d(axes).flatten()

    # Mode ``background`` : le nuage porte la teinte de sa variable. Mode
    # ``text`` : les points passent au gris neutre et la teinte est reportée sur
    # le nom de la variable, en ordonnée, seul support coloré de la figure.
    text_mode = is_text_color_mode(style)
    i = -1
    for i, var in enumerate(variables):
        axes[i].scatter(
            df_sample[target_col], df_sample[var], s=100,
            c=[NEUTRAL_MARK_COLOR if text_mode else series_colors[i]],
            alpha=0.7, edgecolors="black", linewidths=0.5,
        )
        if text_mode:
            axes[i].set_ylabel(mpl_text(var), fontsize=18, color=series_colors[i],
                               fontweight="bold")
        else:
            axes[i].set_ylabel(mpl_text(var), fontsize=18)
        # L'abscisse porte la variable cible, qui est elle même une zone critique
        # lorsque la régularisation en a retenu une : elle reçoit donc sa teinte,
        # faute de quoi un même identifiant serait tantôt teinté en ordonnée et
        # noir en abscisse au sein de la même planche.
        target_color = (colors or {}).get(str(cz)) if text_mode else None
        if target_color is not None:
            axes[i].set_xlabel(mpl_text(cz), fontsize=18, color=target_color,
                               fontweight="bold")
        else:
            axes[i].set_xlabel(mpl_text(cz), fontsize=18)
        axes[i].tick_params(axis="x", labelsize=18)
        axes[i].tick_params(axis="y", labelsize=18)
        # Le coefficient de Pearson n'est défini que pour au moins deux
        # observations et deux séries non constantes : sinon aucun appel n'est fait
        # et la figure l'annonce (« variance nulle »).
        constant = []
        if df_sample[var].nunique() < 2:
            constant.append(str(var))
        if df_sample[target_col].nunique() < 2 and str(target_col) not in constant:
            constant.append(str(target_col))
        if len(df_sample) < 2 or constant:
            label = "r non défini (variance nulle)" if constant else "r non défini"
            if constant:
                _warn_once(
                    f"Nuage de points : variance nulle sur {constant} ; le coefficient de "
                    "Pearson n'est pas défini et n'est pas calculé.")
            fontsize = 16
        else:
            corr, _ = pearsonr(df_sample[var], df_sample[target_col])
            label = f"r={corr:.2f}"
            fontsize = 22
        ypos = df_sample[var].min() + (df_sample[var].max() - df_sample[var].min()) * 0.66
        xpos = df_sample[target_col].max() * 0.66
        axes[i].text(xpos, ypos, label, fontsize=fontsize, bbox=dict(facecolor="white", alpha=0.66))

    for ax in axes[i + 1:]:
        ax.set_visible(False)
    _tight_layout(fig)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_hist(
    df_merged: pd.DataFrame, critical_zones: List[str], path: str,
    random_state: int = 6, style: Optional[Dict[str, Any]] = None,
    colors: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> str:
    """Trace les histogrammes des distributions des cibles critiques.

    Args:
        df_merged: DataFrame fusionné.
        critical_zones: Variables affichées.
        path: Chemin du PDF.
        random_state: Graine de l'échantillonnage.

    Returns:
        Le chemin du PDF.
    """
    df_sample = df_merged.sample(frac=1.0, random_state=random_state)
    variables = critical_zones
    series_colors = resolve_colors(variables, colors, style)
    n = len(variables)
    ncols = 3
    nrows = n // ncols + (n % ncols > 0)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 24))
    axes = np.atleast_1d(axes).flatten()

    # Mode ``background`` : les barres portent la teinte de leur variable. Mode
    # ``text`` : elles passent au gris neutre et la teinte est reportée sur le
    # nom de la variable, en abscisse.
    text_mode = is_text_color_mode(style)
    i = -1
    for i, var in enumerate(variables):
        axes[i].hist(df_sample[var],
                     color=NEUTRAL_MARK_COLOR if text_mode else series_colors[i],
                     edgecolor="black", alpha=0.5)
        if text_mode:
            axes[i].set_xlabel(mpl_text(var), fontsize=18, color=series_colors[i],
                               fontweight="bold")
        else:
            axes[i].set_xlabel(mpl_text(var), fontsize=18)
        axes[i].set_ylabel("Count", fontsize=18)
        axes[i].tick_params(axis="x", labelsize=18)
        axes[i].tick_params(axis="y", labelsize=18)

    for ax in axes[i + 1:]:
        ax.set_visible(False)
    _tight_layout(fig)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_boxplot(df_merged: pd.DataFrame, critical_zones: List[str], path: str,
                 style: Optional[Dict[str, Any]] = None,
                 colors: Optional[Dict[str, Tuple[float, float, float]]] = None,
                 ranking: Optional[Sequence[str]] = None) -> str:
    """Trace les boîtes à moustaches comparatives des distributions (V7).

    La figure porte **deux panneaux** superposés, décrivant les mêmes
    distributions et ne différant que par la disposition horizontale de leurs
    colonnes :

        1. le panneau supérieur restitue les zones dans leur **ordre de
           déclaration**, tel que l'utilisateur l'a écrit dans ``input.json``, soit
           l'état antérieur à toute hiérarchisation ;
        2. le panneau inférieur les dispose par **score de dominance ordinale
           croissant**, soit l'état postérieur au classement de l'étape 1.

    Les deux panneaux correspondent ainsi terme à terme aux deux tableaux du
    livrable 04, et leur confrontation donne à voir la restructuration des
    colonnes qu'opère le classement. La teinte d'une zone, attribuée d'après son
    rang, est commune aux deux panneaux comme elle l'est aux deux tableaux : c'est
    elle qui permet de suivre le déplacement d'une distribution d'un panneau à
    l'autre.

    Aucun titre n'est apposé, conformément aux conventions de l'application. Les
    deux panneaux se distinguent d'eux mêmes : la teinte suivant le rang, le
    panneau inférieur présente une progression chromatique régulière que le
    panneau supérieur n'a pas.

    Args:
        df_merged: DataFrame fusionné.
        critical_zones: Variables affichées, dans leur ordre de déclaration.
        path: Chemin du PDF.
        style: Style de couleurs normalisé par :func:`color_style`.
        colors: Association de chaque zone vers sa teinte, établie une fois pour
            toute l'exécution par :func:`zone_color_map`.
        ranking: Zones dans l'ordre de dominance croissante. En son absence,
            l'ordre de déclaration est repris et les deux panneaux coïncident.

    Returns:
        Le chemin du PDF.
    """
    declared = [zone for zone in critical_zones if zone in df_merged.columns]
    if ranking:
        ordered = [zone for zone in ranking if zone in declared]
        ordered += [zone for zone in declared if zone not in ordered]
    else:
        ordered = list(declared)
    # Les deux panneaux sont tracés même lorsque les deux ordres coïncident, de
    # sorte que la structure du livrable ne dépende pas des données. C'est la règle
    # déjà retenue pour les deux tableaux du livrable 04.
    panels = [declared, ordered]

    fig, axes = plt.subplots(nrows=len(panels), ncols=1, figsize=(12, 12))
    axes = np.atleast_1d(axes).flatten()
    boxprops = dict(linewidth=2, color="black")
    whiskerprops = dict(linewidth=2, color="black")
    capprops = dict(linewidth=2, color="black")
    # Médiane en noir, doublée d'un liseré blanc. Le rouge employé jusqu'à la
    # version 6 se confondait avec une boîte que la carte de couleurs rendait
    # elle même rouge, et le repère central de la distribution devenait
    # indiscernable de son enveloppe. Le noir suffit sur les teintes claires,
    # mais une rampe continue sombre le masquerait tout autant : mesuré, le
    # contraste du noir tombe à 1,38 sur le violet initial de « viridis » et à
    # 1,31 sur le bleu initial de « jet ». Le liseré blanc garantit le repère sur
    # une aire de teinte quelconque, sans dépendre de la carte choisie.
    medianprops = dict(linewidth=4, color="black",
                       path_effects=[path_effects.withStroke(
                           linewidth=6.5, foreground="white")])
    flierprops = dict(marker="o", markerfacecolor="none", markeredgecolor="black", markersize=5,
                      linestyle="none", markeredgewidth=0.75)
    text_mode = is_text_color_mode(style)
    for axis, order in zip(axes, panels):
        # ``patch_artist`` est requis pour que les boîtes admettent un fond : sans
        # lui, matplotlib ne trace que leur contour et la teinte reste sans effet.
        shown = df_merged[order].copy()
        shown.columns = [mpl_text(name) for name in order]
        shown.boxplot(
            ax=axis, widths=0.7, boxprops=boxprops, whiskerprops=whiskerprops,
            capprops=capprops, medianprops=medianprops, flierprops=flierprops,
            patch_artist=True, return_type="axes",
        )
        # La teinte est prélevée sur l'association commune à tous les livrables et
        # appariée au nom de la zone, non à sa position : elle suit donc la même
        # distribution d'un panneau à l'autre malgré la permutation des colonnes.
        series_colors = resolve_colors(order, colors, style)
        # Mode ``background`` : la boîte porte la teinte. Mode ``text`` : la boîte
        # reste blanche et la teinte passe au nom de la variable, en abscisse.
        for patch, color in zip(axis.patches, series_colors):
            patch.set_facecolor(NEUTRAL_FILL_COLOR if text_mode else color)
        for label in axis.get_xticklabels():
            label.set_fontsize(14)
            label.set_rotation(90)
        if text_mode:
            for label, color in zip(axis.get_xticklabels(), series_colors):
                label.set_color(color)
                label.set_fontweight("bold")
        axis.set_ylabel("Value of the critical zones", fontsize=16)
        axis.tick_params(axis="y", labelsize=16)
        axis.grid(True, which="major", linestyle="--", linewidth=0.5)

    _tight_layout(fig)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ordinal_corr_matrix(
    prop_matrix: pd.DataFrame,
    path: str,
    fmt: str = ".2f",
    threshold: Optional[float] = None,
    style: Optional[Dict[str, Any]] = None,
    colors: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> str:
    """Trace la heatmap d'une matrice de corrélation ordinale (ordre de Pareto).

    La case ``(i, j)`` porte la proportion d'observations pour lesquelles la
    variable ``j`` est majorée par la variable ``i``. Une valeur voisine de 1
    atteste une relation de dominance quasi systématique de ``i`` sur ``j`` ; la
    diagonale est vide par construction.

    Cette fonction sert les deux matrices ordinales du pipeline : la matrice
    initiale, établie sur l'intégralité des zones critiques avant toute sélection,
    et la matrice de l'étape 2, augmentée de la colonne enveloppe régularisée.

    Révision V4 (cohérence matrice et graphe) : lorsque ``threshold`` est fourni,
    les cellules atteignant le seuil de dominance sont encadrées. Le lecteur
    dénombre ainsi directement, sur la matrice, les arêtes du graphe associé. Sans
    ce marquage, deux proportions telles que 0,9945 et 0,9956 se rendent par une
    teinte identique sur une échelle saturée à l'unité, alors que la première ne
    produit aucune arête et la seconde une arête : la figure ne permettait pas de
    reconstituer le graphe, d'où une incohérence apparente entre les deux.

    Aucun titre n'est apposé : la légende du document appelant décrit la figure.

    Args:
        prop_matrix: Matrice des proportions de dominance (diagonale NaN).
        path: Chemin du PDF.
        fmt: Format d'annotation des cellules. La matrice initiale emploie trois
            décimales, la discrimination des proportions au voisinage du seuil de
            dominance (typiquement 0,995) exigeant cette précision.
        threshold: Seuil de dominance. Les cellules l'atteignant sont encadrées,
            matérialisant exactement les arêtes du graphe de dominance.

    Returns:
        Le chemin du PDF.
    """
    n_vars = len(prop_matrix.columns)
    side = max(8.0, min(16.0, 2.0 + 1.6 * n_vars))
    plt.figure(figsize=(side, side * 0.86))
    shown_matrix = prop_matrix.copy()
    shown_matrix.index = [mpl_text(name) for name in prop_matrix.index]
    shown_matrix.columns = [mpl_text(name) for name in prop_matrix.columns]
    ax = sns.heatmap(
        shown_matrix, annot=True, fmt=fmt, cmap=matrix_color_map(style), vmin=0, vmax=1,
        cbar_kws={"shrink": 0.99}, annot_kws={"size": 12},
    )
    plt.xticks(fontsize=12, rotation=90)
    # Rotation nulle imposée : seaborn bascule les étiquettes d'ordonnée à la
    # verticale lorsqu'elles sont longues, ce qui les rend illisibles.
    plt.yticks(fontsize=12, rotation=0)
    ax.collections[0].colorbar.ax.tick_params(labelsize=12)
    _recolor_cell_annotations(ax, prop_matrix, matrix_color_map(style))

    # Les cases de la matrice encodent une grandeur continue : elles conservent
    # une rampe de couleurs dans les deux modes, sans quoi la figure perdrait son
    # objet. Le mode ``text`` teinte en revanche les étiquettes d'axes, de sorte
    # qu'une variable garde d'un livrable à l'autre la teinte qui la désigne.
    if is_text_color_mode(style):
        names = [str(name) for name in prop_matrix.columns]
        by_name = dict(zip([mpl_text(name) for name in names],
                           resolve_colors(names, colors, style)))
        for labels in (ax.get_xticklabels(), ax.get_yticklabels()):
            for label in labels:
                color = by_name.get(label.get_text())
                if color is not None:
                    label.set_color(color)
                    label.set_fontweight("bold")

    if threshold is not None:
        _frame_cells_above_threshold(ax, prop_matrix, threshold)

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def _recolor_cell_annotations(ax: "plt.Axes", prop_matrix: pd.DataFrame,
                              color_map_name: str) -> None:
    """Recolore chaque annotation de cellule pour maximiser son contraste.

    Seaborn décide de la couleur du chiffre porté par une cellule d'après un
    seuil unique de luminance du fond, fixé à 0,408 : au delà le chiffre est
    sombre, en deçà il est blanc. Ce seuil n'est pas celui qui maximise le
    contraste, dont le point d'égalité entre le blanc et le noir se situe à une
    luminance de 0,179. Les tons moyens d'une rampe divergente tombent
    précisément dans l'intervalle où la règle de seaborn choisit le blanc alors
    que le noir contrasterait bien mieux.

    La conséquence était mesurable sur le cas d'application livré : quatre
    cellules de la matrice de l'étape 1 portaient un chiffre blanc sur un ton
    moyen, à un rapport de contraste de 2,38, très en deçà du seuil de 4,5 que
    l'application s'impose partout ailleurs, et même du seuil de 3,0 réservé aux
    gros caractères. La rampe ``RdBu_r`` place ses tons moyens juste sous le seuil
    de seaborn, ``coolwarm`` juste au dessus, ce qui n'était affaire que de
    chance : les deux traversent ce point.

    La décision est donc reprise cellule par cellule, et le noir ou le blanc
    retenu selon celui qui contraste le mieux avec le fond effectivement calculé.
    Le remède vaut pour **toute** rampe que l'utilisateur déclarerait, et non pour
    les seules deux qui ont été mesurées.

    Args:
        ax: Axes portant la heatmap et ses annotations.
        prop_matrix: Matrice des proportions, telle que rendue.
        color_map_name: Nom de la carte de couleurs employée pour le fond.
    """
    color_map = _resolve_color_map(color_map_name)
    rows, columns = list(prop_matrix.index), list(prop_matrix.columns)
    for text in ax.texts:
        # La position d'une annotation est le centre de sa cellule : les indices
        # s'en déduisent, ce qui évite de présumer de l'ordre dans lequel seaborn
        # a créé les objets, et reste juste en présence de cases vides.
        x, y = text.get_position()
        column_index, row_index = int(x), int(y)
        if not (0 <= row_index < len(rows) and 0 <= column_index < len(columns)):
            continue
        value = prop_matrix.iloc[row_index, column_index]
        if pd.isna(value):
            continue
        background = color_map(min(1.0, max(0.0, float(value))))[:3]
        on_white = contrast_ratio((1.0, 1.0, 1.0), background)
        on_black = contrast_ratio((0.0, 0.0, 0.0), background)
        text.set_color("white" if on_white >= on_black else "black")


def _frame_cells_above_threshold(
    ax: "plt.Axes",
    prop_matrix: pd.DataFrame,
    threshold: float,
) -> None:
    """Encadre les cellules atteignant le seuil de dominance.

    Le contour est appliqué exactement aux couples ``(i, j)`` vérifiant
    ``prop_matrix[i, j] >= threshold``, soit précisément le critère de création
    d'une arête dans :func:`oca_model.build_dominance_graph`. Le dénombrement des
    cellules encadrées d'une ligne restitue donc le degré sortant du nœud
    correspondant, et celui d'une colonne son degré entrant.

    Args:
        ax: Axes portant la heatmap.
        prop_matrix: Matrice des proportions de dominance.
        threshold: Seuil de dominance.
    """
    rows = list(prop_matrix.index)
    columns = list(prop_matrix.columns)
    for row_index, var_i in enumerate(rows):
        for column_index, var_j in enumerate(columns):
            value = prop_matrix.loc[var_i, var_j]
            if pd.isna(value) or float(value) < threshold:
                continue
            ax.add_patch(
                plt.Rectangle(
                    (column_index, row_index), 1, 1,
                    fill=False, edgecolor="black", linewidth=2.6, zorder=5,
                )
            )


# Marge de sécurité des graphes orientés, en fraction de l'étendue du contenu.
# Elle s'applique après mesure des étiquettes effectivement rendues, et non à
# l'étendue des seuls nœuds.
GRAPH_SAFETY_MARGIN = 0.12
# Marge physique, en pouces, ménagée autour du contenu à l'enregistrement. Elle
# garantit un blanc tournant même après le recadrage serré, lequel rognerait
# autrement jusqu'au dernier pixel des étiquettes.
GRAPH_SAVE_PAD_INCHES = 0.35


def _frame_graph_content(fig: "plt.Figure", ax: "plt.Axes",
                         labels: Sequence[Any],
                         margin: float = GRAPH_SAFETY_MARGIN) -> None:
    """Centre le contenu d'un graphe et ménage une marge autour des étiquettes.

    Les limites des axes sont recalculées d'après l'emprise **réellement rendue**
    du contenu, étiquettes textuelles comprises, et non d'après la seule position
    des nœuds. C'est indispensable : un libellé de zone mesure plusieurs dizaines
    de fois la largeur du disque qui le porte, et les variables synthétiques
    engendrent des libellés plus longs encore, ainsi ``1.53×TRESCA_LIN_Z1_L``.

    Une marge relative à l'étendue des seuls nœuds ne suffisait pas. Sur un graphe
    de trois nœuds, l'étendue des positions est de l'ordre de l'unité, et une marge
    de huit centièmes ne représente qu'une fraction de la largeur d'une étiquette :
    celle ci sortait du cadre et se trouvait tronquée.

    La mesure est conduite en coordonnées d'affichage puis reportée en coordonnées
    de données, ce qui la rend indépendante de l'échelle courante.

    Le calcul est itératif, et il doit l'être : l'emprise d'une étiquette est fixe
    en pixels, de sorte qu'élargir les limites la réduit exprimée en données, ce qui
    déplace le centre du contenu. Quatre passes suffisent, la suite convergeant
    géométriquement. Deux passes laissaient un résidu de dissymétrie de l'ordre de
    trois pixels, mesurable et donc à corriger.

    Le contenu est enfin centré : les limites sont établies symétriquement autour
    du milieu de son emprise, sur les deux axes.

    Args:
        fig: Figure portant les axes.
        ax: Axes du graphe.
        labels: Objets textuels des étiquettes de nœuds.
        margin: Marge de sécurité, en fraction de l'étendue du contenu.
    """
    for _ in range(4):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = []
        for artist in list(labels) + list(ax.collections) + list(ax.patches):
            try:
                box = artist.get_window_extent(renderer=renderer)
            except Exception:  # pragma: no cover - artiste sans emprise mesurable
                continue
            if box.width > 0 and box.height > 0:
                boxes.append(box)
        if not boxes:
            return
        inverse = ax.transData.inverted()
        xs, ys = [], []
        for box in boxes:
            (x0, y0), (x1, y1) = inverse.transform([(box.x0, box.y0),
                                                    (box.x1, box.y1)])
            xs.extend([x0, x1])
            ys.extend([y0, y1])
        # Étendue non nulle imposée : un graphe réduit à un nœud unique produirait
        # une emprise dégénérée et des limites inversées.
        span_x = max(max(xs) - min(xs), 1e-9)
        span_y = max(max(ys) - min(ys), 1e-9)
        centre_x = 0.5 * (max(xs) + min(xs))
        centre_y = 0.5 * (max(ys) + min(ys))
        half_x = 0.5 * span_x * (1.0 + 2.0 * margin)
        half_y = 0.5 * span_y * (1.0 + 2.0 * margin)
        ax.set_xlim(centre_x - half_x, centre_x + half_x)
        ax.set_ylim(centre_y - half_y, centre_y + half_y)


def plot_ordinal_graph(graph_data: Dict[str, Any], path: str,
                       style: Optional[Dict[str, Any]] = None,
                       colors: Optional[Dict[str, Tuple[float, float, float]]] = None) -> str:
    """Trace le graphe de dominance orienté par niveaux hiérarchiques.

    Le tracé porte sur l'intégralité des nœuds du graphe, y compris les variables
    dépourvues de relation d'ordre franchissant le seuil. Ces dernières sont
    regroupées sur une rangée dédiée sous la hiérarchie et distinguées par un
    remplissage propre, de sorte qu'aucune zone critique ne puisse être omise.

    Révision V4 (lisibilité et cohérence) : la figure ne porte ni titre ni légende,
    la légende du document appelant décrivant la figure. La courbure de chaque
    arête dépend de sa portée hiérarchique et de son rang parmi les arêtes de même
    source, en sorte que deux arêtes issues d'un même nœud ne se superposent pas et
    qu'une arête franchissant plusieurs niveaux ne puisse être confondue avec une
    arête vers un nœud intermédiaire. Le dénombrement visuel des arêtes incidentes
    à un nœud restitue ainsi exactement le nombre de cellules encadrées de la ligne
    correspondante de la matrice ordinale.

    Args:
        graph_data: Structure renvoyée par ``oca_model.build_dominance_graph``.
        path: Chemin du PDF.

    Returns:
        Le chemin du PDF.
    """
    graph = graph_data["graph"]
    pos = graph_data["pos"]
    levels = graph_data["levels"]
    isolated = list(graph_data["isolated"])

    # Teintes issues du style configuré. Chaque variable reçoit celle de son rang
    # dans l'ordre des variables du graphe : le nœud et le faisceau d'arêtes qu'il
    # émet partagent donc la même couleur, ce qui rend le faisceau imputable à sa
    # source. Les nœuds isolés, dépourvus d'arête, sont rendus en gris neutre : la
    # teinte de rang n'aurait rien à désigner pour eux.
    variables = list(graph_data.get("variables", list(graph.nodes())))
    variable_colors = dict(zip(variables, resolve_colors(variables, colors, style)))
    isolated_set = set(isolated)
    # Mode ``background`` : le disque du nœud porte la teinte. Mode ``text`` : le
    # disque reste blanc et la teinte passe à l'étiquette, composée en gras. Les
    # arêtes conservent en revanche la teinte de leur source dans les deux modes,
    # cette teinte imputant le faisceau à son nœud émetteur : elle code une
    # information de structure, non un ornement, et la retirer priverait la
    # figure dense de son seul moyen de dénombrement visuel.
    # Les nœuds isolés demeurent distingués par un remplissage propre dans les
    # deux modes, plus discret en mode ``text`` : leur libellé n'y est pas teinté,
    # la teinte de rang n'ayant rien à désigner pour une variable dépourvue de
    # relation d'ordre.
    text_mode = is_text_color_mode(style)
    node_colors = [
        ((0.93, 0.93, 0.93) if text_mode else (0.88, 0.88, 0.88))
        if node in isolated_set
        else NEUTRAL_FILL_COLOR if text_mode
        else variable_colors.get(node, (0.80, 0.80, 0.80))
        for node in graph.nodes()
    ]
    color_cycle = [variable_colors[name] for name in variables] or [(0.2, 0.4, 0.8)]

    pos_flipped = {node: (x, -y) for node, (x, y) in pos.items()}
    n_nodes = graph.number_of_nodes()
    width = max(10.0, min(26.0, 3.0 + 2.4 * max(1, n_nodes)))
    height = 3.2 + 1.9 * (graph_data["max_level"] + (2 if isolated else 1))
    fig, ax = plt.subplots(figsize=(width, min(height, 14.0)))

    node_size = 2500
    node_collection = nx.draw_networkx_nodes(
        graph, pos_flipped, ax=ax, node_color=node_colors, node_size=node_size,
        edgecolors="black", linewidths=0.8,
    )
    node_collection.set_zorder(5)
    if text_mode:
        # Une passe par nœud : ``draw_networkx_labels`` n'admet qu'une couleur de
        # police pour l'ensemble des étiquettes, et c'est ici chaque étiquette qui
        # doit porter la teinte de sa variable.
        #
        # Un cartouche blanc est glissé sous chaque étiquette. Les libellés de
        # zones excèdent en largeur le disque du nœud, en sorte que les arêtes
        # incidentes les traversent ; le disque teinté du mode ``background`` en
        # masquait la portion centrale, mais un disque blanc n'y suffit plus. Le
        # cartouche rétablit la lisibilité sans déplacer l'étiquette, ce qui
        # préserve l'imputation de chaque libellé à son nœud.
        plate = dict(facecolor="white", edgecolor="none", pad=0.15)
        label_dict: Dict[str, Any] = {}
        for node in graph.nodes():
            color = variable_colors.get(node)
            drawn = nx.draw_networkx_labels(
                graph, pos_flipped, ax=ax, labels={node: mpl_text(node)}, font_size=10,
                font_weight="bold", bbox=plate, clip_on=False,
                font_color="black" if node in isolated_set or color is None else color,
            )
            label_dict.update(drawn)
    else:
        label_dict = nx.draw_networkx_labels(
            graph, pos_flipped, ax=ax, labels={node: mpl_text(node) for node in graph.nodes()},
            font_size=10, clip_on=False)
    # ``clip_on`` est vrai par défaut dans ``draw_networkx_labels`` : une étiquette
    # dépassant les limites des axes s'y trouvait **tronquée**, et non simplement
    # débordante. C'est la cause première des libellés incomplets constatés sur le
    # graphe de l'étape 2, où la variable régularisée porte le libellé le plus long.
    # Le découpage est donc levé, et le cadrage confié à :func:`_frame_graph_content`
    # qui mesure l'emprise réelle des étiquettes.
    for text in label_dict.values():
        text.set_zorder(6)
        text.set_clip_on(False)

    # Teinte par nœud source : le faisceau émis par chaque variable dominante
    # devient identifiable, donc dénombrable. Un codage par niveau attribuait une
    # teinte commune à toutes les variables d'un même rang, ce qui interdisait
    # d'imputer visuellement une arête à sa source sur les figures denses.
    source_rank = graph_data.get("source_rank", {})
    for source, target, rad in _edge_curvatures(graph, levels):
        source_level = levels.get(source, 0)
        edge_color = color_cycle[source_rank.get(source, source_level) % len(color_cycle)]
        # ``node_size`` doit être transmis : il conditionne le retrait appliqué
        # aux extrémités de l'arête, faute de quoi les pointes de flèche sont
        # masquées sous les disques des nœuds et le sens de dominance devient
        # illisible.
        edge_collection = nx.draw_networkx_edges(
            graph, pos_flipped, ax=ax, edgelist=[(source, target)], edge_color=edge_color,
            arrowsize=20, connectionstyle=f"arc3,rad={rad}", width=2,
            node_size=node_size,
        )
        for artist in edge_collection:
            artist.set_zorder(1 + source_level)

    ax.axis("off")
    # Marge initiale, qui contient les arêtes de forte courbure. Une arête reliant
    # deux nœuds de même niveau, configuration possible après condensation d'un
    # cycle, contourne la rangée par un arc dont le sommet échappe à la boîte
    # englobante calculée sur les seuls nœuds.
    ax.margins(x=0.08, y=0.26)
    plt.tight_layout()
    # Cadrage définitif (V7) : le contenu est centré et les limites établies d'après
    # l'emprise réellement rendue des étiquettes, arêtes et nœuds compris.
    _frame_graph_content(fig, ax, list(label_dict.values()))
    # ``pad_inches`` ménage un blanc tournant physique. Le recadrage serré rogne
    # jusqu'au dernier pixel des artistes tracés : sans ce retrait, les étiquettes
    # affleureraient le bord du document quand bien même les limites des axes leur
    # laisseraient de la place.
    plt.savefig(path, bbox_inches="tight", pad_inches=GRAPH_SAVE_PAD_INCHES)
    plt.close(fig)
    return path


def _edge_curvatures(
    graph: "nx.DiGraph",
    levels: Dict[str, int],
) -> List[Tuple[str, str, float]]:
    """Attribue une courbure distincte à chaque arête, par source.

    Trois exigences de lisibilité sont satisfaites simultanément :

        1. deux arêtes partageant la même source reçoivent des courbures
           distinctes, réparties symétriquement autour de la corde, ce qui interdit
           leur superposition et rend leur dénombrement visuel fiable ;
        2. la courbure croît avec la portée hiérarchique, de sorte qu'une arête
           franchissant plusieurs niveaux s'écarte des nœuds intermédiaires et ne
           puisse leur être imputée ;
        3. une arête reliant deux nœuds de même niveau, configuration possible
           après condensation d'un cycle, reçoit une courbure accentuée qui la fait
           contourner la rangée au lieu de la traverser.

    Args:
        graph: Graphe orienté de dominance.
        levels: Niveau hiérarchique de chaque nœud.

    Returns:
        Liste de triplets ``(source, cible, courbure)`` couvrant toutes les arêtes.
    """
    curvatures: List[Tuple[str, str, float]] = []
    for source in graph.nodes():
        successors = list(graph.successors(source))
        if not successors:
            continue
        # Tri par abscisse cible implicite : l'ordre des successeurs suit celui de
        # déclaration des nœuds, déjà déterministe, donc reproductible.
        count = len(successors)
        offset = (count - 1) / 2.0
        for rank, target in enumerate(successors):
            span = abs(levels.get(target, 0) - levels.get(source, 0))
            base = 0.10 if span <= 1 else 0.10 + 0.09 * (span - 1)
            if span == 0:
                base = 0.42  # contournement d'une rangée (cycle condensé)
            spread = 0.0 if count == 1 else 0.055 * (rank - offset)
            curvatures.append((source, target, round(base + spread, 4)))
    return curvatures


# ===========================================================================
# Mise en forme numérique adaptative des tableaux (V6)
# ===========================================================================
# Format standard, hérité des versions antérieures : trois chiffres significatifs.
STANDARD_NUMBER_FORMAT = "{:.3g}"
# Format décimal de haute précision, pour les colonnes dont les valeurs sont
# petites mais restent lisibles en notation positionnelle.
DECIMAL_NUMBER_FORMAT = "{:.5f}"
# Notation scientifique, pour les colonnes dont les valeurs sont si faibles que
# la notation positionnelle les afficherait toutes à zéro.
SCIENTIFIC_NUMBER_FORMAT = "{:.3e}"
# Seuils de bascule, appliqués à la médiane des valeurs absolues non nulles.
SMALL_VALUE_THRESHOLD = 1.0
SCIENTIFIC_THRESHOLD = 1e-3


def column_number_format(values: "np.ndarray") -> str:
    """Choisit le format d'affichage d'une colonne d'après sa distribution (V6).

    Le format ``.3g`` employé jusqu'à la version 5 tronque à zéro toute valeur
    inférieure à un millième, ce qui rend illisibles les jeux de données dont les
    grandeurs sont petites, par exemple des contraintes exprimées en unités
    réduites ou des rapports adimensionnels. La règle retenue observe la
    **médiane des valeurs absolues non nulles**, statistique insensible aux
    valeurs aberrantes, et bascule sur une représentation adaptée.

    Règle de décision :

        1. médiane supérieure ou égale à 1, format standard ``.3g``, soit le
           comportement des versions antérieures ;
        2. médiane comprise entre un millième et 1, format décimal ``.5f``, qui
           conserve cinq décimales significatives ;
        3. médiane inférieure à un millième, notation scientifique ``.3e``.

    Une colonne entièrement nulle, vide ou non finie conserve le format standard :
    il n'y a alors aucune précision à préserver.

    Args:
        values: Valeurs numériques de la colonne.

    Returns:
        Gabarit de formatage, au format des chaînes ``str.format``.
    """
    finite = np.abs(np.asarray(values, dtype=float))
    finite = finite[np.isfinite(finite)]
    nonzero = finite[finite > 0.0]
    if nonzero.size == 0:
        return STANDARD_NUMBER_FORMAT
    median = float(np.median(nonzero))
    if median >= SMALL_VALUE_THRESHOLD:
        return STANDARD_NUMBER_FORMAT
    if median >= SCIENTIFIC_THRESHOLD:
        return DECIMAL_NUMBER_FORMAT
    return SCIENTIFIC_NUMBER_FORMAT


def format_frame_for_latex(frame: pd.DataFrame) -> pd.DataFrame:
    """Formate un tableau numérique colonne par colonne, selon sa distribution.

    Le choix du format est établi indépendamment pour chaque colonne, de sorte
    qu'une colonne de faibles amplitudes ne soit pas écrasée par le voisinage
    d'une colonne de grandes amplitudes.

    Args:
        frame: Tableau de valeurs numériques.

    Returns:
        Tableau de chaînes, de mêmes index et colonnes que l'entrée.
    """
    formatted: Dict[Any, pd.Series] = {}
    for column in frame.columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        template = column_number_format(numeric.to_numpy(dtype=float))
        formatted[column] = frame[column].map(
            lambda value, spec=template: spec.format(value))
    return pd.DataFrame(formatted, index=frame.index, columns=frame.columns)


# ===========================================================================
# Découpage adaptatif des tableaux LaTeX de grande dimension (V6)
# ===========================================================================
DEFAULT_MAX_COLUMNS_PER_BLOCK = 6
# Taille de police des blocs de tableaux. Fixe et identique pour tous les blocs de
# tous les tableaux : la cohérence typographique prime sur l'occupation de la
# justification.
#
# Le choix de ``scriptsize`` procède d'une mesure et non d'une préférence. Un
# tableau de six colonnes dont les libellés sont des identifiants de quinze
# caractères, cas du jeu de démonstration, occupe naturellement 509 pt pour une
# justification de 452 pt. La mise à l'échelle du texte étant proscrite, les seuls
# leviers sont le corps de caractères et l'espacement inter colonnes. Les
# débordements mesurés, en points, sont les suivants :
#
#     footnotesize / 6 pt : 86.8      scriptsize / 4 pt : 27.8
#     footnotesize / 4 pt : 58.8      scriptsize / 3 pt : 13.8
#     footnotesize / 2 pt : 30.8      scriptsize / 2 pt : tient
#     footnotesize / 1 pt : 16.8
#
# ``footnotesize`` ne tient donc pas, même à espacement quasi nul. ``scriptsize``
# assorti d'un espacement de 2 pt tient, et demeure la plus douce des réductions
# qui satisfasse l'exigence de composition sans dépassement de marge. Un jeu de
# données aux libellés plus courts peut revenir à ``footnotesize`` par la clé
# ``latex_output.table_font_size``.
SPLIT_BLOCK_FONT_SIZE = "scriptsize"
# Demi espacement inter colonnes. Resserrer les colonnes réduit la largeur du
# tableau sans toucher au corps de caractères, ce qui préserve l'uniformité
# typographique exigée.
TABLE_COLUMN_SEPARATION = "2pt"


def split_column_bounds(n_columns: int, max_per_block: int) -> List[Tuple[int, int]]:
    """Répartit ``n_columns`` colonnes en blocs de largeurs les plus homogènes.

    Un tableau dont le nombre de colonnes excède le seuil de lisibilité est
    découpé en plusieurs sous-tableaux successifs, empilés verticalement. La
    répartition est équilibrée plutôt que gloutonne : découper 25 colonnes avec
    un seuil de 8 produit quatre blocs de 7, 6, 6 et 6 colonnes, et non trois
    blocs de 8 suivis d'un bloc résiduel d'une seule colonne, dont la lecture
    serait déséquilibrée.

    Args:
        n_columns: Nombre total de colonnes du tableau.
        max_per_block: Nombre maximal de colonnes admis dans un bloc.

    Returns:
        Liste de couples ``(début, fin)`` d'indices de colonnes, bornes de fin
        exclues, couvrant l'intégralité du tableau dans l'ordre d'origine. Une
        liste à un seul élément signifie qu'aucun découpage n'est nécessaire.
    """
    max_per_block = max(1, int(max_per_block))
    if n_columns <= max_per_block:
        return [(0, n_columns)]
    n_blocks = math.ceil(n_columns / max_per_block)
    base, extra = divmod(n_columns, n_blocks)
    bounds: List[Tuple[int, int]] = []
    start = 0
    for index in range(n_blocks):
        width = base + (1 if index < extra else 0)
        bounds.append((start, start + width))
        start += width
    return bounds


def _escape_percent(text: str) -> str:
    """Échappe les pourcents LaTeX non déjà échappés.

    Args:
        text: Fragment LaTeX.

    Returns:
        Fragment dont les ``%`` isolés sont précédés d'une contre-oblique.
    """
    return re.sub(r"(?<!\\)%", r"\\%", text)


# Caractères que LaTeX interprète : ils sont remplacés en UNE passe, caractère par
# caractère, de sorte qu'aucun échappement ne soit lui même ré-échappé. Le trait
# bas n'y figure pas : le paquet ``underscore``, chargé par tous les préambules,
# le compose tel quel dans le texte courant.
_TEX_SPECIALS = {
    "&": r"\&", "#": r"\#", "%": r"\%", "{": r"\{", "}": r"\}", "$": r"\$",
    "^": r"\^{}", "~": r"\~{}", "\\": r"\textbackslash{}",
    "<": r"\textless{}", ">": r"\textgreater{}", "|": r"\textbar{}",
}

# Lettres grecques : absentes des polices de texte, composées en mode mathématique.
_TEX_GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "varepsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "κ": "kappa", "λ": "lambda", "μ": "mu",
    "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau",
    "υ": "upsilon", "φ": "varphi", "χ": "chi", "ψ": "psi", "ω": "omega",
    "Γ": "Gamma", "Δ": "Delta", "Θ": "Theta", "Λ": "Lambda", "Ξ": "Xi", "Π": "Pi",
    "Σ": "Sigma", "Φ": "Phi", "Ψ": "Psi", "Ω": "Omega",
}

# Fragments mathématiques produits par le programme lui même dans les en-têtes
# (colonnes enveloppe et β) : LaTeX de confiance, jamais échappé.
_TRUSTED_MATH_TOKENS = (r"$\alpha$", r"$\beta$", r"$\times$")


def tex_escape_name(text: Any, keep_math_tokens: bool = False) -> str:
    """Échappe un nom brut (zone, variable) avant insertion dans du LaTeX.

    Les caractères ``& # % { } $ ^ ~ \\ < > |`` sont échappés en une seule passe ;
    les lettres grecques passent en mode mathématique ; tout autre caractère hors
    Latin-1 devient ``?``. Le trait bas est laissé tel quel (paquet ``underscore``).

    Args:
        text: Nom brut.
        keep_math_tokens: Si vrai, les fragments ``$\\alpha$``, ``$\\beta$`` et
            ``$\\times$`` produits par le programme sont conservés.

    Returns:
        Texte sûr pour LaTeX.
    """
    source = str(text)
    if keep_math_tokens:
        pattern = "(" + "|".join(re.escape(token) for token in _TRUSTED_MATH_TOKENS) + ")"
        pieces = re.split(pattern, source)
        return "".join(
            piece if piece in _TRUSTED_MATH_TOKENS else tex_escape_name(piece)
            for piece in pieces)
    out: List[str] = []
    for char in source:
        if char in _TEX_SPECIALS:
            out.append(_TEX_SPECIALS[char])
        elif char in _TEX_GREEK:
            out.append(r"\ensuremath{\%s}" % _TEX_GREEK[char])
        elif ord(char) > 255:
            out.append("?")
        else:
            out.append(char)
    return "".join(out)


# ===========================================================================
# Style de couleurs configurable (V6)
# ===========================================================================
# Carte de couleurs par défaut. ``tab10`` est le cycle catégoriel de matplotlib,
# celui des couleurs C0, C1, C2 et suivantes : ses teintes sont distinctes deux à
# deux, ce qui convient au repérage de catégories, là où une rampe continue
# rapproche les teintes voisines.
DEFAULT_COLOR_MAP = "tab10"
# Cartes par défaut de la version 7. Les deux usages de la couleur sont dissociés,
# car ils n'appellent pas la même famille de cartes.
#
#   1. Le fond des matrices de corrélation ordinale encode une **grandeur
#      continue**, une proportion de majoration dans [0, 1]. Il appelle une rampe
#      continue, et de préférence divergente : ``RdBu_r`` distingue nettement les
#      proportions faibles des proportions fortes de part et d'autre de la
#      demie, ce qui est précisément la lecture qu'une matrice de dominance
#      demande.
#   2. La palette des zones critiques désigne des **catégories** à distinguer
#      deux à deux, sans ordre de grandeur intrinsèque autre que leur rang de
#      dominance. Une rampe telle que ``jet`` y matérialise la progression du
#      classement, du froid pour les moins dominantes au chaud pour les plus
#      dominantes.
#
# Jusqu'à la version 6, une carte unique servait les deux usages, et la seconde
# devait céder à la première par un repli automatique dès qu'elle était
# catégorielle. La dissociation supprime ce compromis.
DEFAULT_COLOR_MAP_MATRIX = "RdBu_r"
DEFAULT_COLOR_MAP_CRITICAL_ZONES = "jet"
# Saturation par défaut, dans [0, 1]. La valeur 1 restitue la teinte de la carte
# sans atténuation ; la valeur 0 la mêle à moitié de blanc, produisant un pastel.
DEFAULT_COLOR_SATURATION = 1.0
# Opacité par défaut, dans [0, 1].
DEFAULT_COLOR_ALPHA = 1.0
# Au delà de ce nombre de teintes, une carte est tenue pour continue et convient
# donc au fond des matrices ; en deçà, elle est catégorielle.
CATEGORICAL_COLOR_MAP_LIMIT = 20
# Carte employée pour les matrices lorsque la carte configurée est catégorielle.
MATRIX_FALLBACK_COLOR_MAP = "coolwarm"
# Destination de la couleur. ``background`` teinte les aires, fonds de colonne,
# disques des nœuds, boîtes et marques de données : c'est le rendu des supports
# de présentation. ``text`` teinte les seuls identifiants, noms de variables
# composés en gras, et laisse les aires neutres : c'est le rendu attendu d'une
# figure destinée à un article scientifique, où la parcimonie chromatique est de
# règle et où l'impression peut être monochrome hors des libellés.
DEFAULT_COLOR_APPLY_TO = "background"
COLOR_APPLY_MODES = ("background", "text")
# Gris des marques de données en mode ``text``. Les barres et les points ne
# portent alors plus d'information de catégorie, celle ci étant reportée sur le
# libellé : une teinte neutre suffit et n'entre pas en concurrence avec lui.
NEUTRAL_MARK_COLOR = (0.72, 0.72, 0.72)
# Remplissage des boîtes et des disques de nœuds en mode ``text``.
NEUTRAL_FILL_COLOR = (1.0, 1.0, 1.0)
# Rapport de contraste minimal exigé d'un libellé teinté sur fond blanc. C'est le
# seuil AA des règles d'accessibilité pour le texte courant. En mode ``text`` la
# teinte porte l'identification de la variable : une teinte illisible ne prive pas
# seulement la figure d'un ornement, elle la prive de son repérage.
TEXT_MODE_MIN_CONTRAST = 4.5
# Plancher de l'assombrissement automatique. En deçà, les teintes convergeraient
# vers le noir et cesseraient de se distinguer les unes des autres.
TEXT_MODE_MIN_DARKENING = 0.10
# Carte continue employée lorsqu'une carte catégorielle ne porte pas assez de
# teintes distinctes pour le nombre de variables à désigner.
CONTINUOUS_FALLBACK_COLOR_MAP = "turbo"
# Avertissements déjà émis, afin qu'un repli ne soit signalé qu'une fois par
# exécution. Sans cette garde, la résolution d'une carte inconnue, appelée une
# fois par figure et par tableau, inonde la sortie standard du même message.
_REPORTED_WARNINGS: set = set()


def _warn_once(message: str) -> None:
    """Consigne un avertissement au journal, une seule fois par exécution.

    Le message est porté au journal ``oca``, et non à la seule sortie standard.
    Une classe entière d'avertissements y échappait auparavant : les replis de
    carte de couleurs et les corrections de contraste s'affichaient à la console
    et ne figuraient ni au journal d'exécution, ni à la section du rapport
    consolidé qui le reproduit, ni au fichier de résultats. Le rapport archivé
    d'une exécution ne portait donc pas trace du mécanisme de rattrapage qui
    l'avait pourtant modifiée.

    Args:
        message: Texte de l'avertissement.
    """
    if message in _REPORTED_WARNINGS:
        return
    _REPORTED_WARNINGS.add(message)
    logging.getLogger("oca").warning(message)


def _inform_once(message: str) -> None:
    """Consigne une information au journal, une seule fois par exécution.

    Réservé aux ajustements automatiques documentés et voulus, tel le
    rehaussement du contraste d'une palette. Ce ne sont pas des écarts : les
    porter au rang d'avertissement banaliserait ceux qui en sont, et conduirait à
    chercher une correction là où le comportement est nominal.

    Args:
        message: Texte de l'information.
    """
    if message in _REPORTED_WARNINGS:
        return
    _REPORTED_WARNINGS.add(message)
    logging.getLogger("oca").info(message)


def reset_style_notices() -> None:
    """Vide la mémoire des avertissements déjà émis.

    Appelée au début de chaque sous-dossier d'un traitement par lot : chaque
    exécution étant strictement isolée, son journal doit porter ses propres
    avertissements et non hériter du silence de la précédente.
    """
    _REPORTED_WARNINGS.clear()


def color_style(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Normalise le style de couleurs déclaré par l'utilisateur.

    Deux cartes distinctes sont déclarées depuis la version 7,
    ``color_map_matrix`` pour le fond des matrices et
    ``color_map_critical_zones`` pour la palette des zones.

    La clé unique ``color_map`` des versions antérieures reste acceptée, mais
    **seulement si aucune des deux clés nouvelles n'est déclarée** : elle alimente
    alors les deux usages. Cette règle est celle de
    ``config_manager._expand_legacy_color_map``, et elle est reproduite ici à
    l'identique. Traiter l'ancienne clé comme un repli de celle qui manque
    conduirait le module à résoudre une palette que le pipeline n'emploie pas, et
    tout contrôle appelant cette fonction seule mesurerait autre chose que le
    rendu produit.

    Une configuration écrite pour la version 6 retient donc les mêmes cartes et
    produit les mêmes grandeurs. Son rendu n'est pas identique au point près : la
    garantie de lisibilité éclaircit les fonds dont le texte noir n'atteindrait pas
    le seuil de contraste, ce qui est le cas de ``tab10``. L'ordre des teintes,
    leur nombre et leur identité chromatique sont préservés ; seule leur clarté est
    ajustée.

    Args:
        config: Bloc ``color_style`` d'``input.json``, éventuellement partiel.

    Returns:
        Dictionnaire portant ``color_map_matrix``,
        ``color_map_critical_zones``, ``saturation``, ``alpha`` et ``apply_to``,
        chacun borné à son domaine de définition.
    """
    config = config or {}
    apply_to = str(config.get("apply_to", DEFAULT_COLOR_APPLY_TO)).strip().lower()
    if apply_to not in COLOR_APPLY_MODES:
        _warn_once(f"Mode de coloration '{apply_to}' inconnu : "
                   f"'{DEFAULT_COLOR_APPLY_TO}' employé à la place.")
        apply_to = DEFAULT_COLOR_APPLY_TO
    # L'ancienne clé n'est retenue que si aucune des deux nouvelles n'est déclarée,
    # règle identique à celle de ``config_manager._expand_legacy_color_map``.
    legacy = config.get("color_map")
    if legacy and ("color_map_matrix" in config
                   or "color_map_critical_zones" in config):
        legacy = None
    return {
        "color_map_matrix": str(config.get(
            "color_map_matrix", legacy if legacy else DEFAULT_COLOR_MAP_MATRIX)),
        "color_map_critical_zones": str(config.get(
            "color_map_critical_zones",
            legacy if legacy else DEFAULT_COLOR_MAP_CRITICAL_ZONES)),
        "saturation": min(1.0, max(0.0, float(
            config.get("saturation", DEFAULT_COLOR_SATURATION)))),
        "alpha": min(1.0, max(0.0, float(config.get("alpha", DEFAULT_COLOR_ALPHA)))),
        "apply_to": apply_to,
    }


def is_text_color_mode(style: Optional[Dict[str, Any]] = None) -> bool:
    """Indique si la couleur habille les identifiants plutôt que les aires.

    Args:
        style: Style de couleurs, brut ou normalisé par :func:`color_style`.

    Returns:
        ``True`` en mode ``text``, ``False`` en mode ``background``.
    """
    return color_style(style)["apply_to"] == "text"


def _resolve_color_map(name: str) -> Any:
    """Retourne la carte de couleurs matplotlib portant ce nom.

    Args:
        name: Nom de la carte, par exemple ``"tab10"``, ``"jet"`` ou ``"viridis"``.

    Returns:
        Objet ``Colormap``. La carte par défaut est retournée si le nom est
        inconnu, l'absence d'une carte ne devant pas interrompre le calcul.
    """
    try:
        return matplotlib.colormaps[name]
    except (KeyError, AttributeError):
        _warn_once(f"Carte de couleurs '{name}' inconnue : "
                   f"'{DEFAULT_COLOR_MAP}' employée à la place.")
        return matplotlib.colormaps[DEFAULT_COLOR_MAP]


def relative_luminance(color: Sequence[float]) -> float:
    """Luminance relative d'une couleur, au sens des règles d'accessibilité.

    Args:
        color: Composantes rouge, vert et bleu dans [0, 1], espace sRGB.

    Returns:
        Luminance relative dans [0, 1].
    """
    channels = []
    for value in tuple(color)[:3]:
        component = min(1.0, max(0.0, float(value)))
        channels.append(component / 12.92 if component <= 0.03928
                        else ((component + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def contrast_ratio(color: Sequence[float],
                   background: Sequence[float] = (1.0, 1.0, 1.0)) -> float:
    """Rapport de contraste entre une couleur et son fond.

    Args:
        color: Couleur du premier plan, composantes dans [0, 1].
        background: Couleur du fond, blanche par défaut.

    Returns:
        Rapport de contraste, compris entre 1 et 21.
    """
    first, second = relative_luminance(color), relative_luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def enforce_text_contrast(colors: Sequence[Sequence[float]],
                          minimum: float = TEXT_MODE_MIN_CONTRAST
                          ) -> List[Tuple[float, float, float]]:
    """Assombrit une palette jusqu'à ce que toutes ses teintes soient lisibles.

    En mode ``text``, la teinte n'orne pas le libellé : elle l'identifie. Une
    teinte trop claire sur une page blanche prive donc la figure de son repérage,
    et non de son seul agrément. Or les cartes usuelles portent des teintes très
    claires : l'orangé de ``tab10`` n'atteint qu'un rapport de 2,53 sur blanc, le
    jaune terminal de ``viridis`` 1,26, pour un seuil de lisibilité de 4,5.

    Un facteur d'assombrissement **commun** à toutes les teintes est recherché,
    le plus grand qui porte la moins contrastée d'entre elles au seuil. Le facteur
    étant commun, les rapports entre teintes sont préservés et la palette conserve
    son allure ; l'assombrissement accroît d'ailleurs leur séparabilité plutôt
    qu'il ne la réduit. La recherche est dichotomique, le contraste sur fond blanc
    croissant lorsque le facteur décroît.

    Args:
        colors: Palette à contrôler, composantes dans [0, 1].
        minimum: Rapport de contraste exigé sur fond blanc.

    Returns:
        Palette conforme, inchangée si elle l'était déjà.
    """
    palette = [tuple(float(channel) for channel in tuple(color)[:3]) for color in colors]
    if not palette:
        return []
    # Marge de sûreté absorbant l'arrondi. Une couleur est déclarée à LaTeX avec
    # quatre décimales par composante : viser le seuil au plus juste laisserait la
    # teinte publiée retomber d'un millième sous celui ci, et un contrôle mené sur
    # le fichier produit, seul document qui fasse foi, la déclarerait non conforme.
    target = minimum + 0.01
    worst = min(contrast_ratio(color) for color in palette)
    if worst >= target:
        return palette
    minimum = target

    def darkened(factor: float) -> List[Tuple[float, float, float]]:
        return [tuple(factor * channel for channel in color) for color in palette]

    low, high = TEXT_MODE_MIN_DARKENING, 1.0
    if min(contrast_ratio(color) for color in darkened(low)) < minimum:
        # Le plancher lui même ne suffit pas : il est appliqué tel quel, la
        # convergence vers le noir étant préférable à une palette illisible.
        _warn_once(
            f"Palette de couleurs : contraste minimal {worst:.2f} sur blanc, "
            f"inférieur au seuil {minimum} ; assombrissement plafonné au facteur "
            f"{low:.2f}, le seuil demeurant hors d'atteinte.")
        return darkened(low)
    for _ in range(40):
        middle = 0.5 * (low + high)
        if min(contrast_ratio(color) for color in darkened(middle)) >= minimum:
            low = middle
        else:
            high = middle
    _inform_once(
        f"Palette de couleurs : contraste minimal {worst:.2f} sur blanc, "
        f"inférieur au seuil {minimum} ; teintes assombries d'un facteur commun "
        f"{low:.3f} pour garantir la lisibilité des libellés.")
    return darkened(low)


def is_categorical_color_map(name: str) -> bool:
    """Indique si une carte de couleurs est catégorielle plutôt que continue.

    Une carte catégorielle, tel ``tab10``, porte un petit nombre de teintes
    distinctes destinées à désigner des catégories. Une carte continue, telle
    ``viridis`` ou ``jet``, décrit une progression et convient au fond d'une
    matrice de valeurs.

    Args:
        name: Nom de la carte.

    Returns:
        ``True`` si la carte compte au plus
        :data:`CATEGORICAL_COLOR_MAP_LIMIT` teintes.
    """
    return int(getattr(_resolve_color_map(name), "N", 256)) <= CATEGORICAL_COLOR_MAP_LIMIT


def apply_saturation_and_alpha(color: Sequence[float], saturation: float,
                               alpha: float,
                               apply_to: str = DEFAULT_COLOR_APPLY_TO) -> Tuple[float, float, float]:
    """Atténue une teinte et la compose sur fond blanc.

    Deux transformations successives sont appliquées.

        1. **Saturation.** La teinte est mêlée d'un pôle neutre dans une
           proportion qui décroît avec le paramètre : à ``saturation`` valant 1
           la teinte est restituée telle quelle, à 0 elle est mêlée à parts
           égales. La fraction de teinte conservée vaut donc
           ``0.5 + 0.5 * saturation``.

           Le pôle dépend de ce que la couleur habille. Une **aire** est mêlée de
           **blanc** : elle s'éclaircit et devient pastel, ce qui préserve la
           lisibilité du texte noir qu'elle porte. Un **identifiant** est mêlé de
           **noir** : il s'assombrit et gagne en contraste sur la page. Mêler du
           blanc à un libellé le ferait au contraire disparaître, l'atténuation
           d'un texte et celle d'un fond appelant des sens opposés.
        2. **Opacité.** Le résultat est composé selon ``alpha`` sur le même pôle
           neutre que la saturation, blanc pour une aire et noir pour un
           identifiant. La composition est effectuée en amont afin que le fond
           des cellules d'un tableau LaTeX puisse en bénéficier, l'extension
           ``xcolor`` ne gérant pas la transparence des fonds de colonne.

           Composer un identifiant sur blanc reviendrait à le faire disparaître :
           à ``alpha`` nul, les libellés étaient rendus en blanc sur blanc, et le
           tableau perdait purement et simplement sa ligne d'en-tête sans qu'aucun
           contrôle ne s'en émeuve. Les deux paramètres jouent donc dans le même
           sens, et ``alpha`` devient en mode ``text`` un second levier
           d'assombrissement.

    Args:
        color: Teinte d'origine, composantes rouge, vert et bleu dans [0, 1].
        saturation: Intensité de la teinte, dans [0, 1].
        alpha: Opacité, dans [0, 1].
        apply_to: Destination de la couleur, ``"background"`` ou ``"text"``.
            Elle fixe le pôle du mélange d'atténuation.

    Returns:
        Triplet rouge, vert, bleu dans [0, 1].
    """
    keep = 0.5 + 0.5 * min(1.0, max(0.0, float(saturation)))
    opacity = min(1.0, max(0.0, float(alpha)))
    toward_black = str(apply_to).strip().lower() == "text"
    rendered = []
    for channel in tuple(color)[:3]:
        attenuated = (keep * float(channel) if toward_black
                      else 1.0 - keep * (1.0 - float(channel)))
        rendered.append(opacity * attenuated
                        + (0.0 if toward_black else 1.0 - opacity))
    return tuple(rendered)


def effective_critical_zones_color_map(count: int,
                                       style: Optional[Dict[str, Any]] = None) -> str:
    """Nom de la carte réellement employée pour la palette des zones critiques.

    Elle diffère de la carte déclarée lorsqu'un repli s'applique : nom inconnu, ou
    carte catégorielle sollicitée pour plus de variables qu'elle ne porte de
    teintes.

    Args:
        count: Nombre de variables à désigner.
        style: Style de couleurs, brut ou normalisé par :func:`color_style`.

    Returns:
        Nom de carte matplotlib.
    """
    declared = color_style(style)["color_map_critical_zones"]
    try:
        matplotlib.colormaps[declared]
    except (KeyError, AttributeError):
        declared = DEFAULT_COLOR_MAP
    if is_categorical_color_map(declared):
        available = int(getattr(_resolve_color_map(declared), "N", 256))
        if count > available:
            return CONTINUOUS_FALLBACK_COLOR_MAP
    return declared


def _color_style_record(config: Optional[Dict[str, Any]] = None,
                        count: int = 0) -> Dict[str, Any]:
    """Relevé du style de couleurs destiné au fichier de résultats.

    Args:
        config: Bloc ``color_style`` déclaré.
        count: Nombre de variables désignées, qui conditionne un éventuel repli de
            la palette des zones critiques.

    Returns:
        Style normalisé, augmenté des deux cartes effectivement employées.
    """
    record = dict(color_style(config))
    record["color_map_matrix_effective"] = matrix_color_map(config)
    record["color_map_critical_zones_effective"] = (
        effective_critical_zones_color_map(count, config) if count
        else record["color_map_critical_zones"])
    return record


def enforce_background_contrast(colors: Sequence[Sequence[float]],
                                minimum: float = TEXT_MODE_MIN_CONTRAST
                                ) -> List[Tuple[float, float, float]]:
    """Éclaircit une palette jusqu'à ce que le texte noir y soit lisible.

    C'est l'exacte symétrique de :func:`enforce_text_contrast`. En mode
    ``background``, la teinte est un **fond** : ce qui doit rester lisible n'est
    pas elle, mais le texte noir qu'elle porte, libellé de colonne d'un tableau ou
    étiquette d'un nœud de graphe.

    La nécessité en est apparue avec la dissociation des cartes de la version 7.
    ``jet``, retenue par défaut pour les zones critiques, comporte un bleu nuit et
    un rouge sombre dont le contraste avec du texte noir tombe à 1,31 : la colonne
    correspondante du tableau d'échantillonnage devenait illisible. ``viridis``
    descend à 1,38, et ``tab10`` elle même à 3,55, en deçà du seuil.

    Un facteur d'éclaircissement **commun** à toutes les teintes est recherché, le
    plus grand qui porte la moins contrastée d'entre elles au seuil. Le facteur
    étant commun, les rapports entre teintes sont préservés.

    Args:
        colors: Palette à contrôler, composantes dans [0, 1].
        minimum: Rapport de contraste exigé entre le texte noir et le fond.

    Returns:
        Palette conforme, inchangée si elle l'était déjà.
    """
    palette = [tuple(float(channel) for channel in tuple(color)[:3]) for color in colors]
    if not palette:
        return []
    target = minimum + 0.01
    on_black = lambda color: (relative_luminance(color) + 0.05) / 0.05
    if min(on_black(color) for color in palette) >= target:
        return palette

    def lightened(factor: float) -> List[Tuple[float, float, float]]:
        return [tuple(1.0 - factor * (1.0 - channel) for channel in color)
                for color in palette]

    worst = min(on_black(color) for color in palette)
    low, high = 0.0, 1.0
    for _ in range(40):
        middle = 0.5 * (low + high)
        if min(on_black(color) for color in lightened(middle)) >= target:
            low = middle
        else:
            high = middle
    _inform_once(
        f"Palette de couleurs : contraste minimal {worst:.2f} entre le texte noir "
        f"et le fond, inférieur au seuil {target} ; teintes éclaircies d'un facteur "
        f"commun {low:.3f} pour garantir la lisibilité des libellés.")
    return lightened(low)


def palette_colors(count: int,
                   style: Optional[Dict[str, Any]] = None) -> List[Tuple[float, float, float]]:
    """Produit une suite de teintes issues de la carte configurée.

    Une carte catégorielle est parcourue par indices successifs, en bouclant si
    les catégories excèdent le nombre de teintes disponibles. Une carte continue
    est échantillonnée régulièrement sur toute son étendue, de sorte que la
    première catégorie reçoive l'extrémité froide et la dernière l'extrémité
    chaude.

    Args:
        count: Nombre de teintes demandées.
        style: Style de couleurs normalisé par :func:`color_style`.

    Returns:
        Liste de triplets rouge, vert, bleu, atténués et composés sur blanc.
    """
    if count <= 0:
        return []
    resolved = color_style(style)
    name = resolved["color_map_critical_zones"]
    color_map = _resolve_color_map(name)
    categorical = is_categorical_color_map(name)
    if categorical and count > int(color_map.N):
        # Une carte catégorielle ne porte qu'un nombre fini de teintes. Les
        # parcourir cycliquement attribuerait la même teinte à deux variables
        # distinctes, ce qui ruine le repérage sans que rien ne le signale. Une
        # rampe continue, échantillonnable à volonté, prend le relais.
        _warn_once(
            f"Carte de couleurs '{name}' : {int(color_map.N)} teintes pour "
            f"{count} variables à désigner ; '{CONTINUOUS_FALLBACK_COLOR_MAP}' "
            "employée à la place afin que deux variables ne partagent pas la "
            "même teinte.")
        color_map = _resolve_color_map(CONTINUOUS_FALLBACK_COLOR_MAP)
        categorical = False
    if categorical:
        raw = [color_map(index % int(color_map.N)) for index in range(count)]
    else:
        denominator = max(1, count - 1)
        raw = [color_map(index / denominator) for index in range(count)]
    rendered = [apply_saturation_and_alpha(color, resolved["saturation"],
                                           resolved["alpha"], resolved["apply_to"])
                for color in raw]
    # Garantie de lisibilité, dans les deux modes. En mode ``text``, c'est la
    # teinte elle même qui porte le libellé et doit donc se détacher du blanc de
    # la page ; en mode ``background``, c'est le texte noir posé sur la teinte qui
    # doit s'en détacher. Les deux exigences sont symétriques et appellent le même
    # seuil, mais des corrections de sens opposé.
    if resolved["apply_to"] == "text":
        rendered = enforce_text_contrast(rendered)
    else:
        rendered = enforce_background_contrast(rendered)
    return rendered


def zone_color_map(ordering: Sequence[str],
                   style: Optional[Dict[str, Any]] = None) -> Dict[str, Tuple[float, float, float]]:
    """Associe une teinte à chaque variable, une fois pour toute l'exécution.

    L'association est établie **en un seul endroit** et transmise à tous les
    livrables. Sans cette centralisation, chaque figure et chaque tableau
    indexaient la palette sur leur propre ordre de colonnes : une même zone
    recevait alors une teinte au tableau d'échantillonnage, une autre au graphe et
    une troisième à la matrice de l'étape 2. Le repérage d'une variable d'un
    livrable à l'autre, que la couleur est précisément censée assurer, s'en
    trouvait ruiné, et le mode ``text`` en souffrait doublement puisque la teinte
    y est le seul support d'identification.

    L'ordre retenu est celui du classement de dominance de l'étape 1 : la zone la
    moins dominante reçoit l'extrémité froide de la rampe, la plus dominante
    l'extrémité chaude. La teinte porte ainsi une information, et non un simple
    numéro d'ordre de déclaration.

    Args:
        ordering: Variables, de la moins dominante à la plus dominante. Les
            doublons sont écartés, seule la première occurrence étant retenue.
        style: Style de couleurs normalisé par :func:`color_style`.

    Returns:
        Association de chaque variable vers son triplet rouge, vert, bleu.
    """
    unique: List[str] = []
    for variable in ordering:
        name = str(variable)
        if name not in unique:
            unique.append(name)
    return dict(zip(unique, palette_colors(len(unique), style)))


def resolve_colors(variables: Sequence[str],
                   colors: Optional[Dict[str, Tuple[float, float, float]]],
                   style: Optional[Dict[str, Any]] = None
                   ) -> List[Tuple[float, float, float]]:
    """Retourne la teinte de chaque variable, dans l'ordre demandé.

    Args:
        variables: Variables à teinter, dans leur ordre d'affichage.
        colors: Association établie par :func:`zone_color_map`. En son absence,
            une palette est construite sur l'ordre reçu, ce qui préserve le
            comportement des appels qui ne transmettent pas d'association.
        style: Style de couleurs normalisé par :func:`color_style`.

    Returns:
        Liste de triplets rouge, vert, bleu, de même longueur que ``variables``.
    """
    if not colors:
        return palette_colors(len(variables), style)
    fallback = palette_colors(len(variables), style)
    return [colors.get(str(variable), fallback[index])
            for index, variable in enumerate(variables)]


def matrix_color_map(style: Optional[Dict[str, Any]] = None) -> str:
    """Nom de la carte à employer pour le fond des matrices de valeurs.

    La carte est celle que ``color_map_matrix`` désigne, dissociée depuis la
    version 7 de la palette des zones critiques. Une matrice de proportions
    appelle une progression continue : si la carte déclarée se révèle
    catégorielle, elle ne convient pas à cet usage et la carte de repli est
    retenue, le repli étant signalé une fois.

    Args:
        style: Style de couleurs normalisé par :func:`color_style`.

    Returns:
        Nom de carte matplotlib.
    """
    name = color_style(style)["color_map_matrix"]
    if not is_categorical_color_map(name):
        return name
    _warn_once(
        f"Carte de couleurs '{name}' catégorielle : elle ne convient pas au fond "
        f"d'une matrice de proportions, '{MATRIX_FALLBACK_COLOR_MAP}' employée à "
        "la place. Renseigner 'color_style.color_map_matrix' avec une rampe "
        "continue lève cet avertissement.")
    return MATRIX_FALLBACK_COLOR_MAP


def dominance_color_map(ranking: Sequence[str],
                        prefix: str = "ocaZone",
                        style: Optional[Dict[str, Any]] = None,
                        colors: Optional[Dict[str, Tuple[float, float, float]]] = None
                        ) -> Tuple[Dict[str, str], str]:
    """Associe une teinte à chaque zone critique selon son rang de dominance.

    La couleur est attribuée **après** le classement, en fonction de la position
    de la zone dans la hiérarchie et non de son nom : la zone la moins dominante
    reçoit l'extrémité froide de la rampe, la plus dominante l'extrémité chaude.
    Les deux tableaux du livrable 04 partagent cette même association, de sorte
    que l'œil suive le déplacement de chaque colonne d'un tableau à l'autre.

    Un éventuel doublon dans ``ranking`` est écarté, seule sa première occurrence
    étant retenue. Sans cette garde, la seconde occurrence écraserait
    l'association de la première tout en déclarant une couleur supplémentaire :
    le document porterait une teinte définie et jamais employée, et le rang
    attribué à la zone ne serait plus celui de sa première apparition. La
    configuration interdit déjà les doublons de zones critiques, mais la fonction
    ne présume pas de son appelant.

    Args:
        ranking: Zones critiques, de la moins dominante à la plus dominante.
        prefix: Préfixe des noms de couleurs LaTeX déclarés.
        style: Style de couleurs normalisé par :func:`color_style`.
        colors: Association de chaque zone vers sa teinte, établie une fois pour
            toute l'exécution par :func:`zone_color_map`. Elle garantit qu'une
            zone porte ici la teinte même qu'elle porte dans les figures.

    Returns:
        Couple ``(association, définitions)`` où ``association`` relie chaque
        zone au nom de sa couleur LaTeX, et ``définitions`` porte les commandes
        ``\\definecolor`` correspondantes. Chaque couleur déclarée est employée,
        et chaque zone en reçoit exactement une.
    """
    zones: List[str] = []
    for zone in ranking:
        name = str(zone)
        if name not in zones:
            zones.append(name)
    if not zones:
        return {}, ""
    mapping: Dict[str, str] = {}
    definitions: List[str] = []
    for index, (zone, color) in enumerate(zip(zones, resolve_colors(zones, colors, style))):
        name = f"{prefix}{index}"
        red, green, blue = color
        mapping[zone] = name
        definitions.append(
            f"\\definecolor{{{name}}}{{rgb}}{{{red:.4f},{green:.4f},{blue:.4f}}}")
    return mapping, "\n".join(definitions)


def colored_column_format(columns: Sequence[str],
                          colors: Dict[str, str]) -> Callable[[int, int, int], str]:
    """Construit un générateur de spécification de colonnes teintées.

    Args:
        columns: Colonnes du tableau, dans leur ordre d'affichage.
        colors: Association zone vers nom de couleur LaTeX.

    Returns:
        Fonction ``(début, fin, total)`` retournant la spécification
        ``column_format`` du bloc considéré, colonne d'index comprise.
    """
    names = [str(column) for column in columns]

    def specification(low: int, high: int, total: int) -> str:
        del total  # la teinte dépend du nom de la colonne, non de sa position
        rendered = "l"
        for index in range(low, high):
            color = colors.get(names[index]) if index < len(names) else None
            rendered += f">{{\\columncolor{{{color}}}}}r" if color else "r"
        return rendered

    return specification


def colored_header_labels(columns: Sequence[str],
                          colors: Dict[str, str]) -> Dict[str, str]:
    """Décore les noms de colonnes d'une teinte et d'une graisse (mode ``text``).

    Seul l'en-tête est habillé, le corps du tableau demeurant composé en noir.
    C'est la forme parcimonieuse attendue d'un tableau destiné à un article : la
    couleur y désigne une variable sans envahir les valeurs, et le repérage d'une
    colonne d'un tableau à l'autre reste immédiat. La graisse est indispensable,
    un libellé fin coloré perdant en lisibilité ce que le fond teinté lui donnait.

    La graisse est prise **sans empattement**. Ce choix n'est pas ornemental : à
    l'échelle et à l'espacement inter colonnes de l'application, un jeu de six
    libellés de quinze caractères mesure 448,8 pt en romain maigre pour 453,0 pt
    de justification, mais 505,5 pt en romain gras, soit un débordement de plus de
    cinquante points ; la même graisse sans empattement ne mesure que 405,1 pt et
    laisse près de quarante huit points de réserve. La distinction typographique
    de l'en-tête est ainsi obtenue sans mise à l'échelle, sans resserrement
    supplémentaire des colonnes et sans réduction du nombre de colonnes par bloc,
    trois leviers que la version 6 s'interdit. C'est en outre la convention des
    en-têtes de tableau dans l'édition scientifique.

    Args:
        columns: Colonnes du tableau, dans leur ordre d'affichage.
        colors: Association zone vers nom de couleur LaTeX.

    Returns:
        Association du nom d'origine vers son libellé décoré, propre à être
        transmise à ``DataFrame.rename``. Une colonne dépourvue de teinte est
        laissée telle quelle.
    """
    decorated: Dict[str, str] = {}
    for column in columns:
        name = str(column)
        color = colors.get(name)
        if color:
            decorated[name] = (
                f"\\textcolor{{{color}}}{{\\sffamily\\bfseries "
                f"{tex_escape_name(name)}}}")
    return decorated


def plain_column_format(low: int, high: int, total: int) -> str:
    """Spécification de colonnes sans commande de taille, pour un bloc de tableau.

    Args:
        low: Indice de la première colonne du bloc, incluse.
        high: Indice de la colonne suivant le bloc, exclue.
        total: Nombre total de colonnes du tableau. Inutilisé ici ; présent pour
            respecter la signature commune des générateurs de spécification.

    Returns:
        Spécification ``column_format`` LaTeX, colonne d'index comprise.
    """
    del total  # aucune colonne privilégiée dans ce gabarit
    return "l" + "r" * (high - low)


def highlighted_tail_column_format(low: int, high: int, total: int,
                                   n_highlighted: int = 2) -> str:
    """Spécification de colonnes mettant en évidence les dernières du tableau.

    Les ``n_highlighted`` dernières colonnes du tableau, quel que soit le bloc où
    le découpage les fait tomber, sont composées sur fond sombre et texte clair.
    C'est la mise en évidence des colonnes de régularisation du tableau de
    l'étape 2, à savoir l'enveloppe κ·cz et l'indice de parcimonie β.

    Args:
        low: Indice de la première colonne du bloc, incluse.
        high: Indice de la colonne suivant le bloc, exclue.
        total: Nombre total de colonnes du tableau.
        n_highlighted: Nombre de colonnes terminales à mettre en évidence.

    Returns:
        Spécification ``column_format`` LaTeX, colonne d'index comprise.
    """
    specification = "l"
    for index in range(low, high):
        if index >= total - n_highlighted:
            specification += "|>{\\columncolor{black!50!white}\\color{white}}r"
        else:
            specification += "r"
    return specification


def emphasized_tail_column_format(low: int, high: int, total: int,
                                  n_highlighted: int = 2) -> str:
    """Variante sans aplat de :func:`highlighted_tail_column_format`.

    Le fond sombre est remplacé par une composition en gras, le filet vertical
    séparant les colonnes de régularisation étant conservé. C'est la mise en
    évidence retenue en mode de coloration ``text``, où aucune aire n'est
    teintée : un aplat noir y détonnerait, et une figure destinée à un article
    est fréquemment imprimée sans couleur.

    Args:
        low: Indice de la première colonne du bloc, incluse.
        high: Indice de la colonne suivant le bloc, exclue.
        total: Nombre total de colonnes du tableau.
        n_highlighted: Nombre de colonnes terminales à mettre en évidence.

    Returns:
        Spécification ``column_format`` LaTeX, colonne d'index comprise.
    """
    specification = "l"
    for index in range(low, high):
        specification += ("|>{\\bfseries}r" if index >= total - n_highlighted
                          else "r")
    return specification


def tail_column_format_for(style: Optional[Dict[str, Any]] = None
                           ) -> Callable[[int, int, int], str]:
    """Choisit la mise en évidence des colonnes terminales selon le mode.

    Args:
        style: Style de couleurs, brut ou normalisé par :func:`color_style`.

    Returns:
        Générateur de spécification ``column_format``, aplat sombre en mode
        ``background``, graisse en mode ``text``.
    """
    return (emphasized_tail_column_format if is_text_color_mode(style)
            else highlighted_tail_column_format)


def _table_style(style: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Normalise le style typographique d'un tableau en arguments nommés.

    Args:
        style: Dictionnaire optionnel portant ``font_size`` et
            ``column_separation``. Les clés absentes prennent la valeur par
            défaut du module.

    Returns:
        Arguments nommés destinés à :func:`latex_stacked_table`.
    """
    style = style or {}
    return {
        "font_size": str(style.get("font_size", SPLIT_BLOCK_FONT_SIZE)),
        "column_separation": str(style.get("column_separation", TABLE_COLUMN_SEPARATION)),
    }


def latex_stacked_table(
    frame: pd.DataFrame,
    caption: str,
    label: str,
    max_per_block: int = DEFAULT_MAX_COLUMNS_PER_BLOCK,
    column_format_for: Callable[[int, int, int], str] = plain_column_format,
    font_size: str = SPLIT_BLOCK_FONT_SIZE,
    column_separation: str = TABLE_COLUMN_SEPARATION,
) -> str:
    """Compose un tableau LaTeX, découpé en blocs empilés si nécessaire (V6).

    Régime unique : le tableau est rendu comme une suite de blocs ``tabular``
    empilés verticalement au sein d'un unique environnement ``table``, sous une
    légende et une étiquette communes. Un tableau dont le nombre de colonnes
    n'excède pas le seuil ne comporte qu'un seul bloc ; au delà, les colonnes sont
    réparties en blocs équilibrés, chacun répétant la colonne d'index. C'est la
    mise en page de la Table 3 du papier de référence.

    Homogénéité typographique : **aucune mise à l'échelle n'est appliquée**. Le
    redimensionnement proportionnel employé jusqu'à la version 5 faisait dépendre
    le corps de caractères du contenu, en sorte que deux tableaux, ou deux blocs
    d'un même tableau, se trouvaient composés dans des tailles distinctes selon la
    longueur de leurs libellés. Tous les blocs de tous les tableaux de données de
    l'application sont désormais composés dans une taille **fixe, réduite et
    rigoureusement identique**, déclarée une seule fois en tête du groupe par
    ``font_size``, et occupent leur largeur naturelle.

    Args:
        frame: Tableau à rendre. L'index constitue la première colonne.
        caption: Légende du tableau, en syntaxe LaTeX.
        label: Étiquette de référence croisée.
        max_per_block: Seuil de lisibilité horizontale, en nombre de colonnes.
        column_format_for: Fonction ``(début, fin, total)`` retournant la
            spécification ``column_format`` du bloc considéré, colonne d'index
            comprise. Recevoir les bornes, et non le seul nombre de colonnes,
            permet de mettre en évidence des colonnes déterminées quel que soit le
            bloc où le découpage les fait tomber.
        font_size: Nom de la commande LaTeX de taille appliquée à tous les blocs,
            par exemple ``footnotesize`` ou ``small``.
        column_separation: Demi espacement inter colonnes, exprimé en unités
            LaTeX. Resserrer les colonnes est le levier retenu pour contenir un
            tableau dans la justification, la mise à l'échelle du texte étant
            proscrite et le corps de caractères devant rester uniforme.

    Returns:
        Bloc LaTeX complet, de ``\\begin{table}`` à ``\\end{table}``.
    """
    total = len(frame.columns)
    bounds = split_column_bounds(total, max_per_block)

    parts: List[str] = [
        "\\begin{table}[htbp!]",
        "\\centering",
        f"\\caption{{{_escape_percent(caption)}}}",
        f"\\label{{{label}}}",
        # La commande de taille et l'espacement inter colonnes sont déclarés une
        # seule fois, en tête du groupe, et valent donc rigoureusement pour tous
        # les blocs. Le groupe en circonscrit la portée au seul tableau.
        f"\\begingroup\\{font_size}",
        f"\\setlength{{\\tabcolsep}}{{{column_separation}}}",
    ]
    for position, (low, high) in enumerate(bounds):
        block = frame.iloc[:, low:high]
        tabular = block.style.to_latex(
            hrules=True, column_format=column_format_for(low, high, total))
        parts.append(_escape_percent(tabular).strip())
        if position < len(bounds) - 1:
            parts.append("\\vspace{0.35cm}\n")
    parts.append("\\endgroup")
    parts.append("\\end{table}")
    return "\n".join(parts) + "\n"


# ===========================================================================
# Rapports LaTeX
# ===========================================================================
def _latex_document_result(latex_tab1: str, descriptions: str) -> str:
    """Assemble le document LaTeX du rapport GPO (``Result``).

    Args:
        latex_tab1: Bloc tabulaire LaTeX.
        descriptions: Bloc de descriptions (sélection GPO).

    Returns:
        Document LaTeX complet.
    """
    return rf"""
    \documentclass{{article}}
    \usepackage[utf8]{{inputenc}}
    \usepackage{{longtable}}
    \usepackage[a4paper, margin=1in]{{geometry}}
    \usepackage{{booktabs}}
    \usepackage{{graphicx}}
    \usepackage{{underscore}}
    \usepackage{{caption}}
    \usepackage{{subcaption}}
    \usepackage{{amsmath}}
    \usepackage{{hyperref}}
    \usepackage{{siunitx}}
    \usepackage{{babel}}
    \usepackage[table]{{xcolor}}

    \begin{{document}}
    {latex_tab1}
    {descriptions}
    \end{{document}}
    """


def generate_result_table(
    df_results: pd.DataFrame,
    dico_results: Dict[str, List[str]],
    n_sample: int,
    filedir: str,
    filename: str = "Result",
    max_columns_per_block: int = DEFAULT_MAX_COLUMNS_PER_BLOCK,
    table_style: Optional[Dict[str, str]] = None,
) -> str:
    """Génère et écrit le rapport LaTeX GPO ``Result.tex``.

    Le tableau porte une colonne par zone critique. Au delà du seuil de
    lisibilité, il est découpé en sous-tableaux empilés verticalement (V6), selon
    la mise en page de la Table 3 du papier de référence : les lignes ``n``,
    ``p``, ``N`` et ``P`` sont répétées sous chaque groupe de zones.

    Args:
        df_results: Tableau des dominations (n, p, N, P).
        dico_results: Dictionnaire de sélection.
        n_sample: Nombre de cas examinés.
        filedir: Dossier de sortie.
        filename: Nom de base du fichier.
        max_columns_per_block: Seuil de lisibilité horizontale, en colonnes.

    Returns:
        Chemin du ``.tex`` écrit.
    """
    caption = (
        f"\\centering \\small Comptage de dominance un contre tous (One-vs-All) : "
        "pour chacune des configurations de la base, la zone portant la contrainte "
        "maximale est relevée, et la fréquence $p$ qui en résulte fonde la "
        "hiérarchie puis le filtrage des zones. Ce comptage relève strictement de "
        "l'étape 1 de l'algorithme, la sélection par l'ordre de Pareto généralisé. "
        f"Nombre de cas examinés = {n_sample}. Colonnes disposées par fréquence "
        "$p$ croissante, de la zone la moins dominante à la plus dominante ; les "
        "cumuls $N$ et $P$ suivent cet ordre. Le comptage porte sur l'intégralité "
        "des observations. Voir Amuzuga et Iwaniack, Int. J. Pressure Vessels and "
        "Piping 222 (2026) 105801, section 2.5.2, algorithme 1 et Table 3. "
    )
    label = "tab:result"

    latex_tab1 = latex_stacked_table(
        df_results.rename(columns=tex_escape_name), caption, label, max_columns_per_block,
        **_table_style(table_style))

    # Les libellés de zones sont de longs identifiants ininterrompables, que LaTeX
    # ne sait ni couper ni resserrer : une énumération en compose donc des lignes
    # qui débordent de quelques points dans la marge. L'environnement
    # ``sloppypar`` autorise l'élargissement des espaces inter mots, ce qui lève
    # le débordement sans toucher au corps de caractères ni au contenu.
    descriptions = "\\vspace*{0.5cm}\n\\begin{sloppypar}\n\\begin{description}\n"
    for key, value in dico_results.items():
        if isinstance(value, (list, tuple)):
            shown = "[" + ", ".join(f"'{tex_escape_name(name)}'" for name in value) + "]"
        else:
            shown = tex_escape_name(value)
        descriptions += f"\\item[{_escape_percent(str(key))}] {shown}\n"
    descriptions += "\\end{description}\n\\end{sloppypar}\n\\vspace*{0.5cm}\n"

    latex_document = _latex_document_result(latex_tab1, descriptions)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    filepath = os.path.join(filedir, f"{filename}.tex")
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(latex_document)
    return filepath


def generate_step1_zones_sample_table(
    df_merged: pd.DataFrame,
    critical_zones: List[str],
    sample_size: int,
    filedir: str,
    random_state: int = 6,
    filename: str = "04-Step1_Critical_Zones_Sample_Result",
    max_columns_per_block: int = DEFAULT_MAX_COLUMNS_PER_BLOCK,
    table_style: Optional[Dict[str, str]] = None,
    ranking: Optional[Sequence[str]] = None,
    color_style_config: Optional[Dict[str, Any]] = None,
    colors: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> str:
    """Génère le livrable d'échantillonnage des zones critiques de l'étape 1 (V6).

    Le document porte **deux tableaux** décrivant les mêmes observations et les
    mêmes grandeurs, et ne différant que par la disposition horizontale de leurs
    colonnes :

        1. le premier restitue les zones dans leur **ordre de déclaration**, tel
           que l'utilisateur l'a écrit dans ``input.json``, soit l'état antérieur
           à toute hiérarchisation ;
        2. le second les dispose par **fréquence de dominance croissante**, de la
           zone la moins dominante à la plus dominante, soit l'état postérieur au
           classement de l'étape 1.

    Leur confrontation donne à voir le déplacement de chaque zone, c'est à dire
    l'effet même de la hiérarchisation sur l'ordonnancement des variables.

    Repérage par la couleur : une teinte est attribuée à chaque zone d'après son
    rang dans la hiérarchie, selon une rampe « jet » atténuée, du bleu pour les
    moins dominantes au rouge pour les plus dominantes. **La même teinte suit la
    même zone dans les deux tableaux**, ce qui permet de suivre son déplacement
    d'un coup d'œil. L'attribution est algorithmique et s'établit après le
    classement : elle ne dépend ni du nom des zones ni de leur nombre.

    Périmètre : **les seules zones critiques**, variables synthétiques déclarées
    comprises. Les variables de conception de ``feature_cols_init`` en sont
    exclues : elles décrivent la géométrie et non les contraintes, et leur
    présence diluait la lecture comparative des grandeurs que l'ordre partiel de
    dominance met en jeu.

    Périmètre statistique : ces extraits relèvent de la seule illustration. Les
    indicateurs de l'algorithme, comptage de dominance, classement, sélection,
    coefficient de régularisation et indice de parcimonie, sont établis sur
    l'intégralité des observations. Le paramètre ``sample_size`` ne gouverne que
    le nombre de lignes affichées.

    Args:
        df_merged: Jeu de données fusionné de l'étape 1.
        critical_zones: Zones critiques dans leur ordre de déclaration.
        sample_size: Nombre de lignes échantillonnées.
        filedir: Dossier de sortie.
        random_state: Graine de l'échantillonnage.
        filename: Nom de base du fichier.
        max_columns_per_block: Seuil de lisibilité horizontale, en colonnes.
        table_style: Corps de caractères et espacement inter colonnes.
        ranking: Zones dans l'ordre de dominance croissante. En son absence,
            l'ordre de déclaration est repris et les deux tableaux coïncident.
        color_style_config: Style de couleurs déclaré par l'utilisateur, tel que
            normalisé par :func:`color_style`.

    Returns:
        Chemin du ``.tex`` écrit.
    """
    declared = [column for column in critical_zones if column in df_merged.columns]
    if ranking:
        sorted_zones = [zone for zone in ranking if zone in declared]
        sorted_zones += [zone for zone in declared if zone not in sorted_zones]
    else:
        sorted_zones = list(declared)

    # L'échantillon ne peut excéder l'effectif disponible : un jeu de données plus
    # petit que ``sample_size`` est présenté intégralement plutôt que rejeté.
    effective_size = min(int(sample_size), len(df_merged))
    df_sample = df_merged[declared].sample(n=effective_size, random_state=random_state)
    df_sample = df_sample.reset_index(drop=True)
    df_sample.index = df_sample.index + 1
    # Les deux tableaux procèdent du même échantillon formaté : ils portent donc
    # rigoureusement les mêmes valeurs, à la seule permutation des colonnes près.
    formatted = format_frame_for_latex(df_sample)

    tints, color_definitions = dominance_color_map(
        sorted_zones, style=color_style_config, colors=colors)
    style = _table_style(table_style)
    text_mode = is_text_color_mode(color_style_config)

    common = (
        f"Sélection de {effective_size} lignes aléatoirement dans les données, "
        f"portant les {len(declared)} zones critiques de l'étape 1, avant "
        "sélection des variables. Cet extrait relève de la seule illustration : "
        "les indicateurs de l'algorithme sont établis sur l'intégralité des "
        "observations. "
    )
    support = "de l'en-tête de chaque colonne" if text_mode else "de chaque colonne"
    caption_before = (
        "\\centering \\small Avant hiérarchisation. Colonnes dans leur ordre de "
        "déclaration. " + common
    )
    caption_after = (
        "\\centering \\small Après hiérarchisation. Colonnes disposées par score "
        "de dominance ordinale croissant, de la zone la moins dominante à la plus "
        "dominante. Ce score est la moyenne des proportions de majoration de la "
        "matrice de corrélation ordinale de l'étape 1, établie par comparaison "
        "des colonnes deux à deux. " + common
        + f"La teinte {support}, identique dans les deux tableaux, permet de "
        "suivre son déplacement. "
    )

    # En mode de coloration ``text``, la teinte est portée par le seul en-tête et
    # les colonnes ne reçoivent aucun aplat ; en mode ``background``, le fond de
    # colonne la porte et les libellés restent noirs. Les deux tableaux emploient
    # rigoureusement la même association zone vers teinte dans l'un et l'autre cas.
    if text_mode:
        decorated = colored_header_labels(sorted_zones, tints)
        labels = {zone: decorated.get(zone, tex_escape_name(zone))
                  for zone in dict.fromkeys(list(declared) + list(sorted_zones))}
        frame_before = formatted[declared].rename(columns=labels)
        frame_after = formatted[sorted_zones].rename(columns=labels)
        format_before: Callable[[int, int, int], str] = plain_column_format
        format_after: Callable[[int, int, int], str] = plain_column_format
    else:
        frame_before = formatted[declared].rename(columns=tex_escape_name)
        frame_after = formatted[sorted_zones].rename(columns=tex_escape_name)
        format_before = colored_column_format(declared, tints)
        format_after = colored_column_format(sorted_zones, tints)

    blocks = [
        latex_stacked_table(
            frame_before, caption_before, "tab:step1_zones_declared",
            max_columns_per_block, column_format_for=format_before, **style),
        latex_stacked_table(
            frame_after, caption_after, "tab:step1_zones_sample",
            max_columns_per_block, column_format_for=format_after, **style),
    ]

    latex_document = _latex_document_step1_zones(color_definitions, blocks)
    os.makedirs(filedir, exist_ok=True)
    filepath = os.path.join(filedir, f"{filename}.tex")
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(latex_document)
    return filepath


def _latex_document_step1_zones(color_definitions: str, blocks: List[str]) -> str:
    """Assemble le document LaTeX du livrable d'échantillonnage de l'étape 1.

    Args:
        color_definitions: Commandes ``\\definecolor`` de la palette de repérage.
        blocks: Blocs tabulaires à empiler, dans l'ordre d'apparition.

    Returns:
        Document LaTeX complet.
    """
    preamble = rf"""
    \documentclass{{article}}
    \usepackage[utf8]{{inputenc}}
    \usepackage{{longtable}}
    \usepackage[a4paper, margin=1in]{{geometry}}
    \usepackage{{booktabs}}
    \usepackage{{graphicx}}
    \usepackage{{underscore}}
    \usepackage{{caption}}
    \usepackage{{subcaption}}
    \usepackage{{amsmath}}
    \usepackage{{hyperref}}
    \usepackage{{siunitx}}
    \usepackage{{babel}}
    \usepackage[table]{{xcolor}}

{color_definitions}

    \begin{{document}}
    """
    return preamble + "\n\n\\vspace{0.6cm}\n\n".join(blocks) + "\n" + r"\end{document}"


def _latex_document_criticalzones(latex_tab1: str) -> str:
    """Assemble le document LaTeX du tableau ``CriticalZones_Targets``.

    Args:
        latex_tab1: Bloc tabulaire LaTeX.

    Returns:
        Document LaTeX complet.
    """
    # Construction par concaténation, à l'identique du notebook d'origine
    # (preambule + latex_tab1 + \end{document}), afin d'obtenir un fichier
    # byte-identique à la reference (fin de document sans indentation).
    latex_document = rf"""
    \documentclass{{article}}
    \usepackage[utf8]{{inputenc}}
    \usepackage{{longtable}}
    \usepackage[a4paper, margin=1in]{{geometry}}
    \usepackage{{booktabs}}
    \usepackage{{graphicx}}
    %\usepackage{{lscape}}
    \usepackage{{underscore}}
    \usepackage{{caption}}
    \usepackage{{subcaption}}
    %\usepackage{{float}}
    \usepackage{{amsmath}}  % Pour des environnements mathématiques supplémentaires si nécessaire
    \usepackage{{hyperref}} % Pour les liens hypertextes
    %\usepackage{{cleveref}} % Pour des références plus intelligentes
    \usepackage{{siunitx}}  % Pour une meilleure gestion des unités
    \usepackage{{babel}}    % Pour la gestion de la langue
    \usepackage[table]{{xcolor}} % pour utiliser les couleurs dans les tableaux

    \begin{{document}}
    """
    latex_document += latex_tab1
    latex_document += r"\end{document}"
    return latex_document


def generate_criticalzones_table(
    df_sample_full: pd.DataFrame,
    cz: str,
    kappa: float,
    max_beta: float,
    mean_beta: float,
    sample_size: int,
    filedir: str,
    random_state: int = 6,
    filename: str = "CriticalZones_Targets",
    max_columns_per_block: int = DEFAULT_MAX_COLUMNS_PER_BLOCK,
    table_style: Optional[Dict[str, str]] = None,
    color_style_config: Optional[Dict[str, Any]] = None,
    kappa_decimals: int = 2,
    beta_defined: bool = True,
) -> str:
    """Génère et écrit le rapport LaTeX ``CriticalZones_Targets.tex`` (échantillon β).

    Échantillonne ``sample_size`` lignes (graine fixe), applique la mise en forme
    numérique adaptative et met en évidence les deux dernières colonnes, κ·cz et β.

    Note:
        Ce tableau est soumis, depuis la version 6, au même découpage adaptatif et
        à la même taille de caractères fixe que les tableaux de l'étape 1. Lorsque
        le nombre de colonnes retenues après sélection excède le seuil de
        lisibilité, il est scindé en blocs empilés. Les deux colonnes terminales
        demeurent mises en évidence dans le bloc où elles tombent, de sorte que la
        lecture comparative avec les zones critiques reste possible.

    Args:
        df_sample_full: DataFrame complet (zones + ``$\\alpha$cz`` + ``$\\beta$``).
        cz: Variable synthétique enveloppe.
        kappa: Coefficient multiplicateur optimal.
        max_beta: β maximal.
        mean_beta: β moyen.
        sample_size: Nombre de lignes échantillonnées.
        filedir: Dossier de sortie.
        random_state: Graine de l'échantillonnage.
        filename: Nom de base du fichier.
        max_columns_per_block: Seuil de lisibilité horizontale, en colonnes.
        color_style_config: Style de couleurs déclaré par l'utilisateur. Il
            gouverne la mise en évidence des deux colonnes terminales, aplat
            sombre ou composition en gras.
        kappa_decimals: Décimales du coefficient κ affiché en légende.
        beta_defined: Faux si des valeurs négatives ou nulles rendent β non
            interprétable ; la légende le signale.

    Returns:
        Chemin du ``.tex`` écrit.
    """
    # L'echantillon ne peut exceder l'effectif disponible. Sans cette borne,
    # pandas leve 'Cannot take a larger sample than population' et l'execution
    # entiere meurt, sans 'output.json' : mesure sur cinq observations avec un
    # 'sample_size' de trente. Le livrable de l'echantillon, lui, bornait deja ;
    # les deux politiques divergeaient pour un meme reglage.
    effective_size = min(int(sample_size), len(df_sample_full))
    df_sample = df_sample_full.sample(n=effective_size, random_state=random_state)
    df_sample = df_sample.reset_index(drop=True)
    df_sample.index = df_sample.index + 1
    # Mise en forme adaptative (V6) : chaque colonne reçoit le format qu'impose sa
    # distribution. Une colonne dont la médiane des valeurs absolues atteint 1
    # conserve le format ``.3g`` des versions antérieures ; les colonnes de
    # faibles amplitudes basculent en décimal étendu ou en notation scientifique
    # plutôt que d'être tronquées à zéro.
    df_sample = format_frame_for_latex(df_sample).rename(
        columns=lambda name: tex_escape_name(name, keep_math_tokens=True))

    caption1 = (f"\\centering \\small Sélection de {effective_size} lignes aléatoirement "
                "dans les données. ")
    caption2 = (fr"Le coefficient de majoration $\alpha=$"
                f"{kappa:.{max(2, int(kappa_decimals))}f}. ")
    caption3 = fr"La valeur maximale de $\beta$ est {max_beta:.2f}. "
    caption4 = fr"La valeur moyenne de $\beta$ est {mean_beta:.2f}. "
    # Périmètre statistique : les colonnes sont disposées de la zone la moins
    # dominante à la plus dominante, et les grandeurs reportées en légende sont
    # établies sur la totalité des observations, non sur l'extrait affiché.
    caption5 = (
        "Colonnes disposées par score de dominance ordinale croissant, moyenne "
        "des proportions de majoration de la matrice de corrélation ordinale de "
        "l'étape 2, établie par comparaison des colonnes deux à deux. "
        "Cet extrait relève de la seule illustration : $\\kappa$, $\\beta$ et le "
        "classement des zones sont établis sur l'intégralité des observations. "
        "Les zones ici présentes sont celles que la sélection de l'étape 1 a "
        "retenues sur la fréquence de dominance un contre tous (One-vs-All), "
        "laquelle gouverne le filtrage et non la disposition. La valeur de "
        "$\\beta$ reportée est le rapport \\emph{minimal} de l'enveloppe aux "
        "grandeurs qu'elle majore, conformément aux valeurs de la Table 4 du "
        "papier de référence. Voir Amuzuga et Iwaniack, Int. J. Pressure Vessels "
        "and Piping 222 (2026) 105801, section 2.5.2, algorithme 2 et Table 4. "
    )
    caption = caption1 + caption2 + caption3 + caption4 + caption5
    if not beta_defined:
        caption += (
            "Des valeurs négatives ou nulles figurent parmi les zones retenues : "
            "$\\beta$ n'est alors pas interprétable comme marge de conservatisme.")
    label = "tab:desc_stats"

    # Découpage adaptatif (V6) : ce tableau obéit désormais à la même règle que
    # ceux de l'étape 1. Les deux colonnes terminales, κ·cz et β, restent mises en
    # évidence quel que soit le bloc où le découpage les fait tomber, par un aplat
    # sombre en mode de coloration ``background`` et par la graisse en mode ``text``.
    latex_tab1 = latex_stacked_table(
        df_sample, caption, label, max_columns_per_block,
        column_format_for=tail_column_format_for(color_style_config),
        **_table_style(table_style))

    latex_document = _latex_document_criticalzones(latex_tab1)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    filepath = os.path.join(filedir, f"{filename}.tex")
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(latex_document)
    return filepath


# Tolérance de dépassement horizontal, en points typographiques. En deçà, le
# débordement est invisible à l'œil et relève de l'arrondi de composition.
OVERFULL_TOLERANCE_PT = 1.0


def read_overfull_boxes(log_path: str,
                        tolerance_pt: float = OVERFULL_TOLERANCE_PT) -> List[str]:
    """Relève les dépassements de marge horizontale d'un journal ``pdflatex``.

    Un tableau plus large que la justification produit un avertissement
    ``Overfull \\hbox``, assorti de l'ampleur du débordement. Comme la version 6
    proscrit toute mise à l'échelle des tableaux, c'est le découpage en blocs qui
    doit garantir la tenue dans la page : ce relevé en apporte la preuve.

    Seuls les débordements **horizontaux** sont retenus. Les ``Overfull \\vbox``,
    qui signalent une page trop chargée verticalement, relèvent de la mise en page
    du document et non de la largeur des tableaux.

    Args:
        log_path: Chemin du journal ``.log`` produit par ``pdflatex``.
        tolerance_pt: Ampleur en deçà de laquelle un dépassement est négligé.

    Returns:
        Liste des dépassements retenus, décrits par leur ampleur.
    """
    if not os.path.isfile(log_path):
        return []
    with open(log_path, "r", encoding="utf-8", errors="replace") as stream:
        text = stream.read()
    overfull: List[str] = []
    for amount in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", text):
        if float(amount) > tolerance_pt:
            overfull.append(f"{float(amount):.1f}pt")
    return overfull


#: Durée maximale d'une passe de compilation LaTeX, en secondes.
LATEX_PASS_TIMEOUT_S = 300


def first_latex_error(log_path: str) -> str:
    """Première erreur d'un journal de compilation LaTeX, pour le journal d'exécution.

    Args:
        log_path: Chemin du ``.log`` produit par le compilateur.

    Returns:
        La première ligne d'erreur (commençant par ``!``) suivie de la ligne de
        contexte, ou une chaîne vide si le journal est absent ou sans erreur.
    """
    if not os.path.isfile(log_path):
        return ""
    with open(log_path, "r", encoding="utf-8", errors="replace") as stream:
        lines = stream.read().splitlines()
    for index, line in enumerate(lines):
        if line.startswith("!") or line.lower().startswith("error:"):
            context = next((item.strip() for item in lines[index + 1:index + 6]
                            if item.strip()), "")
            return f"{line.strip()} {context}".strip()[:300]
    return ""


def compile_latex(
    filename: str,
    filedir: str,
    original_directory: str,
    do_compile: bool = True,
    passes: int = 2,
    logger: Optional[Any] = None,
) -> str:
    """Compile ``<filename>.tex`` en PDF (``pdflatex`` ou ``tectonic``), sans jamais lever.

    La compilation est répétée afin de stabiliser les références croisées, les
    numéros de tableau et les signets ``hyperref``. Chaque passe est bornée à
    :data:`LATEX_PASS_TIMEOUT_S` secondes et son code retour est contrôlé : un PDF
    issu d'une compilation en erreur n'est **pas** déclaré valide (il est retiré) ;
    le ``.log`` est alors conservé et sa première erreur citée au journal. L'absence
    du compilateur, son impossibilité de démarrer et son dépassement de délai sont
    traités sans interrompre le pipeline.

    Avant la suppression des fichiers auxiliaires d'une compilation réussie, le
    journal est dépouillé à la recherche de dépassements de marge horizontale.

    Args:
        filename: Nom de base (sans extension).
        filedir: Dossier contenant le ``.tex`` (dossier de travail du compilateur).
        original_directory: Conservé pour compatibilité ; le répertoire courant du
            processus n'est plus modifié.
        do_compile: Active la compilation.
        passes: Nombre de passes de compilation.
        logger: Enregistreur optionnel, destinataire des avertissements.

    Returns:
        Chemin du PDF produit et valide, ou chaîne vide.
    """
    del original_directory  # plus de changement de répertoire : ``cwd`` suffit

    def report(message: str) -> None:
        if logger is not None:
            logger.warning(message)
        else:
            print(f"[WARN] {message}")

    if not do_compile:
        return ""
    compiler = latex_support.find_latex_compiler()
    if compiler is None:
        latex_support.warn_if_missing("OCA", logger)
        return ""
    kind, compiler_path = compiler
    abs_filedir = os.path.abspath(filedir)
    pdf = os.path.join(abs_filedir, f"{filename}.pdf")
    log_path = os.path.join(abs_filedir, f"{filename}.log")
    if os.path.isfile(pdf):
        try:
            os.remove(pdf)
        except OSError:
            pass

    failure = ""
    for _ in range(max(1, passes)):
        try:
            completed = subprocess.run(
                latex_support.compile_command(kind, compiler_path, f"{filename}.tex"),
                check=False, cwd=abs_filedir, timeout=LATEX_PASS_TIMEOUT_S,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            failure = (f"délai de {LATEX_PASS_TIMEOUT_S} s dépassé (compilateur bloqué, "
                       "par exemple en attente d'un paquet manquant)")
            break
        except OSError as exc:
            failure = (f"le compilateur n'a pas pu être lancé ({type(exc).__name__}) : "
                       f"{compiler_path}")
            break
        if completed.returncode != 0:
            failure = f"code retour {completed.returncode}"
            break
    if not failure and not os.path.isfile(pdf):
        failure = "aucun PDF produit"

    if failure:
        detail = first_latex_error(log_path)
        if os.path.isfile(pdf):
            try:
                os.remove(pdf)
            except OSError:
                pass
        report(
            f"Compilation LaTeX de '{filename}' en échec ({failure}) ; PDF non retenu."
            + (f" Première erreur : {detail}." if detail else "")
            + (f" Journal conservé : {os.path.basename(log_path)}."
               if os.path.isfile(log_path) else ""))
        return ""

    overfull = read_overfull_boxes(log_path)
    if overfull:
        report(f"Dépassement de marge à la composition de '{filename}' : "
               f"{len(overfull)} boîte(s) trop large(s) ({', '.join(overfull)}). "
               "Réduire 'latex_output.max_columns_per_block'.")

    # Compilation aboutie : suppression des fichiers auxiliaires afin que le
    # dossier de sortie ne contienne que les livrables. Le journal de compilation
    # est en revanche conservé en cas d'échec (voir plus haut).
    for extension in (".aux", ".out", ".toc", ".log", ".lof", ".lot", ".nav", ".snm"):
        auxiliary = os.path.join(abs_filedir, f"{filename}{extension}")
        if os.path.isfile(auxiliary):
            try:
                os.remove(auxiliary)
            except OSError:
                pass
    return pdf


def merge_pdfs_report(filedir: str, merged_name: str,
                      order: List[str]) -> Tuple[str, List[str]]:
    """Fusionne les PDF listés et signale ceux qui manquent.

    Args:
        filedir: Dossier contenant les PDF.
        merged_name: Nom du PDF consolidé.
        order: Noms de fichiers PDF à fusionner, dans l'ordre souhaité.

    Returns:
        Couple ``(chemin du consolidé ou chaîne vide, noms absents ou illisibles)``.
        Un rapport dont la seconde composante n'est pas vide est **partiel**.
    """
    if PdfMerger is None:
        print("[WARN] PyPDF2/pypdf indisponible : fusion PDF ignorée.")
        return "", list(order)
    merged_path = os.path.join(filedir, merged_name)
    if os.path.exists(merged_path):
        os.remove(merged_path)
    existing = [name for name in order if os.path.isfile(os.path.join(filedir, name))]
    missing = [name for name in order if name not in existing]
    # Une même section ne peut figurer deux fois : un nom répété n'est annexé qu'une
    # fois (la collision des noms de sortie est refusée en amont, à la configuration).
    existing = list(dict.fromkeys(existing))
    if not existing:
        return "", missing
    merger = PdfMerger()
    appended = 0
    for name in existing:
        try:
            merger.append(os.path.join(filedir, name))
            appended += 1
        except Exception:  # noqa: BLE001 - PDF corrompu : traité comme absent
            missing.append(name)
    if not appended:
        return "", missing
    merger.write(merged_path)
    try:
        merger.close()
    except Exception:  # pragma: no cover - variantes d'API pypdf
        pass
    print(f"[OCA] Rapport consolidé : {merged_path}")
    return merged_path, missing


def write_partial_status_pdf(path: str, missing_sections: Sequence[str],
                             reason: str = "") -> str:
    """Écrit la page de statut d'un rapport consolidé partiel, sans LaTeX.

    La page est produite avec matplotlib : elle ne dépend d'aucun compilateur. Elle
    ouvre le rapport consolidé lorsque la notice analytique n'a pas pu être
    compilée, de sorte que le PDF porte lui-même la mention « Rapport partiel » et
    la liste des sections absentes.

    Args:
        path: Chemin du PDF à écrire.
        missing_sections: Noms des sections absentes du rapport.
        reason: Explication courte (compilateur absent ou en échec).

    Returns:
        Le chemin écrit.
    """
    import textwrap

    lines = [("Rapport partiel", 26, True), ("", 12, False),
             ("Ce rapport consolidé est incomplet : les sections ci-dessous n'ont pas pu "
              "être produites et ne figurent pas dans ce PDF.", 12, False), ("", 12, False)]
    for name in missing_sections:
        lines.append((f"  - {name}", 12, False))
    lines.append(("", 12, False))
    if reason:
        lines.append(("Cause : " + reason, 11, False))
        lines.append(("", 11, False))
    lines.append(("Les figures disponibles, les sources .tex des tableaux, les fichiers "
                  "texte et output.json restent produits dans le dossier de sortie.",
                  11, False))
    with matplotlib.rc_context({"pdf.fonttype": 42}):
        fig = plt.figure(figsize=(8.27, 11.69))
        try:
            y = 0.92
            for text, size, bold in lines:
                if not text:
                    y -= 0.02
                    continue
                for piece in textwrap.wrap(text, width=int(92 * 12 / size)) or [""]:
                    fig.text(0.08, y, piece, fontsize=size, va="top", ha="left",
                             fontweight="bold" if bold else "normal", parse_math=False)
                    y -= 0.012 + size / 520.0
            fig.savefig(path, format="pdf")
        finally:
            plt.close(fig)
    return path


def merge_pdfs(filedir: str, merged_name: str, order: List[str]) -> str:
    """Fusionne les PDF listés en un rapport consolidé.

    Args:
        filedir: Dossier contenant les PDF.
        merged_name: Nom du PDF consolidé.
        order: Noms de fichiers PDF à fusionner, dans l'ordre souhaité.

    Returns:
        Chemin du PDF consolidé, ou chaîne vide si impossible. Voir
        :func:`merge_pdfs_report` pour connaître les sections absentes.
    """
    return merge_pdfs_report(filedir, merged_name, order)[0]


# ===========================================================================
# Rapport introductif + journal d'exécution (page de garde du rapport consolidé)
# ===========================================================================
def _tex_escape_cell(text: str) -> str:
    """Échappe les caractères LaTeX spéciaux d'une cellule de tableau.

    Le paquet ``underscore`` gère les ``_`` ; les autres caractères spéciaux sont
    échappés en une passe par :func:`tex_escape_name`, sans double échappement.

    Args:
        text: Texte brut.

    Returns:
        Texte échappé.
    """
    return tex_escape_name(text)


# Caractères Unicode absents du jeu Latin-1 (ou ambigus en fonte machine à
# écrire), remplacés par une transcription ASCII avant insertion en verbatim.
_ASCII_FALLBACKS = {
    "κ": "kappa", "β": "beta", "α": "alpha", "σ": "sigma", "τ": "tau",
    "θ": "theta", "μ": "mu", "λ": "lambda", "γ": "gamma", "δ": "delta",
    "œ": "oe", "Œ": "OE", "æ": "ae", "Æ": "AE",
    "’": "'", "‘": "'", "“": '"', "”": '"', "…": "...",
    "≥": ">=", "≤": "<=", "→": "->", "⇒": "=>", "·": ".",
}

# Largeur utile d'une ligne de verbatim en fonte machine à écrire de corps 9 sur
# une page A4 à marges de 2,2 cm.
_JOURNAL_LINE_WIDTH = 92


def _sanitize_journal(lines: List[str], prefix: str = "") -> str:
    """Rend le journal d'exécution sûr pour un environnement LaTeX ``verbatim``.

    Trois traitements sont appliqués :

        1. transcription ASCII des caractères absents du jeu Latin-1 (lettres
           grecques, ligatures, guillemets typographiques), puis coercition en
           ``?`` de tout résidu non représentable ;
        2. numérotation hiérarchique de chaque événement selon la racine fournie ;
        3. repliement des lignes excédant la largeur utile de la page, avec
           alignement de la continuation sur la marge du texte, l'environnement
           ``verbatim`` ne pratiquant aucune césure automatique.

    Le fichier ``OCA_run.log`` et ``output.json`` conservent, eux, l'Unicode
    d'origine.

    Args:
        lines: Lignes du journal.
        prefix: Racine de numérotation hiérarchique appliquée à chaque ligne
            (par exemple ``"4."`` produit ``4.1``, ``4.2``). Chaîne vide pour
            n'appliquer aucune numérotation.

    Returns:
        Texte assaini prêt pour ``verbatim``.
    """
    out: List[str] = []
    width = len(str(len(lines)))
    for index, line in enumerate(lines, start=1):
        s = line
        for source, ascii_name in _ASCII_FALLBACKS.items():
            s = s.replace(source, ascii_name)
        s = s.encode("latin-1", "replace").decode("latin-1")
        label = f"{prefix}{index:<{width}}  " if prefix else ""
        out.extend(_wrap_verbatim(label, s, _JOURNAL_LINE_WIDTH))
    return "\n".join(out)


def _wrap_verbatim(label: str, text: str, width: int) -> List[str]:
    """Replie un événement de journal sur plusieurs lignes de largeur bornée.

    Le repliement s'opère sur les espaces lorsque cela est possible ; un mot plus
    long que la largeur utile (chemin de fichier, liste de colonnes) est scindé
    afin de garantir l'absence de débordement horizontal.

    Args:
        label: Préfixe de numérotation de la première ligne.
        text: Texte de l'événement.
        width: Largeur utile totale, préfixe compris.

    Returns:
        Liste des lignes produites, la continuation étant indentée sous le texte.
    """
    indent = " " * len(label)
    avail = max(20, width - len(label))
    words = text.split(" ")
    lines: List[str] = []
    current = ""
    for word in words:
        while len(word) > avail:
            if current:
                lines.append(current)
                current = ""
            lines.append(word[:avail])
            word = word[avail:]
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= avail:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current or not lines:
        lines.append(current)
    return [f"{label}{lines[0]}"] + [f"{indent}{line}" for line in lines[1:]]


def generate_report_intro(
    summary: Dict[str, Any],
    log_records: List[str],
    filedir: str,
    filename: str,
    sections: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Génère la notice analytique du rapport consolidé (LaTeX).

    Produit les pages liminaires du rapport ``00-OCA_Report.pdf`` : objet de
    l'étude, chaîne de traitement, configuration analysée, synthèse des
    indicateurs quantitatifs, journal d'exécution et composition du document.

    Conventions rédactionnelles imposées à la V3 : registre scientifique et
    technique, absence de tiret cadratin ou de tiret employé comme marqueur de
    liste ou de rupture de paragraphe, structuration exclusivement assurée par une
    numérotation hiérarchique (sections ``1.``, sous sections ``1.1``, énumérations
    ``a.``).

    Args:
        summary: Indicateurs clés (kappa, kappa_found, beta_mean, beta_max,
            n_sample, target_zone, target_auto, dimensioning_zones,
            retained_zones, dropped_zones, n_nodes_step1, n_relations_step1,
            config).
        log_records: Lignes du journal d'exécution.
        filedir: Dossier de sortie.
        filename: Nom de base du fichier (sans extension).
        sections: Composition du rapport, sous forme de couples
            ``(nom de fichier, objet de la section)``.

    Returns:
        Chemin du ``.tex`` écrit.
    """
    def esc(x: Any) -> str:
        return _tex_escape_cell(x)

    def _join(lst: Any) -> str:
        return esc(", ".join(str(x) for x in (lst or [])) or "néant")

    target_label = esc(summary.get("target_zone", "non résolue"))
    if summary.get("target_auto"):
        target_label += " (résolution automatique par fréquence de dominance maximale)"

    kappa_places = max(2, int(summary.get("kappa_decimals", 2)))
    kappa_label = f"{summary.get('kappa', float('nan')):.{kappa_places}f}"
    if not summary.get("kappa_found", True):
        kappa_label += " (seuil de dominance non atteint sur la plage explorée)"

    # Les libellés (colonne de gauche) sont du LaTeX de confiance, susceptible de
    # contenir des mathématiques ($\kappa$, $\bar{\beta}$) : ils ne sont pas
    # échappés. Seules les valeurs issues des données le sont, via esc().
    indicator_rows = [
        ("Cible de régularisation", target_label),
        ("Coefficient multiplicateur optimal $\\kappa$", kappa_label),
        ("Indice de parcimonie moyen $\\bar{\\beta}$",
         f"{summary.get('beta_mean', float('nan')):.3f}"),
        ("Indice de parcimonie maximal $\\beta_{\\max}$",
         f"{summary.get('beta_max', float('nan')):.3f}"),
        ("Effectif analysé $N$", f"{summary.get('n_sample', 0)}"),
        ("Zones critiques soumises à l'analyse",
         f"{len(summary.get('config', {}).get('critical_zones', []))}"),
        ("Nœuds représentés au graphe de l'étape 1",
         f"{summary.get('n_nodes_step1', 0)}"),
        ("Relations d'ordre retenues à l'étape 1",
         f"{summary.get('n_relations_step1', 0)}"),
        ("Zones dimensionnantes", _join(summary.get("dimensioning_zones", []))),
        ("Zones retenues après sélection", _join(summary.get("retained_zones", []))),
        ("Zones écartées par la sélection", _join(summary.get("dropped_zones", []))),
    ]
    indicator_rows_latex = "\n".join(f"{k} & {v} \\\\" for k, v in indicator_rows)

    # Scores globaux de dominance de l'étape 1, présentés de la zone la moins
    # dominante à la plus dominante, dans l'ordre même où les colonnes sont
    # disposées dans les tableaux d'échantillonnage.
    frequencies = summary.get("dominance_frequencies", {}) or {}
    ranking = summary.get("dominance_ranking", []) or list(frequencies)
    step1_scores = summary.get("matrix_dominance_scores", {}) or {}
    step2_scores = summary.get("matrix_dominance_scores_step2", {}) or {}
    if frequencies:
        def _score(table: Dict[str, Any], zone: str) -> str:
            # La mention « s.o. » plutôt qu'un tiret : les conventions
            # rédactionnelles de l'application proscrivent le tiret, et la double
            # frappe qui compose un demi cadratin en LaTeX se lirait ici comme
            # une valeur soustractive dans une colonne de nombres.
            value = table.get(zone)
            return "s.o." if value is None else f"{float(value):.4f}"

        dominance_rows_latex = "\n".join(
            f"{index}. & {esc(zone)} & "
            f"{frequencies.get(zone, float('nan')):.2f}\\% & "
            f"{_score(step1_scores, zone)} & {_score(step2_scores, zone)} \\\\"
            for index, zone in enumerate(ranking, start=1)
        )
        dominance_block = rf"""
\subsection*{{2.1 Scores globaux de dominance}}
Deux grandeurs distinctes qualifient la dominance d'une zone critique, et elles
ne remplissent pas le même office.

\begin{{enumerate}}
\item La \textbf{{fréquence $p$}} procède du comptage \textbf{{un contre tous}}
      (One-vs-All) : pour chacune des $N$ configurations de la base, la zone
      portant la contrainte maximale est relevée, et $p$ est la proportion des
      configurations où la zone considérée l'emporte sur toutes les autres
      simultanément. Cette fréquence gouverne la \textbf{{sélection}} des zones
      conservées pour l'étape 2, ainsi que l'ordre du tableau
      {{\ttfamily 01-Step1\_Dominance\_Hierarchy\_Result}}, dont les cumuls $N$
      et $P$ en dépendent.
\item Le \textbf{{score ordinal}} procède de la comparaison \textbf{{par
      paires}} (colonne par colonne) : pour chaque couple de variables, la
      matrice de corrélation ordinale porte la proportion des observations où
      l'une majore l'autre, et le score d'une zone est la moyenne des
      proportions de sa ligne. Ce score gouverne la \textbf{{disposition
      horizontale}} des colonnes des tableaux d'échantillonnage.
\end{{enumerate}}

\noindent La distinction n'est pas de pure forme. La fréquence $p$ exige qu'une
zone majore toutes les autres à la fois, en sorte qu'elle sature dès qu'une zone
domine largement : les autres tombent alors à zéro conjointement et ne sont plus
départagées. Le score ordinal, fondé sur des comparaisons deux à deux, les
départage toujours et restitue fidèlement la hiérarchie que la matrice donne à
lire.

\begin{{center}}
\begin{{tabular}}{{rlrrr}}
\toprule
\textbf{{Rang}} & \textbf{{Zone critique}} & \textbf{{Fréquence $p$}}
& \textbf{{Score étape 1}} & \textbf{{Score étape 2}} \\
& & {{\small (One-vs-All)}} & {{\small (paires)}} & {{\small (paires)}} \\
\midrule
{dominance_rows_latex}
\bottomrule
\end{{tabular}}
\end{{center}}

\noindent Le rang 1 désigne la zone la moins dominante au sens du score de
l'étape 1, le dernier rang la plus dominante. Cet ordre est celui que reprend, de
gauche à droite, le second tableau de
{{\ttfamily 04-Step1\_Critical\_Zones\_Sample\_Result}}. Le score de l'étape 1
est établi sur la matrice {{\ttfamily 02-OCA\_CorrMat\_Targets}}, laquelle porte
l'intégralité des zones critiques avant toute sélection. Le score de l'étape 2
est établi sur la matrice {{\ttfamily 05-OCA\_OrdinalCorrMat\_Targets}}, restreinte
aux zones retenues et augmentée de la colonne enveloppe régularisée ; il ordonne
le tableau {{\ttfamily 07-CriticalZones\_Targets}}, où les colonnes terminales
$\kappa \cdot c_z$ et $\beta$ demeurent en fin de tableau. La mention
{{\ttfamily s.o.}}, sans objet, désigne les zones que la sélection a écartées :
absentes de la matrice de l'étape 2, elles n'y reçoivent aucun score.

\noindent Les définitions et les algorithmes correspondants figurent à la
section 2.5.2 de Amuzuga et Iwaniack, \emph{{Int. J. Pressure Vessels and
Piping}} \textbf{{222}} (2026) 105801 : l'algorithme 1 et la Table 3 pour le
comptage un contre tous, l'algorithme 2 et la figure 11 pour la matrice de
corrélation ordinale.
"""
    else:
        dominance_block = ""

    cfg = summary.get("config", {})
    config_rows = [
        ("Zones critiques " + ("déduites du classeur" if cfg.get("critical_zones_inferred")
                               else "déclarées"), _join(cfg.get("critical_zones", []))),
        ("Variables de conception " + ("déduites du classeur"
                                       if cfg.get("feature_cols_inferred") else "déclarées"),
         _join(cfg.get("feature_cols_init", []))),
        ("Variables synthétiques", _join(cfg.get("user_defined_vars_cols", []))),
        ("Cible de régularisation configurée",
         "résolution automatique par fréquence de dominance maximale"
         if str(cfg.get("target_zone_cfg", "")).strip().lower() == "none"
         else esc(str(cfg.get("target_zone_cfg", "non définie")))),
        ("Seuils de sélection GPO sur $p$",
         f"seuil bas {cfg.get('gpo_low', 'non défini')}\\%, "
         f"seuil haut {cfg.get('gpo_high', 'non défini')}\\%"),
        ("Régularisation $\\kappa$",
         f"seuil de dominance {cfg.get('reg_threshold', 'non défini')}, "
         f"$\\kappa\\in[{cfg.get('kappa_min', '?')}, {cfg.get('kappa_max', '?')}]$, "
         f"pas {cfg.get('kappa_step', 'non défini')}"),
        ("Seuil du graphe de dominance", f"{cfg.get('graph_threshold', 'non défini')}"),
        ("Effectif de l'échantillon tabulé", f"{cfg.get('sample_size', 'non défini')}"),
        ("Fichier de données", esc(str(cfg.get("data_file", "non défini")))),
    ]
    config_rows_latex = "\n".join(f"{k} & {v} \\\\" for k, v in config_rows)

    journal = _sanitize_journal(log_records, prefix="4.") if log_records \
        else "(aucun événement enregistré)"

    sections_latex = "\n".join(
        f"\\item \\texttt{{{esc(name)}}} : {esc(desc)}" for name, desc in (sections or [])
    ) or "\\item Aucune section annexée."

    run_stamp = esc(summary.get("generated_at", "date non renseignée"))
    version_label = esc(summary.get("oca_version", ""))
    if version_label:
        run_stamp += f", OCA version {version_label}"

    # Rapport partiel : les sections dont le PDF n'a pas pu être produit sont
    # nommées, afin que le lecteur ne prenne pas ce document pour complet.
    missing_sections = [str(name) for name in (summary.get("missing_sections") or [])]
    if missing_sections:
        partial_latex = (
            "\\noindent \\textbf{Rapport partiel.} Les sections suivantes n'ont pas pu "
            "être produites, la compilation LaTeX étant indisponible ou en échec ; les "
            "fichiers .tex correspondants sont conservés dans le dossier de sortie : "
            + ", ".join(f"\\texttt{{{esc(name)}}}" for name in missing_sections) + ".")
    else:
        partial_latex = ""
    beta_note = esc(summary.get("beta_note", ""))
    if beta_note:
        beta_note_latex = "\\noindent \\textbf{Domaine de $\\beta$.} " + beta_note
    else:
        beta_note_latex = ""

    latex_document = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage[french]{{babel}}
\usepackage{{lmodern}}
\usepackage[a4paper,margin=2.2cm]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{tabularx}}
\usepackage{{enumitem}}
\usepackage[nohyphen]{{underscore}}
\usepackage{{array}}
\usepackage{{xcolor}}
\usepackage{{amsmath}}
\usepackage[hidelinks]{{hyperref}}

\setlist[enumerate,1]{{label=\alph*., leftmargin=1.6em, itemsep=0.15em, topsep=0.3em}}
\setlist[enumerate,2]{{label=\roman*., leftmargin=1.6em, itemsep=0.1em}}
\newcolumntype{{L}}{{>{{\raggedright\arraybackslash}}X}}

\begin{{document}}

\begin{{center}}
{{\LARGE\bfseries Rapport OCA}}\\[0.20cm]
{{\large Algorithme de corrélation ordinale}}\\[0.15cm]
{{\normalsize Analyse ordinale de dominance et régularisation par enveloppe majorante}}\\[0.25cm]
{{\small Exécution du {run_stamp}}}
\end{{center}}
\vspace{{0.35cm}}

\section*{{1. Objet de l'étude}}
L'algorithme OCA établit un ordre partiel de dominance sur un ensemble de zones
critiques, puis construit une variable synthétique enveloppe $\kappa\cdot c_z$
destinée à majorer l'intégralité des grandeurs retenues. La réduction de
dimension procède ainsi d'une comparaison ordinale terme à terme, indépendante de
toute hypothèse de linéarité entre les variables. Le présent document consigne
l'intégralité des résultats numériques, tabulaires et graphiques de l'exécution
dont les paramètres sont rappelés à la section 1.2.

\subsection*{{1.1 Chaîne de traitement}}
\begin{{enumerate}}
\item Chargement des données, fusion des sources et calcul des variables
      synthétiques déclarées dans la configuration.
\item Établissement de la matrice de corrélation ordinale initiale sur
      l'intégralité des zones critiques, préalablement à toute sélection, par
      comparaison des colonnes \textbf{{deux à deux}}. Cette matrice constitue
      l'état des lieux exhaustif des relations hiérarchiques ordinales entre les
      grandeurs mesurées.
\item Construction du graphe d'ordonnancement de l'étape 1 à partir de cette
      matrice. Toutes les zones critiques y sont représentées, y compris celles
      dépourvues de relation d'ordre franchissant le seuil.
\item Sélection par comptage de dominance \textbf{{un contre tous}} (One-vs-All),
      puis classement des zones selon la fréquence de dominance $p$ et
      conservation des zones dont $p$ atteint le seuil bas (le seuil haut
      désigne seulement les zones dimensionnantes, sans rien filtrer). Ce
      comptage appartient
      strictement à l'étape 1 de l'algorithme, la sélection par l'ordre de Pareto
      généralisé.
\item Recherche du coefficient multiplicateur optimal $\kappa$ assurant à la
      variable cible un rôle d'enveloppe majorante au seuil de dominance requis.
\item Calcul de la matrice de corrélation ordinale de l'étape 2, augmentée de la
      colonne régularisée, et construction du graphe de dominance final. Cette
      seconde matrice procède du même mode de comparaison \textbf{{par paires}}
      que la matrice initiale, mais se rattache algorithmiquement à l'étape 2, la
      régularisation par l'ordre de Pareto généralisé.
\item Quantification du conservatisme de la formule d'enveloppe par l'indice de
      parcimonie $\beta$, puis échantillonnage tabulaire reproductible.
\end{{enumerate}}

\noindent La méthode et ses deux étapes sont exposées à la section 2.5.2 de
Amuzuga et Iwaniack, \emph{{Int. J. Pressure Vessels and Piping}} \textbf{{222}}
(2026) 105801, DOI 10.1016/j.ijpvp.2026.105801, à laquelle le lecteur se
reportera pour le détail des développements. Les notations $q$, $f$, $Q$, $F$,
$\gamma$ et $\tau$ du papier y correspondent respectivement à $n$, $p$, $N$, $P$,
au seuil de dominance et à $\beta$ dans la présente application.

\subsection*{{1.2 Configuration analysée}}
Le problème traité est entièrement défini par les paramètres du fichier
\texttt{{input.json}} reproduits au tableau suivant.
\begin{{center}}
\begin{{tabularx}}{{\textwidth}}{{lL}}
\toprule
\textbf{{Paramètre}} & \textbf{{Valeur}} \\
\midrule
{config_rows_latex}
\bottomrule
\end{{tabularx}}
\end{{center}}

\section*{{2. Synthèse des indicateurs quantitatifs}}
\begin{{center}}
\begin{{tabularx}}{{\textwidth}}{{lL}}
\toprule
\textbf{{Indicateur}} & \textbf{{Valeur}} \\
\midrule
{indicator_rows_latex}
\bottomrule
\end{{tabularx}}
\end{{center}}

\noindent L'indice de parcimonie $\beta$ mesure, pour chaque observation, le
rapport \textbf{{minimal}} entre l'enveloppe $\kappa\cdot c_z$ et les grandeurs
qu'elle majore. Une valeur proche de l'unité atteste une enveloppe serrée, donc
peu conservative ; une valeur élevée traduit un surdimensionnement de la formule
d'enveloppe. C'est bien le rapport minimal, et non maximal, qui est retenu : il
mesure la marge la plus étroite, celle qui sépare l'enveloppe de la plus grande
des grandeurs majorées, et c'est lui seul qui renseigne sur le caractère serré de
la formule. Les valeurs de la Table 4 du papier de référence procèdent de ce même
calcul, la formule imprimée à la section 2.5.2 du papier comportant une
inadvertance sur ce point.
{beta_note_latex}
{dominance_block}
\subsection*{{2.2 Périmètre statistique des indicateurs}}
L'intégralité des indicateurs reportés ci dessus est établie sur la
\textbf{{totalité des observations}} du jeu de données, soit $N$ configurations.
Sont concernés le comptage un contre tous (One-vs-All) et les fréquences de
dominance $p$ de l'étape 1, la sélection par l'ordre de Pareto généralisé et le
classement des zones critiques qui en découle, les comparaisons par paires dont
procèdent les matrices de corrélation ordinale des deux étapes, les graphes de
dominance, la recherche du coefficient de régularisation $\kappa$ et le calcul de
l'indice de parcimonie $\beta$. Aucune de ces grandeurs n'est estimée sur un
sous ensemble.

Le paramètre {{\ttfamily sample\_size}} de la configuration ne gouverne
\textbf{{que l'affichage}} : il fixe le nombre de lignes extraites, par tirage à
graine fixe, pour l'illustration des tableaux
{{\ttfamily 04-Step1\_Critical\_Zones\_Sample\_Result}} et
{{\ttfamily 07-CriticalZones\_Targets}}. Aucune décision de l'algorithme n'est
prise sur ce sous ensemble. Restreindre ou étendre cet effectif modifie la
présentation des extraits sans altérer d'aucune façon les grandeurs calculées.

\section*{{3. Composition du rapport consolidé}}
Les sections suivantes sont annexées au présent document, dans cet ordre.
\begin{{enumerate}}
{sections_latex}
\end{{enumerate}}
{partial_latex}

\section*{{4. Journal d'exécution}}
Les événements consignés lors de l'exécution sont numérotés selon leur ordre
d'apparition. Les avertissements signalent les situations traitées par un
mécanisme de repli sans interruption du calcul.
{{\footnotesize
\begin{{verbatim}}
{journal}
\end{{verbatim}}
}}

\end{{document}}
"""
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    filepath = os.path.join(filedir, f"{filename}.tex")
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(latex_document)
    return filepath


# ===========================================================================
# output.json
# ===========================================================================
def build_output_json(
    config: Any,
    df_results: pd.DataFrame,
    dico_results: Dict[str, List[str]],
    updated_zones: List[str],
    kappa: float,
    kappa_found: bool,
    prop_matrix: pd.DataFrame,
    graph_data: Dict[str, Any],
    max_beta: float,
    mean_beta: float,
    n_sample: int,
    resources: Dict[str, str],
    target_zone: str,
    target_auto: bool,
    initial_prop_matrix: pd.DataFrame,
    step1_graph_data: Dict[str, Any],
    initial_critical_zones: List[str],
    log_records: Optional[List[str]] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construit la structure centralisée ``output.json``.

    Args:
        config: Configuration OCA.
        df_results: Tableau des dominations.
        dico_results: Sélection GPO.
        updated_zones: Zones critiques après mise à jour.
        kappa: Coefficient optimal.
        kappa_found: Indique si κ satisfait le seuil.
        prop_matrix: Matrice de corrélation ordinale de l'étape 2 (avec enveloppe).
        graph_data: Structure du graphe de dominance final.
        max_beta: β maximal.
        mean_beta: β moyen.
        n_sample: Nombre de cas examinés.
        resources: Chemins des ressources générées.
        target_zone: Cible de régularisation effectivement retenue.
        target_auto: Vrai si la cible a été sélectionnée automatiquement.
        initial_prop_matrix: Matrice de corrélation ordinale initiale, établie sur
            l'intégralité des zones critiques avant sélection (V3).
        step1_graph_data: Structure du graphe d'ordonnancement de l'étape 1 (V3).
        initial_critical_zones: Zones critiques soumises à l'analyse, avant
            sélection.
        log_records: Journal d'exécution (avertissements/événements).
        extras: Compléments de la version 9 : ``oca_version``, ``kappa_decimals``,
            ``kappa_search``, ``beta_domain``, ``column_resolution``,
            ``target_ties``, ``report``, ``configuration_notes``. Toute clé absente
            est simplement omise.

    Returns:
        Dictionnaire natif prêt à sérialiser.
    """
    extras = extras or {}
    kappa_places = max(2, int(extras.get("kappa_decimals", 2)))
    beta_domain = extras.get("beta_domain") or {}
    ordinal_relations = [
        {"dominant": vi, "dominated": vj, "proportion": round(float(p), 2)}
        for vi, vj, p in graph_data["pairs_above"]
    ]
    step1_relations = [
        {"dominant": vi, "dominated": vj, "proportion": round(float(p), 4)}
        for vi, vj, p in step1_graph_data["pairs_above"]
    ]

    def _near(graph_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Couples quasi dominants, sous le seuil et dans la marge de signalement."""
        return [
            {"dominant": vi, "dominated": vj, "proportion": round(float(p), 6)}
            for vi, vj, p in graph_data.get("near_threshold", [])
        ]

    records = log_records or []
    metrics: Dict[str, Any] = {
        # Le flottant réellement éprouvé, sans réarrondi : seul l'affichage (libellés,
        # journal) est arrondi à ``kappa_decimals`` décimales.
        "kappa_optimal": float(kappa),
        "kappa_found": bool(kappa_found),
        "beta_max": round(float(max_beta), 2),
        "beta_mean": round(float(mean_beta), 2),
        "n_sample": int(n_sample),
    }
    if extras.get("kappa_decimals") is not None:
        metrics["kappa_decimals"] = kappa_places
    if beta_domain:
        metrics["beta_defined"] = bool(beta_domain.get("beta_defined", True))
        if beta_domain.get("note"):
            metrics["beta_note"] = beta_domain["note"]
    target_block: Dict[str, Any] = {
        "resolved": target_zone,
        "auto_selected": bool(target_auto),
        "configured": config.synthetic_target,
    }
    target_block.update(extras.get("target_ties") or {})
    document_head: Dict[str, Any] = {}
    if extras.get("oca_version"):
        document_head["oca_version"] = extras["oca_version"]
    result = {
        **document_head,
        "metrics": metrics,
        "target_zone": target_block,
        # Étape 1 (V3) : état des lieux ordinal exhaustif, antérieur à toute
        # sélection de variables, et graphe d'ordonnancement associé.
        "initial_ordinal_state": {
            "critical_zones": list(initial_critical_zones),
            "ordinal_correlation_matrix": to_native(initial_prop_matrix.round(4).to_dict()),
            "graph_threshold": config.ordinal_graph["threshold"],
            "n_nodes": int(step1_graph_data["graph"].number_of_nodes()),
            "n_edges": int(step1_graph_data["graph"].number_of_edges()),
            "nodes": list(step1_graph_data["graph"].nodes()),
            "isolated_nodes": list(step1_graph_data["isolated"]),
            "levels": to_native(step1_graph_data["levels"]),
            "relations": step1_relations,
            "chains": [" < ".join(chain) for chain in step1_graph_data["chains"]],
            "chains_reading": NOTATION_SENTENCE,
            # Cohérence matrice et graphe (V4) : les degrés doivent égaler le
            # dénombrement des cellules encadrées de la ligne et de la colonne.
            "degrees": to_native(step1_graph_data["degrees"]),
            "near_threshold_pairs": _near(step1_graph_data),
        },
        "gpo_selection": {
            "thresholds": {
                "low_percent": config.gpo_selection["threshold_low"],
                "high_percent": config.gpo_selection["threshold_high"],
            },
            "dominance_table": to_native(df_results.to_dict()),
            "selection": to_native(dico_results),
        },
        "updated_critical_zones": list(updated_zones),
        "ordinal_correlation_matrix": to_native(prop_matrix.round(4).to_dict()),
        "ordinal_relations": ordinal_relations,
        "ordinal_graph_degrees": to_native(graph_data["degrees"]),
        "ordinal_near_threshold_pairs": _near(graph_data),
        "hyperparameters": {
            "gpo_regularization": to_native(config.gpo_regularization),
            "ordinal_graph_threshold": config.ordinal_graph["threshold"],
            "sample_random_state": config.sample_random_state,
            # Style de couleurs déclaré, normalisé, ET cartes effectivement
            # employées. Sans cette consignation, deux jeux de livrables
            # visuellement très différents produisaient des fichiers de résultats
            # rigoureusement indiscernables, et le mode de rendu d'une exécution
            # archivée n'était plus reconstituable.
            #
            # Les deux relevés sont distincts et tous deux nécessaires : une carte
            # déclarée peut être écartée par un repli, ainsi d'une carte
            # catégorielle demandée au fond d'une matrice. Ne consigner que la
            # valeur déclarée eût prêté à la carte de repli le nom de celle qu'elle
            # remplaçait.
            "color_style": _color_style_record(
                getattr(config, "color_style", None),
                len(getattr(config, "critical_zones", []) or [])),
        },
        "output_files": to_native(config.output_files),
        "execution_log": {"n_events": len(records), "records": records},
        "resources": {k: v.replace("\\", "/") for k, v in resources.items() if v},
    }
    for key in ("kappa_search", "beta_domain", "column_resolution", "report",
                "configuration_notes"):
        if extras.get(key) is not None:
            result[key] = extras[key]
    return result
