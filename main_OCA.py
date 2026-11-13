"""Point d'entrée principal du pipeline OCA (version 9.0.0).

Orchestre le flux complet, piloté par ``input.json`` :

    1. journalisation et lecture/validation de la configuration ;
    2. chargement/fusion des données, calcul des variables synthétiques ;
    3. **matrice de corrélation ordinale initiale** établie sur l'intégralité des
       zones critiques, préalablement à toute sélection de variables (V3) ;
    4. **graphe d'ordonnancement de l'étape 1**, déduit de cette matrice, où
       l'intégralité des zones critiques est représentée (V3) ;
    5. graphiques descriptifs (boîtes à moustaches, histogrammes) ;
    6. sélection GPO (One-vs-All) puis **résolution de la cible de régularisation**
       (manuelle, automatique via ``"none"``, ou synthétique ajoutée) ;
    7. régularisation κ, matrice de corrélation ordinale augmentée de l'enveloppe
       et graphe de dominance final ;
    8. indice de parcimonie β et tableau d'échantillonnage ;
    9. rapport consolidé numéroté (notice analytique, journal d'exécution et
       sections 01 à 09) et ``output.json``.

Toutes les ressources sont numérotées de ``00-`` à ``11-`` par ordre
d'importance, soit douze livrables.

Révisions de la version 3 :

    a. la ressource ``02-OCA_CorrMat_Targets.pdf`` porte une matrice de corrélation
       ordinale, et non plus linéaire, calculée avant la première sélection ;
    b. la ressource ``03-Step1_OrdinalGraph_Targets.pdf`` représente
       systématiquement toutes les zones critiques, y compris celles dépourvues de
       relation d'ordre franchissant le seuil.

Usage:
    python main_OCA.py                     # utilise ./input.json
    python main_OCA.py --config x.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _require_dependencies() -> None:
    """Vérifie les dépendances avant d'importer les modules qui les utilisent.

    Une dépendance absente donne une ligne qui nomme le paquet et la commande
    d'installation (code de sortie 3), jamais une trace d'importation.
    """
    import importlib.util
    required = (("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"),
                ("matplotlib", "matplotlib"), ("seaborn", "seaborn"),
                ("networkx", "networkx"), ("openpyxl", "openpyxl"),
                ("jinja2", "Jinja2"))
    missing = [package for module, package in required
               if importlib.util.find_spec(module) is None]
    if missing:
        print(f"[OCA][ERREUR D'INSTALLATION] Paquet(s) Python manquant(s) : "
              f"{', '.join(missing)}. Installez les dépendances avec : "
              "pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(3)


_require_dependencies()

import latex_support  # noqa: E402
from config_manager import (
    OCA_VERSION,
    ConfigError,
    check_data_sources,
    check_finite_columns,
    clean_output_main,
    load_config,
    load_data_with_resolution,
    prepare_output_dirs,
    setup_logging,
)
from oca_model import (
    beta_domain_report,
    build_dominance_graph,
    check_matrix_graph_coherence,
    compute_parsimony,
    dominance_frequencies,
    dominance_ranking,
    dominance_ties,
    find_kappa_details,
    gpo_selection,
    highest_dominance_column,
    matrix_dominance_ranking,
    matrix_dominance_scores,
    order_columns_by_dominance,
    ordinal_correlation_matrix,
    ordinal_correlation_matrix_with_envelope,
    update_critical_zones,
)
from post_processor import (
    DEFAULT_MAX_COLUMNS_PER_BLOCK,
    SPLIT_BLOCK_FONT_SIZE,
    TABLE_COLUMN_SEPARATION,
    build_output_json,
    compile_latex,
    generate_criticalzones_table,
    generate_report_intro,
    generate_result_table,
    generate_step1_zones_sample_table,
    merge_pdfs_report,
    plot_boxplot,
    plot_hist,
    plot_ordinal_corr_matrix,
    plot_ordinal_graph,
    plot_scatter,
    split_column_bounds,
    to_native,
    reset_style_notices,
    write_ordinal_graph_txt,
    write_partial_status_pdf,
    write_ordinal_matrix_txt,
    write_sort_txt,
    zone_color_map,
)


def audit_dominance_coherence(
    prop_matrix: "Any",
    graph_data: Dict[str, Any],
    threshold: float,
    logger: "Any",
    label: str,
) -> None:
    """Consigne au journal la cohérence entre une matrice ordinale et son graphe.

    Le contrôle structurel est délégué à
    :func:`oca_model.check_matrix_graph_coherence`. Les couples dont la proportion
    approche le seuil sans l'atteindre sont en outre signalés : indiscernables des
    couples retenus sur une échelle de couleur saturée, ils sont la cause des
    incohérences apparentes entre matrice et graphe, et méritent d'être explicités
    lorsque la hiérarchie s'avère sensible au seuil.

    Args:
        prop_matrix: Matrice des proportions de dominance.
        graph_data: Structure du graphe de dominance.
        threshold: Seuil de dominance appliqué.
        logger: Journal d'exécution.
        label: Désignation de l'étape, reprise dans les messages.
    """
    issues = check_matrix_graph_coherence(prop_matrix, graph_data, threshold)
    if issues:
        for issue in issues:
            logger.warning(f"Cohérence matrice et graphe ({label}) : {issue}")
    else:
        degrees = graph_data["degrees"]
        detail = ", ".join(
            f"{node} domine {info['out']}" for node, info in degrees.items() if info["out"]
        ) or "aucune dominance au seuil"
        logger.info(
            f"Cohérence matrice et graphe ({label}) vérifiée : {detail}.")

    for var_i, var_j, proportion in graph_data.get("near_threshold", []):
        logger.warning(
            f"Couple quasi dominant ({label}) : {var_i} majore {var_j} dans "
            f"{proportion:.4f} des observations, soit {threshold - proportion:.4f} "
            f"sous le seuil {threshold} ; aucune arête n'est portée au graphe.")


def run_subdir(config: "Any", config_path: str) -> Dict[str, Any]:
    """Exécute le pipeline OCA sur un unique sous-dossier d'entrée.

    Cette fonction porte l'intégralité de la logique OCA. Elle est appelée une
    fois par sous-dossier déclaré dans ``merged_subdir`` (V6), avec à chaque fois
    une configuration dédiée obtenue par :meth:`OCAConfig.for_subdir`. Aucun état
    n'est partagé d'une exécution à l'autre : les données, le journal, les
    ressources graphiques et le rapport consolidé sont propres au sous-dossier.

    Args:
        config: Configuration validée, dont ``merged_subdir`` désigne le
            sous-dossier à traiter.
        config_path: Chemin du fichier de configuration, repris dans le journal.

    Returns:
        Le bloc ``results`` de ce sous-dossier, prêt à être sérialisé.
    """
    original_directory = os.getcwd()
    # La source de données est vérifiée AVANT tout nettoyage : une configuration
    # sans données ne doit pas vider les résultats d'une exécution précédente.
    check_data_sources(config)
    # Nettoyage des sorties antérieures AVANT l'ouverture du log (évite d'accumuler
    # des fichiers périmés et de supprimer le fichier de log en cours d'écriture).
    clean_output_main(config)
    filedir = prepare_output_dirs(config)
    of = config.output_files
    rs = config.sample_random_state
    max_columns_per_block = int(
        config.latex_output.get("max_columns_per_block", DEFAULT_MAX_COLUMNS_PER_BLOCK))
    # Style de couleurs commun a l'ensemble des sorties graphiques et tabulaires.
    style = config.color_style
    # Typographie des tableaux de données (V6) : fixe, réduite et identique pour
    # les trois tableaux et tous leurs blocs.
    table_style = {
        "font_size": str(config.latex_output.get(
            "table_font_size", SPLIT_BLOCK_FONT_SIZE)),
        "column_separation": str(config.latex_output.get(
            "table_column_separation", TABLE_COLUMN_SEPARATION)),
    }

    logger, mem = setup_logging(os.path.join(filedir, of["log"]))
    # Chaque exécution d'un lot est strictement isolée : la mémoire des
    # avertissements de style est vidée pour que le journal du sous-dossier porte
    # les siens, et n'hérite pas du silence de la précédente exécution.
    reset_style_notices()
    logger.info(f"Démarrage du pipeline OCA, version {OCA_VERSION}.")
    for note in config.normalization_notes:
        logger.info(f"Configuration '{config_path}' : {note}")
    for warning in config.normalization_warnings:
        logger.warning(f"Configuration '{config_path}' : {warning}")
    if latex_support.find_latex_compiler() is None:
        logger.warning(latex_support.missing_message("OCA"))
    if config.is_multi_run:
        logger.info(
            f"Traitement du sous-dossier '{config.merged_subdir}' "
            f"({config.merged_subdirs.index(config.merged_subdir) + 1} sur "
            f"{len(config.merged_subdirs)}).")
    # V7 : les commentaires sont retirés en amont, systématiquement et sans bruit.
    # Leur présence est consignée pour mémoire, au niveau informatif : l'écriture
    # commentée est admise et ne constitue pas un écart à corriger.
    for note in config.comment_notes:
        logger.info(f"Configuration '{config_path}' : {note}.")
    for repair in config.json_repairs:
        logger.warning(
            f"Configuration '{config_path}' non conforme à la norme JSON : {repair}. "
            "L'écart a été neutralisé à la lecture ; il convient de corriger le fichier.")
    # V7 : les clés d'auto-documentation sont consignées pour mémoire, non comme un
    # écart. Elles énumèrent à l'intention du lecteur du fichier les valeurs qu'un
    # paramètre admet, et sont retirées avant tout traitement.
    if config.option_keys:
        logger.info(
            f"{len(config.option_keys)} clé(s) d'auto-documentation détectée(s) dans "
            f"'{config_path}' et écartée(s) du paramétrage : "
            f"{', '.join(config.option_keys)}.")
    # V7 : écritures des versions antérieures reconnues et transposées. Ce n'est
    # pas un écart mais une compatibilité assumée, consignée pour mémoire.
    for note in config.legacy_notes:
        logger.info(f"Configuration '{config_path}' : {note}.")

    resources: Dict[str, str] = {"log": os.path.join(filedir, of["log"])}

    # 1. Données + variables synthétiques (critical_zones inclut les variables
    #    synthétiques déclarées, par exemple De/E).
    df_merged, critical_zones, resolution = load_data_with_resolution(config, logger)
    logger.info(f"{len(df_merged)} configurations chargées ; zones analysées : {critical_zones}")

    # 2. Matrice de corrélation ordinale INITIALE (V3) : état des lieux exhaustif
    #    des relations hiérarchiques ordinales, établi sur l'intégralité des zones
    #    critiques et antérieur à toute sélection de variables. Elle remplace la
    #    matrice de corrélation linéaire de Pearson de la version 2.
    graph_threshold = config.ordinal_graph["threshold"]
    presel_matrix, presel_vars = ordinal_correlation_matrix(df_merged, critical_zones)

    # Association zone vers teinte, établie UNE SEULE FOIS pour l'exécution et
    # transmise à tous les livrables. Chaque figure et chaque tableau indexaient
    # auparavant la palette sur leur propre ordre de colonnes, en sorte qu'une même
    # zone recevait une teinte au tableau d'échantillonnage, une autre au graphe et
    # une troisième à la matrice de l'étape 2 : le repérage d'une variable d'un
    # livrable à l'autre, que la couleur est précisément censée assurer, s'en
    # trouvait ruiné. L'ordre retenu est celui du classement de dominance de
    # l'étape 1, de sorte que la teinte porte une information et non un simple
    # numéro d'ordre de déclaration.
    ranking = matrix_dominance_ranking(presel_matrix)
    matrix_scores = matrix_dominance_scores(presel_matrix)
    zone_colors = zone_color_map(ranking, style)
    logger.info(
        "Classement des zones critiques par score de majoration croissant, établi "
        "sur la matrice de corrélation ordinale de l'étape 1 et l'intégralité des "
        f"observations : {ranking}. Ce classement fixe également l'attribution des "
        "teintes, commune à l'ensemble des livrables.")

    resources["corr_matrix_pdf"] = plot_ordinal_corr_matrix(
        presel_matrix, os.path.join(filedir, of["corr_matrix"]),
        fmt=".3f", threshold=graph_threshold, style=style, colors=zone_colors)
    resources["step1_corr_matrix_txt"] = write_ordinal_matrix_txt(
        presel_matrix, os.path.join(filedir, of["step1_corr_matrix_txt"]),
        header=("Matrice de correlation ordinale initiale (avant selection des variables) "
                f"sur {len(presel_vars)} zones critiques."),
        threshold=graph_threshold)
    logger.info(
        f"Matrice de corrélation ordinale initiale calculée sur {len(presel_vars)} "
        "zones critiques, avant sélection.")

    # 3. Graphe d'ordonnancement de l'étape 1 (V3) : déduit de la matrice ordinale
    #    initiale, il représente TOUTES les zones critiques, y compris celles
    #    dépourvues de relation d'ordre franchissant le seuil.
    step1_graph = build_dominance_graph(presel_matrix, presel_vars, graph_threshold)
    n_nodes_step1 = step1_graph["graph"].number_of_nodes()
    resources["step1_graph_pdf"] = plot_ordinal_graph(
        step1_graph, os.path.join(filedir, of["step1_graph"]), style=style,
        colors=zone_colors)
    resources["step1_graph_txt"] = write_ordinal_graph_txt(
        step1_graph, graph_threshold, os.path.join(filedir, of["step1_graph_txt"]))
    logger.info(
        f"Graphe étape 1 : {n_nodes_step1} nœud(s) affiché(s) sur "
        f"{len(critical_zones)} zone(s) critique(s), "
        f"{len(step1_graph['pairs_above'])} relation(s) au seuil {graph_threshold}.")
    if n_nodes_step1 != len(presel_vars):
        logger.warning(
            f"Graphe étape 1 : {len(presel_vars) - n_nodes_step1} nœud(s) absent(s) "
            "du graphe ; exhaustivité des zones critiques non vérifiée.")
    if step1_graph["isolated"]:
        logger.info(
            f"Zones sans relation d'ordre au seuil, affichées isolément : "
            f"{list(step1_graph['isolated'])}.")
    # Contrôle de cohérence entre la matrice et le graphe (V4) : le degré sortant de
    # chaque nœud doit égaler le nombre de cellules de sa ligne atteignant le seuil.
    audit_dominance_coherence(presel_matrix, step1_graph, graph_threshold, logger,
                              label="étape 1")

    # 4. Graphiques descriptifs (zones AVANT sélection GPO).
    # Le diagramme en boîtes porte deux panneaux (V7), l'un dans l'ordre de
    # déclaration et l'autre dans l'ordre du classement de dominance, en regard
    # terme à terme des deux tableaux du livrable 04. Leur confrontation donne à
    # voir la restructuration des colonnes qu'opère le classement.
    resources["boxplot_pdf"] = plot_boxplot(
        df_merged, critical_zones, os.path.join(filedir, of["boxplot"]), style=style,
        colors=zone_colors, ranking=ranking)
    resources["hist_pdf"] = plot_hist(
        df_merged, critical_zones, os.path.join(filedir, of["hist"]), rs, style=style,
        colors=zone_colors)

    # 5. Sélection GPO (One-vs-All).
    sel = gpo_selection(
        df_merged, critical_zones,
        config.gpo_selection["threshold_low"], config.gpo_selection["threshold_high"], rs)

    # 6. Résolution de la cible de régularisation.
    target_ties: Dict[str, Any] = {}
    if config.auto_target:
        cz = highest_dominance_column(sel["df_results"])
        best_p, tied = dominance_ties(sel["df_results"])
        logger.info(
            f"Cible de régularisation sélectionnée automatiquement : '{cz}' (fréquence p "
            f"maximale, {best_p:.2f} % ; les décisions de sélection portent sur les "
            "pourcentages arrondis à 0,01 %).")
        tie_rule = ("première zone selon l'ordre des colonnes du tableau de comptage "
                    "(nombre de dominations croissant, puis ordre de déclaration)")
        target_ties = {"tie_candidates": tied, "tie_rule": tie_rule}
        if len(tied) > 1:
            logger.warning(
                f"{len(tied)} zones sont ex æquo au maximum de la fréquence p "
                f"({best_p:.2f} %) : {tied}. Règle déterministe : {tie_rule} ; ici "
                f"'{cz}'. Si ce choix compte, fixez 'gpo_regularization.target_zone'.")
    else:
        cz = config.synthetic_target
        logger.info(f"Cible de régularisation (manuelle) : '{cz}'.")
    if cz not in df_merged.columns:
        try:
            df_merged[cz] = df_merged.eval(cz)
        except Exception as exc:  # noqa: BLE001 - expression invalide
            raise ConfigError(
                f"La cible '{cz}' n'est ni une colonne ni une expression calculable "
                f"({type(exc).__name__}).") from None
        check_finite_columns(df_merged, [cz], "cible calculée")
        logger.warning(f"Cible '{cz}' calculée à la volée (colonne synthétique).")
    target_reintroduced = cz in sel["columns_below"]
    target_ties["target_below_low_threshold"] = bool(target_reintroduced)
    if target_reintroduced:
        logger.info(
            f"La cible '{cz}' est sous le seuil bas de sélection : elle est écartée par "
            "la sélection statistique mais réintroduite comme cible de régularisation, "
            "donc retenue pour l'étape 2.")

    # Nuage de points cibles vs cz (requiert cz résolu).
    # La cible de régularisation peut être une variable synthétique absente du
    # classement de l'étape 1 : elle reçoit alors la teinte de la zone la plus
    # dominante, faute de quoi son libellé serait le seul de la planche à ne pas
    # être teinté en mode ``text``.
    if cz not in zone_colors and zone_colors:
        zone_colors[cz] = zone_colors[ranking[-1]]
    resources["scatter_pdf"] = plot_scatter(
        df_merged, critical_zones, cz, os.path.join(filedir, of["scatter"]), rs,
        style=style, colors=zone_colors)

    updated_zones = update_critical_zones(critical_zones, sel["columns_below"], cz)
    logger.info(f"Zones écartées (p faible) : {sel['columns_below']} ; zones retenues : {updated_zones}")

    resources["sort_txt"] = write_sort_txt(
        sel["df_results"], sel["dico_results"], updated_zones, os.path.join(filedir, of["sort_txt"]))

    # 7. Rapport de tri GPO (étape 1) : 01-Step1_Dominance_Hierarchy_Result.
    #    Les colonnes sont disposées de la zone la moins dominante à la plus
    #    dominante. Le tableau issu de la sélection porte déjà cet ordre, le
    #    classement GPO étant trié par nombre de dominations croissant ; la
    #    réorganisation explicite ci dessous le garantit indépendamment de toute
    #    évolution ultérieure de l'étape de sélection.
    # Deux critères de hiérarchisation coexistent, et il importe de ne pas les
    # confondre.
    #
    #   a. La fréquence ``p`` du comptage un contre tous gouverne la SÉLECTION :
    #      c'est elle qui décide des zones retenues pour l'étape 2, et c'est elle
    #      que rapporte le tableau 01, dont les cumuls N et P en dépendent.
    #      Elle exige qu'une zone majore simultanément toutes les autres, si bien
    #      qu'elle sature dès qu'une zone domine largement : les autres tombent
    #      alors à zéro et ne sont plus départagées.
    #
    #   b. Le score moyen de majoration, issu de la matrice de corrélation
    #      ordinale, gouverne la DISPOSITION des colonnes des tableaux 04 et 07.
    #      Fondé sur les comparaisons deux à deux, il départage toutes les zones
    #      et reflète fidèlement la hiérarchie que la matrice donne à lire.
    #
    # ``ranking`` et ``matrix_scores`` sont établis dès la construction de la
    # matrice initiale, en tête de cette fonction : le classement gouverne aussi
    # l'attribution des teintes, laquelle doit précéder la première figure.

    # 7 bis. Livrable d'échantillonnage des zones critiques de l'étape 1 (V6).
    #        Il porte les seules zones critiques, variables synthétiques
    #        comprises, avant toute sélection ; les variables de conception en
    #        sont exclues.
    #
    #        Deux tableaux y figurent : les zones dans leur ordre de déclaration,
    #        puis dans l'ordre de dominance croissante. Leur confrontation donne à
    #        voir le déplacement de chaque colonne, c'est à dire l'effet de la
    #        hiérarchisation. La teinte d'une colonne, attribuée d'après son rang,
    #        est commune aux deux tableaux et permet de suivre ce déplacement.
    #
    #        La composition intervient après le classement GPO, seul moment où les
    #        fréquences de dominance sont connues. La position du document dans le
    #        rapport consolidé, fixée par sa numérotation, demeure inchangée.
    step1_zones_tex = generate_step1_zones_sample_table(
        df_merged, critical_zones, config.latex_output["sample_size"], filedir,
        rs, of["step1_zones_sample"], max_columns_per_block, table_style,
        ranking=ranking, color_style_config=style, colors=zone_colors)
    resources["step1_zones_sample_tex"] = step1_zones_tex
    step1_zones_pdf = compile_latex(
        of["step1_zones_sample"], filedir, original_directory, logger=logger)
    if step1_zones_pdf:
        resources["step1_zones_sample_pdf"] = step1_zones_pdf
    n_blocks_step1_zones = len(
        split_column_bounds(len(critical_zones), max_columns_per_block))
    logger.info(
        f"Livrable d'échantillonnage des zones critiques de l'étape 1 : deux "
        f"tableaux de {len(critical_zones)} colonne(s), avant et après "
        f"hiérarchisation, présentés en {n_blocks_step1_zones} bloc(s) de "
        f"{max_columns_per_block} colonne(s) au plus. Les teintes suivent le rang "
        "de dominance et sont communes aux deux tableaux.")

    # Le tableau 01 conserve l'ordre du classement GPO, par fréquence ``p``
    # croissante : ses cumuls N et P sont établis dans cet ordre et perdraient
    # tout sens sous une autre disposition. C'est le tableau du critère de
    # sélection, non celui de la hiérarchie ordinale.
    result_tex = generate_result_table(
        sel["df_results"], sel["dico_results"], sel["n_sample"], filedir,
        of["step1_result"], max_columns_per_block, table_style)
    resources["step1_result_tex"] = result_tex
    result_pdf = compile_latex(
        of["step1_result"], filedir, original_directory, logger=logger)
    if result_pdf:
        resources["step1_result_pdf"] = result_pdf

    # 8. Régularisation κ.
    reg = config.gpo_regularization
    # L'enveloppe κ·cz porte sur les zones CONSERVÉES par la sélection (cible
    # comprise), non sur celles que le seuil bas a écartées.
    kappa_search = find_kappa_details(
        df_merged, updated_zones, cz, reg["threshold"], reg["kappa_min"],
        reg["kappa_max"], reg["kappa_step"])
    kappa, kappa_found = kappa_search["kappa"], kappa_search["found"]
    places = int(kappa_search["decimals"])
    if kappa_found:
        logger.info(
            f"Coefficient κ optimal = {kappa:.{places}f} (plus petite valeur de la grille "
            f"de {kappa_search['n_points']} point(s) qui satisfait le seuil "
            f"{reg['threshold']} pour toutes les zones conservées).")
    else:
        logger.warning(
            f"Aucun κ ne satisfait le seuil {reg['threshold']} dans [{reg['kappa_min']}, "
            f"{reg['kappa_max']}] ; κ={kappa:.{places}f} est le dernier point testé "
            f"({kappa_search['n_tested']} sur {kappa_search['n_points']}), où la plus "
            f"faible couverture est {kappa_search['min_coverage']:.4f}.")

    # 9. Matrice de corrélation ordinale de l'étape 2 (avec enveloppe) + graphe final.
    prop_matrix, variables, alpha_col = ordinal_correlation_matrix_with_envelope(
        df_merged, updated_zones, cz, kappa, places)
    # La colonne enveloppe régularisée hérite de la teinte de la zone dont elle
    # procède : elle n'est pas une zone critique et ne figure donc pas au
    # classement de l'étape 1, mais elle en est le multiple direct.
    if alpha_col not in zone_colors and cz in zone_colors:
        zone_colors[alpha_col] = zone_colors[cz]
    resources["ordinal_corr_matrix_pdf"] = plot_ordinal_corr_matrix(
        prop_matrix, os.path.join(filedir, of["ordinal_corr_matrix"]),
        fmt=".3f", threshold=graph_threshold, style=style, colors=zone_colors)

    graph_data = build_dominance_graph(prop_matrix, variables, graph_threshold)
    resources["ordinal_graph_txt"] = write_ordinal_graph_txt(
        graph_data, graph_threshold, os.path.join(filedir, of["ordinal_graph_txt"]))
    resources["ordinal_graph_pdf"] = plot_ordinal_graph(
        graph_data, os.path.join(filedir, of["ordinal_graph"]), style=style,
        colors=zone_colors)
    audit_dominance_coherence(prop_matrix, graph_data, graph_threshold, logger,
                              label="étape 2")

    # 10. Indice de parcimonie β + tableau CriticalZones (06).
    df_sample_full, max_beta, mean_beta = compute_parsimony(df_merged, updated_zones, cz, kappa)
    beta_domain = beta_domain_report(
        df_merged, updated_zones, cz, df_sample_full[r"$\beta$"])
    if not beta_domain["beta_defined"]:
        logger.warning(beta_domain["note"])
    if beta_domain["n_infinite_beta"]:
        logger.warning(
            f"β infini sur {beta_domain['n_infinite_beta']} observation(s) : toutes les "
            "zones conservées y valent zéro (rapport posé à +inf par convention) ; "
            "β maximal et moyen sont alors infinis, publiés null dans output.json.")
    # Les zones retenues sont disposées de la moins dominante à la plus
    # dominante, selon le score de majoration issu de la matrice ordinale de
    # l'**étape 2**, celle qui porte la colonne enveloppe. Le tableau et la
    # matrice de cette étape donnent ainsi à lire la même hiérarchie. Les deux
    # colonnes terminales, l'enveloppe κ·cz et l'indice β, restent en fin de
    # tableau.
    step2_ranking = matrix_dominance_ranking(prop_matrix)
    step2_scores = matrix_dominance_scores(prop_matrix)
    logger.info(
        "Classement des zones retenues par score de majoration croissant, établi "
        f"sur la matrice de corrélation ordinale de l'étape 2 : {step2_ranking}.")
    df_sample_full = order_columns_by_dominance(
        df_sample_full, step2_ranking, n_trailing=2)
    cz_tex = generate_criticalzones_table(
        df_sample_full, cz, kappa, max_beta, mean_beta,
        config.latex_output["sample_size"], filedir, rs, of["criticalzones"],
        max_columns_per_block, table_style, color_style_config=style,
        kappa_decimals=places, beta_defined=beta_domain["beta_defined"])
    resources["criticalzones_tex"] = cz_tex
    cz_pdf = compile_latex(
        of["criticalzones"], filedir, original_directory, logger=logger)
    if cz_pdf:
        resources["criticalzones_pdf"] = cz_pdf

    # 11. Notice analytique + journal d'exécution (pages liminaires du rapport).
    summary = {
        "oca_version": OCA_VERSION,
        "kappa_decimals": places,
        "beta_note": beta_domain["note"],
        "kappa": kappa, "kappa_found": kappa_found, "beta_mean": mean_beta, "beta_max": max_beta,
        "n_sample": sel["n_sample"], "target_zone": cz, "target_auto": config.auto_target,
        "dimensioning_zones": sel["columns_above"], "retained_zones": updated_zones,
        "dropped_zones": sel["columns_below"],
        "n_nodes_step1": n_nodes_step1,
        "n_relations_step1": len(step1_graph["pairs_above"]),
        # Scores globaux de dominance. La fréquence ``p``, issue du comptage un
        # contre tous, gouverne la sélection et l'ordre du tableau 01 ; le score
        # matriciel, issu des comparaisons deux à deux, gouverne la disposition
        # des colonnes des tableaux 04 (matrice de l'étape 1) et 07 (matrice de
        # l'étape 2). Les trois grandeurs figurent au tableau 2.1 de la notice,
        # de sorte que le lecteur retrouve le score exact ayant présidé à chaque
        # disposition sans avoir à le recalculer.
        "dominance_frequencies": dominance_frequencies(sel["df_results"]),
        "dominance_ranking": ranking,
        "matrix_dominance_scores": matrix_scores,
        "matrix_dominance_scores_step2": step2_scores,
        # Horodatage sans deux-points : la typographie française insère une espace
        # fine devant ce signe, ce qui disloque la lecture d'une heure.
        "generated_at": datetime.datetime.now().strftime("%Y/%m/%d à %Hh%M"),
        # Choix de configuration (input.json) rappelés dans « Configuration analysée ».
        "config": {
            "critical_zones": list(resolution.zones),
            "critical_zones_inferred": resolution.zones_inferred,
            "feature_cols_init": list(resolution.features),
            "feature_cols_inferred": resolution.features_inferred,
            "user_defined_vars_cols": config.user_defined_vars_cols if config.user_defined_vars else [],
            "target_zone_cfg": config.synthetic_target,
            "gpo_low": config.gpo_selection["threshold_low"],
            "gpo_high": config.gpo_selection["threshold_high"],
            "reg_threshold": config.gpo_regularization["threshold"],
            "kappa_min": config.gpo_regularization["kappa_min"],
            "kappa_max": config.gpo_regularization["kappa_max"],
            "kappa_step": config.gpo_regularization["kappa_step"],
            "graph_threshold": config.ordinal_graph["threshold"],
            "sample_size": config.latex_output["sample_size"],
            "data_file": os.path.join(config.merged_subdir, config.data_files).replace("\\", "/"),
        },
    }

    # Composition du rapport consolidé : source unique de vérité, exploitée à la
    # fois par la notice analytique et par la fusion des PDF.
    report_sections: List[Tuple[str, str]] = [
        (f"{of['step1_result']}.pdf",
         "tableau de comptage de dominance un contre tous et classement des zones "
         "critiques par fréquence de dominance croissante"),
        (of["corr_matrix"],
         "matrice de corrélation ordinale initiale, établie sur l'intégralité des "
         "zones critiques avant sélection des variables"),
        (of["step1_graph"],
         "graphe d'ordonnancement ordinal de l'étape 1, représentant l'ensemble "
         "des zones critiques et leurs relations d'ordre au seuil retenu"),
        (f"{of['step1_zones_sample']}.pdf",
         "échantillon tabulé des observations de l'étape 1, portant les zones "
         "critiques avant sélection des variables"),
        (of["ordinal_corr_matrix"],
         "matrice de corrélation ordinale de l'étape 2, augmentée de la colonne "
         "enveloppe régularisée"),
        (of["ordinal_graph"],
         "graphe de dominance ordinal de l'étape 2, attestant le rôle d'enveloppe "
         "majorante de la variable régularisée"),
        (f"{of['criticalzones']}.pdf",
         "échantillon tabulé des observations, coefficient multiplicateur et indice "
         "de parcimonie associé"),
        (of["boxplot"],
         "boîtes à moustaches comparatives des distributions des zones critiques, "
         "en deux panneaux : ordre de déclaration puis ordre du classement de "
         "dominance, en regard des deux tableaux du livrable d'échantillonnage"),
        (of["hist"],
         "histogrammes des distributions individuelles des zones critiques"),
        (of["scatter"],
         "nuages de points des zones critiques en fonction de la variable cible, "
         "assortis du coefficient de corrélation linéaire de Pearson"),
    ]

    logger.info("Génération du rapport consolidé.")
    # La liste des sections à fusionner provient des ressources VALIDÉES de la
    # compilation (un PDF figure dans ``resources`` seulement si sa production a
    # réussi), non de l'existence des fichiers : un PDF dont la compilation a échoué
    # n'est jamais fusionné, même si son retrait du disque a échoué.
    validated = {
        f"{of['step1_result']}.pdf": "step1_result_pdf",
        of["corr_matrix"]: "corr_matrix_pdf",
        of["step1_graph"]: "step1_graph_pdf",
        f"{of['step1_zones_sample']}.pdf": "step1_zones_sample_pdf",
        of["ordinal_corr_matrix"]: "ordinal_corr_matrix_pdf",
        of["ordinal_graph"]: "ordinal_graph_pdf",
        f"{of['criticalzones']}.pdf": "criticalzones_pdf",
        of["boxplot"]: "boxplot_pdf",
        of["hist"]: "hist_pdf",
        of["scatter"]: "scatter_pdf",
    }
    summary["missing_sections"] = [
        name for name, _ in report_sections
        if not resources.get(validated[name])
        or not os.path.isfile(os.path.join(filedir, name))]
    intro_tex = generate_report_intro(
        summary, list(mem.records), filedir, of["report_intro"], sections=report_sections)
    resources["report_intro_tex"] = intro_tex
    intro_pdf = compile_latex(
        of["report_intro"], filedir, original_directory, logger=logger)

    # 12. Fusion du rapport consolidé 00 : notice analytique puis sections 01 à 10.
    absent_before = list(summary["missing_sections"])
    status_name = ""
    merge_order = []
    if intro_pdf:
        merge_order.append(f"{of['report_intro']}.pdf")
    elif absent_before:
        # Rapport partiel sans notice compilée : une page de statut, produite sans
        # LaTeX, ouvre le consolidé pour que le PDF porte lui-même la liste des
        # sections absentes.
        status_name = "_statut_rapport_partiel.pdf"
        try:
            write_partial_status_pdf(
                os.path.join(filedir, status_name),
                [f"{of['report_intro']}.pdf"] + absent_before,
                "aucun compilateur LaTeX n'est utilisé ou disponible (voir le message "
                "d'installation du lancement)" if latex_support.find_latex_compiler()
                is None else "la compilation LaTeX de la notice a échoué (voir le journal)")
            merge_order.append(status_name)
        except Exception as exc:  # noqa: BLE001 - la page de statut est un confort
            status_name = ""
            logger.warning(f"Page de statut du rapport partiel non produite "
                           f"({type(exc).__name__}).")
    merge_order += [name for name, _ in report_sections if name not in absent_before]
    merged, absent = merge_pdfs_report(filedir, of["consolidated_report"], merge_order)
    if status_name:
        status_path = os.path.join(filedir, status_name)
        if os.path.isfile(status_path):
            try:
                os.remove(status_path)
            except OSError:
                pass
        absent = [name for name in absent if name != status_name]
    # Les sections écartées avant la fusion, et la notice si elle manque, sont absentes.
    absent = ([] if intro_pdf else [f"{of['report_intro']}.pdf"]) + absent_before + [
        name for name in absent if name not in absent_before]
    report_status = {
        "complete": bool(merged) and not absent,
        "partial": bool(merged) and bool(absent),
        "notice_included": bool(intro_pdf),
        "status_page_included": bool(status_name) and bool(merged),
        "missing_sections": list(absent),
    }
    if merged:
        resources["consolidated_report_pdf"] = merged
        if absent:
            logger.warning(
                f"Rapport consolidé PARTIEL : sections absentes {absent}. Les sources .tex "
                "des tableaux et de la notice sont conservées ; sans LaTeX seules les "
                "figures et les fichiers texte sont produits.")
    else:
        logger.warning("Fusion du rapport consolidé indisponible (pypdf absent ou aucun PDF).")

    # 13. Bloc de résultats du sous-dossier.
    results = to_native(build_output_json(
        config, sel["df_results"], sel["dico_results"], updated_zones, kappa, kappa_found,
        prop_matrix, graph_data, max_beta, mean_beta, sel["n_sample"], resources,
        target_zone=cz, target_auto=config.auto_target,
        initial_prop_matrix=presel_matrix, step1_graph_data=step1_graph,
        initial_critical_zones=critical_zones, log_records=list(mem.records),
        extras={
            "oca_version": OCA_VERSION,
            "kappa_decimals": places,
            "kappa_search": {key: kappa_search[key] for key in (
                "n_points", "n_tested", "last_tested", "min_coverage")},
            "beta_domain": beta_domain,
            "column_resolution": resolution.as_dict(),
            "target_ties": target_ties,
            "report": report_status,
            "configuration_notes": (list(config.normalization_notes)
                                    + list(config.normalization_warnings)),
        }))

    # En configuration multi-dossiers, chaque exécution publie son propre fichier
    # de résultats dans son dossier de sortie. En configuration mono-dossier, seule
    # la racine est écrite, à l'identique des versions antérieures.
    if config.is_multi_run:
        per_run_path = config.output_json_for_subdir()
        per_run = {
            "status": "complete",
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "config_file": _shown_path(config_path),
            "merged_subdir": config.merged_subdir,
            "results": results,
        }
        with open(per_run_path, "w", encoding="utf-8") as handle:
            json.dump(to_native(per_run), handle, indent=2, ensure_ascii=False,
                      allow_nan=False)
        logger.info(f"{per_run_path} écrit. Sous-dossier terminé.")
        print(f"[OCA] Sous-dossier '{config.merged_subdir}' traite : rapport "
              f"{of['consolidated_report']}"
              + (" (PARTIEL)" if report_status["partial"] else ""))
    else:
        logger.info("Pipeline terminé.")
    return results


def run(config_path: str = "input.json") -> Dict[str, Any]:
    """Orchestre le pipeline OCA sur l'ensemble des sous-dossiers déclarés.

    Le paramètre ``merged_subdir`` d'``input.json`` accepte une chaîne unique ou
    une liste de chaînes (V6). Chaque sous-dossier est traité séquentiellement et
    de façon strictement isolée, ses sorties étant écrites dans
    ``<output_dir>/<sous-dossier>/``. L'échec d'un sous-dossier n'interrompt pas
    les suivants : il est consigné et l'orchestrateur poursuit, de sorte qu'un
    jeu de données absent ou mal formé ne compromette pas le lot entier.

    Structure du fichier ``output.json`` produit à la racine :

        1. **un seul sous-dossier** : structure des versions antérieures,
           inchangée (clés ``generated_at``, ``config_file``, ``results``) ;
        2. **plusieurs sous-dossiers** : document agrégé portant en outre
           ``merged_subdirs`` et ``runs``, ce dernier associant à chaque
           sous-dossier son bloc ``results``. Chaque sous-dossier reçoit par
           ailleurs son propre ``output.json`` dans son dossier de sortie.

    Le fichier ``output_json`` est marqué périmé dès le début du lancement : un
    document minimal ``{"status": "running", "started_at": ...}`` remplace l'ancien
    résultat, lui-même remplacé en fin de calcul par le résultat (clé ``status`` :
    ``"complete"``, ou ``"partial"`` si des sous-dossiers d'un lot ont échoué) ou, en
    cas d'échec global, par ``{"status": "failed", "reason": ...}``. Un consommateur
    qui surveille ce seul fichier ne relit donc jamais un succès antérieur.

    Args:
        config_path: Chemin du fichier de configuration.

    Returns:
        Le dictionnaire écrit dans ``output.json``.

    Raises:
        ConfigError: si aucun sous-dossier n'a pu être traité.
    """
    config = load_config(config_path)
    # Le fichier de résultats ne doit jamais être le fichier de configuration lui-même :
    # il est réécrit dès le début du lancement.
    if (os.path.normcase(os.path.realpath(os.path.abspath(config.output_json)))
            == os.path.normcase(os.path.realpath(os.path.abspath(config_path)))):
        raise ConfigError(
            f"'output_json' ('{config.output_json}') désigne le fichier de configuration "
            "lui-même, qui serait écrasé : choisissez un autre nom de fichier de résultats.")
    started = datetime.datetime.now().isoformat(timespec="seconds")
    _write_output_json(config.output_json, {"status": "running", "started_at": started})
    try:
        return _run_configured(config, config_path)
    except BaseException as exc:  # noqa: BLE001 - l'échec est publié puis relancé
        reason = ("interrompu par l'utilisateur" if isinstance(exc, KeyboardInterrupt)
                  else str(exc) if isinstance(exc, ConfigError)
                  else f"{type(exc).__name__} : {exc}")
        try:
            _write_output_json(config.output_json, {
                "status": "failed", "reason": reason[:600], "started_at": started,
                "finished_at": datetime.datetime.now().isoformat(timespec="seconds")})
        except Exception:  # noqa: BLE001 - l'erreur d'origine prime
            pass
        raise


def _run_configured(config: "Any", config_path: str) -> Dict[str, Any]:
    """Corps de :func:`run` une fois la configuration chargée et le document marqué."""
    subdirs = config.merged_subdirs
    # Compilateur LaTeX : réglage lu une fois, absence signalée une seule fois par
    # le message de latex_support. Sans LaTeX tout le reste est produit.
    latex_support.configure(config.latex_compiler)
    latex_support.warn_if_missing("OCA")

    # Dépôt fraîchement cloné, dossier d'entrée vide : un message court renvoie à
    # --init et aux exemples, avant tout traitement et tout nettoyage.
    absent: List[ConfigError] = []
    for subdir in subdirs:
        try:
            check_data_sources(config.for_subdir(subdir))
        except ConfigError as exc:
            absent.append(exc)
    if len(absent) == len(subdirs):
        raise absent[0]

    if not config.is_multi_run:
        results = run_subdir(config.for_subdir(subdirs[0]), config_path)
        output = {
            "status": "complete",
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "config_file": _shown_path(config_path),
            "results": results,
        }
        _write_output_json(config.output_json, output)
        report = results.get("report", {})
        if report.get("partial"):
            print(f"[OCA] Pipeline termine : voir {config.output_json}. Rapport consolide "
                  f"{config.output_files['consolidated_report']} PARTIEL, sections absentes : "
                  f"{report.get('missing_sections', [])}")
        elif not report.get("complete", True):
            print(f"[OCA] Pipeline termine : voir {config.output_json}. Aucun rapport "
                  "consolide n'a pu etre produit (voir le journal).")
        else:
            print(f"[OCA] Pipeline termine : voir {config.output_json} et le rapport "
                  f"{config.output_files['consolidated_report']}")
        return output

    print(f"[OCA] Traitement par lot de {len(subdirs)} sous-dossier(s) : {subdirs}")
    runs: Dict[str, Any] = {}
    failures: Dict[str, str] = {}
    for position, subdir in enumerate(subdirs, start=1):
        print(f"[OCA] ({position}/{len(subdirs)}) sous-dossier '{subdir}'")
        try:
            runs[subdir] = run_subdir(config.for_subdir(subdir), config_path)
        except Exception as exc:  # noqa: BLE001 - un lot ne doit pas s'interrompre
            failures[subdir] = f"{type(exc).__name__}: {exc}"
            print(f"[OCA][ERREUR] sous-dossier '{subdir}' ignoré : {failures[subdir]}")

    if not runs:
        raise ConfigError(
            "Aucun sous-dossier n'a pu être traité. Détail des échecs : "
            + "; ".join(f"{name} -> {reason}" for name, reason in failures.items()))

    output = {
        "status": "partial" if failures else "complete",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "config_file": _shown_path(config_path),
        "merged_subdirs": subdirs,
        "runs": runs,
    }
    if failures:
        output["failed_subdirs"] = failures
    _write_output_json(config.output_json, output)
    print(f"[OCA] Lot termine : {len(runs)}/{len(subdirs)} sous-dossier(s) traite(s). "
          f"Voir {config.output_json}")
    return output


def _shown_path(path: str) -> str:
    """Chemin du fichier de configuration à publier, séparateurs normalisés.

    Un chemin relatif est publié tel que fourni ; un chemin absolu est réduit au nom du
    fichier, afin que ``output.json`` ne révèle aucun nom de dossier local.
    """
    text = str(path)
    if os.path.isabs(text):
        text = os.path.basename(text)
    return text.replace("\\", "/")


def _write_output_json(path: str, payload: Dict[str, Any]) -> None:
    """Écrit un document de résultats au format JSON.

    Args:
        path: Chemin du fichier à écrire.
        payload: Document à sérialiser.

    Raises:
        ConfigError: si le fichier ne peut pas être écrit (dossier non inscriptible).
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            # 'allow_nan=False' fait echouer l'ecriture plutot que de produire un
            # document que la norme JSON n'admet pas. Sans ce refus, une valeur non
            # finie introduite plus tard repasserait en silence.
            json.dump(to_native(payload), handle, indent=2, ensure_ascii=False,
                      allow_nan=False)
    except OSError as exc:
        raise ConfigError(
            f"'output_json' ('{path}') ne peut pas être écrit ({type(exc).__name__}) : "
            "vérifiez que le dossier existe, est inscriptible et que le fichier n'est "
            "pas ouvert dans un autre programme.") from None


#: Copie du fichier ``input.json`` du depot, utilisee par ``--init`` quand le fichier
#: n'est pas present a cote du module (installation par pip). Un test verifie
#: qu'elle reste identique au fichier du depot.
TEMPLATE_INPUT_JSON = r'''{
  "critical_zones": [
    "TRESCA_LIN_Z1_R", "TRESCA_LIN_Z2_R", "TRESCA_LIN_Z3_R",
    "TRESCA_LIN_Z1_L", "TRESCA_LIN_Z2_L", "TRESCA_LIN_Z3_L"
  ], // vide ou absent : déduit du classeur
  "feature_cols_init": ["e", "D1", "ALPHA", "H", "R1", "R2", "H3", "D3"],
  "input_dir": "./input",
  "output_dir": "./output",
  "output_json": "./output.json",
  "data_files": "dataSet_ALL.xlsx",
  "merged_subdir": [
    "01-RMR-P1_Fv0_Fext0", "02-RMR-P0_Fv1_Fext0", "03-RMR-P0_Fv0_Fext1"
  ],
  "user_defined_vars": false,
  "user_defined_vars_cols": [],
  "gpo_selection": {
    "threshold_low": 0.5,
    "threshold_high": 10.0
  },
  "gpo_regularization": {
    "target_zone": "none", // none = cible automatique
    "threshold": 0.99,
    "kappa_min": 1.0,
    "kappa_max": 3.0,
    "kappa_step": 0.01
  },
  "ordinal_graph": {
    "threshold": 0.99
  },
  "latex_output": {
    "sample_size": 30,
    "max_columns_per_block": 6,
    "table_font_size": "scriptsize", // tiny|scriptsize|footnotesize|small|normalsize
    "table_column_separation": "2pt",
    "compiler": "auto" // auto|pdflatex|tectonic|none
  },
  "sample_random_state": 6,
  "color_style": {
    "color_map_matrix": "RdBu_r", // vlag|RdBu_r|coolwarm|viridis|plasma|jet
    "color_map_critical_zones": "jet", // jet|vlag|tab10|tab20|Set1|Dark2
    "saturation": 1.0,
    "alpha": 1.0,
    "apply_to": "background" // text|background
  }
}
'''


def scaffold_working_directory(destination: str = ".") -> int:
    """Ecrit dans le repertoire courant de quoi lancer un premier calcul.

    Le modele d'entree de OCA est un fichier ``input.json`` et un dossier
    ``input/`` pris dans le repertoire courant. Apres une installation par pip,
    l'utilisateur n'a ni l'un ni l'autre : la commande refuserait, faute de
    configuration, et le gabarit serait a chercher dans l'arborescence
    d'installation. Cette fonction fait ce chainon.

    Rien n'est jamais ecrase : un fichier deja present est laisse tel quel et
    signale. La commande est donc sans danger, y compris dans un dossier de travail
    deja constitue.

    Args:
        destination: Dossier ou ecrire. Le repertoire courant par defaut.

    Returns:
        Le code de sortie : 0 si tout s'est bien passe.
    """
    root = Path(destination).resolve()
    # Le gabarit est le fichier input.json du depot ; apres une installation par
    # pip, ce fichier n'est pas livre avec le module : le gabarit integre, identique,
    # le remplace.
    template = Path(__file__).with_name("input.json")
    content = (template.read_bytes() if template.is_file()
               else TEMPLATE_INPUT_JSON.encode("utf-8"))

    target = root / "input.json"
    data_dir = root / "input"
    # Refus nets, en une ligne, plutôt qu'une trace : fichier à la place d'un dossier,
    # dossier à la place d'un fichier, emplacement non inscriptible.
    if root.exists() and not root.is_dir():
        raise ConfigError(f"'{root.name}' existe mais n'est pas un dossier : "
                          "placez-vous dans un dossier de travail.")
    if data_dir.exists() and not data_dir.is_dir():
        raise ConfigError(
            f"'{data_dir.name}' existe déjà mais c'est un fichier, pas un dossier : "
            "renommez-le ou supprimez-le, puis relancez --init.")
    if target.is_dir():
        raise ConfigError(
            f"'{target.name}' existe déjà mais c'est un dossier, pas un fichier : "
            "renommez-le, puis relancez --init.")
    try:
        root.mkdir(parents=True, exist_ok=True)
        if target.exists():
            print(f"[OCA] '{target.name}' existe deja : laisse tel quel.")
        else:
            target.write_bytes(content)
            print(f"[OCA] Gabarit de configuration ecrit : {target}")
        if data_dir.exists():
            print(f"[OCA] Le dossier '{data_dir.name}' existe deja.")
        else:
            data_dir.mkdir(parents=True)
            print(f"[OCA] Dossier de donnees cree : {data_dir}")
    except OSError as exc:
        raise ConfigError(
            f"Impossible d'écrire dans le dossier de travail ({type(exc).__name__}) : "
            "lancez --init dans un dossier inscriptible.") from None

    print(
        f"\n[OCA] Prochaine etape : deposez vos donnees dans '{data_dir.name}',"
        f"\n         adaptez dans '{target.name}' les noms de colonnes et de sous-dossiers"
        f"\n         (zones et variables peuvent rester vides : elles sont deduites du classeur),"
        f"\n         puis relancez la commande sans '--init'."
    )
    return 0


def main() -> None:
    """Analyse les arguments de ligne de commande et lance le pipeline.

    Une erreur de configuration ou de données est rapportée en une ou deux lignes
    (code de sortie 2), jamais par une trace Python. Une erreur imprévue l'est de
    même (code 1) ; l'option ``--debug`` en affiche alors la trace complète.
    """
    # Un nom contenant une lettre grecque ne doit pas faire échouer l'affichage
    # d'un message sur une console ou une redirection en cp1252.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - flux sans reconfigure
            pass
    parser = argparse.ArgumentParser(
        description=f"OCA {OCA_VERSION} : Ordinal Correlation Algorithm.")
    parser.add_argument("--config", default="input.json", help="Chemin du fichier input.json.")
    parser.add_argument(
        "--init", action="store_true",
        help="Ecrire un gabarit 'input.json' et un dossier 'input/' dans le "
             "repertoire courant, puis quitter. N'ecrase aucun fichier existant.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Afficher la trace Python complete en cas d'erreur imprevue.",
    )
    args = parser.parse_args()
    try:
        if args.init:
            raise SystemExit(scaffold_working_directory("."))
        run(args.config)
    except ConfigError as exc:
        # Le message de refus est exact et actionnable : il merite mieux qu'une
        # trace Python de plusieurs dizaines de lignes, ou il se perd.
        print(f"[OCA][ERREUR DE CONFIGURATION] {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except KeyboardInterrupt:
        print("[OCA] Interrompu.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:  # noqa: BLE001 - jamais de trace brute pour l'utilisateur
        if args.debug:
            raise
        print(f"[OCA][ERREUR] {type(exc).__name__} : {exc}\n"
              "Relancez avec --debug pour la trace complete, et signalez le cas.",
              file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
