"""Module de configuration et d'entrée du package OCA (version 9.0.0).

OCA (*Ordinal Correlation Algorithm*) réduit la dimension d'un ensemble de zones
critiques (contraintes) par ordre partiel de dominance et régularise une variable
synthétique enveloppe. Ce module centralise **tous** les paramètres utilisateur,
lit et valide ``input.json``, prépare les dossiers d'entrée/sortie, charge les
données (fusion multi-zones ou lecture d'un fichier fusionné), calcule les
variables synthétiques (par ex. ``De/E``) et les ajoute à la liste des zones
critiques à analyser.

Deux modes d'entrée sont pris en charge, sans codage en dur :
    * **Fusion** : si des fichiers bruts par-zone
      ``<input_dir>/dataSet_<zone>/dataSet_<zone>.xlsx`` existent pour toutes les
      zones déclarées, ils sont fusionnés (identifiant ``config_id``,
      déduplication, intersection des configurations communes) : logique d'origine
      du notebook.
    * **Fusionné** : sinon, un fichier déjà fusionné (contenant les variables et
      les zones) est lu directement. Le classeur peut être un ``.xlsx`` (première
      feuille) ou un ``.csv`` (séparateur virgule ou point-virgule détecté).

Colonnes du classeur : une première colonne sans en-tête, ou nommée ``index``,
``id``, ``config``, ``config_id`` ou ``n``, est un index et n'est pas analysée ;
toute autre première colonne est conservée comme les suivantes. Les listes
``critical_zones`` et ``feature_cols_init`` peuvent être vides : la liste
manquante se déduit alors des colonnes numériques du classeur.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import shutil
from dataclasses import dataclass, field, replace
from functools import reduce
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "OCAConfig",
    "ConfigError",
    "ColumnResolution",
    "OCA_VERSION",
    "load_config",
    "prepare_output_dirs",
    "load_and_merge_data",
    "load_data_with_resolution",
    "check_data_sources",
    "check_finite_columns",
    "setup_logging",
    "clean_output_main",
    "DEFAULT_CONFIG",
]

#: Version du programme, reprise dans ``output.json`` et dans le rapport.
OCA_VERSION = "9.0.0"

#: Noms d'en-tête (insensibles à la casse) d'une première colonne d'index.
INDEX_LIKE_HEADERS = ("index", "id", "config", "config_id", "n")

#: Formats de classeur lus.
READABLE_EXTENSIONS = (".xlsx", ".csv")


class ConfigError(ValueError):
    """Erreur levée lorsque la configuration ``input.json`` est invalide."""


class MemoryLogHandler(logging.Handler):
    """Handler de logs conservant les messages formatés en mémoire.

    Permet d'intégrer un journal d'exécution (avertissements, erreurs, événements
    clés) dans le rapport consolidé.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Mémorise le message formaté.

        Args:
            record: Enregistrement de log.
        """
        self.records.append(self.format(record))


def setup_logging(log_path: str) -> Tuple[logging.Logger, MemoryLogHandler]:
    """Configure le logger ``oca`` (fichier + capture mémoire).

    Args:
        log_path: Chemin du fichier de log.

    Returns:
        Tuple ``(logger, memory_handler)``. Les messages du ``memory_handler``
        alimentent le journal d'exécution du rapport consolidé.
    """
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logger = logging.getLogger("oca")
    logger.setLevel(logging.INFO)
    # Les gestionnaires de la configuration précédente sont fermés avant d'être
    # retirés. En traitement par lot (V6), le journal est réinitialisé à chaque
    # sous-dossier : un simple ``clear()`` abandonnerait le descripteur du fichier
    # de journal du sous-dossier précédent, ce que Python signale par un
    # ``ResourceWarning`` et qui laisserait le fichier ouvert jusqu'au ramasse
    # miettes.
    for handler in list(logger.handlers):
        try:
            handler.close()
        except Exception:  # pragma: no cover - fermeture défensive
            pass
    logger.handlers.clear()
    logger.propagate = False

    # Horodatage sans tiret : le journal est repris tel quel dans le rapport
    # consolidé, dont les conventions rédactionnelles proscrivent le tiret.
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y/%m/%d %H:%M:%S")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    memory_handler = MemoryLogHandler()
    memory_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(memory_handler)
    return logger, memory_handler


DEFAULT_CONFIG: Dict[str, Any] = {
    # Aucune etude par defaut. Ces cles designent l'objet meme de l'analyse :
    # les pourvoir d'une valeur ferait porter un calcul complet sur une etude
    # non declaree. Une liste vide (ou une cle absente) n'est pas une erreur : la
    # liste manquante se deduit des colonnes numeriques du classeur (voir
    # ``resolve_columns``). Si les deux sont vides, toutes les colonnes numeriques
    # sont des zones et il n'y a aucune variable de conception.
    "critical_zones": [],
    "feature_cols_init": [],
    "input_dir": "./input",
    "output_dir": "./output",
    # Emplacement du fichier de resultats centralise. Le rendre configurable permet
    # de faire coexister plusieurs instances d'application sans qu'elles se
    # recouvrent, chaque cas d'application disposant de son propre output.json.
    "output_json": "./output.json",
    "data_files": "dataSet_criticalZones.xlsx",
    "merged_subdir": "dataSet_criticalZones",
    # Aucune variable synthetique par defaut : une etude ne doit jamais faire
    # reference a une colonne (par exemple un rapport De/E) qu'elle n'a pas declaree.
    "user_defined_vars": False,
    "user_defined_vars_cols": [],
    "gpo_selection": {"threshold_low": 0.5, "threshold_high": 10.0},
    "gpo_regularization": {
        # "none" : cible choisie automatiquement parmi les zones.
        "target_zone": "none",
        "threshold": 0.99,
        "kappa_min": 1.0,
        "kappa_max": 3.0,
        "kappa_step": 0.01,
    },
    "ordinal_graph": {"threshold": 0.99},
    # ``max_columns_per_block`` (V6) : seuil de lisibilité horizontale des tableaux
    # LaTeX. Au delà, le tableau est découpé en sous-tableaux empilés verticalement
    # (mise en page de la Table 3 du papier de référence). La valeur 6 présente
    # d'un seul tenant les cas d'application à six zones critiques, tout en
    # découpant automatiquement les configurations plus larges.
    # ``table_font_size`` et ``table_column_separation`` (V6) : corps de caractères
    # et espacement inter colonnes des tableaux de données. Fixes et identiques
    # pour tous les tableaux et tous leurs blocs ; aucune mise à l'échelle du texte
    # n'est appliquée. Voir post_processor.SPLIT_BLOCK_FONT_SIZE pour la mesure
    # justifiant les valeurs par défaut.
    "latex_output": {
        "sample_size": 30,
        "max_columns_per_block": 6,
        "table_font_size": "scriptsize",
        "table_column_separation": "2pt",
        # Compilateur LaTeX : "auto" (pdflatex, sinon tectonic), "pdflatex",
        # "tectonic" ou "none" (sources .tex seules, aucun PDF LaTeX).
        "compiler": "auto",
    },
    # ``color_style`` (V7) : style de couleurs commun à l'ensemble des sorties.
    # Les deux usages de la couleur sont dissociés depuis la version 7, car ils
    # n'appellent pas la même famille de cartes.
    #   color_map_matrix          : fond des matrices de corrélation ordinale, qui
    #                               encode une proportion dans [0, 1] et appelle
    #                               donc une rampe continue, de préférence
    #                               divergente : "RdBu_r", "coolwarm", "jet",
    #                               "viridis", "plasma" ;
    #   color_map_critical_zones  : palette désignant les zones critiques, dont le
    #                               rang de dominance fixe l'attribution : "jet",
    #                               "tab10", "tab20", "Set1", "Dark2" ;
    #   saturation                : intensité de la teinte dans [0, 1] ; 1 la
    #                               restitue telle quelle, 0 la mêle à parts
    #                               égales du pôle neutre (blanc pour une aire,
    #                               noir pour un identifiant) ;
    #   alpha                     : opacité dans [0, 1], composée sur ce même pôle ;
    #   apply_to                  : destination de la couleur. "background" teinte
    #                               les aires, fonds de colonne, disques des
    #                               nœuds, boîtes et marques de données : c'est le
    #                               rendu des supports de présentation. "text"
    #                               teinte les seuls noms de variables, composés
    #                               en gras, et laisse les aires neutres : c'est
    #                               le rendu attendu d'une figure destinée à un
    #                               article scientifique.
    # La clé unique ``color_map`` des versions antérieures reste acceptée et
    # alimente alors les deux usages.
    "color_style": {
        "color_map_matrix": "RdBu_r",
        "color_map_critical_zones": "jet",
        "saturation": 1.0,
        "alpha": 1.0,
        "apply_to": "background",
    },
    "sample_random_state": 6,
    # Noms de sortie numérotés par ordre d'importance (surchargeables). Ce sont
    # ces clés qui figent les noms de fichiers ; il n'y a plus de champs
    # 'filename' ou 'consolidated_pdf_name' séparés.
    "output_files": {
        "consolidated_report": "00-OCA_Report.pdf",
        "step1_result": "01-Step1_Dominance_Hierarchy_Result",
        # V3 : 02- porte désormais la matrice de corrélation ORDINALE initiale,
        # établie sur l'intégralité des zones critiques avant toute sélection
        # (la matrice de corrélation linéaire de Pearson a été retirée).
        "corr_matrix": "02-OCA_CorrMat_Targets.pdf",
        "step1_graph": "03-Step1_OrdinalGraph_Targets.pdf",
        # V6 : 04- est un nouveau livrable tabulaire, inséré immédiatement après le
        # graphe de dominance brute de l'étape 1. Il présente un échantillon des
        # données portant les seules ZONES CRITIQUES, variables synthétiques
        # comprises, avant toute sélection ; les variables de conception de
        # ``feature_cols_init`` en sont exclues. Les ressources suivantes sont
        # décalées d'un rang par rapport à la version 5.
        "step1_zones_sample": "04-Step1_Critical_Zones_Sample_Result",
        "ordinal_corr_matrix": "05-OCA_OrdinalCorrMat_Targets.pdf",
        "ordinal_graph": "06-OCA_OrdinalGraph_Targets.pdf",
        "criticalzones": "07-CriticalZones_Targets",
        "boxplot": "08-OCA_Boxplot_Targets.pdf",
        "hist": "09-OCA_Hist_Targets.pdf",
        "scatter": "10-OCA_Scatter_Targets.pdf",
        "report_intro": "11-OCA_Report_intro",
        "sort_txt": "OCA_Sort_Targets.txt",
        "step1_corr_matrix_txt": "Step1_OrdinalCorrMat_Targets.txt",
        "step1_graph_txt": "Step1_OrdinalGraph_Targets.txt",
        "ordinal_graph_txt": "OCA_OrdinalGraph_Targets.txt",
        "log": "OCA_run.log",
    },
}


@dataclass(frozen=True)
class OCAConfig:
    """Configuration validée et immuable du pipeline OCA.

    Attributes:
        critical_zones: Zones critiques (cibles de contraintes) à analyser.
        feature_cols_init: Variables de conception de base (features).
        input_dir: Racine des données d'entrée.
        output_dir: Racine des résultats de sortie.
        output_json: Chemin du fichier de résultats centralisé.
        data_files: Nom du fichier fusionné (features + cibles).
        merged_subdir: Sous-dossier contenant le fichier fusionné.
        user_defined_vars: Active le calcul de variables synthétiques.
        user_defined_vars_cols: Expressions pandas des variables synthétiques
            (ex. ``"De/E"``) ; la colonne créée porte le nom de l'expression.
        gpo_selection: Seuils bas/haut (%) de sélection GPO.
        gpo_regularization: Paramètres de la régularisation κ (cible, seuil,
            plage et pas de κ).
        ordinal_graph: Seuil de construction du graphe de dominance.
        latex_output: Taille d'échantillon (``sample_size``) du tableau LaTeX.
        sample_random_state: Graine des échantillonnages reproductibles.
        raw: Dictionnaire brut fusionné (défauts + utilisateur), pour audit.
        json_repairs: Écarts d'écriture neutralisés lors de la lecture du fichier
            de configuration (virgule orpheline, commentaire). Vide lorsque le
            fichier est conforme à la norme JSON.
        option_keys: Chemins des clés d'auto-documentation suffixées
            ``__options`` retirées à la lecture (V7). Purement informatif : ces
            clés énumèrent les valeurs admises d'un paramètre à l'intention du
            lecteur du fichier et n'ont aucune incidence sur le calcul.
        legacy_notes: Mentions relatives aux écritures des versions antérieures
            reconnues et transposées (V7), la carte de couleurs unique notamment.
        comment_notes: Dénombrement des commentaires retirés du fichier de
            configuration (V7). Purement informatif : l'écriture commentée est
            admise, elle ne constitue pas un écart et ne figure donc pas parmi les
            réparations.
    """

    critical_zones: List[str]
    feature_cols_init: List[str]
    input_dir: str
    output_dir: str
    output_json: str
    data_files: str
    merged_subdir: str
    user_defined_vars: bool
    user_defined_vars_cols: List[str]
    gpo_selection: Dict[str, float]
    gpo_regularization: Dict[str, Any]
    ordinal_graph: Dict[str, float]
    latex_output: Dict[str, Any]
    color_style: Dict[str, Any]
    sample_random_state: int
    output_files: Dict[str, str]
    # V6 : liste complète des sous-dossiers d'entrée à traiter. ``merged_subdir``
    # désigne celui de l'exécution courante ; ``merged_subdirs`` les énumère tous.
    # En configuration mono-dossier, la liste ne contient que ``merged_subdir``.
    merged_subdirs: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)
    json_repairs: List[str] = field(default_factory=list, repr=False)
    # V7 : chemins des clés d'auto-documentation retirées à la lecture. Purement
    # informatif ; aucune n'a d'incidence sur le calcul.
    option_keys: List[str] = field(default_factory=list, repr=False)
    # V7 : mentions relatives aux écritures des versions antérieures reconnues et
    # transposées, la carte de couleurs unique notamment.
    legacy_notes: List[str] = field(default_factory=list, repr=False)
    # V7 : dénombrement des commentaires retirés du fichier de configuration.
    # Purement informatif : l'écriture commentée est admise et n'est pas un écart.
    comment_notes: List[str] = field(default_factory=list, repr=False)
    # V9 : champs vides ou nuls remplacés par leur défaut annoncé, et clés
    # inconnues ignorées, consignés au journal à chaque exécution.
    normalization_notes: List[str] = field(default_factory=list, repr=False)
    normalization_warnings: List[str] = field(default_factory=list, repr=False)

    @property
    def synthetic_target(self) -> str:
        """Valeur brute du paramètre ``target_zone`` (peut être ``"none"``)."""
        return str(self.gpo_regularization.get("target_zone", "none"))

    @property
    def latex_compiler(self) -> str:
        """Compilateur LaTeX demandé (``auto``, ``pdflatex``, ``tectonic``, ``none``)."""
        return str(self.latex_output.get("compiler", "auto"))

    @property
    def auto_target(self) -> bool:
        """Indique si la cible doit être sélectionnée automatiquement (``"none"``).

        La sentinelle est comparée à la casse près et **seulement** si aucune zone
        critique déclarée ne porte ce nom. Une base légitime peut en effet nommer
        une colonne ``none`` : sans cette réserve, le choix explicite de
        l'utilisateur serait silencieusement remplacé par une résolution
        automatique, ce qui a été mesuré.
        """
        raw = self.synthetic_target.strip()
        if raw in [str(zone) for zone in (self.critical_zones or [])]:
            # Le nom désigne une colonne déclarée : il ne peut pas être la sentinelle.
            return False
        return raw.lower() == "none"

    def merged_file_path(self) -> str:
        """Chemin du fichier fusionné dans le dossier d'entrée.

        Returns:
            ``<input_dir>/<merged_subdir>/<data_files>``.
        """
        return os.path.join(self.input_dir, self.merged_subdir, self.data_files)

    def output_dir_main(self) -> str:
        """Dossier de sortie principal ``<output_dir>/<merged_subdir>``."""
        return os.path.join(self.output_dir, self.merged_subdir)

    @property
    def is_multi_run(self) -> bool:
        """Indique si plusieurs sous-dossiers d'entrée doivent être traités (V6)."""
        return len(self.merged_subdirs) > 1

    def for_subdir(self, subdir: str) -> "OCAConfig":
        """Retourne une configuration dédiée à un sous-dossier d'entrée (V6).

        Chaque sous-dossier déclaré dans ``merged_subdir`` est traité de façon
        strictement isolée : la configuration retournée ne diffère de l'originale
        que par le sous-dossier actif, ce qui redirige à la fois le chemin du
        fichier de données (:meth:`merged_file_path`) et le dossier de sortie
        (:meth:`output_dir_main`). Aucun état n'est partagé entre deux exécutions.

        Args:
            subdir: Sous-dossier d'entrée à activer.

        Returns:
            OCAConfig: copie immuable dont ``merged_subdir`` vaut ``subdir``.

        Raises:
            ConfigError: si ``subdir`` n'est pas déclaré dans la configuration.
        """
        if subdir not in self.merged_subdirs:
            raise ConfigError(
                f"Sous-dossier '{subdir}' absent de 'merged_subdir' : "
                f"{self.merged_subdirs}.")
        return replace(self, merged_subdir=subdir)

    def output_json_for_subdir(self) -> str:
        """Chemin du ``output.json`` propre au sous-dossier courant (V6).

        En configuration multi-dossiers, chaque exécution publie son propre
        fichier de résultats dans son dossier de sortie, et la racine reçoit un
        document agrégé. En configuration mono-dossier, seule la racine est
        écrite, à l'identique des versions antérieures.

        Returns:
            ``<output_dir>/<merged_subdir>/output.json``.
        """
        return os.path.join(self.output_dir_main(), "output.json")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Fusionne récursivement ``override`` dans une copie de ``base``.

    Args:
        base: Dictionnaire de base (défauts).
        override: Dictionnaire prioritaire (utilisateur).

    Returns:
        Un nouveau dictionnaire fusionné.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _normalize_merged_subdirs(value: Any) -> List[str]:
    """Normalise le paramètre ``merged_subdir`` en liste de sous-dossiers (V6).

    Le paramètre accepte deux écritures, afin de rester rétrocompatible avec les
    configurations des versions antérieures :

        1. une chaîne unique, par exemple ``"01-RMR-P1_Fv0_Fext0"`` ;
        2. une liste de chaînes, par exemple
           ``["01-RMR-P1_Fv0_Fext0", "01-RMR-P0_Fv1_Fext0"]``.

    Args:
        value: Valeur brute lue dans ``input.json``.

    Returns:
        Liste des sous-dossiers, dans l'ordre de déclaration, sans entrée vide.
        Liste vide si la valeur est d'un type non pris en charge.
    """
    if isinstance(value, str):
        name = value.strip()
        return [name] if name else []
    if isinstance(value, (list, tuple)):
        names = []
        for item in value:
            if not isinstance(item, str):
                return []
            name = item.strip()
            if name:
                names.append(name)
        return names
    return []


# Suffixe des clés d'auto-documentation (V7). Une clé ainsi nommée n'est pas un
# paramètre : elle énumère, à l'intention du lecteur du fichier de configuration,
# les valeurs qu'un paramètre admet. Le suffixe est double tiret bas afin qu'aucun
# nom de paramètre ne puisse le porter par accident.
OPTION_METADATA_SUFFIX = "__options"


def strip_option_metadata(node: Any, path: str = "") -> Tuple[Any, List[str]]:
    """Retire d'une configuration ses clés d'auto-documentation.

    Le fichier ``input.json`` est destiné à être modifié à la main. Pour éviter à
    l'utilisateur d'avoir à consulter la documentation externe chaque fois qu'il
    veut changer un paramètre énuméré, il peut y déclarer, à côté de chaque
    paramètre, une clé homonyme suffixée de :data:`OPTION_METADATA_SUFFIX`
    énumérant les valeurs admises.

    .. code-block:: json

        "apply_to": "text",
        "apply_to__options": ["text", "background"]

    Ces clés relèvent de la seule documentation. Elles sont donc retirées avant
    toute validation et avant que la configuration ne parvienne aux modules de
    calcul et de post-traitement : aucune ne saurait être confondue avec un
    paramètre, ni provoquer une erreur de type, ni figurer dans les
    hyperparamètres consignés au fichier de résultats.

    Le retrait est récursif et s'applique à toute profondeur, aux objets comme aux
    éléments de tableau, de sorte qu'une clé d'aide déclarée au sein d'un bloc
    imbriqué soit traitée comme une clé déclarée à la racine.

    Args:
        node: Fragment de configuration, objet, tableau ou valeur scalaire.
        path: Chemin de la clé courante, employé dans le compte rendu.

    Returns:
        Couple formé du fragment épuré et de la liste des clés retirées, chacune
        désignée par son chemin complet.
    """
    removed: List[str] = []
    if isinstance(node, dict):
        cleaned: Dict[str, Any] = {}
        for key, value in node.items():
            location = f"{path}.{key}" if path else str(key)
            # La comparaison est insensible à la casse. Une clé écrite
            # ``__OPTIONS`` relève manifestement de la même intention ; la laisser
            # passer la ferait tenir pour un paramètre et la reporterait jusqu'aux
            # hyperparamètres du fichier de résultats.
            if str(key).lower().endswith(OPTION_METADATA_SUFFIX):
                removed.append(location)
                # Le sous-arbre écarté est néanmoins parcouru, aux seules fins du
                # dénombrement : une clé d'aide imbriquée sous une autre doit être
                # comptée, faute de quoi le journal en annoncerait moins que le
                # fichier n'en porte.
                _, nested = strip_option_metadata(value, location)
                removed.extend(nested)
                continue
            child, child_removed = strip_option_metadata(value, location)
            cleaned[key] = child
            removed.extend(child_removed)
        return cleaned, removed
    if isinstance(node, list):
        items = []
        for index, value in enumerate(node):
            child, child_removed = strip_option_metadata(value, f"{path}[{index}]")
            items.append(child)
            removed.extend(child_removed)
        return items, removed
    return node, removed


def _validate_color_map_names(style: Dict[str, Any]) -> None:
    """Valide les noms de cartes de couleurs déclarés (V7).

    Chaque nom doit être une chaîne désignant une carte connue de matplotlib. Le
    module de post-traitement conserve un repli défensif, la fonction ne présumant
    pas de son appelant, mais une valeur venue du fichier de configuration relève
    d'une autre logique : elle est écrite à la main, son domaine est énuméré dans
    le fichier même par une clé d'auto-documentation, et une faute de frappe
    silencieusement rattrapée serait durablement invisible.

    Args:
        style: Bloc ``color_style``, clés d'aide déjà retirées.

    Raises:
        ConfigError: si un nom n'est pas une chaîne, ou ne désigne aucune carte.
    """
    try:
        import matplotlib
        # Seaborn enregistre ses propres cartes auprès de matplotlib au moment de
        # son importation : "vlag", "icefire", "rocket", "mako", "flare", "crest".
        # Aucune n'est connue de matplotlib seul. Or la validation peut s'exécuter
        # avant que le module de post-traitement, qui importe seaborn, ne l'ait été,
        # ainsi lorsque le harnais charge la configuration avant ses contrôles. Une
        # carte parfaitement légitime se voyait alors rejetée selon le seul ordre
        # des importations. L'importation est donc forcée ici, à seule fin
        # d'enregistrement, et son échec est sans conséquence.
        try:
            import seaborn  # noqa: F401 - importé pour enregistrer ses cartes
        except Exception:
            pass
        known = set(matplotlib.colormaps)
    except Exception:
        # L'absence de matplotlib ne saurait faire échouer la validation d'une
        # configuration : le repli du module de post-traitement prend le relais.
        return
    for key in ("color_map_matrix", "color_map_critical_zones", "color_map"):
        if key not in style:
            continue
        value = style[key]
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(
                f"'color_style.{key}' doit être une chaîne non vide désignant une "
                f"carte de couleurs matplotlib ; valeur reçue : {value!r}.")
        if value not in known:
            raise ConfigError(
                f"'color_style.{key}' vaut '{value}', qui ne désigne aucune carte "
                "de couleurs matplotlib. Exemples de valeurs admises : "
                "'RdBu_r', 'coolwarm', 'jet', 'viridis', 'plasma' pour le fond des "
                "matrices ; 'jet', 'tab10', 'tab20', 'Set1', 'Dark2' pour les "
                "zones critiques.")


def _expand_legacy_color_map(user_cfg: Dict[str, Any]) -> List[str]:
    """Développe la carte de couleurs unique des versions antérieures (V7).

    Jusqu'à la version 6, une clé unique ``color_style.color_map`` servait à la
    fois le fond des matrices et la palette des zones critiques. La version 7 les
    dissocie en ``color_map_matrix`` et ``color_map_critical_zones``.

    Le développement doit intervenir **avant** la fusion avec les valeurs par
    défaut : celles ci portant désormais les deux clés nouvelles, la fusion les
    introduirait dans la configuration de l'utilisateur et son choix historique
    resterait sans effet. Une configuration écrite pour la version 6 produirait
    alors un rendu différent sans que rien ne l'en avertisse.

    Le développement n'a lieu que si l'utilisateur n'a déclaré aucune des deux
    clés nouvelles : les avoir écrites signifie qu'il connaît le schéma de la
    version 7, et sa déclaration prévaut alors sur l'ancienne clé.

    Args:
        user_cfg: Configuration de l'utilisateur, modifiée sur place.

    Returns:
        Liste des mentions à consigner au journal, vide si rien n'a été développé.
    """
    style = user_cfg.get("color_style")
    if not isinstance(style, dict):
        return []
    legacy = style.get("color_map")
    if not legacy:
        return []
    declared = [key for key in ("color_map_matrix", "color_map_critical_zones")
                if key in style]
    if declared:
        missing = [key for key in ("color_map_matrix", "color_map_critical_zones")
                   if key not in style]
        note = (f"'color_style.color_map' vaut '{legacy}' et coexiste avec "
                f"{', '.join(declared)} du schéma dissocié de la version 7 ; "
                "l'ancienne clé est ignorée")
        if missing:
            note += (f", et {', '.join(missing)} prend la valeur par défaut de la "
                     "version 7 plutôt que la valeur héritée")
        return [note]
    style["color_map_matrix"] = legacy
    style["color_map_critical_zones"] = legacy
    return [
        f"'color_style.color_map' vaut '{legacy}', écriture des versions "
        "antérieures à la version 7 : la carte est appliquée au fond des matrices "
        "comme à la palette des zones critiques. Déclarer 'color_map_matrix' et "
        "'color_map_critical_zones' permet de les dissocier"
    ]


_MISSING = object()

#: Corps de caractères admis pour les tableaux LaTeX.
_FONT_SIZES = ("tiny", "scriptsize", "footnotesize", "small", "normalsize")

#: Valeurs admises pour ``latex_output.compiler``.
_COMPILERS = ("auto", "pdflatex", "tectonic", "none")

#: Longueur LaTeX simple : nombre suivi d'une unité.
_LENGTH_PATTERN = re.compile(
    r"^\s*([0-9]+(?:\.[0-9]+)?|\.[0-9]+)\s*(pt|mm|cm|in|em|ex|bp|pc|dd|cc|sp)\s*$")


def _show(value: Any) -> str:
    """Représentation courte d'une valeur reçue, avec son type, pour un message."""
    text = repr(value)
    if len(text) > 40:
        text = text[:37] + "..."
    kinds = {list: "liste", dict: "objet", str: "chaîne", bool: "booléen",
             int: "entier", float: "nombre", type(None): "null"}
    return f"{text} ({kinds.get(type(value), type(value).__name__)})"


def _fmt_bound(value: float) -> str:
    """Borne d'intervalle lisible (entier sans décimale, sinon format court)."""
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:g}"


def _is_blank(value: Any) -> bool:
    """Vrai pour ``null`` et pour une chaîne vide ou blanche."""
    return value is None or (isinstance(value, str) and not value.strip())


def _default_note(path: str, default: Any, notes: List[str]) -> None:
    """Consigne le remplacement d'un champ vide ou nul par son défaut."""
    notes.append(f"'{path}' est vide ou nul : valeur par défaut {default!r} appliquée.")


def _as_text(value: Any, path: str, notes: List[str], default: Any,
             allowed: Optional[Tuple[str, ...]] = None) -> Any:
    """Valide une chaîne de configuration ; renvoie ``_MISSING`` si vide ou nulle."""
    if _is_blank(value):
        _default_note(path, default, notes)
        return _MISSING
    if not isinstance(value, str):
        raise ConfigError(
            f"'{path}' doit être une chaîne de caractères ; valeur reçue : {_show(value)}.")
    text = value.strip()
    if "\x00" in text:
        raise ConfigError(f"'{path}' contient un caractère nul.")
    if allowed is not None:
        if text.lower() not in allowed:
            raise ConfigError(
                f"'{path}' vaut '{text}' ; valeurs admises : {list(allowed)}.")
        return text.lower()
    return text


def _as_number(value: Any, path: str, notes: List[str], default: Any,
               minimum: Optional[float] = None, maximum: Optional[float] = None,
               min_open: bool = False, max_open: bool = False,
               integer: bool = False) -> Any:
    """Valide un nombre fini (entier si demandé) ; ``_MISSING`` si vide ou nul."""
    if _is_blank(value):
        _default_note(path, default, notes)
        return _MISSING
    kind = "un entier" if integer else "un nombre"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"'{path}' doit être {kind} (sans guillemets) ; valeur reçue : {_show(value)}.")
    try:
        finite = math.isfinite(float(value))
    except OverflowError:
        finite = False
    if not finite:
        try:
            shown = str(value) if len(str(value)) <= 40 else str(value)[:37] + "..."
        except ValueError:  # entier de plus de 4300 chiffres
            shown = "un entier démesuré"
        raise ConfigError(
            f"'{path}' doit être {kind} fini et de grandeur raisonnable ; valeur reçue : "
            f"{shown}.")
    if integer:
        if isinstance(value, float):
            if not value.is_integer():
                raise ConfigError(
                    f"'{path}' doit être un entier ; valeur reçue : {value}.")
            value = int(value)
    else:
        value = float(value)
    low_ok = minimum is None or (value > minimum if min_open else value >= minimum)
    high_ok = maximum is None or (value < maximum if max_open else value <= maximum)
    if not (low_ok and high_ok):
        left = "]" if min_open else "["
        right = "[" if max_open else "]"
        low_text = "-inf" if minimum is None else _fmt_bound(minimum)
        high_text = "+inf" if maximum is None else _fmt_bound(maximum)
        raise ConfigError(
            f"'{path}' vaut {value} ; la valeur doit appartenir à {left}{low_text}, "
            f"{high_text}{right}.")
    return value


def _as_bool(value: Any, path: str, notes: List[str], default: Any) -> Any:
    """Valide un booléen JSON (``true``/``false``) ; ``_MISSING`` si vide ou nul."""
    if _is_blank(value):
        _default_note(path, default, notes)
        return _MISSING
    if not isinstance(value, bool):
        raise ConfigError(
            f"'{path}' doit valoir true ou false (sans guillemets) ; valeur reçue : "
            f"{_show(value)}.")
    return value


def _as_length(value: Any, path: str, notes: List[str], default: Any) -> Any:
    """Valide une longueur LaTeX simple (par exemple ``2pt``)."""
    text = _as_text(value, path, notes, default)
    if text is _MISSING:
        return _MISSING
    match = _LENGTH_PATTERN.match(text)
    if not match:
        raise ConfigError(
            f"'{path}' vaut '{text}' ; une longueur LaTeX simple est attendue, par "
            "exemple '2pt' ou '1.5mm'.")
    return f"{match.group(1)}{match.group(2)}"


def _as_name_list(value: Any, path: str, notes: List[str], empty_meaning: str) -> List[str]:
    """Valide une liste de noms de colonnes : chaînes non vides, sans doublon.

    Une valeur nulle, une chaîne vide ou une liste vide donnent une liste vide, ce
    qui n'est pas une erreur : ``empty_meaning`` dit ce que le programme en fait.
    """
    if _is_blank(value):
        notes.append(f"'{path}' est vide ou nul : {empty_meaning}.")
        return []
    if not isinstance(value, list):
        raise ConfigError(
            f"'{path}' doit être une liste de noms entre guillemets, par exemple "
            f"[\"A\", \"B\"] ; valeur reçue : {_show(value)}.")
    names: List[str] = []
    for position, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(
                f"'{path}[{position}]' doit être un nom non vide entre guillemets ; "
                f"valeur reçue : {_show(item)}.")
        names.append(item.strip())
    if not names:
        notes.append(f"'{path}' est vide : {empty_meaning}.")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ConfigError(f"'{path}' contient des doublons : {duplicates}.")
    return names


def _as_block(user_cfg: Dict[str, Any], key: str, notes: List[str]) -> Dict[str, Any]:
    """Extrait un bloc objet de la configuration ; vide, nul ou absent donne ``{}``."""
    if key not in user_cfg:
        return {}
    value = user_cfg[key]
    if value is None or (isinstance(value, str) and not value.strip()):
        notes.append(f"'{key}' est vide ou nul : tous les défauts du bloc sont appliqués.")
        return {}
    if not isinstance(value, dict):
        raise ConfigError(
            f"'{key}' doit être un objet JSON entre accolades ; valeur reçue : "
            f"{_show(value)}.")
    return value


#: Sous-clés de chaque bloc : type, bornes ou valeurs admises.
_BLOCK_SPECS: Dict[str, Dict[str, Tuple[str, Dict[str, Any]]]] = {
    "gpo_selection": {
        "threshold_low": ("number", {"minimum": 0.0, "maximum": 100.0}),
        "threshold_high": ("number", {"minimum": 0.0, "maximum": 100.0}),
    },
    "gpo_regularization": {
        "target_zone": ("text", {}),
        "threshold": ("number", {"minimum": 0.0, "maximum": 1.0, "min_open": True}),
        "kappa_min": ("number", {"minimum": 0.0, "min_open": True}),
        "kappa_max": ("number", {"minimum": 0.0, "min_open": True}),
        "kappa_step": ("number", {"minimum": 0.0, "min_open": True}),
    },
    "ordinal_graph": {
        "threshold": ("number", {"minimum": 0.0, "maximum": 1.0, "min_open": True}),
    },
    "latex_output": {
        "sample_size": ("integer", {"minimum": 1}),
        "max_columns_per_block": ("integer", {"minimum": 1}),
        "table_font_size": ("text", {"allowed": _FONT_SIZES}),
        "table_column_separation": ("length", {}),
        "compiler": ("text", {"allowed": _COMPILERS}),
    },
    "color_style": {
        "color_map_matrix": ("text", {}),
        "color_map_critical_zones": ("text", {}),
        "color_map": ("text", {}),
        "saturation": ("number", {"minimum": 0.0, "maximum": 1.0}),
        "alpha": ("number", {"minimum": 0.0, "maximum": 1.0}),
        "apply_to": ("text", {"allowed": ("background", "text")}),
    },
}


def _normalize_user_config(
    user_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Valide les types et les valeurs de la configuration utilisateur.

    Chaque champ est contrôlé avant toute conversion : un champ optionnel nul ou
    vide reprend son défaut annoncé (avec une note), un mauvais type ou une valeur
    hors domaine lève une ``ConfigError`` qui nomme la clé par son chemin complet.
    Aucune exception ``TypeError``, ``ValueError`` ni ``AttributeError`` ne sort de
    cette fonction.

    Args:
        user_cfg: Dictionnaire lu dans ``input.json``, clés d'aide déjà retirées.

    Returns:
        Triplet ``(surcharges, notes, avertissements)`` : les surcharges ne portent
        que les valeurs valides et non vides, à fusionner aux défauts.

    Raises:
        ConfigError: au premier champ invalide.
    """
    notes: List[str] = []
    warnings: List[str] = []
    out: Dict[str, Any] = {}

    for key in user_cfg:
        if key not in DEFAULT_CONFIG:
            warnings.append(
                f"Clé inconnue '{key}' ignorée (faute de frappe ? voir input.json pour "
                "les clés admises).")

    def take(container: Dict[str, Any], key: str, value: Any) -> None:
        if value is not _MISSING:
            container[key] = value

    # Listes de colonnes.
    if "critical_zones" in user_cfg:
        out["critical_zones"] = _as_name_list(
            user_cfg["critical_zones"], "critical_zones", notes,
            "les zones sont déduites du classeur (colonnes numériques qui ne sont pas des "
            "variables)")
    if "feature_cols_init" in user_cfg:
        out["feature_cols_init"] = _as_name_list(
            user_cfg["feature_cols_init"], "feature_cols_init", notes,
            "les variables sont déduites du classeur (colonnes numériques qui ne sont pas des "
            "zones)")
    if "user_defined_vars_cols" in user_cfg:
        out["user_defined_vars_cols"] = _as_name_list(
            user_cfg["user_defined_vars_cols"], "user_defined_vars_cols", notes,
            "aucune variable synthétique")

    # Chemins et noms de fichiers.
    for key in ("input_dir", "output_dir", "output_json", "data_files"):
        if key in user_cfg:
            take(out, key, _as_text(user_cfg[key], key, notes, DEFAULT_CONFIG[key]))

    # Sous-dossiers : chaîne ou liste de chaînes non vides, sans doublon.
    if "merged_subdir" in user_cfg:
        value = user_cfg["merged_subdir"]
        if _is_blank(value) or value == []:
            _default_note("merged_subdir", DEFAULT_CONFIG["merged_subdir"], notes)
        elif isinstance(value, str):
            out["merged_subdir"] = value.strip()
        elif isinstance(value, list):
            names: List[str] = []
            for position, item in enumerate(value):
                if not isinstance(item, str) or not item.strip():
                    raise ConfigError(
                        f"'merged_subdir[{position}]' doit être un nom de sous-dossier non "
                        f"vide ; valeur reçue : {_show(item)}.")
                names.append(item.strip())
            if len(set(names)) != len(names):
                raise ConfigError("'merged_subdir' contient des doublons.")
            out["merged_subdir"] = names
        else:
            raise ConfigError(
                "'merged_subdir' doit être une chaîne ou une liste de chaînes ; valeur "
                f"reçue : {_show(value)}.")

    if "user_defined_vars" in user_cfg:
        take(out, "user_defined_vars", _as_bool(
            user_cfg["user_defined_vars"], "user_defined_vars", notes,
            DEFAULT_CONFIG["user_defined_vars"]))
    if "sample_random_state" in user_cfg:
        take(out, "sample_random_state", _as_number(
            user_cfg["sample_random_state"], "sample_random_state", notes,
            DEFAULT_CONFIG["sample_random_state"], minimum=0, maximum=2 ** 32 - 1,
            integer=True))

    # Blocs objets.
    for block, spec in _BLOCK_SPECS.items():
        content = _as_block(user_cfg, block, notes)
        cleaned: Dict[str, Any] = {}
        for sub_key, sub_value in content.items():
            if sub_key not in spec:
                warnings.append(
                    f"Clé inconnue '{block}.{sub_key}' ignorée (faute de frappe ?).")
                continue
            kind, options = spec[sub_key]
            path = f"{block}.{sub_key}"
            default = DEFAULT_CONFIG[block].get(sub_key)
            if kind == "number":
                checked = _as_number(sub_value, path, notes, default, **options)
            elif kind == "integer":
                checked = _as_number(sub_value, path, notes, default, integer=True,
                                     **options)
            elif kind == "length":
                checked = _as_length(sub_value, path, notes, default)
            else:
                checked = _as_text(sub_value, path, notes, default, **options)
            take(cleaned, sub_key, checked)
        if block in user_cfg:
            out[block] = cleaned

    # Noms de sortie : fichiers simples.
    if "output_files" in user_cfg:
        content = _as_block(user_cfg, "output_files", notes)
        cleaned_files: Dict[str, str] = {}
        for sub_key, sub_value in content.items():
            path = f"output_files.{sub_key}"
            if sub_key not in DEFAULT_CONFIG["output_files"]:
                warnings.append(f"Clé inconnue '{path}' ignorée (faute de frappe ?).")
                continue
            text = _as_text(sub_value, path, notes, DEFAULT_CONFIG["output_files"][sub_key])
            if text is _MISSING:
                continue
            if text != os.path.basename(text) or "/" in text or "\\" in text or text in (".", ".."):
                raise ConfigError(
                    f"'{path}' doit être un nom de fichier simple, sans dossier ; valeur "
                    f"reçue : '{text}'.")
            cleaned_files[sub_key] = text
        out["output_files"] = cleaned_files

    return out, notes, warnings


def _is_within(child: str, parent: str) -> bool:
    """Vrai si ``child`` est ``parent`` ou se trouve sous ``parent`` (chemins résolus)."""
    try:
        return os.path.commonpath([child, parent]) == parent
    except ValueError:  # lecteurs différents
        return False


def _resolved(path: str) -> str:
    """Chemin absolu, liens résolus, normalisé en casse."""
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def check_subdir_name(name: str, path: str = "merged_subdir") -> None:
    """Refuse un nom de sous-dossier qui sortirait de la racine d'entrée ou de sortie.

    Args:
        name: Nom déclaré.
        path: Nom de la clé, pour le message.

    Raises:
        ConfigError: si le nom est absolu, contient un séparateur de chemin, ``..``
            ou une désignation de lecteur.
    """
    drive, _ = os.path.splitdrive(name)
    if (os.path.isabs(name) or name.startswith(("/", "\\")) or drive
            or "/" in name or "\\" in name or ".." in name or ":" in name
            or name == "."):
        raise ConfigError(
            f"'{path}' vaut '{name}' ; un nom de sous-dossier simple est exigé (ni "
            "séparateur de chemin, ni '..', ni chemin absolu) : le dossier de sortie "
            "correspondant est vidé à chaque exécution.")


def check_data_file_name(name: str) -> None:
    """Refuse un ``data_files`` qui n'est pas un nom de fichier simple de format admis.

    Args:
        name: Nom déclaré.

    Raises:
        ConfigError: si le nom contient un dossier, ou si le format n'est pas lu.
    """
    if (os.path.isabs(name) or "/" in name or "\\" in name or ":" in name
            or name in (".", "..")):
        raise ConfigError(
            f"'data_files' vaut '{name}' ; un nom de fichier simple est exigé, sans "
            "dossier ni chemin absolu (le dossier d'entrée est 'input_dir', le "
            "sous-dossier est 'merged_subdir').")
    extension = os.path.splitext(name)[1].lower()
    if extension == ".xls":
        raise ConfigError(
            f"'data_files' vaut '{name}' : le format .xls (ancien Excel) n'est pas pris "
            "en charge. Ouvrez le fichier dans Excel, enregistrez-le au format .xlsx (ou "
            ".csv), puis déclarez le nouveau nom.")
    if extension not in READABLE_EXTENSIONS:
        raise ConfigError(
            f"'data_files' vaut '{name}' : formats lus .xlsx et .csv. Convertissez le "
            "fichier à l'un de ces formats.")


#: Noms de colonnes engendrés par le programme : ils ne peuvent pas être des noms
#: de zones ou de variables, sous peine de collision dans les tableaux de travail.
#: ``2.00$\times$A`` (enveloppe κ·cible), ``$\alpha$A`` et ``$\beta$`` (tableau de
#: parcimonie) ont tous la forme d'un nombre suivi de ``$\times$`` ou débutent par
#: un dollar.
_GENERATED_NAME_PATTERN = re.compile(r"^\s*[0-9]+(?:\.[0-9]+)?\$\\times\$")


def check_reserved_names(names: List[str], label: str) -> None:
    """Refuse les noms de zones ou de variables qui collisionneraient avec OCA.

    Un nom qui commence par ``$`` ou qui a la forme d'un libellé engendré par le
    programme (enveloppe ``2.00$\\times$A``, colonnes ``$\\alpha$A`` et ``$\\beta$``)
    écraserait ou se confondrait avec une colonne calculée.

    Args:
        names: Noms à contrôler.
        label: Désignation de la liste (clé de configuration ou nature du nom).

    Raises:
        ConfigError: au premier nom réservé.
    """
    for name in names:
        text = str(name)
        if text.startswith("$") or _GENERATED_NAME_PATTERN.match(text):
            raise ConfigError(
                f"{label} : le nom '{text}' est réservé (il commence par '$' ou a la "
                "forme d'un libellé engendré par OCA, comme 2.00$\\times$A). Renommez la "
                "colonne dans le classeur et dans la configuration.")


def _source_files(input_dir: str, subdirs: List[str], data_files: str,
                  zones: List[str]) -> Dict[str, str]:
    """Fichiers sources possibles : classeurs fusionnés et fichiers par-zone.

    Returns:
        Association chemin résolu vers désignation lisible.
    """
    sources: Dict[str, str] = {}
    for name in subdirs:
        shown = os.path.join(input_dir, name, data_files)
        sources[_resolved(shown)] = shown
    for zone in zones:
        if any(char in zone for char in "/\\:"):
            continue
        shown = os.path.join(input_dir, f"dataSet_{zone}", f"dataSet_{zone}.xlsx")
        sources[_resolved(shown)] = shown
    return sources


#: Clés de ``output_files`` dont la valeur est un nom sans extension : le programme
#: écrit un ``.tex`` et compile un ``.pdf`` (et des fichiers auxiliaires).
_LATEX_BASE_KEYS = ("step1_result", "step1_zones_sample", "criticalzones", "report_intro")


def planned_output_names(output_files: Dict[str, str], data_files: str,
                         multi_run: bool) -> Dict[str, List[str]]:
    """Noms de fichiers écrits dans un dossier de sortie, avec leur provenance.

    Returns:
        Association nom (casse normalisée) vers désignations des clés qui l'écrivent.
    """
    planned: Dict[str, List[str]] = {}

    def add(file_name: str, label: str) -> None:
        planned.setdefault(os.path.normcase(file_name), [])
        if label not in planned[os.path.normcase(file_name)]:
            planned[os.path.normcase(file_name)].append(label)

    for key, value in output_files.items():
        label = f"output_files.{key}"
        if key in _LATEX_BASE_KEYS:
            for extension in (".tex", ".pdf", ".aux", ".log", ".out"):
                add(value + extension, label)
        else:
            add(value, label)
    saved = data_files
    if os.path.splitext(saved)[1].lower() != ".xlsx":
        saved = os.path.splitext(saved)[0] + ".xlsx"
    add(saved, "data_files (copie de traçabilité)")
    if multi_run:
        add("output.json", "output.json de chaque sous-dossier")
    return planned


def check_output_collisions(input_dir: str, output_dir: str, subdirs: List[str],
                            data_files: str, zones: List[str], output_json: str,
                            output_files: Dict[str, str]) -> None:
    """Refuse les sorties qui s'écraseraient entre elles ou écraseraient une source.

    Contrôles, sur chemins résolus et avant tout nettoyage ou calcul :

        * deux clés de ``output_files`` (ou la copie de traçabilité, ou le JSON de
          chaque sous-dossier) ne doivent pas désigner le même fichier ;
        * aucun fichier de sortie, ni ``output_json``, ne doit désigner un fichier
          source (classeur fusionné ou fichier par-zone) ;
        * ``output_json`` ne doit pas désigner un fichier écrit dans un dossier de
          sortie.

    Raises:
        ConfigError: au premier recouvrement.
    """
    planned = planned_output_names(output_files, data_files, len(subdirs) > 1)
    for file_name, labels in planned.items():
        if len(labels) > 1:
            raise ConfigError(
                f"Les sorties {labels} désignent le même fichier '{file_name}' : l'une "
                "écraserait l'autre. Donnez-leur des noms distincts dans 'output_files'.")

    sources = _source_files(input_dir, subdirs, data_files, zones)
    for name in subdirs:
        folder = os.path.join(output_dir, name)
        for file_name, labels in planned.items():
            target = _resolved(os.path.join(folder, file_name))
            if target in sources:
                raise ConfigError(
                    f"La sortie {labels[0]} ('{file_name}') désigne le fichier source "
                    f"'{sources[target]}' : il serait écrasé. Choisissez un autre nom.")
    json_target = _resolved(output_json)
    if json_target in sources:
        raise ConfigError(
            f"'output_json' ('{output_json}') désigne le fichier source "
            f"'{sources[json_target]}' : il serait écrasé. Choisissez un autre nom.")
    for name in subdirs:
        folder = os.path.join(output_dir, name)
        for file_name, labels in planned.items():
            if json_target == _resolved(os.path.join(folder, file_name)):
                raise ConfigError(
                    f"'output_json' ('{output_json}') désigne le même fichier que la "
                    f"sortie {labels[0]} : l'un écraserait l'autre. Choisissez un autre nom.")


def check_paths_safe(input_dir: str, output_dir: str, subdirs: List[str],
                     data_files: str, zones: List[str], output_json: str,
                     output_files: Optional[Dict[str, str]] = None) -> None:
    """Vérifie, avant tout nettoyage, que les sorties ne recouvrent pas les entrées.

    Le dossier de sortie de chaque sous-dossier est vidé à chaque exécution. Cette
    fonction refuse tout recouvrement avec les données d'entrée, ainsi que tout
    ``output_json`` ou nom de sortie qui désignerait un fichier source, ou qui
    entrerait en collision avec un autre fichier de sortie.

    Args:
        input_dir: Dossier d'entrée déclaré.
        output_dir: Dossier de sortie déclaré.
        subdirs: Sous-dossiers déclarés.
        data_files: Nom du classeur de données.
        zones: Zones déclarées (fichiers par-zone éventuels).
        output_json: Chemin du fichier de résultats.
        output_files: Noms des fichiers de sortie (défauts si absent).

    Raises:
        ConfigError: au premier recouvrement dangereux.
    """
    for name in subdirs:
        check_subdir_name(name)
    check_data_file_name(data_files)

    source = _resolved(input_dir)
    target = _resolved(output_dir)
    if target == source or _is_within(source, target):
        raise ConfigError(
            f"'output_dir' ('{output_dir}') est identique à 'input_dir' ('{input_dir}') "
            "ou le contient : le nettoyage des sorties détruirait les données. Choisissez "
            "un dossier de sortie distinct.")
    protected = [_resolved(os.path.join(input_dir, f"dataSet_{zone}"))
                 for zone in zones if not any(c in zone for c in "/\\:")]
    for name in subdirs:
        cleaned = _resolved(os.path.join(output_dir, name))
        shown = os.path.join(output_dir, name)
        for other in protected + [source, _resolved(os.path.join(input_dir, name))]:
            if cleaned == other or _is_within(other, cleaned):
                raise ConfigError(
                    f"Le dossier de sortie '{shown}' recouvre un dossier d'entrée ; il est "
                    "vidé à chaque exécution et détruirait les données. Séparez "
                    "'output_dir' et 'input_dir'.")
        if os.path.isfile(cleaned):
            raise ConfigError(f"Le chemin de sortie '{shown}' est un fichier, pas un dossier.")
    json_target = _resolved(output_json)
    if os.path.isdir(json_target):
        raise ConfigError(f"'output_json' ('{output_json}') désigne un dossier.")
    check_output_collisions(
        input_dir, output_dir, subdirs, data_files, zones, output_json,
        output_files if output_files is not None else DEFAULT_CONFIG["output_files"])


def _validate(cfg: Dict[str, Any]) -> None:
    """Valide les relations entre champs d'une configuration normalisée.

    Les types et domaines de chaque champ sont déjà contrôlés par
    :func:`_normalize_user_config` ; il reste les contraintes croisées.

    Args:
        cfg: Dictionnaire de configuration (défauts + utilisateur).

    Raises:
        ConfigError: à la première incohérence rencontrée.
    """
    from oca_model import MAX_KAPPA_POINTS, kappa_grid, kappa_point_count

    common = [zone for zone in cfg["critical_zones"] if zone in cfg["feature_cols_init"]]
    if common:
        raise ConfigError(
            f"Les listes 'critical_zones' et 'feature_cols_init' se recouvrent : {common}. "
            "Une colonne est soit une variable de conception, soit une zone critique.")

    subdirs = _normalize_merged_subdirs(cfg["merged_subdir"])
    if not subdirs:
        raise ConfigError(
            "'merged_subdir' doit être une chaîne non vide ou une liste non vide de "
            "chaînes.")
    if len(set(subdirs)) != len(subdirs):
        raise ConfigError("'merged_subdir' contient des doublons.")

    gpo = cfg["gpo_selection"]
    if gpo["threshold_low"] > gpo["threshold_high"]:
        raise ConfigError(
            "'gpo_selection.threshold_low' doit être <= 'gpo_selection.threshold_high'.")

    reg = cfg["gpo_regularization"]
    if reg["kappa_min"] > reg["kappa_max"]:
        raise ConfigError(
            "'gpo_regularization.kappa_min' doit être <= 'gpo_regularization.kappa_max'.")
    n_points = kappa_point_count(reg["kappa_min"], reg["kappa_max"], reg["kappa_step"])
    if n_points > MAX_KAPPA_POINTS:
        raise ConfigError(
            f"'gpo_regularization.kappa_step' = {reg['kappa_step']:g} donnerait plus de "
            f"{MAX_KAPPA_POINTS} valeurs de κ sur [{reg['kappa_min']:g}, "
            f"{reg['kappa_max']:g}] (maximum admis : {MAX_KAPPA_POINTS}). Augmentez le "
            "pas ou resserrez la plage.")
    try:
        kappa_grid(reg["kappa_min"], reg["kappa_max"], reg["kappa_step"])
    except ValueError as exc:
        raise ConfigError(f"'gpo_regularization' (kappa_min, kappa_max, kappa_step) : "
                          f"{exc}") from None

    check_reserved_names(cfg["critical_zones"], "'critical_zones'")
    check_reserved_names(cfg["feature_cols_init"], "'feature_cols_init'")
    check_reserved_names(cfg["user_defined_vars_cols"], "'user_defined_vars_cols'")
    check_reserved_names([str(reg["target_zone"])], "'gpo_regularization.target_zone'")

    # Validation des noms de cartes de couleurs (valeurs de chaîne déjà garanties).
    _validate_color_map_names(cfg.get("color_style", {}))

    if cfg["user_defined_vars"] and not cfg["user_defined_vars_cols"]:
        raise ConfigError(
            "'user_defined_vars' est vrai mais 'user_defined_vars_cols' est vide : "
            "listez les expressions à calculer, ou mettez 'user_defined_vars' à false.")

    check_paths_safe(cfg["input_dir"], cfg["output_dir"], subdirs, cfg["data_files"],
                     cfg["critical_zones"], cfg["output_json"], cfg["output_files"])


def strip_json_comments(text: str) -> Tuple[str, int, int]:
    """Retire les commentaires d'un fichier de configuration (V7).

    Les commentaires de fin de ligne ``//`` et les commentaires encadrés
    ``/* ... */`` sont supprimés. La norme JSON ne les admet pas, mais un fichier
    de configuration est un document que l'on annote : les y proscrire condamnerait
    l'utilisateur à documenter son paramétrage ailleurs que là où il le lit.

    Le retrait est donc **systématique et silencieux**. Il intervient avant toute
    tentative d'analyse, et non en réparation d'un échec : un commentaire n'est pas
    une anomalie de ponctuation, et le signaler comme telle inciterait à corriger
    un fichier que l'application admet délibérément. Le dénombrement est retourné
    à seule fin de consignation au journal, au niveau informatif.

    Le commentaire est remplacé par autant d'espaces qu'il comptait de caractères,
    les sauts de ligne étant préservés. Les offsets et les numéros de ligne que
    rapporterait le décodeur JSON sur le texte assaini désignent ainsi exactement
    la même position que dans le fichier d'origine, ce qui rend tout message
    d'erreur ultérieur directement exploitable.

    Le balayage suit l'état de chaîne et d'échappement : une barre oblique ou une
    astérisque interne à une valeur textuelle n'est jamais altérée. C'est
    indispensable ici, les variables synthétiques de l'application s'écrivant
    précisément avec des barres obliques et des astérisques, ainsi
    ``"D1/e + 0.5*H3/e"``.

    Args:
        text: Contenu brut du fichier de configuration.

    Returns:
        Triplet formé du texte assaini, du nombre de commentaires de fin de ligne
        et du nombre de commentaires encadrés retirés.
    """
    out: List[str] = []
    index = 0
    length = len(text)
    in_string = False
    escaped = False
    n_line = 0
    n_block = 0

    while index < length:
        char = text[index]

        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue

        # Commentaire de fin de ligne : le saut de ligne terminal est conservé.
        if char == "/" and index + 1 < length and text[index + 1] == "/":
            end = text.find("\n", index)
            end = length if end == -1 else end
            out.append(" " * (end - index))
            n_line += 1
            index = end
            continue

        # Commentaire encadré : les sauts de ligne internes sont conservés, afin
        # que la numérotation des lignes demeure celle du fichier d'origine.
        if char == "/" and index + 1 < length and text[index + 1] == "*":
            end = text.find("*/", index + 2)
            end = length if end == -1 else end + 2
            out.append("".join("\n" if c == "\n" else " " for c in text[index:end]))
            n_block += 1
            index = end
            continue

        out.append(char)
        index += 1

    return "".join(out), n_line, n_block


def _repair_json_text(text: str) -> Tuple[str, List[str]]:
    """Neutralise les véritables anomalies de ponctuation d'une configuration.

    Les fichiers ``input.json`` sont édités à la main : la suppression d'un champ
    y laisse fréquemment une virgule orpheline avant l'accolade fermante. La norme
    JSON la rejette. Plutôt que d'échouer, la lecture la neutralise et consigne la
    réparation, l'utilisateur restant informé d'un écart qu'il lui appartient de
    corriger.

    Les commentaires ne relèvent **plus** de cette fonction : ils sont retirés en
    amont et sans bruit par :func:`strip_json_comments`, l'application les
    admettant délibérément. Cette fonction ne traite donc que ce qui constitue
    réellement une anomalie.

    Le balayage suit l'état de chaîne et d'échappement, en sorte qu'une virgule
    interne à une valeur textuelle ne soit jamais altérée.

    Args:
        text: Contenu de la configuration, commentaires déjà retirés.

    Returns:
        Couple formé du texte assaini et de la liste des réparations appliquées.
    """
    repairs: List[str] = []
    out: List[str] = []
    index = 0
    length = len(text)
    in_string = False
    escaped = False

    while index < length:
        char = text[index]

        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue

        # Virgule orpheline précédant une accolade ou un crochet fermant.
        if char == ",":
            probe = index + 1
            while probe < length and text[probe] in " \t\r\n":
                probe += 1
            if probe < length and text[probe] in "}]":
                repairs.append(f"virgule orpheline supprimée à l'offset {index}")
                index += 1
                continue

        out.append(char)
        index += 1

    return "".join(out), repairs


def load_config(config_path: str = "input.json") -> OCAConfig:
    """Lit, fusionne, valide et retourne la configuration OCA.

    La lecture tolère les écarts d'écriture usuels décrits par
    :func:`_repair_json_text`. Les réparations appliquées sont exposées par
    l'attribut ``json_repairs`` de la configuration retournée, afin d'être
    consignées au journal d'exécution.

    Args:
        config_path: Chemin du fichier ``input.json``.

    Returns:
        OCAConfig: configuration validée et immuable.

    Raises:
        ConfigError: si le fichier est absent, mal formé ou incohérent.
    """
    if not os.path.isfile(config_path):
        raise ConfigError(f"Fichier de configuration introuvable : {config_path}")
    # ``utf-8-sig`` retire un éventuel indicateur d'ordre des octets (BOM) placé en
    # tête par les éditeurs Windows, que le décodeur JSON rejette. En son absence,
    # le codec se comporte exactement comme ``utf-8``.
    try:
        with open(config_path, "r", encoding="utf-8-sig") as config_stream:
            raw_text = config_stream.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(
            f"Le fichier de configuration '{config_path}' n'a pas pu être lu "
            f"({type(exc).__name__}) : enregistrez-le en UTF-8.") from None

    # V7 : les commentaires sont retirés SYSTÉMATIQUEMENT et en amont de toute
    # tentative d'analyse, et non en réparation d'un échec. L'application les admet
    # délibérément : un fichier de configuration est un document que l'on annote, et
    # les proscrire condamnerait l'utilisateur à documenter son paramétrage ailleurs
    # que là où il le lit. Leur dénombrement est conservé pour une consignation
    # informative, jamais pour un avertissement.
    clean_text, n_line, n_block = strip_json_comments(raw_text)
    comment_notes: List[str] = []
    if n_line or n_block:
        parts = []
        if n_line:
            parts.append(f"{n_line} commentaire(s) de fin de ligne")
        if n_block:
            parts.append(f"{n_block} commentaire(s) encadré(s)")
        comment_notes.append(
            " et ".join(parts) + " retiré(s) avant analyse ; l'écriture commentée "
            "est admise et ne constitue pas un écart")

    repairs: List[str] = []
    try:
        user_cfg = json.loads(clean_text)
    except (json.JSONDecodeError, RecursionError) as first_error:
        # Un entier de plus de 4300 chiffres lève aussi une erreur de valeur du
        # décodeur : elle est présentée comme un JSON invalide, sans trace.
        repaired_text, repairs = _repair_json_text(clean_text)
        if not repairs:
            raise ConfigError(
                f"JSON invalide dans {config_path} : {first_error}") from None
        try:
            user_cfg = json.loads(repaired_text)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ConfigError(
                f"JSON invalide dans {config_path} : {exc}. Réparations tentées : "
                f"{'; '.join(repairs)}.") from None
    except ValueError as exc:
        raise ConfigError(
            f"JSON invalide dans {config_path} : nombre démesuré ou illisible "
            f"({str(exc).splitlines()[0] if str(exc) else 'valeur rejetée'}).") from None
    if not isinstance(user_cfg, dict):
        raise ConfigError("La racine de input.json doit être un objet JSON.")

    # V7 : les clés d'auto-documentation sont retirées avant toute autre
    # opération. Elles ne relèvent pas du paramétrage et ne doivent donc ni être
    # validées, ni fusionnées, ni parvenir aux modules de calcul.
    # Elles sont consignées à part des réparations : une clé d'aide est un
    # document parfaitement conforme à la norme JSON, non un écart d'écriture, et
    # la signaler comme tel induirait l'utilisateur à corriger un fichier correct.
    user_cfg, option_keys = strip_option_metadata(user_cfg)
    legacy_notes = _expand_legacy_color_map(user_cfg)

    # V9 : types et valeurs contrôlés champ par champ avant toute conversion ; un
    # champ vide ou nul reprend son défaut, un mauvais type nomme la clé fautive.
    cleaned, normalization_notes, normalization_warnings = _normalize_user_config(user_cfg)
    merged = _deep_merge(DEFAULT_CONFIG, cleaned)
    _validate(merged)

    # V6 : 'merged_subdir' est normalisé en liste ; le premier élément désigne le
    # sous-dossier actif par défaut, l'orchestrateur parcourant ensuite la liste.
    subdirs = _normalize_merged_subdirs(merged["merged_subdir"])

    return OCAConfig(
        critical_zones=list(merged["critical_zones"]),
        feature_cols_init=list(merged["feature_cols_init"]),
        input_dir=str(merged["input_dir"]),
        output_dir=str(merged["output_dir"]),
        output_json=str(merged["output_json"]),
        data_files=str(merged["data_files"]),
        merged_subdir=subdirs[0],
        user_defined_vars=bool(merged["user_defined_vars"]),
        user_defined_vars_cols=list(merged["user_defined_vars_cols"]),
        gpo_selection=dict(merged["gpo_selection"]),
        gpo_regularization=copy.deepcopy(merged["gpo_regularization"]),
        ordinal_graph=dict(merged["ordinal_graph"]),
        latex_output=dict(merged["latex_output"]),
        color_style=dict(merged["color_style"]),
        sample_random_state=int(merged["sample_random_state"]),
        output_files=dict(merged["output_files"]),
        merged_subdirs=subdirs,
        raw=merged,
        json_repairs=repairs,
        option_keys=option_keys,
        legacy_notes=legacy_notes,
        comment_notes=comment_notes,
        normalization_notes=normalization_notes,
        normalization_warnings=normalization_warnings,
    )


def _rows_text(positions: List[int], limit: int = 8) -> str:
    """Numéros d'observations (à partir de 1) sous forme lisible et bornée."""
    shown = ", ".join(str(position + 1) for position in positions[:limit])
    extra = len(positions) - limit
    return f"{shown} (et {extra} autre(s))" if extra > 0 else shown


def check_finite_columns(df: "pd.DataFrame", columns: "List[str]", where: str = "") -> None:
    """Refuse les valeurs manquantes ou infinies dans les colonnes comparées.

    Le contrôle porte sur toutes les colonnes effectivement comparées : colonnes du
    classeur, variables synthétiques et cible ajoutée. Une comparaison impliquant
    une valeur manquante vaut faux (la proportion de majoration serait diluée) ; un
    infini majorerait toutes les autres colonnes.

    Args:
        df: Tableau de travail.
        columns: Colonnes comparées présentes dans ``df``.
        where: Précision ajoutée au message (étape ou origine).

    Raises:
        ConfigError: avec colonne, nombre et observations fautives.
    """
    problems: List[str] = []
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        missing = np.flatnonzero(np.isnan(values)).tolist()
        infinite = np.flatnonzero(np.isinf(values)).tolist()
        if missing:
            problems.append(
                f"'{column}' : {len(missing)} valeur(s) manquante(s), observation(s) n° "
                f"{_rows_text(missing)}")
        if infinite:
            problems.append(
                f"'{column}' : {len(infinite)} valeur(s) infinie(s), observation(s) n° "
                f"{_rows_text(infinite)}")
    if problems:
        suffix = f" ({where})" if where else ""
        raise ConfigError(
            f"Valeurs non utilisables sur les zones comparées{suffix}, sur {len(df)} "
            f"observation(s) : " + " ; ".join(problems) + ". Une valeur manquante ou "
            "infinie fausserait les proportions de majoration : complétez ou retirez ces "
            "observations avant l'analyse.")


def _certify_working_table(
    df_merged: "pd.DataFrame",
    critical_zones: "List[str]",
    logger: Any = None,
) -> None:
    """Refuse un jeu de travail hors du domaine de l'algorithme ordinal.

    Préconditions, toutes vérifiées avant d'être imposées ici.

        1. **Au moins deux observations.** Sans observation, le comptage divise par
           zéro ; avec une seule, la corrélation linéaire (nuages de points) n'est
           pas définie et l'ordre n'apprend rien.
        2. **Au moins deux zones critiques.** L'analyse un contre tous est vide pour
           une zone unique : elle serait déclarée écartée et retenue à la fois.
        3. **Aucune valeur manquante ni infinie sur les zones comparées.** Une
           comparaison impliquant ``NaN`` vaut faux : la proportion de majoration est
           diluée par le taux de trous (vingt pour cent de valeurs manquantes
           ramènent une dominance totale de 1,0000 à 0,8000). Un infini majore toutes
           les colonnes.

    Le contrôle est posé APRÈS la construction des variables synthétiques, de sorte
    qu'il porte sur les colonnes effectivement comparées, et AVANT tout calcul.

    Args:
        df_merged: Tableau de travail fusionné.
        critical_zones: Zones critiques, variables synthétiques comprises.
        logger: Journal, le cas échéant.

    Raises:
        ConfigError: à la première précondition violée.
    """
    if len(df_merged) == 0:
        raise ConfigError(
            "Le jeu de travail ne comporte aucune observation : tout comptage de "
            "dominance diviserait par zéro. Vérifiez le classeur (feuille, lignes de "
            "données sous l'en-tête).")
    if len(df_merged) < 2:
        raise ConfigError(
            "Le jeu de travail ne comporte qu'une observation : au moins 2 observations "
            "sont nécessaires (la corrélation des nuages de points n'est pas définie, et "
            "un ordre ne s'établit pas sur un seul point).")
    if len(critical_zones) < 2:
        raise ConfigError(
            f"Une seule zone à comparer ({critical_zones}). L'analyse de dominance un "
            "contre tous est vide dans ce cas. Déclarez au moins deux zones critiques, "
            "variables synthétiques comprises.")

    present = [str(zone) for zone in critical_zones if str(zone) in df_merged.columns]
    check_finite_columns(df_merged, present)
    if logger:
        logger.info(
            f"Jeu de travail certifié : {len(df_merged)} observation(s), "
            f"{len(critical_zones)} zone(s) comparée(s), aucune valeur manquante ni "
            "infinie.")


def prepare_output_dirs(config: OCAConfig) -> str:
    """Crée le dossier de sortie global et le sous-dossier principal.

    Args:
        config: Configuration validée.

    Returns:
        Chemin du sous-dossier de sortie principal créé.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    main_dir = config.output_dir_main()
    os.makedirs(main_dir, exist_ok=True)
    return main_dir


def clean_output_main(config: OCAConfig) -> str:
    """Vide (puis recrée) le sous-dossier de sortie principal.

    Supprime les résultats d'exécutions antérieures afin que la sortie ne
    contienne que les fichiers de l'exécution courante (schéma de numérotation à
    jour). À appeler **avant** l'ouverture du fichier de log. Les chemins sont
    contrôlés une seconde fois ici, quelle que soit l'origine de la configuration :
    aucun dossier d'entrée ne peut être supprimé.

    Args:
        config: Configuration validée.

    Returns:
        Chemin du sous-dossier de sortie principal (vide et recréé).

    Raises:
        ConfigError: si le dossier à vider recouvre des données d'entrée.
    """
    check_paths_safe(config.input_dir, config.output_dir, [config.merged_subdir],
                     config.data_files, list(config.critical_zones), config.output_json,
                     config.output_files)
    main_dir = config.output_dir_main()
    if os.path.isdir(main_dir):
        shutil.rmtree(main_dir, ignore_errors=True)
    os.makedirs(main_dir, exist_ok=True)
    return main_dir


# ===========================================================================
# Lecture des classeurs
# ===========================================================================
@dataclass
class ColumnResolution:
    """Listes de colonnes résolues pour une exécution.

    Attributes:
        zones: Zones critiques lues dans le classeur (hors variables synthétiques).
        features: Variables de conception.
        zones_inferred: Vrai si les zones ont été déduites des colonnes numériques.
        features_inferred: Vrai si les variables ont été déduites.
        ignored: Colonnes du classeur ni zones ni variables (textuelles, vides ou
            non retenues).
        source: Classeur lu.
        sheet: Feuille lue (la première du classeur).
        index_column: Première colonne retirée comme index, ou ``None``.
        messages: Notes et avertissements ``(niveau, texte)`` à consigner.
    """

    zones: List[str]
    features: List[str]
    zones_inferred: bool = False
    features_inferred: bool = False
    ignored: List[str] = field(default_factory=list)
    source: str = ""
    sheet: str = ""
    index_column: Optional[str] = None
    messages: List[Tuple[str, str]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """Structure sérialisable pour ``output.json``."""
        return {
            "critical_zones": list(self.zones),
            "feature_cols": list(self.features),
            "critical_zones_inferred": bool(self.zones_inferred),
            "feature_cols_inferred": bool(self.features_inferred),
            "ignored_columns": list(self.ignored),
            "sheet": self.sheet,
            "index_column_removed": self.index_column,
        }


def _clean_header(cell: Any) -> str:
    """Nom de colonne normalisé : chaîne sans espaces de bord, entier sans ``.0``."""
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return ""
    if isinstance(cell, float) and cell.is_integer():
        return str(int(cell))
    return str(cell).strip()


def _name_is_cited(name: str, cited: "Sequence[str]", expressions: "Sequence[str]") -> bool:
    """Vrai si ``name`` est déclaré (zones, variables) ou utilisé par une expression."""
    if name in cited:
        return True
    pattern = re.compile(r"(?<![\w.])" + re.escape(name) + r"(?![\w])")
    return any(pattern.search(expression) for expression in expressions)


def _read_raw_table(
    path: str,
    cited: "Sequence[str]" = (),
    expressions: "Sequence[str]" = (),
) -> Tuple["pd.DataFrame", Dict[str, Any]]:
    """Lit la première feuille d'un classeur ``.xlsx`` (ou un ``.csv``) sans conversion.

    La première ligne est l'en-tête. Une première colonne sans en-tête (ou dont
    l'en-tête commence par ``Unnamed``) ou nommée ``index``, ``id``, ``config``,
    ``config_id`` ou ``n`` est un index d'observations : elle est retirée, **sauf**
    si son nom est cité dans ``critical_zones``, ``feature_cols_init`` ou une
    expression de variable synthétique (``cited`` et ``expressions``) : elle est
    alors conservée comme grandeur. Toute autre première colonne est conservée. Les
    colonnes à en-tête vide sont ignorées.

    Args:
        path: Chemin du classeur.
        cited: Noms déclarés (zones, variables), qui protègent une première colonne.
        expressions: Expressions de variables synthétiques et de cible.

    Returns:
        Couple ``(tableau, infos)`` : colonnes nommées d'après l'en-tête, cellules
        non converties ; ``infos`` porte la feuille, l'index retiré (ou conservé parce
        que cité, clé ``index_kept``) et le format.

    Raises:
        ConfigError: si le fichier est illisible, vide, de format non pris en
            charge, ou si deux colonnes portent le même nom.
    """
    extension = os.path.splitext(path)[1].lower()
    name = os.path.basename(path)
    info: Dict[str, Any] = {"format": extension, "sheet": "", "index_column": None,
                            "index_kept": None, "decimal_comma": False}
    if extension == ".xls":
        raise ConfigError(
            f"Le format .xls (ancien Excel) de '{name}' n'est pas pris en charge : "
            "enregistrez le fichier au format .xlsx (ou .csv) depuis Excel.")
    try:
        if extension == ".csv":
            raw = _read_csv_rows(path, info)
        elif extension == ".xlsx":
            with pd.ExcelFile(path, engine="openpyxl") as workbook:
                info["sheet"] = str(workbook.sheet_names[0])
                raw = workbook.parse(workbook.sheet_names[0], header=None, dtype=object)
        else:
            raise ConfigError(
                f"Format de '{name}' non pris en charge : formats lus .xlsx et .csv.")
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001 - fichier illisible : message clair
        raise ConfigError(
            f"Le fichier '{path}' n'a pas pu être lu ({type(exc).__name__} : "
            f"{str(exc).splitlines()[0] if str(exc) else 'erreur inconnue'}). Vérifiez "
            "qu'il est un classeur .xlsx (ou .csv) valide et fermé dans Excel.") from None

    if raw.shape[0] < 2 or raw.shape[1] < 1:
        raise ConfigError(
            f"Le fichier '{name}' est vide ou ne comporte pas de ligne de données sous "
            "l'en-tête (première ligne de la première feuille).")
    headers = [_clean_header(cell) for cell in raw.iloc[0].tolist()]
    body = raw.iloc[1:].reset_index(drop=True)
    body.columns = list(range(len(headers)))
    # Les lignes entièrement vides en fin de feuille (mise en forme laissée par
    # Excel) ne portent aucune observation : elles sont retirées, sans décaler la
    # numérotation des lignes qui les précèdent.
    filled = body.map(_cell_present).any(axis=1).to_numpy()
    last_filled = int(np.flatnonzero(filled).max()) if filled.any() else -1
    info["trailing_empty_rows"] = len(body) - (last_filled + 1)
    body = body.iloc[:last_filled + 1].reset_index(drop=True)
    if len(body) == 0:
        raise ConfigError(
            f"Le fichier '{name}' ne comporte aucune ligne de données sous l'en-tête "
            "(première ligne de la première feuille).")

    keep = list(range(len(headers)))
    first = headers[0]
    index_like = (first == "" or first.lower().startswith("unnamed")
                  or first.lower() in INDEX_LIKE_HEADERS)
    if index_like and first != "" and _name_is_cited(first, cited, expressions):
        # Le nom usuel d'un index désigne ici une grandeur déclarée : elle est gardée.
        info["index_kept"] = first
    elif index_like:
        info["index_column"] = first or "(sans en-tête)"
        keep = keep[1:]
    blank = [position for position in keep if headers[position] == ""]
    if blank:
        keep = [position for position in keep if headers[position] != ""]
    names = [headers[position] for position in keep]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ConfigError(
            f"Le fichier '{name}' a des en-têtes de colonne dupliqués : {duplicates}. "
            "Renommez-les : chaque colonne doit avoir un nom unique.")
    table = body[keep].copy()
    table.columns = names
    return table, info


def _read_csv_rows(path: str, info: Dict[str, Any]) -> "pd.DataFrame":
    """Lit un ``.csv`` brut : séparateur ``,`` ou ``;`` détecté, encodage UTF-8 ou cp1252."""
    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            with open(path, "r", encoding=encoding) as stream:
                head = ""
                for line in stream:
                    if line.strip():
                        head = line
                        break
            separator = ";" if head.count(";") > head.count(",") else ","
            info["decimal_comma"] = separator == ";"
            info["separator"] = separator
            return pd.read_csv(path, header=None, dtype=object, sep=separator,
                               encoding=encoding, skip_blank_lines=True)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ConfigError(
        f"Le fichier '{os.path.basename(path)}' n'est ni en UTF-8 ni en cp1252 "
        f"({last_error}). Réenregistrez-le en UTF-8.")


def _coerce_numeric(series: "pd.Series", decimal_comma: bool) -> "pd.Series":
    """Convertit une colonne en nombres ; ce qui n'est pas un nombre devient ``NaN``."""
    def clean(value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            if text == "":
                return np.nan
            return text.replace(",", ".") if decimal_comma else text
        return value
    return pd.to_numeric(series.map(clean), errors="coerce")


def _cell_present(value: Any) -> bool:
    """Vrai si une cellule n'est ni nulle, ni ``NaN``, ni une chaîne blanche."""
    return not (value is None or (isinstance(value, float) and math.isnan(value))
                or (isinstance(value, str) and not value.strip()))


def _present_mask(series: "pd.Series") -> "pd.Series":
    """Cellules non vides (ni nulles ni chaîne blanche)."""
    return series.map(_cell_present).astype(bool)


@dataclass
class _ColumnKinds:
    """Nature des colonnes d'un classeur lu."""

    names: List[str]
    numeric: Dict[str, "pd.Series"] = field(default_factory=dict)
    textual: List[str] = field(default_factory=list)
    empty: List[str] = field(default_factory=list)
    mixed: Dict[str, List[Tuple[int, Any]]] = field(default_factory=dict)
    mostly_numeric: List[str] = field(default_factory=list)


def _classify_columns(table: "pd.DataFrame", decimal_comma: bool) -> _ColumnKinds:
    """Classe chaque colonne : numérique, textuelle, vide ou mixte."""
    kinds = _ColumnKinds(names=list(table.columns))
    for name in table.columns:
        raw = table[name]
        present = _present_mask(raw)
        coerced = _coerce_numeric(raw, decimal_comma)
        n_present = int(present.sum())
        n_ok = int((coerced.notna() & present).sum())
        if n_present == 0:
            kinds.empty.append(name)
        elif n_ok == n_present:
            kinds.numeric[name] = coerced.astype(float)
        elif n_ok == 0:
            kinds.textual.append(name)
        else:
            bad = [(int(pos), raw.iloc[pos]) for pos in
                   np.flatnonzero((present & coerced.isna()).to_numpy())]
            kinds.mixed[name] = bad
            if n_ok / n_present > 0.5:
                kinds.mostly_numeric.append(name)
    return kinds


def _cells_text(bad: List[Tuple[int, Any]], limit: int = 4) -> str:
    """Cellules fautives : ligne du classeur (en-tête en ligne 1) et contenu."""
    shown = "; ".join(f"ligne {pos + 2} : {str(value)[:20]!r}" for pos, value in bad[:limit])
    extra = len(bad) - limit
    return shown + (f" (et {extra} autre(s))" if extra > 0 else "")


def resolve_columns(
    declared_zones: List[str],
    declared_features: List[str],
    kinds: _ColumnKinds,
    source: str,
) -> ColumnResolution:
    """Résout les listes de zones et de variables d'après les colonnes du classeur.

    Règles :

        * les deux listes déclarées : chaque nom doit exister, être numérique et
          n'appartenir qu'à une liste ;
        * seules les variables déclarées : les zones sont les colonnes numériques
          qui ne sont pas des variables ;
        * seules les zones déclarées : les variables sont les colonnes numériques qui
          ne sont pas des zones ;
        * aucune des deux : toutes les colonnes numériques sont des zones, il n'y a
          aucune variable de conception (avertissement).

    Les colonnes textuelles ou vides ne sont jamais déduites. Une colonne mixte, en
    majorité numérique, est refusée (faute de saisie probable) ; sinon elle est
    ignorée avec un avertissement.

    Args:
        declared_zones: Zones déclarées (éventuellement vide).
        declared_features: Variables déclarées (éventuellement vide).
        kinds: Nature des colonnes du classeur.
        source: Nom du classeur, pour les messages.

    Returns:
        La résolution, avec ses messages.

    Raises:
        ConfigError: nom introuvable ou non numérique, ou aucune colonne à déduire.
    """
    messages: List[Tuple[str, str]] = []
    available = list(kinds.names)
    for label, names in (("critical_zones", declared_zones),
                         ("feature_cols_init", declared_features)):
        missing = [name for name in names if name not in available]
        if missing:
            raise ConfigError(
                f"Colonne(s) introuvable(s) dans '{source}' pour '{label}' : {missing}. "
                f"Colonnes disponibles : {available}.")
        for name in names:
            if name in kinds.mixed:
                raise ConfigError(
                    f"La colonne '{name}' ('{label}') contient du texte dans '{source}' : "
                    f"{_cells_text(kinds.mixed[name])}. Corrigez ces cellules.")
            if name in kinds.textual:
                raise ConfigError(
                    f"La colonne '{name}' ('{label}') ne contient pas de nombres dans "
                    f"'{source}'.")

    inferring = not declared_zones or not declared_features
    if inferring:
        suspicious = [name for name in kinds.mostly_numeric
                      if name not in declared_zones and name not in declared_features]
        if suspicious:
            name = suspicious[0]
            raise ConfigError(
                f"La colonne '{name}' de '{source}' est presque entièrement numérique mais "
                f"contient du texte : {_cells_text(kinds.mixed[name])}. Corrigez ces "
                "cellules, ou déclarez explicitement les zones et les variables.")
        for name in kinds.mixed:
            if name not in kinds.mostly_numeric:
                messages.append(("warning",
                                 f"Colonne '{name}' ignorée : texte et nombres mêlés "
                                 "(colonne de commentaire probable)."))
    candidates = [name for name in available if name in kinds.numeric]

    zones_inferred = features_inferred = False
    if declared_zones and declared_features:
        zones, features = list(declared_zones), list(declared_features)
    elif declared_features:
        zones = [name for name in candidates if name not in declared_features]
        features, zones_inferred = list(declared_features), True
    elif declared_zones:
        features = [name for name in candidates if name not in declared_zones]
        zones, features_inferred = list(declared_zones), True
    else:
        zones, features = list(candidates), []
        zones_inferred = features_inferred = True
        messages.append((
            "warning",
            "Ni 'critical_zones' ni 'feature_cols_init' ne sont renseignées : toutes les "
            f"colonnes numériques ({len(zones)}) sont traitées comme zones critiques et il "
            "n'y a aucune variable de conception. Renseignez l'une des deux listes pour "
            "distinguer variables et zones."))
    if not zones:
        raise ConfigError(
            f"Aucune colonne numérique de '{source}' ne reste pour les zones critiques "
            f"(variables : {features}). Déclarez 'critical_zones'.")
    check_reserved_names(zones, f"Colonnes de zones de '{source}'")
    check_reserved_names(features, f"Colonnes de variables de '{source}'")
    used = set(zones) | set(features)
    ignored = [name for name in available if name not in used]
    messages.append((
        "info",
        f"Colonnes résolues ({'déduites' if zones_inferred else 'déclarées'} pour les "
        f"zones, {'déduites' if features_inferred else 'déclarées'} pour les variables) : "
        f"zones = {zones} ; variables de conception = {features}."))
    if ignored:
        messages.append(("info", f"Colonnes du classeur non utilisées : {ignored}."))
    return ColumnResolution(
        zones=zones, features=features, zones_inferred=zones_inferred,
        features_inferred=features_inferred, ignored=ignored, source=source,
        messages=messages)


def _numeric_frame(kinds: _ColumnKinds, wanted: List[str], n_rows: int) -> "pd.DataFrame":
    """Tableau des colonnes numériques (en flottants) et des colonnes vides voulues."""
    columns: Dict[str, "pd.Series"] = {}
    for name in kinds.names:
        if name in kinds.numeric:
            columns[name] = kinds.numeric[name].astype(float)
        elif name in kinds.empty and name in wanted:
            columns[name] = pd.Series(np.nan, index=range(n_rows), dtype=float)
    return pd.DataFrame(columns, index=range(n_rows)).reset_index(drop=True)


def _declared_names(config: OCAConfig) -> List[str]:
    """Noms de colonnes déclarés : zones et variables de conception."""
    return list(config.critical_zones) + list(config.feature_cols_init)


def _expressions(config: OCAConfig) -> List[str]:
    """Expressions de la configuration : variables synthétiques et cible explicite."""
    expressions = list(config.user_defined_vars_cols) if config.user_defined_vars else []
    if not config.auto_target:
        expressions.append(config.synthetic_target)
    return expressions


def _per_zone_paths(config: OCAConfig) -> Dict[str, str]:
    """Chemins des fichiers bruts par-zone, ou dictionnaire vide si zones non déclarées."""
    paths: Dict[str, str] = {}
    for zone in config.critical_zones:
        if any(char in zone for char in "/\\:"):
            return {}
        paths[zone] = os.path.join(config.input_dir, f"dataSet_{zone}", f"dataSet_{zone}.xlsx")
    return paths


def check_data_sources(config: OCAConfig) -> None:
    """Vérifie qu'une source de données existe, avant tout nettoyage de sortie.

    Args:
        config: Configuration validée (sous-dossier actif).

    Raises:
        ConfigError: message de deux lignes renvoyant à ``--init`` et à ``examples/``
            si aucune source n'existe.
    """
    if os.path.isfile(config.input_dir):
        raise ConfigError(
            f"'input_dir' ('{config.input_dir}') désigne un fichier, pas un dossier : "
            "indiquez le dossier qui contient les sous-dossiers de données.")
    per_zone = _per_zone_paths(config)
    if per_zone and not config.is_multi_run and all(os.path.isfile(p) for p in per_zone.values()):
        return
    merged = config.merged_file_path()
    if os.path.isfile(merged):
        return
    hint = "" if config.critical_zones else (
        " ('critical_zones' est vide : les fichiers par-zone, qui exigent des zones "
        "déclarées, ne sont pas cherchés)")
    raise ConfigError(
        f"Aucune donnée trouvée : '{merged}' n'existe pas{hint}.\n"
        "Premier essai : python main_OCA.py --config examples/CAS_TEST_00/input.json "
        "(voir examples/README.md) ; votre propre étude : python main_OCA.py --init.")


def _merge_per_zone(
    config: OCAConfig,
    per_zone_paths: Dict[str, str],
    logger: Optional[logging.Logger] = None,
) -> Tuple["pd.DataFrame", ColumnResolution]:
    """Fusionne les fichiers bruts par-zone (logique d'origine du notebook).

    Pour chaque zone : renomme la colonne cible (dernière colonne) avec le nom de la
    zone, construit un ``config_id`` à partir des variables, déduplique (première
    occurrence), puis restreint à l'intersection des configurations communes à
    toutes les zones et concatène variables et cibles. Les doublons retirés, les
    doublons aux valeurs contradictoires et les configurations hors intersection
    sont comptés et consignés au journal.

    Args:
        config: Configuration validée.
        per_zone_paths: Association zone vers chemin du fichier brut.
        logger: Journal.

    Returns:
        ``(tableau fusionné, résolution des colonnes)``.

    Raises:
        ConfigError: fichier illisible, variable introuvable, intersection vide.
    """
    def log(level: str, text: str) -> None:
        if logger:
            getattr(logger, level)(text)

    target_frames: Dict[str, pd.DataFrame] = {}
    indices_list: List[set] = []
    df_features = None
    feature_names: List[str] = []
    declared = list(config.feature_cols_init)
    first_source = ""
    check_reserved_names(list(config.critical_zones), "'critical_zones'")
    for reserved in ("config_id",):
        if reserved in config.critical_zones or reserved in declared:
            raise ConfigError(
                f"Fusion par-zone : le nom '{reserved}' est réservé à la clé de fusion ; "
                "renommez cette colonne.")

    for zone in config.critical_zones:
        table, info = _read_raw_table(
            per_zone_paths[zone], cited=_declared_names(config),
            expressions=_expressions(config))
        source = os.path.basename(per_zone_paths[zone])
        names = list(table.columns)
        if len(names) < 2:
            raise ConfigError(
                f"Le fichier par-zone '{source}' doit porter au moins une variable et une "
                "colonne cible (la dernière).")
        target_col = names[-1]
        if declared:
            missing = [name for name in declared if name not in names[:-1]]
            if missing:
                raise ConfigError(
                    f"Variable(s) {missing} ('feature_cols_init') introuvable(s) dans "
                    f"'{source}' (hors colonne cible '{target_col}'). Colonnes : {names}.")
            feature_cols = list(declared)
        elif not feature_names:
            # Variables inférées : l'ordre et l'ensemble sont fixés par le premier
            # fichier (toutes les colonnes sauf la dernière).
            feature_cols = names[:-1]
            first_source = source
            check_reserved_names(feature_cols, f"Variables de '{source}'")
            if "config_id" in feature_cols:
                raise ConfigError(
                    f"Fusion par-zone : le nom 'config_id' est réservé à la clé de fusion ; "
                    f"renommez cette colonne de '{source}'.")
        else:
            # Les fichiers suivants doivent porter exactement le même ensemble de
            # variables ; leurs colonnes sont réordonnées sur le premier fichier avant
            # de construire les clés, de sorte qu'une permutation de colonnes entre
            # deux zones ne change rien au résultat.
            others = names[:-1]
            absent = [name for name in feature_names if name not in others]
            extra = [name for name in others if name not in feature_names]
            if absent or extra:
                raise ConfigError(
                    f"Fusion par-zone : les variables de '{source}' diffèrent de celles de "
                    f"'{first_source}' (fixées par le premier fichier, toutes les colonnes "
                    f"sauf la dernière : {feature_names}). Colonne(s) absente(s) de "
                    f"'{source}' : {absent} ; colonne(s) en trop : {extra}. Corrigez les "
                    "fichiers ou déclarez 'feature_cols_init'.")
            feature_cols = list(feature_names)
        kinds = _classify_columns(table[feature_cols + [target_col]], info["decimal_comma"])
        for name in feature_cols + [target_col]:
            if name in kinds.mixed or name in kinds.textual:
                bad = kinds.mixed.get(name, [])
                raise ConfigError(
                    f"La colonne '{name}' de '{source}' n'est pas numérique"
                    + (f" : {_cells_text(bad)}." if bad else "."))
        df = _numeric_frame(kinds, feature_cols + [target_col], len(table))
        df = df.rename(columns={target_col: zone})

        df["config_id"] = df[feature_cols].astype(str).agg("_".join, axis=1)
        n_rows = len(df)
        conflicts = int((df.groupby("config_id")[zone].nunique(dropna=False) > 1).sum())
        n_duplicates = int(df["config_id"].duplicated().sum())
        if n_duplicates:
            df = df.drop_duplicates(subset="config_id", keep="first")
            log("info", f"Fusion par-zone, zone '{zone}' : {n_duplicates} doublon(s) de "
                        f"configuration retiré(s) sur {n_rows} ligne(s) (première occurrence "
                        "conservée).")
        if conflicts:
            log("warning",
                f"Fusion par-zone, zone '{zone}' : {conflicts} configuration(s) répétée(s) "
                "avec des valeurs différentes ; la première occurrence est conservée, le "
                "résultat dépend de l'ordre des lignes. Corrigez les doublons du fichier.")
        df = df.set_index("config_id")

        indices_list.append(set(df.index))
        if df_features is None:
            df_features = df[feature_cols].copy()
            feature_names = list(feature_cols)
        target_frames[zone] = df[[zone]]

    common_indices = sorted(list(reduce(set.intersection, indices_list)))
    if not common_indices:
        raise ConfigError(
            "Fusion par-zone : aucune configuration n'est commune à toutes les zones. "
            "Vérifiez que les fichiers portent les mêmes variables et les mêmes valeurs.")
    for zone, index in zip(config.critical_zones, indices_list):
        outside = len(index) - len(common_indices)
        if outside:
            log("warning", f"Fusion par-zone, zone '{zone}' : {outside} configuration(s) "
                           "absente(s) d'au moins une autre zone, exclue(s) de l'intersection.")
    df_features = df_features.loc[common_indices]
    for zone in config.critical_zones:
        target_frames[zone] = target_frames[zone].loc[common_indices]

    df_targets = pd.concat([target_frames[zone] for zone in config.critical_zones], axis=1)
    df_merged = pd.concat([df_features, df_targets], axis=1).reset_index(drop=True)
    resolution = ColumnResolution(
        zones=list(config.critical_zones), features=feature_names,
        zones_inferred=False, features_inferred=not declared,
        source="fichiers par-zone", sheet="(première feuille de chaque fichier)")
    resolution.messages.append((
        "info",
        f"Fusion par-zone : {len(common_indices)} configuration(s) commune(s) ; variables "
        f"{'déclarées' if declared else 'déduites (toutes les colonnes sauf la dernière)'} "
        f"= {feature_names}."))
    return df_merged, resolution


def load_data_with_resolution(
    config: OCAConfig, logger: Optional[logging.Logger] = None
) -> Tuple["pd.DataFrame", List[str], ColumnResolution]:
    """Charge les données, résout les colonnes, calcule les variables synthétiques.

    Mode fusion si tous les fichiers bruts par-zone existent (zones déclarées),
    sinon lecture du classeur fusionné (première feuille d'un ``.xlsx``, ou
    ``.csv``). Ajoute les variables synthétiques déclarées puis, si la cible
    explicite est une expression non déclarée, la calcule. La cible ``"none"``
    (sélection automatique) est résolue plus tard, après le classement.

    Args:
        config: Configuration validée.
        logger: Journal optionnel.

    Returns:
        ``(df_merged, critical_zones, resolution)`` : ``critical_zones`` inclut les
        variables synthétiques ; ``resolution`` consigne les listes de colonnes
        effectivement utilisées.

    Raises:
        ConfigError: si aucune source valide n'est trouvée ou si une colonne ou une
            expression est inutilisable.
    """
    per_zone = _per_zone_paths(config)
    use_per_zone = (
        bool(per_zone)
        and not config.is_multi_run
        and all(os.path.isfile(path) for path in per_zone.values())
    )
    if use_per_zone:
        if logger:
            logger.info("Fusion des fichiers bruts par-zone.")
        df_merged, resolution = _merge_per_zone(config, per_zone, logger)
    else:
        merged_path = config.merged_file_path()
        if not os.path.isfile(merged_path):
            check_data_sources(config)
        if logger:
            logger.info(f"Lecture du fichier fusionné : {merged_path}")
        table, info = _read_raw_table(
            merged_path, cited=_declared_names(config), expressions=_expressions(config))
        kinds = _classify_columns(table, info["decimal_comma"])
        source = os.path.basename(merged_path)
        resolution = resolve_columns(
            list(config.critical_zones), list(config.feature_cols_init), kinds, source)
        resolution.sheet = info["sheet"] or "(fichier .csv)"
        resolution.index_column = info["index_column"]
        wanted = resolution.zones + resolution.features
        df_merged = _numeric_frame(kinds, wanted, len(table))
        if logger:
            if info["sheet"]:
                logger.info(f"Classeur '{source}' : première feuille '{info['sheet']}' lue "
                            "(les autres feuilles ne sont pas lues).")
            if info.get("trailing_empty_rows"):
                logger.info(f"{info['trailing_empty_rows']} ligne(s) entièrement vide(s) en "
                            "fin de feuille ignorée(s).")
            if info["index_column"] is not None:
                logger.info(f"Première colonne '{info['index_column']}' reconnue comme index "
                            "d'observations et retirée (si c'est une grandeur à analyser, "
                            "citez-la dans 'critical_zones' ou 'feature_cols_init').")
            elif info.get("index_kept"):
                logger.info(f"Première colonne '{info['index_kept']}' conservée : son nom est "
                            "celui d'un index usuel, mais elle est citée dans la "
                            "configuration (zones, variables ou expression).")
            else:
                logger.info("Première colonne conservée : elle n'est pas un index (en-tête "
                            "présent et différent de index, id, config, config_id, n).")
    if logger:
        for level, text in resolution.messages:
            getattr(logger, level)(text)

    # Sauvegarde du fichier fusionné dans la sortie (traçabilité). Le nom suit celui
    # déclaré dans la configuration ; un fichier .csv est enregistré en .xlsx.
    main_dir = prepare_output_dirs(config)
    saved_name = config.data_files
    if os.path.splitext(saved_name)[1].lower() != ".xlsx":
        saved_name = os.path.splitext(saved_name)[0] + ".xlsx"
    with pd.ExcelWriter(os.path.join(main_dir, saved_name), mode="w") as writer:
        df_merged.to_excel(writer, sheet_name="criticalZones")

    # Variables synthétiques déclarées (ex. un rapport) ajoutées aux zones critiques.
    critical_zones = list(resolution.zones)
    if config.user_defined_vars:
        for expr in config.user_defined_vars_cols:
            if expr not in df_merged.columns:
                try:
                    df_merged[expr] = df_merged.eval(expr)
                except Exception as exc:  # noqa: BLE001 - expression invalide
                    raise ConfigError(
                        f"La variable synthétique '{expr}' ('user_defined_vars_cols') n'a "
                        f"pas pu être calculée ({type(exc).__name__}). Colonnes disponibles : "
                        f"{list(df_merged.columns)}.") from None
                # Une expression divisant par zéro ne lève pas : elle produit
                # l'infini. Or « x <= inf » est vrai partout, si bien que l'artefact
                # dominerait toutes les zones réelles à 1,000 dans la matrice.
                non_finite = int((~np.isfinite(
                    pd.to_numeric(df_merged[expr], errors="coerce"))).sum())
                if non_finite:
                    raise ConfigError(
                        f"La variable synthétique '{expr}' produit {non_finite} valeur(s) "
                        f"non finie(s) sur {len(df_merged)} observation(s) : le dénominateur "
                        "s'annule ou approche zéro. Une valeur infinie majorerait toutes "
                        "les autres colonnes : le calcul est refusé.")
                if logger:
                    logger.info(f"Variable synthétique déclarée calculée : '{expr}'.")
            if expr not in critical_zones:
                critical_zones.append(expr)

    # Ajout transparent de la cible explicite si synthétique/oubliée (hors "none").
    tz = config.synthetic_target
    if not config.auto_target and tz not in critical_zones:
        if tz not in df_merged.columns:
            try:
                df_merged[tz] = df_merged.eval(tz)
            except Exception as exc:  # noqa: BLE001 - colonne inexistante et non calculable
                raise ConfigError(
                    f"'gpo_regularization.target_zone' = '{tz}' : colonne introuvable et "
                    f"expression non calculable ({type(exc).__name__}). Colonnes "
                    f"disponibles : {list(df_merged.columns)} ; ou utilisez \"none\" pour "
                    "une cible automatique.") from None
            if logger:
                logger.warning(
                    f"target_zone synthétique '{tz}' non déclarée dans "
                    "'user_defined_vars_cols' : calculée et ajoutée automatiquement.")
        critical_zones.append(tz)

    # Certification du jeu de travail : effectif, nombre de zones, valeurs
    # manquantes ou infinies. Elle porte sur les colonnes effectivement comparées.
    _certify_working_table(df_merged, critical_zones, logger)

    return df_merged, critical_zones, resolution


def load_and_merge_data(
    config: OCAConfig, logger: Optional[logging.Logger] = None
) -> Tuple["pd.DataFrame", List[str]]:
    """Charge les données et calcule les variables synthétiques.

    Voir :func:`load_data_with_resolution`, dont cette fonction ne retient que le
    tableau et la liste des zones.

    Args:
        config: Configuration validée.
        logger: Logger optionnel pour tracer les avertissements et événements.

    Returns:
        ``(df_merged, critical_zones)`` ; ``critical_zones`` inclut les variables
        synthétiques et la cible ajoutée.
    """
    df_merged, critical_zones, _ = load_data_with_resolution(config, logger)
    return df_merged, critical_zones
