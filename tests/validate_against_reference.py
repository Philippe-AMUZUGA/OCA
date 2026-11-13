"""Validation de non régression du pipeline OCA contre le cas test.

Exécute le pipeline puis compare ses sorties textuelles et LaTeX aux références de
``examples/CAS_TEST_00/dataSet_criticalZones`` et vérifie les valeurs clés (κ, β).
La comparaison des fichiers texte est faite à l'identique (octet près), les
tableaux LaTeX sont comparés sur leur contenu logique ; les valeurs numériques sont
en outre extraites et contrôlées.

Portée exacte : les références sont l'état de la version 8.0, les sorties du calcul
d'origine n'ayant pas été conservées. Le harnais établit donc l'invariance depuis
la version 8.0, non l'exactitude absolue (celle-ci relève de
``validate_pipeline.py``).

Usage, depuis la racine du dépôt :
    python tests/validate_against_reference.py            # calcule puis valide
    python tests/validate_against_reference.py --no-run   # valide des sorties déjà calculées
    python tests/validate_against_reference.py --config examples/CAS_TEST_00/input.json
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
from typing import List, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

#: Configuration de l'exemple livré, chemins relatifs à la racine du dépôt.
DEFAULT_CONFIG_PATH = "examples/CAS_TEST_00/input.json"

#: Dossier des references de non-regression. Il est CHERCHE et non presume : le
#: dossier 'tests' est livre a la fois dans 'CAS_TEST_00' et dans 'GitHub', et une
#: composition figee ne pouvait valoir pour les deux. Une ecriture anterieure
#: designait 'PROJECT_ROOT/CAS_TEST_00/dataSet_criticalZones', chemin inexistant
#: depuis l'une comme depuis l'autre : le harnais rendait alors un succes sans
#: avoir rien compare.
_REFERENCE_CANDIDATES = (
    os.path.join(PROJECT_ROOT, "dataSet_criticalZones"),
    # Disposition du depot : les exemples portent leurs propres references, et le
    # code reste a la racine. Le harnais s'y lance donc depuis la racine.
    os.path.join(PROJECT_ROOT, "examples", "CAS_TEST_00", "dataSet_criticalZones"),
    os.path.join(PROJECT_ROOT, "CAS_TEST_00", "dataSet_criticalZones"),
    os.path.join(os.path.dirname(PROJECT_ROOT), "CAS_TEST_00",
                 "dataSet_criticalZones"),
)
REFERENCE_DIR = next(
    (candidate for candidate in _REFERENCE_CANDIDATES if os.path.isdir(candidate)),
    _REFERENCE_CANDIDATES[0],
)
REFERENCE_DIR_FOUND = os.path.isdir(REFERENCE_DIR)
if not REFERENCE_DIR_FOUND:
    print(
        "[REF][ERREUR] Dossier de references introuvable. Emplacements essayes :\n"
        + "\n".join(f"    {candidate}" for candidate in _REFERENCE_CANDIDATES)
        + "\nAucune comparaison ne peut etre conduite : le harnais s'arrete plutot "
          "que de rendre un succes vide.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def build_name_map(config) -> "dict[str, str]":
    """Construit le mapping fichier produit -> fichier de référence.

    Les fichiers de sortie sont renommés/numérotés (ex. ``01-Step1_...``) mais
    leur contenu doit rester identique aux références d'origine.

    Args:
        config: Configuration OCA (fournit ``output_files``).

    Returns:
        Dictionnaire ``{nom_produit: nom_reference}``.
    """
    of = config.output_files
    return {
        of["sort_txt"]: "OCA_Sort_Targets.txt",
        of["ordinal_graph_txt"]: "OCA_OrdinalGraph_Targets.txt",
        of["step1_result"] + ".tex": "Result.tex",
        of["criticalzones"] + ".tex": "CriticalZones_Targets.tex",
    }


def _read_text(path: str) -> str:
    """Lit un fichier texte en essayant UTF-8 puis cp1252.

    Les ``.txt`` de référence sont en cp1252, les ``.tex`` en UTF-8.

    Args:
        path: Chemin du fichier.

    Returns:
        Contenu décodé.
    """
    with open(path, "rb") as stream:
        data = stream.read()
    for enc in ("utf-8", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


#: Début de la phrase de lecture de la notation ``A < B``, ajoutée en tête des
#: relevés de graphe depuis la version 9.0. Les références sont celles de la
#: version 8.0 : cette seule ligne est écartée avant la comparaison.
NOTATION_PREFIX = "Lecture : « A < B »"


def _without_added_header(data: bytes) -> bytes:
    """Retire d'un relevé texte (cp1252) la phrase de lecture ajoutée en 9.0."""
    text = data.decode("cp1252", errors="replace")
    kept = [line for line in text.splitlines(keepends=True)
            if not line.startswith(NOTATION_PREFIX)]
    return "".join(kept).encode("cp1252", errors="replace")


def compare_text(produced_name: str, ref_name: str, produced_dir: str) -> List[str]:
    """Compare un fichier produit à sa référence, octet par octet.

    La comparaison binaire garantit une identité stricte, y compris l'encodage
    (cp1252 pour les ``.txt``, UTF-8 pour les ``.tex``). Le contenu doit être
    identique même si le fichier a été renommé/numéroté. Seule la phrase de
    lecture de la notation ``A < B``, ajoutée en 9.0, est ignorée (voir
    :data:`NOTATION_PREFIX`).

    Args:
        produced_name: Nom du fichier produit.
        ref_name: Nom du fichier de référence correspondant.
        produced_dir: Dossier de sortie produit.

    Returns:
        Liste d'écarts (vide si identique octet pour octet).
    """
    ref_path = os.path.join(REFERENCE_DIR, ref_name)
    got_path = os.path.join(produced_dir, produced_name)
    if not os.path.isfile(ref_path):
        return [f"Référence absente : {ref_path}"]
    if not os.path.isfile(got_path):
        return [f"Sortie absente : {got_path}"]

    with open(ref_path, "rb") as ref_stream:
        ref_bytes = ref_stream.read()
    with open(got_path, "rb") as got_stream:
        got_bytes = got_stream.read()
    if produced_name.endswith(".txt"):
        got_bytes = _without_added_header(got_bytes)
    if ref_bytes == got_bytes:
        return []
    diff = list(difflib.unified_diff(
        _read_text(ref_path).splitlines(), _read_text(got_path).splitlines(),
        fromfile=f"ref/{ref_name}", tofile=f"out/{produced_name}", lineterm=""))
    return diff[:30] if diff else [f"Diff binaire ({len(ref_bytes)} vs {len(got_bytes)} octets) sans diff texte"]


def parse_latex_table(text: str) -> Tuple[List[str], List[str], "dict[str, List[str]]"]:
    """Reconstruit le tableau logique porté par un document LaTeX.

    Un tableau peut être rendu d'un seul tenant ou, depuis la version 6, découpé
    en plusieurs blocs empilés lorsque son encombrement horizontal excède le
    seuil de lisibilité. Les deux rendus décrivent le même tableau logique : la
    concaténation des blocs, dans l'ordre, restitue l'intégralité des colonnes,
    et chaque bloc répète les mêmes libellés de lignes.

    Args:
        text: Contenu du document LaTeX.

    Returns:
        Triplet ``(colonnes, ordre_des_lignes, valeurs)`` où ``valeurs`` associe
        à chaque libellé de ligne la suite de ses cellules, colonne par colonne.
    """
    columns: List[str] = []
    row_order: List[str] = []
    values: "dict[str, List[str]]" = {}

    for block in re.findall(r"\\toprule\s*\n(.*?)\\bottomrule", text, flags=re.S):
        lines = [line.strip() for line in block.strip().splitlines()
                 if line.strip() and line.strip() != r"\midrule"]
        if not lines:
            continue
        strip_row_end = lambda item: re.sub(r"\\\\\s*$", "", item).strip()
        header = [cell.strip() for cell in strip_row_end(lines[0]).split("&")]
        columns.extend(header[1:])
        for line in lines[1:]:
            cells = [cell.strip() for cell in strip_row_end(line).split("&")]
            label = cells[0]
            if label not in values:
                values[label] = []
                row_order.append(label)
            values[label].extend(cells[1:])
    return columns, row_order, values


# Nombres désignant un rang de division du texte ou du papier, et non une
# grandeur mesurée : numéro d'étape de l'algorithme, de section, d'algorithme, de
# table ou de figure de la référence bibliographique.
_ORDINAL_REFERENCE = re.compile(
    r"(?:étapes?|etapes?|sections?|algorithmes?|tables?|figures?|équations?"
    r"|equations?|volume)\s*[0-9]+(?:\.[0-9]+)*", flags=re.I)


def caption_quantities(caption: str) -> List[str]:
    """Extrait d'une légende les seules grandeurs qu'elle rapporte.

    Les légendes de la version 6 précisent la méthode, le périmètre statistique
    et la référence bibliographique. Ces précisions introduisent des nombres qui
    ne rapportent aucun résultat : volume et année de la revue, numéro d'article,
    numéros de section, d'algorithme, de table et d'étape. Les confronter aux
    légendes historiques ferait échouer la comparaison sur la seule présence
    d'une citation, ce qui n'atteste rien de la fidélité numérique. Ils sont donc
    retranchés avant l'extraction, tandis que toute grandeur du calcul, effectif
    examiné, coefficient de majoration ou indice de parcimonie, est conservée.

    Args:
        caption: Légende LaTeX, telle qu'elle figure dans le document.

    Returns:
        Nombres rapportés par la légende, dans leur ordre d'apparition.
    """
    text = re.split(r"Voir Amuzuga", caption)[0]
    text = _ORDINAL_REFERENCE.sub(" ", text)
    return re.findall(r"[0-9]+(?:\.[0-9]+)?", text)


def compare_table_content(produced_name: str, ref_name: str, produced_dir: str,
                          ordered_columns: bool = True) -> List[str]:
    """Compare deux tableaux LaTeX sur leur contenu logique, non sur leur mise en page.

    La comparaison octet par octet reste la règle. Elle ne peut toutefois
    s'appliquer aux tableaux soumis au découpage adaptatif de la version 6, dont
    la mise en page diffère volontairement de celle des références historiques
    dès que le nombre de colonnes excède le seuil de lisibilité. Pour ceux là, la
    fidélité est établie sur ce qui la constitue réellement : la légende, la
    liste ordonnée des colonnes, l'ordre des lignes et la valeur de chaque
    cellule.

    Args:
        produced_name: Nom du fichier produit.
        ref_name: Nom du fichier de référence correspondant.
        produced_dir: Dossier de sortie produit.
        ordered_columns: Impose l'identité de l'ordre des colonnes. Mettre à
            ``False`` lorsque la version 6 réordonne délibérément les colonnes,
            par exemple par fréquence de dominance croissante : la comparaison
            porte alors sur l'ensemble des colonnes et sur la valeur de chaque
            cellule, appariée par nom de colonne. C'est précisément ce qui établit
            que la réorganisation est une permutation pure, sans perte ni
            décalage de données.

    Returns:
        Liste d'écarts (vide si les deux tableaux logiques coïncident).
    """
    ref_path = os.path.join(REFERENCE_DIR, ref_name)
    got_path = os.path.join(produced_dir, produced_name)
    if not os.path.isfile(ref_path):
        return [f"Référence absente : {ref_path}"]
    if not os.path.isfile(got_path):
        return [f"Sortie absente : {got_path}"]

    ref_text, got_text = _read_text(ref_path), _read_text(got_path)
    issues: List[str] = []

    ref_cols, ref_order, ref_values = parse_latex_table(ref_text)
    got_cols, got_order, got_values = parse_latex_table(got_text)

    # Les légendes sont enrichies en version 6 : elles précisent désormais l'ordre
    # des colonnes, le périmètre statistique des indicateurs et la référence
    # bibliographique de la méthode. La comparaison porte donc sur les grandeurs
    # qu'elles rapportent, seules porteuses du résultat, et non sur leur libellé
    # littéral.
    #
    # La référence bibliographique est retranchée avant l'extraction : volume,
    # année, numéro d'article et numéros de section y forment des nombres qui ne
    # rapportent aucune grandeur du calcul. Les confronter aux références
    # historiques ferait échouer la comparaison sur la seule présence d'une
    # citation, ce qui n'atteste rien de la fidélité numérique.
    caption = lambda text: re.search(r"\\caption\{(.*?)\}\s*\n", text, flags=re.S)
    ref_caption = caption(ref_text).group(1).strip() if caption(ref_text) else ""
    got_caption = caption(got_text).group(1).strip() if caption(got_text) else ""
    ref_numbers = caption_quantities(ref_caption)
    got_numbers = caption_quantities(got_caption)
    if ref_numbers != got_numbers:
        issues.append(
            f"grandeurs de la légende : {ref_numbers} vs {got_numbers}")

    if ordered_columns:
        if ref_cols != got_cols:
            issues.append(f"colonnes : {ref_cols} vs {got_cols}")
    elif sorted(ref_cols) != sorted(got_cols):
        issues.append(
            f"ensemble des colonnes : {sorted(ref_cols)} vs {sorted(got_cols)}")

    if ref_order != got_order:
        issues.append(f"ordre des lignes : {ref_order} vs {got_order}")

    if ordered_columns:
        for label in ref_order:
            if ref_values.get(label) != got_values.get(label):
                issues.append(
                    f"ligne '{label}' : {ref_values.get(label)} vs {got_values.get(label)}")
        return issues

    # Comparaison cellule par cellule, appariée par nom de colonne.
    for label in ref_order:
        reference = dict(zip(ref_cols, ref_values.get(label, [])))
        obtained = dict(zip(got_cols, got_values.get(label, [])))
        for column, expected in reference.items():
            if obtained.get(column) != expected:
                issues.append(
                    f"ligne '{label}', colonne '{column}' : "
                    f"{expected!r} vs {obtained.get(column)!r}")
    return issues


def _is_split(produced_dir: str, produced_name: str) -> bool:
    """Indique si un tableau produit a été rendu en plusieurs blocs empilés.

    Args:
        produced_dir: Dossier de sortie produit.
        produced_name: Nom du fichier LaTeX produit.

    Returns:
        ``True`` si le document porte plus d'un bloc tabulaire.
    """
    path = os.path.join(produced_dir, produced_name)
    if not os.path.isfile(path):
        return False
    return len(re.findall(r"\\toprule", _read_text(path))) > 1


def _extract_kappa(text: str) -> List[str]:
    """Extrait toutes les valeurs κ (motif ``N.NN$\\times$``) d'un texte.

    Args:
        text: Contenu texte.

    Returns:
        Liste des valeurs κ trouvées.
    """
    return re.findall(r"([0-9]+\.[0-9]{2})\$\\times\$", text)


def check_key_values(produced_dir: str, name_map: "dict[str, str]") -> List[str]:
    """Vérifie κ et β entre sortie et référence.

    Args:
        produced_dir: Dossier de sortie produit.
        name_map: Mapping fichier produit -> fichier de référence.

    Returns:
        Liste d'écarts (vide si conformes).
    """
    issues: List[str] = []
    # Retrouver les noms produits pour le graphe et le tableau CriticalZones.
    graph_name = next((p for p, r in name_map.items() if r == "OCA_OrdinalGraph_Targets.txt"), None)
    cz_name = next((p for p, r in name_map.items() if r == "CriticalZones_Targets.tex"), None)

    # κ depuis le graphe ordinal.
    ref_graph = _read_text(os.path.join(REFERENCE_DIR, "OCA_OrdinalGraph_Targets.txt"))
    got_graph = _read_text(os.path.join(produced_dir, graph_name)) if graph_name else ""
    ref_k, got_k = set(_extract_kappa(ref_graph)), set(_extract_kappa(got_graph))
    if ref_k != got_k:
        issues.append(f"κ divergent (graphe) : ref={sorted(ref_k)} vs sortie={sorted(got_k)}")

    def caption_vals(text: str) -> Tuple[str, str, str]:
        alpha = re.search(r"alpha=\$([0-9.]+)", text)
        bmax = re.search(r"maximale de \$\\beta\$ est ([0-9.]+)", text)
        bmean = re.search(r"moyenne de \$\\beta\$ est ([0-9.]+)", text)
        return (alpha.group(1) if alpha else "?", bmax.group(1) if bmax else "?", bmean.group(1) if bmean else "?")

    ref_vals = caption_vals(_read_text(os.path.join(REFERENCE_DIR, "CriticalZones_Targets.tex")))
    got_vals = caption_vals(_read_text(os.path.join(produced_dir, cz_name))) if cz_name else ("?", "?", "?")
    if ref_vals != got_vals:
        issues.append(f"(α, β_max, β_moyen) divergents : ref={ref_vals} vs sortie={got_vals}")
    return issues


def main() -> int:
    """Exécute (optionnellement) le pipeline puis lance toutes les comparaisons.

    Returns:
        Code de sortie 0 si toutes les validations réussissent, 1 sinon.
    """
    parser = argparse.ArgumentParser(description="Validation OCA contre le cas test.")
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH,
        help="Configuration de l'exemple à valider (défaut : %(default)s).")
    parser.add_argument(
        "--no-run", action="store_true",
        help="Comparer des sorties déjà calculées au lieu de relancer le calcul.")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    os.chdir(PROJECT_ROOT)
    from config_manager import ConfigError, load_config

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"[REF][ERREUR] {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    produced_dir = config.output_dir_main()
    name_map = build_name_map(config)

    # Sans '--no-run', le calcul est refait AVANT toute vérification de présence des
    # sorties : sur un dépôt fraîchement cloné elles n'existent pas encore, et c'est
    # précisément ce lancement qui les produit. Le contrôle de présence ne vaut donc
    # qu'après le calcul, ou avec '--no-run', qui compare des sorties déjà calculées.
    if not args.no_run:
        print("[VALIDATION] Exécution du pipeline OCA modulaire...")
        from main_OCA import run

        try:
            run(args.config)
        except ConfigError as exc:
            print(f"[REF][ERREUR] {exc}", file=sys.stderr)
            raise SystemExit(2) from None

    if not os.path.isdir(produced_dir):
        print(
            f"""[REF][ERREUR] Dossier de sortie introuvable : {produced_dir}
    Configuration lue : {args.config}

    Avec '--no-run', ce harnais compare des sorties DEJA calculees aux references.
    Calculez-les d'abord, depuis la racine du depot :

        python main_OCA.py --config {args.config}

    ou relancez ce harnais sans '--no-run' : il lance le calcul lui-meme.""",
            file=sys.stderr,
        )
        raise SystemExit(2)

    # Les tableaux de données sont soumis, depuis la version 6, à une mise en page
    # unifiée : taille de caractères fixe, aucune mise à l'échelle, et découpage en
    # blocs au delà du seuil de lisibilité. Leur présentation diffère donc
    # volontairement de celle des références historiques. La fidélité est alors
    # établie sur le contenu logique du tableau, seul porteur du résultat :
    # légende, liste ordonnée des colonnes, ordre des lignes et valeur de chaque
    # cellule. Les relevés textuels restent comparés octet par octet.
    # Le tableau de l'étape 1 conserve l'ordre de ses colonnes, le classement GPO
    # les disposant déjà par dominance croissante. Celui de l'étape 2 les
    # réordonne délibérément selon ce même classement : sa comparaison se fait
    # donc par appariement des noms de colonnes, ce qui établit que la
    # réorganisation est une permutation pure, sans décalage de données.
    layout_revised = {
        config.output_files["step1_result"] + ".tex": True,
        config.output_files["criticalzones"] + ".tex": False,
    }
    results = {}
    for produced, ref in name_map.items():
        if produced in layout_revised:
            ordered = layout_revised[produced]
            mention = "mise en page unifiee V6" if ordered else \
                "colonnes triees par dominance V6"
            label = f"{produced} (= {ref}) [contenu, {mention}]"
            results[label] = compare_table_content(
                produced, ref, produced_dir, ordered_columns=ordered)
        else:
            results[f"{produced} (= {ref})"] = compare_text(produced, ref, produced_dir)
    results["valeurs_cles (kappa, beta)"] = check_key_values(produced_dir, name_map)

    print("\n" + "=" * 64)
    print("RÉSULTAT DE LA VALIDATION OCA (comparaison au cas test)")
    print("=" * 64)
    total_fail = 0
    for name, issues in results.items():
        if issues:
            total_fail += 1
            print(f"[FAIL] {name} : {len(issues)} écart(s)")
            for line in issues[:15]:
                print(f"     {line}")
        else:
            print(f"[ OK ] {name} : identique aux références")
    print("=" * 64)
    if total_fail == 0:
        print("SUCCÈS : le pipeline OCA modularisé reproduit strictement les références locales.")
        return 0
    print(f"ÉCHEC : {total_fail} comparaison(s) en écart.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
