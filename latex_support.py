"""Détection et appel du compilateur LaTeX, partagés par tous les rapports OCA (9.0.0).

La qualité typographique des rapports PDF repose sur ``pdflatex``. Il n'existe pas
de paquet Python qui l'apporte avec la même composition : le programme ne tente donc
aucune installation silencieuse. Il cherche ``pdflatex`` (variable ``PATH`` puis
emplacements d'installation usuels de TeX Live et de MiKTeX), et, à défaut, avertit
l'utilisateur UNE fois, de façon exacte, sans interrompre le calcul : les figures,
les tableaux ``.tex``, les fichiers texte et ``output.json`` sont produits quand même.

Le réglage ``latex_output.compiler`` d'``input.json`` admet :

* ``"auto"`` (défaut) : ``pdflatex`` s'il est présent, sinon ``tectonic`` s'il est
  installé (repli signalé), sinon aucun rapport PDF compilé ;
* ``"pdflatex"`` : ``pdflatex`` uniquement ;
* ``"tectonic"`` : ``tectonic`` uniquement (moteur TeX autonome, composition proche
  mais non identique) ;
* ``"none"`` : aucune compilation, les sources ``.tex`` seules sont écrites.
"""

from __future__ import annotations

import glob
import os
import shutil
from typing import List, Optional, Tuple

__all__ = [
    "ALLOWED_COMPILERS",
    "configure",
    "find_latex_compiler",
    "compile_command",
    "missing_message",
    "warn_if_missing",
]

ALLOWED_COMPILERS = ("auto", "pdflatex", "tectonic", "none")

_PREFERENCE = "auto"
_WARNED = False


def configure(preference: str = "auto") -> None:
    """Fixe la préférence de compilateur pour la suite de l'exécution.

    Args:
        preference: ``"auto"``, ``"pdflatex"``, ``"tectonic"`` ou ``"none"``.

    Raises:
        ValueError: si la valeur n'est pas admise.
    """
    global _PREFERENCE, _WARNED
    value = str(preference or "auto").strip().lower()
    if value not in ALLOWED_COMPILERS:
        raise ValueError(
            f"'latex_output.compiler' vaut '{preference}' ; valeurs admises : "
            f"{list(ALLOWED_COMPILERS)}.")
    _PREFERENCE = value
    _WARNED = False


def _candidate_dirs() -> List[str]:
    """Dossiers d'installation usuels de TeX Live et de MiKTeX."""
    patterns = [
        r"C:\texlive\*\bin\windows",
        r"C:\texlive\*\bin\win64",
        r"C:\Program Files\MiKTeX\miktex\bin\x64",
        r"C:\Program Files (x86)\MiKTeX\miktex\bin",
        "/Library/TeX/texbin",
        "/usr/local/texlive/*/bin/*",
        "/usr/local/bin",
        "/opt/homebrew/bin",
        "/usr/bin",
    ]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        patterns.append(os.path.join(local, "Programs", "MiKTeX", "miktex", "bin", "x64"))
    found: List[str] = []
    for pattern in patterns:
        found.extend(sorted(glob.glob(pattern), reverse=True))
    return found


def _locate(program: str) -> Optional[str]:
    """Chemin d'un exécutable : ``PATH`` d'abord, puis emplacements usuels."""
    path = shutil.which(program)
    if path:
        return path
    for folder in _candidate_dirs():
        path = shutil.which(program, path=folder)
        if path:
            return path
    return None


def find_latex_compiler() -> Optional[Tuple[str, str]]:
    """Retourne ``(nature, chemin)`` du compilateur retenu, ou ``None``.

    La nature vaut ``"pdflatex"`` ou ``"tectonic"``. Le résultat dépend du réglage
    posé par :func:`configure`.
    """
    if _PREFERENCE == "none":
        return None
    if _PREFERENCE in ("auto", "pdflatex"):
        path = _locate("pdflatex")
        if path:
            return "pdflatex", path
    if _PREFERENCE in ("auto", "tectonic"):
        path = _locate("tectonic")
        if path:
            return "tectonic", path
    return None


def compile_command(kind: str, path: str, tex_name: str) -> List[str]:
    """Commande de compilation d'un fichier ``.tex`` du dossier courant."""
    if kind == "tectonic":
        return [path, "--keep-logs", tex_name]
    return [path, "-interaction=nonstopmode", tex_name]


def missing_message(program: str = "OCA") -> str:
    """Message exact et actionnable lorsque aucun compilateur n'est disponible."""
    if _PREFERENCE == "none":
        return (f"[{program}] Compilation LaTeX désactivée ('latex_output.compiler' = \"none\") : "
                "les sources .tex sont écrites, aucun rapport PDF n'est compilé.")
    return (
        f"[{program}] LaTeX (pdflatex) est introuvable sur ce poste. Les figures PDF, les "
        "tableaux .tex, les fichiers texte et output.json sont produits ; seuls les "
        "rapports PDF compilés manquent. Pour les obtenir, installez TeX Live "
        "(https://tug.org/texlive/) ou MiKTeX (https://miktex.org/download), puis relancez. "
        "pdflatex ne s'installe pas par pip et aucun paquet Python n'offre la même qualité "
        "de composition. Variante : si 'tectonic' est installé, 'latex_output.compiler' = "
        "\"tectonic\" produit les rapports avec une composition proche mais non identique.")


def warn_if_missing(program: str = "OCA", logger: object = None) -> bool:
    """Avertit une seule fois par exécution si aucun compilateur n'est disponible.

    Returns:
        ``True`` si un compilateur est disponible.
    """
    global _WARNED
    if find_latex_compiler() is not None:
        return True
    if not _WARNED:
        _WARNED = True
        message = missing_message(program)
        print(message)
        if logger is not None and hasattr(logger, "warning"):
            logger.warning(message)
    return False
