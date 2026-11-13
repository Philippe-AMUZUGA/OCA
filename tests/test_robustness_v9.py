"""Tests de robustesse d'OCA 9.0.0 (unittest, sans dépendance supplémentaire).

Ils couvrent les entrées que la version 8.1 traitait mal : configuration minimale,
champs vides ou nuls, mauvais types, chemins dangereux, classeurs sans index,
valeurs infinies, effectifs limites, colonnes constantes, grille de κ, noms de
zones à caractères spéciaux, absence de LaTeX, formats de fichiers et manifestes
d'installation.

Chaque test travaille dans un dossier temporaire : aucun fichier du dépôt n'est
modifié.

Usage (depuis la racine du dépôt) :

    python -m unittest tests.test_robustness_v9 -v
    python tests/test_robustness_v9.py
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config_manager as cm  # noqa: E402
import latex_support  # noqa: E402
import main_OCA  # noqa: E402
import oca_model  # noqa: E402
import post_processor  # noqa: E402
from config_manager import ConfigError  # noqa: E402

PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Outils
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def workdir():
    """Dossier temporaire devenu répertoire courant, restauré et supprimé ensuite."""
    previous = os.getcwd()
    folder = tempfile.mkdtemp(prefix="oca9_")
    os.chdir(folder)
    try:
        yield folder
    finally:
        os.chdir(previous)
        shutil.rmtree(folder, ignore_errors=True)


def base_frame(rows: int = 80, seed: int = 3) -> pd.DataFrame:
    """Données d'essai : trois variables et trois zones qui se chevauchent."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "V1": rng.uniform(1, 5, rows), "V2": rng.uniform(1, 5, rows),
        "V3": rng.uniform(1, 5, rows),
        "ZoneA": rng.uniform(1, 10, rows), "ZoneB": rng.uniform(1, 10, rows),
        "ZoneC": rng.uniform(1, 9, rows),
    })


def write_book(folder: str, frame: pd.DataFrame, name: str = "data.xlsx",
               sub: str = "run", index: bool = False) -> str:
    """Écrit un classeur ``input/<sub>/<name>`` (avec ou sans colonne d'index)."""
    target = os.path.join(folder, "input", sub)
    os.makedirs(target, exist_ok=True)
    path = os.path.join(target, name)
    if name.endswith(".csv"):
        frame.to_csv(path, index=index)
    else:
        frame.to_excel(path, index=index, sheet_name="criticalZones")
    return path


def minimal_config(**overrides) -> dict:
    """Configuration minimale : quatre clés."""
    cfg = {"critical_zones": ["ZoneA", "ZoneB", "ZoneC"],
           "feature_cols_init": ["V1", "V2", "V3"],
           "data_files": "data.xlsx", "merged_subdir": ["run"]}
    cfg.update(overrides)
    return cfg


def write_config(folder: str, cfg, name: str = "input.json") -> str:
    """Écrit un fichier de configuration (dictionnaire ou texte brut)."""
    path = os.path.join(folder, name)
    text = cfg if isinstance(cfg, str) else json.dumps(cfg)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def run_pipeline(folder: str, frame: pd.DataFrame, compiler: str = "none",
                 index: bool = False, **overrides) -> dict:
    """Écrit les données, lance le pipeline et retourne le bloc ``results``."""
    write_book(folder, frame, index=index)
    cfg = minimal_config(latex_output={"compiler": compiler}, **overrides)
    path = write_config(folder, cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        document = main_OCA.run(path)
    return document["results"]


def has_latex() -> bool:
    """Vrai si un compilateur LaTeX est disponible sur ce poste."""
    latex_support.configure("auto")
    return latex_support.find_latex_compiler() is not None


class MemoryHandler(logging.Handler):
    """Collecte les messages d'un journal."""

    def __init__(self) -> None:
        super().__init__()
        self.records = []

    def emit(self, record) -> None:
        self.records.append((record.levelname, record.getMessage()))


def make_logger():
    """Journal en mémoire, indépendant du journal ``oca`` du pipeline."""
    logger = logging.getLogger("oca_test_" + str(time.time_ns()))
    logger.setLevel(logging.INFO)
    handler = MemoryHandler()
    logger.addHandler(handler)
    logger.propagate = False
    return logger, handler


# ---------------------------------------------------------------------------
# F03 : défauts internes et JSON minimal
# ---------------------------------------------------------------------------
class TestMinimalConfiguration(unittest.TestCase):
    def test_defaults_are_neutral(self):
        with workdir() as folder:
            config = cm.load_config(write_config(folder, minimal_config()))
        self.assertFalse(config.user_defined_vars)
        self.assertEqual(config.user_defined_vars_cols, [])
        self.assertEqual(config.gpo_regularization["target_zone"], "none")
        self.assertEqual(config.synthetic_target, "none")
        self.assertTrue(config.auto_target)
        self.assertEqual(config.gpo_regularization["threshold"], 0.99)
        self.assertEqual(config.latex_compiler, "auto")
        self.assertEqual(cm.DEFAULT_CONFIG["gpo_regularization"]["target_zone"], "none")

    def test_four_key_json_runs_as_is(self):
        with workdir() as folder:
            results = run_pipeline(folder, base_frame())
        self.assertEqual(results["oca_version"], "9.0.0")
        self.assertTrue(results["target_zone"]["auto_selected"])
        self.assertNotIn("De/E", json.dumps(results["updated_critical_zones"]))
        self.assertEqual(results["column_resolution"]["feature_cols"], ["V1", "V2", "V3"])


# ---------------------------------------------------------------------------
# F01, F02, F05 : listes de colonnes vides et résolution
# ---------------------------------------------------------------------------
class TestColumnResolution(unittest.TestCase):
    def resolve(self, folder, frame, **cfg_overrides):
        write_book(folder, frame)
        cfg = minimal_config(**cfg_overrides)
        config = cm.load_config(write_config(folder, cfg))
        df, zones, resolution = cm.load_data_with_resolution(config.for_subdir("run"))
        return df, zones, resolution

    def test_feature_list_empty_null_empty_string_or_absent(self):
        for label, value in (("liste vide", []), ("null", None), ("chaîne vide", ""),
                             ("absente", "ABSENT")):
            with self.subTest(label), workdir() as folder:
                overrides = {}
                cfg = minimal_config()
                if value == "ABSENT":
                    del cfg["feature_cols_init"]
                else:
                    cfg["feature_cols_init"] = value
                write_book(folder, base_frame())
                config = cm.load_config(write_config(folder, cfg))
                _, zones, resolution = cm.load_data_with_resolution(config.for_subdir("run"))
                self.assertEqual(zones, ["ZoneA", "ZoneB", "ZoneC"])
                self.assertEqual(resolution.features, ["V1", "V2", "V3"])
                self.assertTrue(resolution.features_inferred)
                self.assertFalse(resolution.zones_inferred)
                del overrides

    def test_zone_list_empty_null_empty_string_or_absent(self):
        for label, value in (("liste vide", []), ("null", None), ("chaîne vide", ""),
                             ("absente", "ABSENT")):
            with self.subTest(label), workdir() as folder:
                cfg = minimal_config()
                if value == "ABSENT":
                    del cfg["critical_zones"]
                else:
                    cfg["critical_zones"] = value
                write_book(folder, base_frame())
                config = cm.load_config(write_config(folder, cfg))
                _, zones, resolution = cm.load_data_with_resolution(config.for_subdir("run"))
                self.assertEqual(zones, ["ZoneA", "ZoneB", "ZoneC"])
                self.assertTrue(resolution.zones_inferred)
                self.assertEqual(resolution.features, ["V1", "V2", "V3"])

    def test_both_empty_means_all_numeric_columns_are_zones_with_warning(self):
        with workdir() as folder:
            write_book(folder, base_frame())
            cfg = minimal_config(critical_zones=[], feature_cols_init=[])
            config = cm.load_config(write_config(folder, cfg))
            logger, handler = make_logger()
            _, zones, resolution = cm.load_data_with_resolution(
                config.for_subdir("run"), logger)
        self.assertEqual(zones, ["V1", "V2", "V3", "ZoneA", "ZoneB", "ZoneC"])
        self.assertEqual(resolution.features, [])
        self.assertTrue(any(level == "WARNING" and "aucune variable de conception" in text
                            for level, text in handler.records))

    def test_text_columns_are_ignored_by_inference(self):
        with workdir() as folder:
            frame = base_frame()
            frame["Commentaire"] = ["ok"] * len(frame)
            _, zones, resolution = self.resolve(folder, frame, critical_zones=[])
        self.assertEqual(zones, ["ZoneA", "ZoneB", "ZoneC"])
        self.assertIn("Commentaire", resolution.ignored)

    def test_unknown_and_overlapping_names_are_refused(self):
        with workdir() as folder:
            write_book(folder, base_frame())
            config = cm.load_config(write_config(
                folder, minimal_config(feature_cols_init=["V1", "Inexistante"])))
            with self.assertRaises(ConfigError) as ctx:
                cm.load_data_with_resolution(config.for_subdir("run"))
            self.assertIn("Inexistante", str(ctx.exception))
            self.assertIn("Colonnes disponibles", str(ctx.exception))
            with self.assertRaises(ConfigError) as ctx:
                cm.load_config(write_config(folder, minimal_config(
                    feature_cols_init=["V1", "ZoneA"])))
            self.assertIn("ZoneA", str(ctx.exception))
            with self.assertRaises(ConfigError) as ctx:
                cm.load_config(write_config(folder, minimal_config(
                    critical_zones=["ZoneA", "ZoneA"])))
            self.assertIn("doublons", str(ctx.exception))
            with self.assertRaises(ConfigError):
                cm.load_config(write_config(folder, minimal_config(critical_zones=["A", " "])))


# ---------------------------------------------------------------------------
# F04 : champs nuls, vides, mauvais types (les 32 entrées de l'audit)
# ---------------------------------------------------------------------------
FULL_BASE = {
    "critical_zones": ["ZoneA", "ZoneB", "ZoneC"],
    "feature_cols_init": ["V1", "V2", "V3"],
    "input_dir": "./input", "output_dir": "./output", "output_json": "./output.json",
    "data_files": "data.xlsx", "merged_subdir": ["run"],
    "user_defined_vars": False, "user_defined_vars_cols": [],
    "gpo_selection": {"threshold_low": 0.5, "threshold_high": 10.0},
    "gpo_regularization": {"target_zone": "none", "threshold": 0.99, "kappa_min": 1.0,
                           "kappa_max": 3.0, "kappa_step": 0.01},
    "ordinal_graph": {"threshold": 0.99},
    "latex_output": {"sample_size": 30, "max_columns_per_block": 6,
                     "table_font_size": "scriptsize", "table_column_separation": "2pt",
                     "compiler": "auto"},
    "sample_random_state": 6,
    "color_style": {"color_map_matrix": "RdBu_r", "color_map_critical_zones": "jet",
                    "saturation": 1.0, "alpha": 1.0, "apply_to": "background"},
}

SIMPLE_KEYS = ["critical_zones", "feature_cols_init", "input_dir", "output_dir",
               "output_json", "data_files", "merged_subdir", "user_defined_vars",
               "user_defined_vars_cols", "sample_random_state"]
BLOCK_KEYS = ["gpo_selection", "gpo_regularization", "ordinal_graph", "latex_output",
              "color_style"]
SUB_KEYS = [(block, sub) for block in BLOCK_KEYS for sub in FULL_BASE[block]
            if (block, sub) != ("latex_output", "compiler")]


class TestEmptyAndWrongTypes(unittest.TestCase):
    def load(self, cfg):
        with workdir() as folder:
            return cm.load_config(write_config(folder, cfg))

    def test_audit_matrix_never_raises_a_raw_exception(self):
        """Pour chaque clé : absente, [], "", null, {} -> défaut ou ConfigError nommé."""
        cases = []
        for key in SIMPLE_KEYS + BLOCK_KEYS:
            cases.append((key, None))
        for block, sub in SUB_KEYS:
            cases.append((f"{block}.{sub}", None))
        self.assertEqual(len(SIMPLE_KEYS) + len(BLOCK_KEYS) + len(SUB_KEYS), 32)
        for path, _ in cases:
            for label, value in (("absente", "ABSENT"), ("[]", []), ("vide", ""),
                                 ("null", None), ("{}", {})):
                cfg = json.loads(json.dumps(FULL_BASE))
                target = cfg
                parts = path.split(".")
                for part in parts[:-1]:
                    target = target[part]
                if value == "ABSENT":
                    del target[parts[-1]]
                else:
                    target[parts[-1]] = value
                with self.subTest(path=path, valeur=label):
                    try:
                        config = self.load(cfg)
                    except ConfigError as exc:
                        self.assertTrue(len(str(exc)) > 10)
                        self.assertLessEqual(str(exc).count("\n"), 1)
                        continue
                    for text in (config.input_dir, config.output_dir, config.output_json,
                                 config.data_files):
                        self.assertNotIn(text, ("None", "[]", ""))
                    self.assertIsInstance(config.user_defined_vars, bool)
                    self.assertIsInstance(config.sample_random_state, int)
                    self.assertIsInstance(config.gpo_selection["threshold_low"], float)
                    self.assertIsInstance(config.gpo_regularization["kappa_step"], float)
                    self.assertIsInstance(config.latex_output["sample_size"], int)
                    self.assertIn(config.latex_compiler, latex_support.ALLOWED_COMPILERS)

    def test_null_optional_fields_take_their_default_with_a_note(self):
        cfg = json.loads(json.dumps(FULL_BASE))
        cfg["gpo_regularization"]["kappa_step"] = None
        cfg["gpo_regularization"]["target_zone"] = None
        cfg["user_defined_vars_cols"] = None
        cfg["latex_output"] = None
        config = self.load(cfg)
        self.assertEqual(config.gpo_regularization["kappa_step"], 0.01)
        self.assertEqual(config.gpo_regularization["target_zone"], "none")
        self.assertEqual(config.latex_output["sample_size"], 30)
        joined = " ".join(config.normalization_notes)
        self.assertIn("gpo_regularization.kappa_step", joined)
        self.assertIn("latex_output", joined)

    def test_wrong_types_name_the_key(self):
        bad = [
            ("gpo_selection.threshold_low", "abc"), ("gpo_selection.threshold_low", [1]),
            ("gpo_selection.threshold_high", True), ("gpo_selection.threshold_low", 120),
            ("gpo_selection.threshold_low", -1),
            ("gpo_regularization.kappa_step", "0.01"), ("gpo_regularization.kappa_step", 0),
            ("gpo_regularization.kappa_step", -0.5), ("gpo_regularization.threshold", 1.5),
            ("gpo_regularization.threshold", 0), ("gpo_regularization.kappa_min", 0),
            ("gpo_regularization.target_zone", ["A"]), ("gpo_regularization.target_zone", 3),
            ("ordinal_graph.threshold", "0.99"),
            ("latex_output.sample_size", "30"), ("latex_output.sample_size", 2.5),
            ("latex_output.sample_size", 0), ("latex_output.max_columns_per_block", [6]),
            ("latex_output.table_font_size", "huge"), ("latex_output.table_font_size", 12),
            ("latex_output.table_column_separation", "abc"),
            ("latex_output.table_column_separation", ["2pt"]),
            ("latex_output.compiler", "word"), ("latex_output.compiler", 1),
            ("color_style.alpha", 2), ("color_style.saturation", "1"),
            ("color_style.apply_to", "both"), ("color_style.color_map_matrix", ["jet"]),
            ("color_style.color_map_matrix", "carte_inconnue"),
            ("user_defined_vars", "false"), ("user_defined_vars", []),
            ("user_defined_vars", 1), ("user_defined_vars_cols", "De/E"),
            ("user_defined_vars_cols", [3]),
            ("sample_random_state", 1.5), ("sample_random_state", "6"),
            ("sample_random_state", -1), ("sample_random_state", [6]),
            ("input_dir", ["a"]), ("input_dir", 3), ("output_dir", {}), ("output_json", 7),
            ("data_files", []), ("merged_subdir", 3), ("merged_subdir", ["a", 2]),
            ("merged_subdir", ["a", "a"]),
            ("critical_zones", "ZoneA"), ("critical_zones", [1]), ("critical_zones", ["A", ""]),
            ("feature_cols_init", {"a": 1}), ("feature_cols_init", ["V1", "V1"]),
            ("gpo_selection", []), ("gpo_regularization", "x"), ("ordinal_graph", [1]),
            ("latex_output", 3), ("color_style", ["jet"]),
        ]
        for path, value in bad:
            cfg = json.loads(json.dumps(FULL_BASE))
            target = cfg
            parts = path.split(".")
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
            with self.subTest(path=path, valeur=value):
                with self.assertRaises(ConfigError) as ctx:
                    self.load(cfg)
                self.assertIn(parts[-1] if len(parts) == 1 else path, str(ctx.exception))

    def test_non_finite_literals_are_refused(self):
        for literal in ("Infinity", "NaN", "-Infinity"):
            text = ('{"critical_zones": ["A", "B"], "feature_cols_init": ["V"], '
                    f'"gpo_regularization": {{"kappa_max": {literal}}}}}')
            with self.subTest(literal), workdir() as folder:
                with self.assertRaises(ConfigError) as ctx:
                    cm.load_config(write_config(folder, text))
                self.assertIn("gpo_regularization.kappa_max", str(ctx.exception))

    def test_inconsistent_bounds_are_refused(self):
        cfg = json.loads(json.dumps(FULL_BASE))
        cfg["gpo_regularization"]["kappa_min"] = 2.0
        cfg["gpo_regularization"]["kappa_max"] = 1.0
        with self.assertRaises(ConfigError) as ctx:
            self.load(cfg)
        self.assertIn("kappa_min", str(ctx.exception))
        cfg = json.loads(json.dumps(FULL_BASE))
        cfg["gpo_selection"] = {"threshold_low": 20.0, "threshold_high": 10.0}
        with self.assertRaises(ConfigError):
            self.load(cfg)

    def test_invalid_json_and_missing_file_are_config_errors(self):
        with workdir() as folder:
            with self.assertRaises(ConfigError):
                cm.load_config(os.path.join(folder, "absent.json"))
            with self.assertRaises(ConfigError):
                cm.load_config(write_config(folder, "{ pas du json"))
            with self.assertRaises(ConfigError):
                cm.load_config(write_config(folder, "[1, 2]"))

    def test_unknown_keys_are_reported_not_fatal(self):
        cfg = json.loads(json.dumps(FULL_BASE))
        cfg["critical_zone"] = ["x"]
        cfg["gpo_selection"]["threshold_lo"] = 1
        config = self.load(cfg)
        joined = " ".join(config.normalization_warnings)
        self.assertIn("critical_zone", joined)
        self.assertIn("gpo_selection.threshold_lo", joined)


# ---------------------------------------------------------------------------
# F06 : chemins dangereux
# ---------------------------------------------------------------------------
class TestDangerousPaths(unittest.TestCase):
    def setUp(self):
        self.ctx = workdir()
        self.folder = self.ctx.__enter__()
        self.book = write_book(self.folder, base_frame())
        self.sentinel = os.path.join(self.folder, "input", "run", "sentinelle.txt")
        with open(self.sentinel, "w", encoding="utf-8") as handle:
            handle.write("ne pas supprimer")
        with open(self.book, "rb") as handle:
            self.book_bytes = handle.read()

    def tearDown(self):
        self.ctx.__exit__(None, None, None)

    def assert_intact(self):
        self.assertTrue(os.path.isfile(self.sentinel))
        with open(self.book, "rb") as handle:
            self.assertEqual(handle.read(), self.book_bytes)

    def refused(self, **overrides):
        cfg = minimal_config(latex_output={"compiler": "none"}, **overrides)
        path = write_config(self.folder, cfg)
        with self.assertRaises(ConfigError):
            with contextlib.redirect_stdout(io.StringIO()):
                main_OCA.run(path)
        self.assert_intact()

    def test_subdir_escaping_the_output_root(self):
        for value in ("../input/run", "..", "a/b", "a\\b", "/absolu", "C:\\x", "./run"):
            with self.subTest(value):
                self.refused(merged_subdir=value)
        self.refused(merged_subdir=["run", "../input/run"])

    def test_data_files_must_be_a_simple_file_name(self):
        for value in (self.book, "../x.xlsx", "sub/data.xlsx", "C:data.xlsx", ".."):
            with self.subTest(value):
                self.refused(data_files=value)

    def test_output_dir_equal_to_or_containing_the_input_dir(self):
        self.refused(output_dir="./input")
        self.refused(output_dir=".")
        self.refused(input_dir="./out/input", output_dir="./out")

    def test_clean_output_guard_without_load_config(self):
        config = cm.load_config(write_config(self.folder, minimal_config()))
        dangerous = replace(config, output_dir=config.input_dir)
        with self.assertRaises(ConfigError):
            cm.clean_output_main(dangerous.for_subdir("run"))
        self.assert_intact()

    def test_output_json_cannot_overwrite_the_workbook(self):
        self.refused(output_json="./input/run/data.xlsx")

    def test_safe_configuration_still_works_and_cleans_only_the_output(self):
        stale = os.path.join(self.folder, "output", "run", "ancien.txt")
        os.makedirs(os.path.dirname(stale))
        with open(stale, "w", encoding="utf-8") as handle:
            handle.write("périmé")
        run_pipeline(self.folder, base_frame())
        self.assertFalse(os.path.exists(stale))
        self.assert_intact()


# ---------------------------------------------------------------------------
# F07, F26, F27 : lecture des classeurs
# ---------------------------------------------------------------------------
class TestWorkbookReading(unittest.TestCase):
    def load_df(self, folder, frame, index=False, name="data.xlsx", **cfg):
        write_book(folder, frame, name=name, index=index)
        config = cm.load_config(write_config(
            folder, minimal_config(data_files=name, **cfg)))
        return cm.load_data_with_resolution(config.for_subdir("run"))

    def test_workbook_without_index_keeps_its_first_column(self):
        with workdir() as folder:
            df, zones, resolution = self.load_df(folder, base_frame(), index=False)
        self.assertIsNone(resolution.index_column)
        self.assertIn("V1", df.columns)
        self.assertEqual(len(df), 80)

    def test_workbook_with_index_removes_it_and_gives_the_same_table(self):
        with workdir() as folder:
            df_plain, _, _ = self.load_df(folder, base_frame(), index=False)
        with workdir() as folder:
            df_indexed, _, resolution = self.load_df(folder, base_frame(), index=True)
        self.assertEqual(resolution.index_column, "(sans en-tête)")
        pd.testing.assert_frame_equal(df_plain[["V1", "ZoneA"]], df_indexed[["V1", "ZoneA"]])

    def test_named_index_columns_are_removed(self):
        for header in ("index", "ID", "config", "config_id", "n"):
            with self.subTest(header), workdir() as folder:
                frame = base_frame()
                frame.insert(0, header, range(len(frame)))
                df, _, resolution = self.load_df(folder, frame)
                self.assertEqual(resolution.index_column, header)
                self.assertNotIn(header, df.columns)

    def test_first_zone_column_is_not_swallowed(self):
        with workdir() as folder:
            frame = base_frame()[["ZoneA", "ZoneB", "ZoneC", "V1", "V2", "V3"]]
            df, zones, _ = self.load_df(folder, frame)
        self.assertEqual(zones, ["ZoneA", "ZoneB", "ZoneC"])

    def test_same_results_with_and_without_index(self):
        frame = base_frame()
        with workdir() as folder:
            plain = run_pipeline(folder, frame, index=False)
        with workdir() as folder:
            indexed = run_pipeline(folder, frame, index=True)
        self.assertEqual(plain["initial_ordinal_state"]["ordinal_correlation_matrix"],
                         indexed["initial_ordinal_state"]["ordinal_correlation_matrix"])
        self.assertEqual(plain["metrics"]["kappa_optimal"], indexed["metrics"]["kappa_optimal"])
        self.assertEqual(plain["metrics"]["n_sample"], indexed["metrics"]["n_sample"])

    def test_text_cell_in_a_zone_names_file_and_column(self):
        with workdir() as folder:
            frame = base_frame().astype(object)
            frame.loc[5, "ZoneB"] = "abc"
            with self.assertRaises(ConfigError) as ctx:
                self.load_df(folder, frame)
        message = str(ctx.exception)
        self.assertIn("data.xlsx", message)
        self.assertIn("ZoneB", message)
        self.assertIn("ligne 7", message)

    def test_text_comment_column_is_ignored(self):
        with workdir() as folder:
            frame = base_frame()
            frame["Remarque"] = "à vérifier"
            df, zones, resolution = self.load_df(folder, frame)
        self.assertNotIn("Remarque", df.columns)
        self.assertEqual(zones, ["ZoneA", "ZoneB", "ZoneC"])

    def test_trailing_empty_rows_are_ignored(self):
        with workdir() as folder:
            os.makedirs("input/run")
            frame = base_frame(40)
            padded = pd.concat([frame, pd.DataFrame({c: [np.nan] * 3 for c in frame.columns})],
                               ignore_index=True)
            padded.to_excel("input/run/data.xlsx", index=False)
            config = cm.load_config(write_config(folder, minimal_config()))
            df, _, _ = cm.load_data_with_resolution(config.for_subdir("run"))
        self.assertEqual(len(df), 40)

    def test_only_the_first_sheet_is_read(self):
        with workdir() as folder:
            os.makedirs("input/run")
            with pd.ExcelWriter("input/run/data.xlsx") as writer:
                base_frame(40).to_excel(writer, sheet_name="premiere", index=False)
                base_frame(10, seed=9).to_excel(writer, sheet_name="seconde", index=False)
            config = cm.load_config(write_config(folder, minimal_config()))
            logger, handler = make_logger()
            df, _, resolution = cm.load_data_with_resolution(config.for_subdir("run"), logger)
        self.assertEqual(len(df), 40)
        self.assertEqual(resolution.sheet, "premiere")
        self.assertTrue(any("première feuille" in text for _, text in handler.records))

    def test_xls_is_refused_with_a_conversion_message(self):
        with workdir() as folder:
            with self.assertRaises(ConfigError) as ctx:
                cm.load_config(write_config(folder, minimal_config(data_files="data.xls")))
        self.assertIn(".xlsx", str(ctx.exception))
        with workdir() as folder:
            with self.assertRaises(ConfigError):
                cm.load_config(write_config(folder, minimal_config(data_files="data.txt")))

    def test_csv_with_comma_or_semicolon(self):
        frame = base_frame(30)
        with workdir() as folder:
            expected, _, _ = self.load_df(folder, frame)
        with workdir() as folder:
            write_book(folder, frame, name="data.csv")
            config = cm.load_config(write_config(folder, minimal_config(data_files="data.csv")))
            comma, _, _ = cm.load_data_with_resolution(config.for_subdir("run"))
            main_dir = config.output_dir_main()
            self.assertTrue(os.path.isfile(os.path.join(main_dir, "data.xlsx")))
        with workdir() as folder:
            target = os.path.join(folder, "input", "run")
            os.makedirs(target)
            text = frame.to_csv(sep=";", index=False, decimal=",")
            with open(os.path.join(target, "data.csv"), "w", encoding="utf-8") as handle:
                handle.write(text)
            config = cm.load_config(write_config(folder, minimal_config(data_files="data.csv")))
            semicolon, _, _ = cm.load_data_with_resolution(config.for_subdir("run"))
        pd.testing.assert_frame_equal(expected, comma, check_exact=False)
        pd.testing.assert_frame_equal(expected, semicolon, check_exact=False)

    def test_corrupted_workbook_is_a_config_error(self):
        with workdir() as folder:
            os.makedirs("input/run")
            with open("input/run/data.xlsx", "wb") as handle:
                handle.write(b"ce n'est pas un classeur")
            config = cm.load_config(write_config(folder, minimal_config()))
            with self.assertRaises(ConfigError):
                cm.load_data_with_resolution(config.for_subdir("run"))

    def test_missing_data_message_points_to_init_and_examples(self):
        with workdir() as folder:
            os.makedirs("input")
            path = write_config(folder, minimal_config(
                latex_output={"compiler": "none"}))
            result = subprocess.run(
                [PYTHON, "-X", "utf8", os.path.join(ROOT, "main_OCA.py"), "--config", path],
                capture_output=True, text=True, cwd=folder, encoding="utf-8")
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("Traceback", result.stderr)
        self.assertIn("--init", result.stderr)
        self.assertIn("examples/", result.stderr)
        self.assertLessEqual(len(result.stderr.strip().splitlines()), 3)


# ---------------------------------------------------------------------------
# F08, F17, F18 : données limites
# ---------------------------------------------------------------------------
class TestDataLimits(unittest.TestCase):
    def check_refused(self, frame, expected_words, **cfg):
        with workdir() as folder:
            write_book(folder, frame)
            config = cm.load_config(write_config(folder, minimal_config(**cfg)))
            with self.assertRaises(ConfigError) as ctx:
                cm.load_data_with_resolution(config.for_subdir("run"))
        for word in expected_words:
            self.assertIn(word, str(ctx.exception))

    def test_infinity_in_a_zone_is_refused_with_column_count_and_rows(self):
        frame = base_frame()
        frame.loc[[3, 9], "ZoneC"] = np.inf
        self.check_refused(frame, ["ZoneC", "infinie", "2 valeur", "4, 10"])

    def test_negative_infinity_and_nan(self):
        frame = base_frame()
        frame.loc[1, "ZoneA"] = -np.inf
        self.check_refused(frame, ["ZoneA", "infinie"])
        frame = base_frame()
        frame.loc[2, "ZoneB"] = np.nan
        self.check_refused(frame, ["ZoneB", "manquante"])

    def test_target_synthetic_division_by_zero_is_refused(self):
        frame = base_frame()
        frame.loc[4, "V2"] = 0.0
        with workdir() as folder:
            write_book(folder, frame)
            config = cm.load_config(write_config(folder, minimal_config(
                user_defined_vars=True, user_defined_vars_cols=["ZoneA/V2"])))
            with self.assertRaises(ConfigError) as ctx:
                cm.load_data_with_resolution(config.for_subdir("run"))
        self.assertIn("ZoneA/V2", str(ctx.exception))
        frame = base_frame()
        frame.loc[4, "V2"] = 0.0
        with workdir() as folder:
            write_book(folder, frame)
            config = cm.load_config(write_config(folder, minimal_config(
                gpo_regularization={"target_zone": "ZoneA/V2"})))
            with self.assertRaises(ConfigError) as ctx:
                cm.load_data_with_resolution(config.for_subdir("run"))
        self.assertIn("ZoneA/V2", str(ctx.exception))

    def test_invalid_expression_is_a_config_error_not_a_trace(self):
        with workdir() as folder:
            write_book(folder, base_frame())
            config = cm.load_config(write_config(folder, minimal_config(
                user_defined_vars=True, user_defined_vars_cols=["Absente/V2"])))
            with self.assertRaises(ConfigError):
                cm.load_data_with_resolution(config.for_subdir("run"))

    def test_single_observation_is_refused_two_is_accepted(self):
        self.check_refused(base_frame(1), ["une observation", "2 observations"])
        with workdir() as folder:
            results = run_pipeline(folder, base_frame(2))
        self.assertEqual(results["metrics"]["n_sample"], 2)

    def test_zero_observation_and_single_zone_are_refused(self):
        with workdir() as folder:
            write_book(folder, base_frame().iloc[0:0])
            config = cm.load_config(write_config(folder, minimal_config()))
            with self.assertRaises(ConfigError):
                cm.load_data_with_resolution(config.for_subdir("run"))
        self.check_refused(base_frame(), ["Une seule zone"], critical_zones=["ZoneA"])

    def test_constant_column_does_not_call_pearson(self):
        frame = base_frame(30)
        frame["ZoneC"] = 4.0
        calls = []

        def forbidden(*args, **kwargs):
            calls.append(args)
            raise AssertionError("pearsonr appelé sur une série constante")

        with workdir() as folder, mock.patch.object(post_processor, "pearsonr", forbidden):
            path = os.path.join(folder, "s.pdf")
            post_processor.plot_scatter(
                frame, ["ZoneA", "ZoneB", "ZoneC"], "ZoneC", path)
            self.assertTrue(os.path.isfile(path))
        self.assertEqual(calls, [])

    def test_constant_column_full_run(self):
        frame = base_frame(40)
        frame["ZoneC"] = 4.0
        with workdir() as folder:
            results = run_pipeline(folder, frame)
        self.assertIn("kappa_optimal", results["metrics"])


# ---------------------------------------------------------------------------
# F11, F12, F13 : grille de κ
# ---------------------------------------------------------------------------
class TestKappaGrid(unittest.TestCase):
    def frame(self, ratio):
        return pd.DataFrame({"A": [1.0] * 10, "B": [ratio] * 10})

    def test_upper_bound_is_included_by_index(self):
        kappa, found = oca_model.find_kappa(self.frame(1.2), ["A", "B"], "A", 1.0,
                                            1.0, 1.2, 0.1)
        self.assertTrue(found)
        self.assertEqual(kappa, 1.2)
        self.assertEqual(oca_model.kappa_grid(1.0, 1.2, 0.1), [1.0, 1.1, 1.2])

    def test_success_exactly_at_kappa_max_is_a_success(self):
        kappa, found = oca_model.find_kappa(self.frame(3.0), ["A", "B"], "A", 1.0,
                                            1.0, 3.0, 0.01)
        self.assertTrue(found)
        self.assertEqual(kappa, 3.0)

    def test_default_grid_has_201_points_ending_at_3(self):
        grid = oca_model.kappa_grid(1.0, 3.0, 0.01)
        self.assertEqual(len(grid), 201)
        self.assertEqual(grid[-1], 3.0)
        self.assertEqual(grid[55], 1.55)

    def test_failure_returns_the_last_tested_point_inside_the_range(self):
        kappa, found = oca_model.find_kappa(self.frame(5.0), ["A", "B"], "A", 1.0,
                                            1.0, 2.0, 0.3)
        self.assertFalse(found)
        self.assertLessEqual(kappa, 2.0)
        self.assertEqual(kappa, 1.9)
        details = oca_model.find_kappa_details(
            self.frame(5.0), ["A", "B"], "A", 1.0, 1.0, 2.0, 0.3)
        self.assertEqual(details["n_tested"], details["n_points"])
        self.assertLess(details["min_coverage"], 1.0)

    def test_tiny_step_is_refused_quickly(self):
        cfg = json.loads(json.dumps(FULL_BASE))
        cfg["gpo_regularization"]["kappa_step"] = 1e-20
        with workdir() as folder:
            started = time.time()
            with self.assertRaises(ConfigError) as ctx:
                cm.load_config(write_config(folder, cfg))
            self.assertLess(time.time() - started, 5)
        self.assertIn("gpo_regularization.kappa_step", str(ctx.exception))
        with self.assertRaises(ValueError):
            oca_model.kappa_grid(1.0, 3.0, 1e-20)
        for bad in ((3.0, 1.0, 0.1), (1.0, float("inf"), 0.1), (1.0, 3.0, 0.0),
                    (1.0, 3.0, -1.0), (1.0, 3.0, float("nan"))):
            with self.subTest(bad), self.assertRaises(ValueError):
                oca_model.kappa_grid(*bad)

    def test_limit_of_100000_points(self):
        self.assertEqual(oca_model.MAX_KAPPA_POINTS, 100000)
        self.assertEqual(oca_model.kappa_point_count(1.0, 3.0, 2e-5), 100001)
        cfg = json.loads(json.dumps(FULL_BASE))
        cfg["gpo_regularization"]["kappa_step"] = 2e-5
        with workdir() as folder:
            with self.assertRaises(ConfigError):
                cm.load_config(write_config(folder, cfg))
        cfg["gpo_regularization"]["kappa_step"] = 3e-5
        with workdir() as folder:
            cm.load_config(write_config(folder, cfg))

    def test_decimals_follow_the_step(self):
        self.assertEqual(oca_model.kappa_decimals(1.0, 0.01), 2)
        self.assertEqual(oca_model.kappa_decimals(1.0, 0.1), 2)
        self.assertEqual(oca_model.kappa_decimals(1.0, 0.001), 3)
        self.assertEqual(oca_model.kappa_decimals(1.0, 0.0001), 4)
        self.assertEqual(oca_model.alpha_column_name(1.004, "A", 3), "1.004$\\times$A")
        self.assertEqual(oca_model.alpha_column_name(1.5, "A"), "1.50$\\times$A")

    def test_published_kappa_is_the_tested_kappa(self):
        frame = pd.DataFrame({
            "V1": np.arange(1, 21, dtype=float),
            "A": [1.0] * 20, "B": [1.004] * 20})
        with workdir() as folder:
            results = run_pipeline(
                folder, frame, critical_zones=["A", "B"], feature_cols_init=["V1"],
                gpo_selection={"threshold_low": 0.0, "threshold_high": 10.0},
                gpo_regularization={"target_zone": "A", "threshold": 1.0,
                                    "kappa_min": 1.004, "kappa_max": 1.004,
                                    "kappa_step": 0.001})
        self.assertTrue(results["metrics"]["kappa_found"])
        self.assertEqual(results["metrics"]["kappa_optimal"], 1.004)
        self.assertEqual(results["metrics"]["kappa_decimals"], 3)
        labels = list(results["ordinal_correlation_matrix"])
        self.assertTrue(any(label.startswith("1.004$\\times$") for label in labels), labels)
        envelope = results["metrics"]["kappa_optimal"] * frame["A"]
        self.assertTrue((envelope >= frame["B"]).all())


# ---------------------------------------------------------------------------
# F14, F15, F16 : ex æquo et domaine de β
# ---------------------------------------------------------------------------
class TestTiesAndBeta(unittest.TestCase):
    def test_ties_are_reported_with_the_deterministic_rule(self):
        frame = base_frame(60)
        frame["ZoneB"] = frame["ZoneA"]
        frame["ZoneC"] = 0.5 * frame["ZoneA"]
        with workdir() as folder:
            results = run_pipeline(
                folder, frame, gpo_selection={"threshold_low": 0.0, "threshold_high": 10.0})
            records = results["execution_log"]["records"]
        target = results["target_zone"]
        self.assertGreaterEqual(len(target["tie_candidates"]), 2)
        self.assertIn("ordre des colonnes", target["tie_rule"])
        self.assertTrue(any("ex æquo" in line for line in records))

    def test_dominance_ties_helper(self):
        frame = pd.DataFrame({"A": [1.0, 1.0], "B": [1.0, 1.0], "C": [0.5, 0.5]})
        selection = oca_model.gpo_selection(frame, ["A", "B", "C"], 0.5, 10.0)
        best, tied = oca_model.dominance_ties(selection["df_results"])
        self.assertEqual(best, 0.0)
        self.assertEqual(tied, ["A", "B", "C"])
        self.assertEqual(oca_model.highest_dominance_column(selection["df_results"]), "A")

    def test_beta_not_defined_with_negative_or_zero_values(self):
        frame = pd.DataFrame({"A": [2.0, 3.0, 4.0], "B": [-1.0, 1.0, 0.0]})
        report = oca_model.beta_domain_report(frame, ["A", "B"], "A")
        self.assertFalse(report["beta_defined"])
        self.assertEqual(report["n_negative_values"], 1)
        self.assertEqual(report["n_zero_values"], 1)
        self.assertIn("non interprétable", report["note"])
        positive = oca_model.beta_domain_report(base_frame(), ["ZoneA", "ZoneB"], "ZoneA")
        self.assertTrue(positive["beta_defined"])

    def test_beta_flag_in_output_json(self):
        frame = base_frame(40)
        frame.loc[0:3, "ZoneB"] = -1.0
        with workdir() as folder:
            results = run_pipeline(folder, frame)
        self.assertFalse(results["metrics"]["beta_defined"])
        self.assertIn("beta_note", results["metrics"])
        self.assertTrue(any("non interprétable" in line
                            for line in results["execution_log"]["records"]))

    def test_all_zero_rows_give_infinite_beta_flagged(self):
        frame = base_frame(30)
        frame.loc[0:4, ["ZoneA", "ZoneB", "ZoneC"]] = 0.0
        with workdir() as folder:
            results = run_pipeline(folder, frame)
        self.assertFalse(results["metrics"]["beta_defined"])
        self.assertGreater(results["beta_domain"]["n_infinite_beta"], 0)


# ---------------------------------------------------------------------------
# F19 : fusion par fichiers par-zone
# ---------------------------------------------------------------------------
class TestPerZoneMerge(unittest.TestCase):
    def write_zone(self, zone, frame):
        target = os.path.join("input", f"dataSet_{zone}")
        os.makedirs(target, exist_ok=True)
        frame.to_excel(os.path.join(target, f"dataSet_{zone}.xlsx"), index=False)

    def test_duplicates_conflicts_and_intersection_are_logged(self):
        with workdir() as folder:
            base = pd.DataFrame({"V1": [1.0, 2.0, 3.0, 4.0], "V2": [5.0, 6.0, 7.0, 8.0]})
            a = base.copy()
            a["T"] = [10.0, 20.0, 30.0, 40.0]
            b = pd.concat([base, base.iloc[[0]]], ignore_index=True)
            b["T"] = [11.0, 21.0, 31.0, 41.0, 99.0]   # doublon contradictoire (config 0)
            c = base.iloc[:3].copy()                    # configuration 4 absente
            c["T"] = [12.0, 22.0, 32.0]
            for zone, frame in (("ZA", a), ("ZB", b), ("ZC", c)):
                self.write_zone(zone, frame)
            config = cm.load_config(write_config(folder, {
                "critical_zones": ["ZA", "ZB", "ZC"], "feature_cols_init": ["V1", "V2"],
                "merged_subdir": "run"}))
            logger, handler = make_logger()
            df, zones, resolution = cm.load_data_with_resolution(
                config.for_subdir("run"), logger)
        texts = [text for _, text in handler.records]
        self.assertEqual(len(df), 3)
        self.assertEqual(resolution.features, ["V1", "V2"])
        self.assertTrue(any("1 doublon" in t for t in texts))
        self.assertTrue(any("valeurs différentes" in t for t in texts))
        self.assertTrue(any("exclue(s) de l'intersection" in t for t in texts))
        warn = [level for level, text in handler.records if "valeurs différentes" in text]
        self.assertEqual(warn, ["WARNING"])

    def test_zones_must_be_declared_for_per_zone_mode(self):
        with workdir() as folder:
            base = pd.DataFrame({"V1": [1.0, 2.0], "V2": [5.0, 6.0], "T": [1.0, 2.0]})
            self.write_zone("ZA", base)
            self.write_zone("ZB", base)
            config = cm.load_config(write_config(folder, {
                "critical_zones": [], "feature_cols_init": ["V1", "V2"],
                "merged_subdir": "run"}))
            with self.assertRaises(ConfigError) as ctx:
                cm.load_data_with_resolution(config.for_subdir("run"))
        self.assertIn("critical_zones", str(ctx.exception))


# ---------------------------------------------------------------------------
# F20, F21, F22 : textes, encodage, échappement LaTeX
# ---------------------------------------------------------------------------
SPECIAL_NAMES = {"ZoneA": "Z&A", "ZoneB": "Z#1", "ZoneC": "σ_max"}


class TestNamesAndEscaping(unittest.TestCase):
    def special_frame(self):
        return base_frame(50).rename(columns=SPECIAL_NAMES)

    def special_config(self):
        return {"critical_zones": list(SPECIAL_NAMES.values())}

    def test_tex_escape_name_single_pass(self):
        self.assertEqual(post_processor.tex_escape_name("A&B"), r"A\&B")
        self.assertEqual(post_processor.tex_escape_name("Z#1"), r"Z\#1")
        self.assertEqual(post_processor.tex_escape_name("50%"), r"50\%")
        self.assertEqual(post_processor.tex_escape_name("{a}"), r"\{a\}")
        self.assertEqual(post_processor.tex_escape_name("$x$"), r"\$x\$")
        self.assertEqual(post_processor.tex_escape_name("a^b~c"), r"a\^{}b\~{}c")
        self.assertEqual(post_processor.tex_escape_name("a\\b"), r"a\textbackslash{}b")
        self.assertEqual(post_processor.tex_escape_name("σ"), r"\ensuremath{\sigma}")
        self.assertEqual(post_processor.tex_escape_name("a_b"), "a_b")
        self.assertEqual(post_processor.tex_escape_name("ж"), "?")
        # pas de double échappement : le résultat d'un échappement n'est pas ré-échappé
        # par les mécanismes internes (pourcentage déjà échappé).
        self.assertEqual(post_processor._escape_percent(r"50\% et 5%"), r"50\% et 5\%")
        # fragments mathématiques du programme
        self.assertEqual(
            post_processor.tex_escape_name(r"$\alpha$De/E&", keep_math_tokens=True),
            r"$\alpha$De/E\&")
        self.assertEqual(post_processor.tex_escape_name(r"$\alpha$"), r"\$\textbackslash{}alpha\$")

    def test_txt_files_never_fail_on_exotic_names_and_json_keeps_them(self):
        with workdir() as folder:
            results = run_pipeline(folder, self.special_frame(),
                                   **self.special_config())
            out = os.path.join(folder, "output", "run")
            for name in ("OCA_Sort_Targets.txt", "Step1_OrdinalCorrMat_Targets.txt",
                         "Step1_OrdinalGraph_Targets.txt", "OCA_OrdinalGraph_Targets.txt"):
                self.assertTrue(os.path.isfile(os.path.join(out, name)), name)
            with open(os.path.join(out, "OCA_Sort_Targets.txt"), "rb") as handle:
                text = handle.read().decode("cp1252")
        self.assertIn("Z&A", text)
        self.assertIn("?_max", text)
        self.assertIn("σ_max", results["initial_ordinal_state"]["critical_zones"])

    def test_notation_sentence_in_graph_text_files(self):
        with workdir() as folder:
            run_pipeline(folder, base_frame(50))
            out = os.path.join(folder, "output", "run")
            for name in ("Step1_OrdinalGraph_Targets.txt", "OCA_OrdinalGraph_Targets.txt"):
                with open(os.path.join(out, name), "rb") as handle:
                    text = handle.read().decode("cp1252")
                self.assertIn("précède B dans le classement de dominance", text)

    def test_tex_sources_escape_special_characters(self):
        with workdir() as folder:
            run_pipeline(folder, self.special_frame(), **self.special_config())
            out = os.path.join(folder, "output", "run")
            sources = {}
            for name in os.listdir(out):
                if name.endswith(".tex"):
                    with open(os.path.join(out, name), encoding="utf-8") as handle:
                        sources[name] = handle.read()
        for name in ("01-Step1_Dominance_Hierarchy_Result.tex",
                     "04-Step1_Critical_Zones_Sample_Result.tex",
                     "07-CriticalZones_Targets.tex", "11-OCA_Report_intro.tex"):
            text = sources[name]
            self.assertIn(r"Z\&A", text, name)
            self.assertIn(r"Z\#1", text, name)
            self.assertIn(r"\ensuremath{\sigma}_max", text, name)
            if name != "11-OCA_Report_intro.tex":   # le journal verbatim garde les noms bruts
                self.assertNotIn("Z&A", text.replace(r"Z\&A", ""), name)

    @unittest.skipUnless(has_latex(), "LaTeX absent")
    def test_names_with_special_characters_compile_without_error(self):
        with workdir() as folder:
            results = run_pipeline(folder, self.special_frame(), compiler="auto",
                                   **self.special_config())
            out = os.path.join(folder, "output", "run")
            for name in ("01-Step1_Dominance_Hierarchy_Result.pdf",
                         "04-Step1_Critical_Zones_Sample_Result.pdf",
                         "07-CriticalZones_Targets.pdf", "11-OCA_Report_intro.pdf",
                         "00-OCA_Report.pdf"):
                self.assertTrue(os.path.isfile(os.path.join(out, name)), name)
        self.assertTrue(results["report"]["complete"], results["report"])
        self.assertFalse(any("en échec" in line
                             for line in results["execution_log"]["records"]))


# ---------------------------------------------------------------------------
# F23, F24, F25 : compilation LaTeX et rapport partiel
# ---------------------------------------------------------------------------
class TestLatexHandling(unittest.TestCase):
    def prepare(self, folder):
        with open(os.path.join(folder, "t.tex"), "w", encoding="utf-8") as handle:
            handle.write("\\documentclass{article}\\begin{document}x\\end{document}")

    def compile_with(self, folder, side_effect=None, returncode=0, make_log=None):
        self.prepare(folder)
        logger, handler = make_logger()

        def fake_run(command, **kwargs):
            if side_effect is not None:
                raise side_effect
            if make_log:
                with open(os.path.join(folder, "t.log"), "w", encoding="utf-8") as stream:
                    stream.write(make_log)
            with open(os.path.join(folder, "t.pdf"), "wb") as stream:
                stream.write(b"%PDF-1.4 faux")
            return subprocess.CompletedProcess(command, returncode)

        with mock.patch.object(latex_support, "find_latex_compiler",
                               return_value=("pdflatex", "pdflatex-fictif")), \
                mock.patch.object(post_processor.subprocess, "run", fake_run):
            result = post_processor.compile_latex("t", folder, folder, logger=logger)
        return result, handler.records

    def test_launch_failure_is_not_fatal(self):
        with workdir() as folder:
            result, records = self.compile_with(folder, side_effect=OSError("introuvable"))
        self.assertEqual(result, "")
        self.assertTrue(any("n'a pas pu être lancé" in text for _, text in records))

    def test_timeout_is_not_fatal_and_is_bounded(self):
        self.assertEqual(post_processor.LATEX_PASS_TIMEOUT_S, 300)
        with workdir() as folder:
            result, records = self.compile_with(
                folder, side_effect=subprocess.TimeoutExpired("x", 300))
        self.assertEqual(result, "")
        self.assertTrue(any("300 s" in text for _, text in records))

    def test_error_return_code_discards_the_pdf_keeps_the_log_and_cites_the_error(self):
        log = "ligne\n! Undefined control sequence.\nl.3 \\badcommand\nfin\n"
        with workdir() as folder:
            result, records = self.compile_with(folder, returncode=1, make_log=log)
            self.assertEqual(result, "")
            self.assertFalse(os.path.exists(os.path.join(folder, "t.pdf")))
            self.assertTrue(os.path.exists(os.path.join(folder, "t.log")))
        self.assertTrue(any("Undefined control sequence" in text for _, text in records))

    def test_success_cleans_auxiliary_files(self):
        with workdir() as folder:
            result, _ = self.compile_with(folder, returncode=0, make_log="tout va bien")
            self.assertTrue(result.endswith("t.pdf"))
            self.assertFalse(os.path.exists(os.path.join(folder, "t.log")))

    def test_merge_reports_missing_sections(self):
        with workdir() as folder:
            for name in ("a.pdf", "b.pdf"):
                figure = post_processor.plt.figure(figsize=(2, 2))
                figure.savefig(os.path.join(folder, name))
                post_processor.plt.close(figure)
            with contextlib.redirect_stdout(io.StringIO()):
                path, missing = post_processor.merge_pdfs_report(
                    folder, "tout.pdf", ["a.pdf", "manquant.pdf", "b.pdf"])
            self.assertTrue(os.path.isfile(path))
            self.assertEqual(missing, ["manquant.pdf"])
            path, missing = post_processor.merge_pdfs_report(folder, "rien.pdf", ["x.pdf"])
            self.assertEqual((path, missing), ("", ["x.pdf"]))


class TestWithoutLatex(unittest.TestCase):
    def check_outputs(self, folder, results):
        out = os.path.join(folder, "output", "run")
        names = set(os.listdir(out))
        for expected in ("02-OCA_CorrMat_Targets.pdf", "03-Step1_OrdinalGraph_Targets.pdf",
                         "05-OCA_OrdinalCorrMat_Targets.pdf", "06-OCA_OrdinalGraph_Targets.pdf",
                         "08-OCA_Boxplot_Targets.pdf", "09-OCA_Hist_Targets.pdf",
                         "10-OCA_Scatter_Targets.pdf", "OCA_Sort_Targets.txt",
                         "Step1_OrdinalCorrMat_Targets.txt", "Step1_OrdinalGraph_Targets.txt",
                         "OCA_OrdinalGraph_Targets.txt", "OCA_run.log", "data.xlsx",
                         "01-Step1_Dominance_Hierarchy_Result.tex",
                         "04-Step1_Critical_Zones_Sample_Result.tex",
                         "07-CriticalZones_Targets.tex", "11-OCA_Report_intro.tex",
                         "00-OCA_Report.pdf"):
            self.assertIn(expected, names)
        self.assertTrue(os.path.isfile(os.path.join(folder, "output.json")))
        report = results["report"]
        self.assertFalse(report["complete"])
        self.assertTrue(report["partial"])
        self.assertFalse(report["notice_included"])
        self.assertIn("01-Step1_Dominance_Hierarchy_Result.pdf", report["missing_sections"])
        self.assertIn("07-CriticalZones_Targets.pdf", report["missing_sections"])

    def test_compiler_none(self):
        with workdir() as folder:
            results = run_pipeline(folder, base_frame(), compiler="none")
            self.check_outputs(folder, results)

    def test_compiler_none_cli_exit_code_zero_and_single_message(self):
        with workdir() as folder:
            write_book(folder, base_frame())
            path = write_config(folder, minimal_config(latex_output={"compiler": "none"}))
            result = subprocess.run(
                [PYTHON, "-X", "utf8", os.path.join(ROOT, "main_OCA.py"), "--config", path],
                capture_output=True, text=True, cwd=folder, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(result.stdout.count("Compilation LaTeX désactivée"), 1)
        self.assertIn("PARTIEL", result.stdout)

    def test_path_emptied_and_no_usual_location(self):
        with workdir() as folder, \
                mock.patch.dict(os.environ, {"PATH": ""}), \
                mock.patch.object(latex_support, "_candidate_dirs", return_value=[]):
            write_book(folder, base_frame())
            path = write_config(folder, minimal_config())
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                document = main_OCA.run(path)
            self.assertEqual(output.getvalue().count("introuvable sur ce poste"), 1)
            self.check_outputs(folder, document["results"])
        latex_support.configure("auto")


# ---------------------------------------------------------------------------
# F34, F35 : légende et axe
# ---------------------------------------------------------------------------
class TestCaptionsAndLabels(unittest.TestCase):
    def test_sample_caption_uses_the_effective_size(self):
        with workdir() as folder:
            frame = base_frame(2)
            frame["alpha"] = 1.0
            frame["beta"] = 1.0
            path = post_processor.generate_criticalzones_table(
                frame, "ZoneA", 1.5, 2.0, 1.2, 30, folder, 6, "t")
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        self.assertIn("Sélection de 2 lignes", text)
        self.assertNotIn("Sélection de 30 lignes", text)

    def test_boxplot_axis_has_no_unit(self):
        labels = []
        original = post_processor.plt.Axes.set_ylabel

        def recorder(self, label, *args, **kwargs):
            labels.append(label)
            return original(self, label, *args, **kwargs)

        with workdir() as folder, mock.patch.object(post_processor.plt.Axes, "set_ylabel",
                                                    recorder):
            post_processor.plot_boxplot(base_frame(20), ["ZoneA", "ZoneB"],
                                        os.path.join(folder, "b.pdf"))
        self.assertTrue(labels)
        self.assertTrue(all("MPa" not in label for label in labels))
        self.assertIn("Value of the critical zones", labels)


# ---------------------------------------------------------------------------
# Versions, manifestes, confidentialité
# ---------------------------------------------------------------------------
class TestPackagingAndVersion(unittest.TestCase):
    def read(self, name):
        with open(os.path.join(ROOT, name), encoding="utf-8") as handle:
            return handle.read()

    def test_version_is_9_0_0_everywhere(self):
        self.assertEqual(cm.OCA_VERSION, "9.0.0")
        project = tomllib.loads(self.read("pyproject.toml"))
        self.assertEqual(project["project"]["version"], "9.0.0")
        for name in ("README.md", "AI_AGENT_GUIDE.md", "CHANGELOG.md"):
            self.assertIn("9.0.0", self.read(name), name)

    def test_embedded_template_matches_input_json_and_init_never_overwrites(self):
        shipped = self.read("input.json").replace("\r\n", "\n")
        self.assertEqual(main_OCA.TEMPLATE_INPUT_JSON.replace("\r\n", "\n"), shipped)
        with workdir() as folder:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main_OCA.scaffold_working_directory(folder), 0)
                target = os.path.join(folder, "input.json")
                with open(target, "w", encoding="utf-8") as handle:
                    handle.write("{}")
                self.assertEqual(main_OCA.scaffold_working_directory(folder), 0)
            with open(target, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "{}")
            self.assertTrue(os.path.isdir(os.path.join(folder, "input")))

    def test_shipped_configurations_load_with_few_comments(self):
        for name in ("input.json", os.path.join("examples", "CAS_TEST_00", "input.json")):
            config = cm.load_config(os.path.join(ROOT, name))
            self.assertEqual(config.latex_compiler, "auto")
            self.assertTrue(config.comment_notes, name)

    def test_pyproject_metadata(self):
        project = tomllib.loads(self.read("pyproject.toml"))
        self.assertEqual(project["project"]["license"], "AGPL-3.0-or-later")
        classifiers = project["project"].get("classifiers", [])
        self.assertFalse([c for c in classifiers if c.startswith("License ::")])
        requires = " ".join(project["build-system"]["requires"])
        self.assertRegex(requires, r"setuptools>=77")
        dependencies = " ".join(project["project"]["dependencies"]).lower()
        for package in ("jinja2", "openpyxl", "pandas", "numpy", "scipy", "matplotlib",
                        "seaborn", "networkx", "pypdf"):
            self.assertIn(package, dependencies)
        self.assertIn("latex_support", project["tool"]["setuptools"]["py-modules"])
        requirements = self.read("requirements.txt").lower()
        self.assertIn("jinja2", requirements)
        self.assertIn("openpyxl", requirements)

    def test_every_third_party_import_is_declared(self):
        declared = (self.read("requirements.txt") + self.read("pyproject.toml")).lower()
        aliases = {"yaml": "pyyaml"}
        stdlib = set(sys.stdlib_module_names)
        local = {"config_manager", "oca_model", "post_processor", "latex_support",
                 "main_OCA"}
        imported = set()
        for name in ("main_OCA.py", "config_manager.py", "oca_model.py", "post_processor.py",
                     "latex_support.py"):
            for match in re.finditer(
                    r"^\s*(?:import\s+([A-Za-z_]\w*)|from\s+([A-Za-z_]\w*)\s+import\b)",
                    self.read(name), flags=re.M):
                imported.add(match.group(1) or match.group(2))
        optional = {"PyPDF2"}   # repli facultatif de pypdf, non déclaré
        for module in sorted(imported - stdlib - local - optional - {"__future__"}):
            self.assertIn(aliases.get(module, module).lower(), declared, module)

    def test_no_personal_data_or_forbidden_vocabulary_in_the_deliverables(self):
        forbidden = [re.compile(pattern, re.I) for pattern in (
            "ce" + "tim", r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}",
            r"[A-Za-z]:\\Users", r"/home/", r"C:/Users", "_coop" + "_runs", "co" + "dex",
            "clau" + "de", r"\bsous-agent\b", r"plan de travail", "anti" + "gravity")]
        names = []
        for folder, _dirs, files in os.walk(ROOT):
            if any(part in folder for part in (".git", "__pycache__", "output")):
                continue
            for file in files:
                if file.endswith((".py", ".md", ".json", ".toml", ".txt", ".tex", ".gitignore")):
                    names.append(os.path.join(folder, file))
        self.assertTrue(names)
        for path in names:
            if os.path.basename(path) == "test_robustness_v9.py":
                continue
            with open(path, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
            for pattern in forbidden:
                match = pattern.search(text)
                self.assertIsNone(match, f"{os.path.relpath(path, ROOT)} : {match and match.group(0)}")


# ===========================================================================
# Seconde passe de corrections (N01 à N18)
# ===========================================================================
def write_zone_file(zone: str, frame: pd.DataFrame, base: str = "input") -> str:
    """Écrit ``<base>/dataSet_<zone>/dataSet_<zone>.xlsx`` (fusion par-zone)."""
    target = os.path.join(base, f"dataSet_{zone}")
    os.makedirs(target, exist_ok=True)
    path = os.path.join(target, f"dataSet_{zone}.xlsx")
    frame.to_excel(path, index=False)
    return path


def pdf_pages(path: str) -> list:
    """Texte extrait de chaque page d'un PDF (pypdf, ou PyPDF2 en repli)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader
    return [page.extract_text() or "" for page in PdfReader(path).pages]


def squeeze(text: str) -> str:
    """Texte sans aucun espace, pour des comparaisons indifférentes au retour à la ligne."""
    return re.sub(r"\s+", "", text)


# ---------------------------------------------------------------------------
# N01 : fusion par-zone, variables inférées
# ---------------------------------------------------------------------------
class TestPerZoneInferredVariables(unittest.TestCase):
    def load(self, folder, zones=("ZA", "ZB")):
        config = cm.load_config(write_config(folder, {
            "critical_zones": list(zones), "merged_subdir": "run"}))
        return cm.load_data_with_resolution(config.for_subdir("run"))

    def test_column_permutation_between_zones_changes_nothing(self):
        # Configurations physiques (x, y) = (1, 2) et (2, 1).
        a = pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 1.0], "T": [3.0, 1.0]})
        b_same = pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 1.0], "T": [2.0, 4.0]})
        b_swapped = pd.DataFrame({"y": [2.0, 1.0], "x": [1.0, 2.0], "T": [2.0, 4.0]})
        with workdir() as folder:
            write_zone_file("ZA", a)
            write_zone_file("ZB", b_same)
            same, zones, resolution_same = self.load(folder)
        with workdir() as folder:
            write_zone_file("ZA", a)
            write_zone_file("ZB", b_swapped)
            swapped, _, resolution_swapped = self.load(folder)
        self.assertEqual(resolution_same.features, ["x", "y"])
        self.assertEqual(resolution_swapped.features, ["x", "y"])
        pd.testing.assert_frame_equal(same, swapped)
        # A >= B sur la moitié des configurations (3 >= 2 oui, 1 >= 4 non).
        self.assertEqual(float((swapped["ZA"] >= swapped["ZB"]).mean()), 0.5)

    def test_different_variable_sets_are_refused_with_files_and_columns(self):
        a = pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 1.0], "T": [3.0, 1.0]})
        b = pd.DataFrame({"x": [1.0, 2.0], "z": [2.0, 1.0], "T": [2.0, 4.0]})
        with workdir() as folder:
            write_zone_file("ZA", a)
            write_zone_file("ZB", b)
            with self.assertRaises(ConfigError) as ctx:
                self.load(folder)
        message = str(ctx.exception)
        for fragment in ("dataSet_ZA.xlsx", "dataSet_ZB.xlsx", "'y'", "'z'"):
            self.assertIn(fragment, message)

    def test_declared_variables_still_work_with_a_comment_column(self):
        a = pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 1.0], "T": [3.0, 1.0]})
        b = pd.DataFrame({"y": [2.0, 1.0], "x": [1.0, 2.0], "T": [2.0, 4.0]})
        with workdir() as folder:
            write_zone_file("ZA", a)
            write_zone_file("ZB", b)
            config = cm.load_config(write_config(folder, {
                "critical_zones": ["ZA", "ZB"], "feature_cols_init": ["x", "y"],
                "merged_subdir": "run"}))
            df, _, _ = cm.load_data_with_resolution(config.for_subdir("run"))
        self.assertEqual(float((df["ZA"] >= df["ZB"]).mean()), 0.5)


# ---------------------------------------------------------------------------
# N02, N16 : sorties qui désignent une source ou se recouvrent
# ---------------------------------------------------------------------------
class TestOutputCollisions(unittest.TestCase):
    def setUp(self):
        self.ctx = workdir()
        self.folder = self.ctx.__enter__()
        base = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 2.0, 1.0]})
        self.sources = {}
        for zone, offset in (("ZA", 0.0), ("ZB", 1.0)):
            frame = base.copy()
            frame["T"] = [3.0 + offset, 1.0, 2.0]
            self.sources[zone] = write_zone_file(zone, frame)
        self.before = {}
        for zone, path in self.sources.items():
            with open(path, "rb") as handle:
                self.before[zone] = handle.read()

    def tearDown(self):
        self.ctx.__exit__(None, None, None)

    def assert_sources_intact(self):
        for zone, path in self.sources.items():
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), self.before[zone])

    def refused(self, **overrides):
        cfg = {"critical_zones": ["ZA", "ZB"], "merged_subdir": "run",
               "latex_output": {"compiler": "none"}}
        cfg.update(overrides)
        path = write_config(self.folder, cfg)
        with self.assertRaises(ConfigError) as ctx:
            with contextlib.redirect_stdout(io.StringIO()):
                main_OCA.run(path)
        self.assert_sources_intact()
        return str(ctx.exception)

    def test_output_json_naming_a_per_zone_source_is_refused_before_any_work(self):
        for spelling in ("./input/dataSet_ZA/dataSet_ZA.xlsx",
                         "input/../input/dataSet_ZB/dataSet_ZB.xlsx"):
            with self.subTest(spelling):
                message = self.refused(output_json=spelling)
                self.assertIn("fichier source", message)
        self.assertFalse(os.path.exists("output"))

    def test_output_json_naming_a_merged_workbook_is_refused(self):
        write_book(self.folder, base_frame(10))
        message = self.refused(
            critical_zones=["ZoneA", "ZoneB"], output_json="./input/run/data.xlsx",
            data_files="data.xlsx")
        self.assertIn("fichier source", message)

    def test_two_output_files_with_the_same_name_are_refused(self):
        message = self.refused(output_files={"corr_matrix": "figure.pdf",
                                             "step1_graph": "figure.pdf"})
        self.assertIn("corr_matrix", message)
        self.assertIn("step1_graph", message)

    def test_consolidated_report_named_like_a_section_is_refused(self):
        message = self.refused(output_files={
            "consolidated_report": "01-Step1_Dominance_Hierarchy_Result.pdf"})
        self.assertIn("consolidated_report", message)

    def test_output_json_naming_an_output_file_is_refused(self):
        message = self.refused(output_json="./output/run/00-OCA_Report.pdf")
        self.assertIn("output_json", message)

    def test_output_file_naming_a_source_is_refused(self):
        with self.assertRaises(ConfigError) as ctx:
            cm.check_output_collisions(
                "in", "in", ["run"], "d.csv", [], "o.json",
                dict(cm.DEFAULT_CONFIG["output_files"], log="d.csv"))
        self.assertIn("fichier source", str(ctx.exception))

    def test_default_names_do_not_collide(self):
        cm.check_output_collisions(
            "in", "out", ["a", "b"], "d.xlsx", ["ZA"], "o.json",
            cm.DEFAULT_CONFIG["output_files"])


# ---------------------------------------------------------------------------
# N03 : première colonne portant un nom d'index usuel
# ---------------------------------------------------------------------------
class TestIndexLikeNames(unittest.TestCase):
    def frame_with_n(self, position):
        frame = base_frame(60)
        frame["n"] = np.random.default_rng(11).permutation(len(frame)).astype(float) * 0.5
        columns = [name for name in frame.columns if name != "n"]
        columns.insert(position, "n")
        return frame[columns]

    def test_declared_n_is_kept_in_first_or_second_position(self):
        results = {}
        for position in (0, 1):
            with workdir() as folder:
                results[position] = run_pipeline(
                    folder, self.frame_with_n(position),
                    feature_cols_init=["n", "V1", "V2", "V3"])
        for position in (0, 1):
            self.assertIn("n", results[position]["column_resolution"]["feature_cols"])
        self.assertEqual(
            results[0]["initial_ordinal_state"]["ordinal_correlation_matrix"],
            results[1]["initial_ordinal_state"]["ordinal_correlation_matrix"])
        self.assertEqual(results[0]["metrics"]["kappa_optimal"],
                         results[1]["metrics"]["kappa_optimal"])

    def test_n_cited_by_a_synthetic_expression_is_kept(self):
        frame = self.frame_with_n(0)
        frame["n"] += 1.0
        with workdir() as folder:
            write_book(folder, frame)
            config = cm.load_config(write_config(folder, minimal_config(
                feature_cols_init=[], user_defined_vars=True,
                user_defined_vars_cols=["n*ZoneA"])))
            df, zones, resolution = cm.load_data_with_resolution(config.for_subdir("run"))
        self.assertIsNone(resolution.index_column)
        self.assertIn("n", df.columns)
        self.assertIn("n*ZoneA", zones)

    def test_undeclared_n_in_first_position_is_dropped_with_a_log_line(self):
        frame = self.frame_with_n(0)
        with workdir() as folder:
            write_book(folder, frame)
            config = cm.load_config(write_config(folder, minimal_config(feature_cols_init=[])))
            logger, handler = make_logger()
            df, _, resolution = cm.load_data_with_resolution(config.for_subdir("run"), logger)
        self.assertEqual(resolution.index_column, "n")
        self.assertNotIn("n", df.columns)
        self.assertTrue(any("reconnue comme index" in text for _, text in handler.records))

    def test_undeclared_n_in_second_position_is_a_variable(self):
        frame = self.frame_with_n(1)
        with workdir() as folder:
            write_book(folder, frame)
            config = cm.load_config(write_config(folder, minimal_config(feature_cols_init=[])))
            df, _, resolution = cm.load_data_with_resolution(config.for_subdir("run"))
        self.assertIsNone(resolution.index_column)
        self.assertIn("n", resolution.features)


# ---------------------------------------------------------------------------
# N04, N05, N06 : κ publié, débordement, noms réservés
# ---------------------------------------------------------------------------
class TestKappaPublicationAndOverflow(unittest.TestCase):
    def test_published_kappa_is_the_tested_float(self):
        tested = 1.0000000000000002
        frame = pd.DataFrame({
            "V1": np.arange(1, 11, dtype=float),
            "A": [1.0] * 10, "B": [tested] * 10})
        with workdir() as folder:
            results = run_pipeline(
                folder, frame, critical_zones=["A", "B"], feature_cols_init=["V1"],
                gpo_selection={"threshold_low": 0.0, "threshold_high": 10.0},
                gpo_regularization={"target_zone": "A", "threshold": 1.0,
                                    "kappa_min": tested, "kappa_max": tested,
                                    "kappa_step": 0.01})
        published = results["metrics"]["kappa_optimal"]
        self.assertTrue(results["metrics"]["kappa_found"])
        self.assertEqual(published, tested)
        self.assertTrue((published * frame["A"] >= frame["B"]).all())

    def test_grid_points_must_be_distinct(self):
        with self.assertRaises(ValueError):
            oca_model.kappa_grid(1e-16, 4e-16, 1e-16)
        with workdir() as folder:
            with self.assertRaises(ConfigError) as ctx:
                cm.load_config(write_config(folder, minimal_config(
                    gpo_regularization={"kappa_min": 1e-16, "kappa_max": 4e-16,
                                        "kappa_step": 1e-16})))
        self.assertIn("gpo_regularization", str(ctx.exception))

    def test_examples_grid_is_unchanged(self):
        grid = oca_model.kappa_grid(1.0, 3.0, 0.01)
        self.assertEqual(len(grid), 201)
        self.assertEqual(grid[55], 1.55)
        self.assertEqual(oca_model.kappa_grid(1.0, 1.2, 0.1), [1.0, 1.1, 1.2])

    def test_envelope_overflow_is_refused_not_a_success(self):
        frame = pd.DataFrame({"A": [1e308] * 3, "B": [1.7e308] * 3})
        with self.assertRaises(ConfigError) as ctx:
            oca_model.find_kappa_details(frame, ["A", "B"], "A", 1.0, 1.0, 2.0, 1.0)
        self.assertIn("Débordement", str(ctx.exception))
        with self.assertRaises(ConfigError):
            oca_model.ordinal_correlation_matrix_with_envelope(frame, ["A", "B"], "A", 2.0)
        with self.assertRaises(ConfigError):
            oca_model.compute_parsimony(frame, ["A", "B"], "A", 2.0)

    def test_envelope_overflow_in_a_full_run_is_a_config_error(self):
        frame = pd.DataFrame({
            "V1": np.arange(1, 6, dtype=float),
            "A": [1e308, 1e308, 1e308, 1e308, 1e308],
            "B": [1.7e308, 1.7e308, 1.7e308, 1.7e308, 1.7e308]})
        # Les figures de grandeur 1e308 échouent déjà dans matplotlib : elles sont
        # neutralisées pour atteindre la recherche de κ.
        patches = [mock.patch.object(main_OCA, name, return_value="x.pdf")
                   for name in ("plot_boxplot", "plot_hist", "plot_scatter",
                                "plot_ordinal_corr_matrix", "plot_ordinal_graph")]
        with workdir() as folder:
            for patch in patches:
                patch.start()
            try:
                with self.assertRaises(ConfigError) as ctx:
                    run_pipeline(
                        folder, frame, critical_zones=["A", "B"], feature_cols_init=["V1"],
                        gpo_selection={"threshold_low": 0.0, "threshold_high": 10.0},
                        gpo_regularization={"target_zone": "A", "threshold": 1.0,
                                            "kappa_min": 1.0, "kappa_max": 2.0,
                                            "kappa_step": 1.0})
            finally:
                for patch in patches:
                    patch.stop()
        self.assertIn("Débordement", str(ctx.exception))


class TestReservedNames(unittest.TestCase):
    def test_names_starting_with_a_dollar_or_generated_are_refused(self):
        for name in ("$A", "$\\alpha$A", "$\\beta$", "2.00$\\times$A", "1$\\times$De/E"):
            with self.subTest(name), workdir() as folder:
                with self.assertRaises(ConfigError) as ctx:
                    cm.load_config(write_config(folder, minimal_config(
                        critical_zones=[name, "ZoneB"])))
                self.assertIn("réservé", str(ctx.exception))
                with self.assertRaises(ConfigError):
                    cm.load_config(write_config(folder, minimal_config(
                        feature_cols_init=["V1", name])))

    def test_workbook_headers_are_checked_when_inferred(self):
        frame = base_frame(20).rename(columns={"ZoneA": "$weird"})
        with workdir() as folder:
            write_book(folder, frame)
            config = cm.load_config(write_config(folder, minimal_config(
                critical_zones=[], feature_cols_init=["V1", "V2", "V3"])))
            with self.assertRaises(ConfigError) as ctx:
                cm.load_data_with_resolution(config.for_subdir("run"))
        self.assertIn("$weird", str(ctx.exception))

    def test_per_zone_key_name_is_reserved(self):
        base = pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 1.0]})
        with workdir() as folder:
            for zone in ("config_id", "ZB"):
                frame = base.copy()
                frame["T"] = [1.0, 2.0]
                write_zone_file(zone, frame)
            config = cm.load_config(write_config(folder, {
                "critical_zones": ["config_id", "ZB"], "merged_subdir": "run"}))
            with self.assertRaises(ConfigError):
                cm.load_data_with_resolution(config.for_subdir("run"))

    def test_envelope_never_overwrites_a_source_column(self):
        frame = pd.DataFrame({"A": [1.0, 2.0], "2.00$\\times$A": [3.0, 4.0]})
        with self.assertRaises(ConfigError):
            oca_model.ordinal_correlation_matrix_with_envelope(
                frame, ["A", "2.00$\\times$A"], "A", 2.0)


# ---------------------------------------------------------------------------
# N07 : noms bruts affichés littéralement par matplotlib
# ---------------------------------------------------------------------------
class TestLiteralNamesInFigures(unittest.TestCase):
    NAMES = {"ZoneA": "Z$x$", "ZoneB": "Z$\\notACommand$", "ZoneC": "Z$"}

    def test_mpl_text_escapes_user_dollars_and_keeps_generated_prefix(self):
        self.assertEqual(post_processor.mpl_text("Z$x$"), "Z\\$x\\$")
        self.assertEqual(post_processor.mpl_text("a_b"), "a_b")
        self.assertEqual(post_processor.mpl_text("2.00$\\times$A$x"),
                         "2.00$\\times$A\\$x")

    def test_full_run_with_dollar_names_renders_every_figure(self):
        frame = base_frame(40).rename(columns=self.NAMES)
        with workdir() as folder:
            results = run_pipeline(folder, frame, critical_zones=list(self.NAMES.values()))
            out = os.path.join(folder, "output", "run")
            for name in ("02-OCA_CorrMat_Targets.pdf", "03-Step1_OrdinalGraph_Targets.pdf",
                         "05-OCA_OrdinalCorrMat_Targets.pdf", "06-OCA_OrdinalGraph_Targets.pdf",
                         "08-OCA_Boxplot_Targets.pdf", "09-OCA_Hist_Targets.pdf",
                         "10-OCA_Scatter_Targets.pdf"):
                self.assertGreater(os.path.getsize(os.path.join(out, name)), 0, name)
        self.assertEqual(sorted(results["initial_ordinal_state"]["critical_zones"]),
                         sorted(self.NAMES.values()))

    def test_labels_are_passed_escaped_to_matplotlib(self):
        labels = []
        original = post_processor.plt.Axes.set_xlabel

        def recorder(self, label, *args, **kwargs):
            labels.append(label)
            return original(self, label, *args, **kwargs)

        frame = base_frame(20).rename(columns=self.NAMES)
        with workdir() as folder, mock.patch.object(post_processor.plt.Axes, "set_xlabel",
                                                    recorder):
            post_processor.plot_hist(frame, list(self.NAMES.values()),
                                     os.path.join(folder, "h.pdf"))
        self.assertIn("Z\\$x\\$", labels)
        self.assertIn("Z\\$\\notACommand\\$", labels)

    @unittest.skipUnless(has_latex(), "LaTeX absent")
    def test_dollar_names_compile_in_the_tables(self):
        frame = base_frame(40).rename(columns=self.NAMES)
        with workdir() as folder:
            results = run_pipeline(folder, frame, compiler="auto",
                                   critical_zones=list(self.NAMES.values()))
        self.assertFalse(any("en échec" in line
                             for line in results["execution_log"]["records"]))


# ---------------------------------------------------------------------------
# N08, N09, N10, N18 : manifestes et exactitude documentaire
# ---------------------------------------------------------------------------
def version_tuple(text: str) -> tuple:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


class TestManifestsAndDocumentation(unittest.TestCase):
    def read(self, *parts):
        with open(os.path.join(ROOT, *parts), encoding="utf-8") as handle:
            return handle.read()

    def test_minimal_versions_match_pandas_requirements(self):
        from pandas.compat._optional import VERSIONS
        lowered = {key.lower(): value for key, value in VERSIONS.items()}
        project = tomllib.loads(self.read("pyproject.toml"))
        manifests = {"pyproject": project["project"]["dependencies"],
                     "requirements": [line.strip() for line in
                                      self.read("requirements.txt").splitlines()
                                      if line.strip() and not line.startswith("#")]}
        for label, lines in manifests.items():
            for line in lines:
                match = re.match(r"([A-Za-z0-9_.-]+)>=([0-9.]+)", line)
                self.assertIsNotNone(match, f"{label} : {line}")
                name, minimum = match.group(1).lower(), match.group(2)
                if name in lowered:
                    self.assertGreaterEqual(
                        version_tuple(minimum), version_tuple(lowered[name]),
                        f"{label} : {line} < minimum de pandas {lowered[name]}")
        for label, lines in manifests.items():
            text = " ".join(lines).lower()
            self.assertIn("openpyxl>=3.1.5", text, label)
            self.assertIn("jinja2>=3.1.5", text, label)
        readme = self.read("README.md")
        self.assertRegex(readme, r"`openpyxl`\s*\|\s*3\.1\.5")
        self.assertRegex(readme, r"`Jinja2`\s*\|\s*3\.1\.5")

    def test_readme_claims_are_exact(self):
        readme = " ".join(self.read("README.md").replace("**", "").split())
        head = readme[:1500]
        self.assertIn("retenues par la sélection", head)
        self.assertNotIn("toutes les autres au seuil", head)
        self.assertNotIn("borne haute incluse", readme)
        self.assertIn("à la tolérance d'arrondi près", readme)
        self.assertNotIn("cible incalculable ou absente, absence de relation", readme)
        self.assertIn("refusés avec un message", readme)
        self.assertIn("arrondies à 4 décimales", readme)
        self.assertIn('"status": "running"', readme)
        guide = " ".join(self.read("AI_AGENT_GUIDE.md").replace("**", "").split())
        self.assertNotIn("The selection band", guide)
        self.assertNotIn("borne haute incluse", guide)
        self.assertIn("Limites connues", guide)

    def test_examples_readme_distinguishes_the_three_studies(self):
        text = self.read("examples", "README.md")
        for fragment in ("907", "400", "1,55", "1,35", "1,53", "output.json"):
            self.assertIn(fragment, text)

    def test_changelog_does_not_overstate_the_config_path(self):
        self.assertNotIn("n'est plus consigné", self.read("CHANGELOG.md"))
        self.assertIn("tel que fourni", self.read("CHANGELOG.md"))

    def test_notice_does_not_present_the_high_threshold_as_a_filter(self):
        source = self.read("post_processor.py")
        self.assertNotIn("seuils bas et haut de filtrage", source)


# ---------------------------------------------------------------------------
# N13, N14 : rapport consolidé partiel
# ---------------------------------------------------------------------------
class TestPartialReport(unittest.TestCase):
    def test_status_page_leads_the_partial_report_without_latex(self):
        with workdir() as folder:
            results = run_pipeline(folder, base_frame(), compiler="none")
            out = os.path.join(folder, "output", "run")
            pages = pdf_pages(os.path.join(out, "00-OCA_Report.pdf"))
            leftovers = [name for name in os.listdir(out) if name.startswith("_statut")]
        self.assertIn("Rapportpartiel", squeeze(pages[0]))
        for name in results["report"]["missing_sections"]:
            self.assertIn(squeeze(name), squeeze(pages[0]))
        self.assertTrue(results["report"]["status_page_included"])
        self.assertFalse(leftovers)
        self.assertEqual(len(pages), 1 + 7)   # page de statut + sept figures

    def test_partial_status_page_function_is_independent_of_latex(self):
        with workdir() as folder:
            path = post_processor.write_partial_status_pdf(
                os.path.join(folder, "s.pdf"), ["a.pdf", "b$.pdf"], "cause d'essai")
            text = squeeze(pdf_pages(path)[0])
        self.assertIn("Rapportpartiel", text)
        self.assertIn("a.pdf", text)
        self.assertIn("b$.pdf", text)

    def test_failed_pdf_is_never_merged_even_if_its_removal_failed(self):
        stale_text = "ANCIENCONTENU"

        def fake_compile(name, filedir, original_directory, logger=None, **kwargs):
            if name.startswith("01-"):
                # compilation en échec dont le PDF n'a pas pu être retiré
                figure = post_processor.plt.figure(figsize=(2, 2))
                figure.text(0.1, 0.5, stale_text)
                figure.savefig(os.path.join(filedir, f"{name}.pdf"))
                post_processor.plt.close(figure)
            return ""

        with workdir() as folder:
            with mock.patch.object(main_OCA, "compile_latex", fake_compile):
                results = run_pipeline(folder, base_frame(), compiler="none")
            out = os.path.join(folder, "output", "run")
            pages = pdf_pages(os.path.join(out, "00-OCA_Report.pdf"))
        self.assertIn("01-Step1_Dominance_Hierarchy_Result.pdf",
                      results["report"]["missing_sections"])
        self.assertFalse(results["report"]["complete"])
        self.assertNotIn(stale_text, squeeze(" ".join(pages)))


# ---------------------------------------------------------------------------
# N15 : chemins d'erreur de la ligne de commande
# ---------------------------------------------------------------------------
class TestCommandLineErrors(unittest.TestCase):
    def cli(self, args, cwd, code=None):
        launcher = ("import sys; sys.path.insert(0, %r); import main_OCA; "
                    "sys.argv = ['oca'] + %r; main_OCA.main()" % (ROOT, list(args)))
        if code:
            launcher = code
        return subprocess.run([PYTHON, "-X", "utf8", "-c", launcher], capture_output=True,
                              text=True, cwd=cwd, encoding="utf-8")

    def test_huge_json_integer_names_the_key(self):
        text = ('{"critical_zones": [], "gpo_regularization": {"kappa_step": 1'
                + "0" * 400 + "}}")
        with workdir() as folder:
            with self.assertRaises(ConfigError) as ctx:
                cm.load_config(write_config(folder, text))
        self.assertIn("gpo_regularization.kappa_step", str(ctx.exception))
        text = '{"sample_random_state": 1' + "0" * 400 + "}"
        with workdir() as folder:
            with self.assertRaises(ConfigError) as ctx:
                cm.load_config(write_config(folder, text))
        self.assertIn("sample_random_state", str(ctx.exception))

    def test_json_integer_beyond_the_decoder_limit_is_a_config_error(self):
        text = '{"sample_random_state": 1' + "0" * 5000 + "}"
        with workdir() as folder:
            with self.assertRaises(ConfigError):
                cm.load_config(write_config(folder, text))

    def test_init_when_input_is_a_file(self):
        with workdir() as folder:
            with open("input", "w", encoding="utf-8") as handle:
                handle.write("je suis un fichier")
            result = self.cli(["--init"], folder)
            with open("input", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "je suis un fichier")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("c'est un fichier", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_init_in_an_unwritable_folder_is_a_short_message(self):
        with workdir() as folder, \
                mock.patch.object(sys, "argv", ["oca", "--init"]), \
                mock.patch("pathlib.Path.write_bytes", side_effect=PermissionError("non")):
            error = io.StringIO()
            with contextlib.redirect_stderr(error), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    main_OCA.main()
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("inscriptible", error.getvalue())
        self.assertNotIn("Traceback", error.getvalue())

    def test_init_twice_keeps_the_configuration(self):
        with workdir() as folder:
            first = self.cli(["--init"], folder)
            with open("input.json", "w", encoding="utf-8") as handle:
                handle.write("{}")
            second = self.cli(["--init"], folder)
            with open("input.json", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "{}")
        self.assertEqual((first.returncode, second.returncode), (0, 0))

    def test_missing_dependency_gives_one_clear_line(self):
        code = ("import sys; sys.modules['openpyxl'] = None; sys.path.insert(0, %r); "
                "import main_OCA" % ROOT)
        with workdir() as folder:
            result = self.cli([], folder, code=code)
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn("openpyxl", result.stderr)
        self.assertIn("pip install -r requirements.txt", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(len(result.stderr.strip().splitlines()), 1)

    def test_input_directory_that_is_a_file(self):
        with workdir() as folder:
            with open("fichier", "w", encoding="utf-8") as handle:
                handle.write("x")
            path = write_config(folder, minimal_config(input_dir="./fichier"))
            with self.assertRaises(ConfigError) as ctx:
                with contextlib.redirect_stdout(io.StringIO()):
                    main_OCA.run(path)
        self.assertIn("input_dir", str(ctx.exception))


# ---------------------------------------------------------------------------
# N17 : état de la tentative dans output.json
# ---------------------------------------------------------------------------
class TestAttemptStatus(unittest.TestCase):
    def read_json(self, folder):
        with open(os.path.join(folder, "output.json"), encoding="utf-8") as handle:
            return json.load(handle)

    def test_success_carries_status_complete_and_unchanged_content(self):
        with workdir() as folder:
            run_pipeline(folder, base_frame())
            document = self.read_json(folder)
        self.assertEqual(document["status"], "complete")
        self.assertEqual(set(document), {"status", "generated_at", "config_file", "results"})

    def test_running_document_is_written_at_the_start(self):
        seen = []

        def spy(config, config_path):
            with open(config.output_json, encoding="utf-8") as handle:
                seen.append(json.load(handle))
            return {}

        with workdir() as folder:
            write_book(folder, base_frame())
            path = write_config(folder, minimal_config())
            with open("output.json", "w", encoding="utf-8") as handle:
                handle.write('{"status": "complete", "ancien": true}')
            with mock.patch.object(main_OCA, "_run_configured", spy):
                main_OCA.run(path)
        self.assertEqual(seen[0]["status"], "running")
        self.assertIn("started_at", seen[0])
        self.assertNotIn("ancien", seen[0])

    def test_total_failure_replaces_the_old_success_with_a_failed_document(self):
        with workdir() as folder:
            run_pipeline(folder, base_frame())
            self.assertEqual(self.read_json(folder)["status"], "complete")
            os.remove(os.path.join(folder, "input", "run", "data.xlsx"))
            path = os.path.join(folder, "input.json")
            with self.assertRaises(ConfigError):
                with contextlib.redirect_stdout(io.StringIO()):
                    main_OCA.run(path)
            document = self.read_json(folder)
        self.assertEqual(document["status"], "failed")
        self.assertIn("Aucune donnée", document["reason"])
        self.assertNotIn("results", document)

    def test_partial_batch_is_marked_partial(self):
        with workdir() as folder:
            write_book(folder, base_frame())
            path = write_config(folder, minimal_config(
                merged_subdir=["run", "absent"], latex_output={"compiler": "none"}))
            with contextlib.redirect_stdout(io.StringIO()):
                main_OCA.run(path)
            document = self.read_json(folder)
            per_run = self.read_json_run(folder)
        self.assertEqual(document["status"], "partial")
        self.assertIn("absent", document["failed_subdirs"])
        self.assertEqual(per_run["status"], "complete")

    def read_json_run(self, folder):
        with open(os.path.join(folder, "output", "run", "output.json"),
                  encoding="utf-8") as handle:
            return json.load(handle)


# ---------------------------------------------------------------------------
# N11, N12 : harnais de validation
# ---------------------------------------------------------------------------
class TestValidationHarness(unittest.TestCase):
    HARNESS = os.path.join(ROOT, "tests", "validate_pipeline.py")

    def harness(self, *args):
        return subprocess.run([PYTHON, "-X", "utf8", self.HARNESS, *args],
                              capture_output=True, text=True, encoding="utf-8")

    def absolute_config(self, folder, **overrides):
        cfg = minimal_config(
            input_dir=os.path.join(folder, "input"), output_dir=os.path.join(folder, "out"),
            output_json=os.path.join(folder, "output.json"),
            latex_output={"compiler": "none"}, **overrides)
        return write_config(folder, cfg)

    def test_no_run_without_outputs_is_a_diagnostic_not_a_trace(self):
        with workdir() as folder:
            result = self.harness("--no-run", "--config", self.absolute_config(folder))
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertNotIn("Traceback", result.stdout + result.stderr)
        self.assertIn("chargement impossible", result.stdout)

    def test_csv_source_zero_values_and_no_latex_pass_the_harness(self):
        frame = base_frame(60)
        frame.loc[0:3, ["ZoneA", "ZoneB", "ZoneC"]] = 0.0   # beta non fini : publie null
        with workdir() as folder:
            write_book(folder, frame, name="data.csv")
            config = self.absolute_config(folder, data_files="data.csv")
            result = self.harness("--config", config)
            document = self.read_json(folder)
        output = result.stdout + result.stderr
        self.assertNotIn("Traceback", output)
        self.assertEqual(result.returncode, 0, output)
        metrics = document["results"]["metrics"]
        self.assertIsNone(metrics["beta_max"])
        self.assertFalse(metrics["beta_defined"])
        self.assertIn("non defini", result.stdout)
        self.assertNotIn("[FAIL]", output)

    def read_json(self, folder):
        with open(os.path.join(folder, "output.json"), encoding="utf-8") as handle:
            return json.load(handle)

    def test_margin_check_uses_the_bounded_compilation_and_honours_the_compiler(self):
        import importlib
        harness = importlib.import_module("tests.validate_pipeline")
        calls = []

        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 1)

        with workdir() as folder:
            config = cm.load_config(write_config(folder, minimal_config(
                latex_output={"compiler": "none"})))
            produced = os.path.join(folder, "produit")
            os.makedirs(produced)
            for name in ("01-Step1_Dominance_Hierarchy_Result",
                         "04-Step1_Critical_Zones_Sample_Result", "07-CriticalZones_Targets"):
                with open(os.path.join(produced, f"{name}.tex"), "w", encoding="utf-8") as h:
                    h.write("\\documentclass{article}\\begin{document}x\\end{document}")
            with mock.patch.object(harness.subprocess, "run", fake_run):
                self.assertEqual(harness.check_latex_margins(produced, config), [])
                self.assertEqual(calls, [])
                latex_support.configure("auto")
                config_auto = cm.load_config(write_config(folder, minimal_config(
                    latex_output={"compiler": "pdflatex"})))
                with mock.patch.object(latex_support, "find_latex_compiler",
                                       return_value=("pdflatex", "pdflatex-fictif")):
                    issues = harness.check_latex_margins(produced, config_auto)
        latex_support.configure("auto")
        self.assertEqual(len(issues), 3)
        self.assertTrue(all("compilation en echec" in issue for issue in issues))
        for command, kwargs in calls:
            self.assertEqual(kwargs["timeout"], post_processor.LATEX_PASS_TIMEOUT_S)
            self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
