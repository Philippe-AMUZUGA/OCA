# Exemples : un cas réel, prêt à rejouer

Ce dossier existe pour que vous puissiez apprendre à configurer OCA en rejouant un
calcul dont le résultat est connu, avant de l'appliquer à vos propres données.

## Les deux façons d'utiliser ce dépôt

**En local, avec tout le dossier.** Vous téléchargez le dépôt entier, vous rejouez
l'exemple, vous modifiez sa configuration pour voir ce que chaque clé change. C'est la
voie recommandée pour commencer.

**En paquet installé.** Une fois le fonctionnement compris, `pip install` vous donne la
commande `oca`, utilisable depuis n'importe quel dossier de travail sur vos propres
données (`oca --init` écrit un gabarit de configuration). Voir le
[README](../README.md) principal.

Les deux voies exécutent exactement le même code.

## Rejouer l'exemple

Depuis **la racine du dépôt**, pas depuis ce dossier :

```bash
python main_OCA.py --config examples/CAS_TEST_00/input.json
```

Les résultats de chaque étude sont écrits dans
`examples/CAS_TEST_00/output/<sous-dossier>/` (avec le `output.json` de cette étude).
Le JSON **agrégé** du lot (`merged_subdirs`, `runs`, `status`) est écrit à la **racine
du dépôt**, dans `./output.json` (clé `output_json` de la configuration de l'exemple),
et non dans `examples/`. Sans compilateur LaTeX
(`pdflatex`, TeX Live ou MiKTeX), le calcul va jusqu'au bout : un avertissement unique
l'indique, les figures, les sources `.tex`, les relevés texte et `output.json` sont
produits, et le rapport consolidé est déclaré partiel.

> **La chose à comprendre, et c'est la principale.** Les clés `input_dir` et
> `output_dir` se résolvent par rapport au **répertoire courant**, et non par rapport
> à l'emplacement du fichier `input.json`. C'est pourquoi la configuration de ce
> dossier préfixe ses chemins par `examples/CAS_TEST_00/`. Si vous la copiez ailleurs,
> ajustez ces chemins.

## Vérifier que vous avez reproduit le résultat

Le dossier `dataSet_criticalZones/` porte les sorties de référence. Depuis la racine
du dépôt :

```bash
python tests/validate_against_reference.py
```

Ce harnais prend par défaut la configuration de l'exemple. Il refait le calcul puis
compare vos sorties aux références, ce qui fonctionne sur un dépôt fraîchement cloné.
Pour comparer des sorties déjà calculées, sans refaire le calcul :

```bash
python tests/validate_against_reference.py --no-run
```

Le harnais énonce chaque écart. Il compare le **premier** sous-dossier, dont les
valeurs clés attendues sont **κ = 1,55** et **β moyen = 1,21** sur 907 observations.
Les trois études de l'exemple n'ont ni les mêmes effectifs ni le même κ :

| Étude | Observations | κ |
|---|---|---|
| `01-RMR-P1_Fv0_Fext0` | 907 | 1,55 |
| `02-RMR-P0_Fv1_Fext0` | 907 | 1,35 |
| `03-RMR-P0_Fv0_Fext1` | 400 | 1,53 |

> **Portée exacte de cette garantie.** Les références sont l'état de la version 8.0,
> et non les sorties du calcul d'origine, qui n'avaient pas été conservées. Le harnais
> vérifie donc l'invariance depuis la version 8.0 : c'est plus faible que ce que son
> nom suggère, et cela vaut d'être su plutôt que supposé. L'exactitude absolue est
> contrôlée par `python tests/validate_pipeline.py`, qui recalcule chaque grandeur
> publiée.

## Ce que contient l'exemple

| Fichier | Rôle |
|---|---|
| `CAS_TEST_00/input.json` | La configuration complète, commentée |
| `CAS_TEST_00/input/` | Les données, au format Excel (une feuille par classeur) |
| `CAS_TEST_00/dataSet_criticalZones/` | Les sorties de référence, pour la non-régression |

Le dossier `output/` **n'est pas livré** : il pèse plusieurs mégaoctets et se
régénère en une commande.

## Format de vos propres classeurs

La première feuille d'un classeur `.xlsx` est lue (les autres ne le sont pas), ainsi
qu'un `.csv` ; la première ligne est l'en-tête. Les classeurs de l'exemple portent
une première colonne sans en-tête, qui est un index et n'est pas analysée ; un
classeur sans cette colonne d'index est lu de la même façon, sa première colonne
restant une variable ou une zone. Les listes `critical_zones` et `feature_cols_init`
peuvent être vides : la liste manquante est déduite des colonnes numériques du
classeur. Voir la section 3 du [README](../README.md).

## Ce que l'exemple illustre

Six zones critiques (`TRESCA_LIN_Z1_R` à `TRESCA_LIN_Z3_L`) et deux variables
synthétiques, comparées sur 907 observations dans les deux premières études et 400
dans la troisième. Le calcul produit **deux classements**, et c'est le point de méthode
à ne pas manquer :

- la **fréquence un contre tous** *sélectionne* : elle désigne un vainqueur net, mais
  elle sature, car dès qu'une zone domine largement, toutes les autres tombent à zéro ;
- le **score matriciel par paires** *dispose* : il ordonne l'ensemble du peloton, et
  ne sature pas.

Regardez les deux sorties correspondantes et comparez-les : c'est le meilleur moyen de
voir pourquoi l'un ne remplace pas l'autre. Le détail figure au §7 de
[`../AI_AGENT_GUIDE.md`](../AI_AGENT_GUIDE.md).

Les trois sous-dossiers de `input/` sont trois **études indépendantes** (un
traitement par lot), non une population unique de lignes réunies.

## Aller plus loin

- [`../AI_AGENT_GUIDE.md`](../AI_AGENT_GUIDE.md) : la référence complète (contrat de
  données, toutes les clés, le calcul étape par étape, les pièges mesurés).
- [`../README.md`](../README.md) : installation et vue d'ensemble.
- [`../CHANGELOG.md`](../CHANGELOG.md) : ce que chaque version a changé.
