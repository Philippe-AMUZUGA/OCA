# Journal des versions — OCA

Les versions suivent le versionnage sémantique : `MAJEUR.MINEUR.CORRECTIF`.

---

## v9.0.0 : robustesse des entrées, compilateur LaTeX, grille de κ

Les résultats numériques de l'exemple `CAS_TEST_00` sont **inchangés** par rapport à
la version 8.1 : `output.json` (hors horodatages, chemins et clés nouvelles), relevés
texte, tableaux `.tex` et classeurs de traçabilité sont identiques, à l'exception de
la phrase de lecture de la notation `A < B` ajoutée en tête des relevés de graphe et
de la notice. La logique de calcul n'est pas modifiée.

### Entrées et configuration

- `critical_zones` et `feature_cols_init` peuvent être vides, `null`, `""` ou
  absentes : la liste manquante se déduit des colonnes numériques du classeur ; si les
  deux sont vides, toutes les colonnes numériques sont des zones (avertissement). La
  liste résolue est consignée au journal, dans `output.json` et dans le rapport.
- Défauts internes neutres : `user_defined_vars` faux, `user_defined_vars_cols` vide,
  `target_zone` à `"none"`, `gpo_regularization.threshold` à 0,99. Un fichier de
  quatre clés fonctionne tel quel.
- Un champ optionnel nul ou vide reprend son défaut avec une note au journal ; un
  mauvais type ou une valeur hors domaine lève une erreur d'une ou deux lignes qui
  nomme la clé par son chemin complet. Plus aucune `TypeError`, `ValueError` ni
  `AttributeError` ne sort de la lecture de la configuration. Contrôles ajoutés :
  `kappa_min <= kappa_max`, `kappa_step > 0`, bornes finies, seuils dans leurs
  domaines, listes de noms sans vide ni doublon, listes disjointes, noms confrontés
  aux colonnes du classeur (la liste des colonnes disponibles est donnée).
- Chemins sûrs : refus, avant tout nettoyage, d'un `merged_subdir` qui sort de la
  racine, d'un `data_files` qui n'est pas un nom simple, d'un `output_dir` identique
  à `input_dir` ou qui le contient, d'un `output_json` qui désigne un classeur source.
- Lecture des classeurs : la première colonne n'est plus consommée systématiquement.
  Seule une colonne sans en-tête ou nommée `index`, `id`, `config`, `config_id`, `n`
  est un index. En-têtes et colonnes utiles vérifiés avant conversion en nombres ;
  colonnes de commentaires ignorées ; première feuille lue (et annoncée) ;
  `.xlsx` et `.csv` (séparateur `,` ou `;` détecté) pris en charge, `.xls` refusé avec
  un message de conversion.
- Données : valeurs manquantes ou infinies refusées sur toutes les zones comparées
  (colonne, nombre, observations fautives) ; au moins 2 observations ; coefficient de
  Pearson non calculé sur une série constante (annotation « variance nulle »).
- Fusion par fichiers par-zone : zones obligatoirement déclarées, identifiants bâtis
  sur les variables déclarées, doublons retirés, doublons contradictoires et
  configurations hors intersection comptés et consignés.
- Absence de données (dépôt fraîchement cloné) : un message de deux lignes renvoie à
  `python main_OCA.py --init` et à `examples/`, sans trace Python ; le nettoyage des
  sorties n'a lieu qu'une fois la source de données vérifiée. Une erreur imprévue est
  résumée en une ligne (`--debug` donne la trace).

### Calcul et publication

- Grille de κ construite par indice (`kappa_min + i * kappa_step`), borne haute
  incluse si elle appartient à la grille (tolérance d'arrondi), limitée à 100 000 points : plus de borne manquée
  par accumulation flottante (cas `[1 ; 1,2]` au pas de 0,1), plus de boucle sans fin
  pour un pas dérisoire. Le κ publié garde autant de décimales que le pas l'exige, au
  moins 2 : les valeurs publiées de l'exemple sont inchangées.
- Cible automatique : zones ex æquo au maximum et règle de choix déterministe
  consignées au journal et dans `output.json` ; cible sous le seuil bas, écartée par
  la sélection mais réintroduite, signalée.
- β : avertissement et `metrics.beta_defined: false` si des valeurs négatives ou
  nulles le rendent non interprétable ; `kappa_search` publie le nombre de points, le
  dernier point testé et la plus faible couverture. Le calcul n'est pas modifié.
- Relevés texte : écriture sans échec quel que soit le nom de zone (`errors="replace"`,
  encodage de référence conservé) ; phrase de lecture de la notation `A < B`.
- Tableaux LaTeX et notice : échappement en une passe de `& # % { } $ ^ ~ \`, lettres
  grecques composées en mode mathématique ; légende du tableau `07-` avec l'effectif
  réellement affiché ; axe des boîtes à moustaches au libellé neutre.
- `output.json` : version, zones et variables résolues, statut du rapport, domaine de
  β ; `config_file` reproduit le chemin de configuration **tel que fourni** s'il est
  relatif, séparateurs normalisés ; un chemin absolu est réduit au nom du fichier.
- Clé `status` du document de résultats et état de la tentative : au début d'un
  lancement, l'ancien `output_json` est remplacé par
  `{"status": "running", "started_at": ...}` ; en fin de calcul il porte
  `"status": "complete"` (`"partial"` si des sous-dossiers d'un lot ont échoué), le
  reste du contenu étant inchangé ; en cas d'échec global, `{"status": "failed",
  "reason": ...}`. Un consommateur qui surveille ce seul fichier ne relit plus un
  succès périmé.
- Sorties : un `output_json`, un `output_files.*` ou un nom de rapport consolidé qui
  désigne un fichier source (classeurs fusionnés ou par-zone, chemins résolus), ou
  deux sorties qui désignent le même fichier, sont refusés avant tout nettoyage.
- Fusion par fichiers par-zone avec variables déduites : ordre et ensemble des
  variables fixés par le premier fichier, fichier divergent refusé (fichiers et
  colonnes nommés), colonnes des suivants réordonnées avant de construire les clés.
- Première colonne à nom d'index usuel (`n`, `id`...) : conservée si elle est citée
  dans `critical_zones`, `feature_cols_init` ou une expression synthétique.
- Noms réservés refusés (`$` initial, libellés engendrés par OCA, `config_id` en
  fusion par-zone) ; noms bruts affichés littéralement dans les figures (les `$` ne
  sont plus interprétés par matplotlib).
- κ publié : le flottant réellement éprouvé, sans réarrondi (seuls les libellés sont
  arrondis) ; grille aux points distincts, sinon refus ; enveloppe κ·cible ou rapport
  de β non fini : débordement refusé avec un message (les valeurs publiées de l'exemple
  sont inchangées).
- Entier JSON démesuré : erreur de configuration qui nomme la clé ; `--init` dans le
  traitement d'erreurs (dossier non inscriptible, `input` qui est un fichier : message
  court) ; dépendance manquante : une ligne qui nomme le paquet et la commande
  `pip install -r requirements.txt` (code 3).
- Rapport consolidé : la liste des sections fusionnées provient des ressources
  validées de la compilation (un PDF dont la compilation a échoué n'est jamais fusionné,
  même si son retrait a échoué) ; rapport partiel sans notice compilée : page « Rapport
  partiel » produite avec matplotlib, sans LaTeX, en tête du consolidé.
- Dépendances : `openpyxl` et `Jinja2` relevés à 3.1.5, `scipy` à 1.14.1 et
  `matplotlib` à 3.9.3 (exigences minimales de pandas 3.0 pour ses dépendances
  optionnelles).
- Harnais `validate_pipeline.py` : classeur de traçabilité lu sous son nom réel
  (`.xlsx` même pour une source `.csv`), chargement initial protégé, β publié `null`
  accepté, PDF facultatifs non exigés sans compilateur, contrôle des marges passant par
  le lancement borné (délai, code retour) et respectant `latex_output.compiler`,
  matrices arrondies à 4 décimales tolérées au voisinage du seuil, optimalité de κ
  contrôlée sur la grille indépendante.

### Compilateur LaTeX

- Nouveau réglage `latex_output.compiler` : `auto` (défaut), `pdflatex`, `tectonic`,
  `none`. Module `latex_support.py` : détection (PATH puis emplacements usuels), un
  seul message exact si aucun compilateur n'est disponible.
- Compilation bornée à 300 s par passe, code retour contrôlé, démarrage impossible
  traité : un PDF issu d'une compilation en erreur n'est pas retenu, le `.log` est
  conservé et sa première erreur citée au journal.
- Sans LaTeX, tout le reste est produit et le code de sortie est 0. Les PDF
  manquants rendent le rapport consolidé **partiel** (journal, notice, `output.json`,
  message final), avec la liste des sections absentes.

### Installation et dépôt

- `pyproject.toml` : classificateur de licence redondant retiré (l'expression SPDX
  `AGPL-3.0-or-later` suffit), `setuptools>=77`, module `latex_support` déclaré ;
  `Jinja2` et `openpyxl` déclarés dans `pyproject.toml` et `requirements.txt`.
- Harnais : `validate_against_reference.py` refait le calcul avant de vérifier la
  présence des sorties (sauf `--no-run`) ; les deux harnais prennent par défaut la
  configuration de l'exemple ; nouveau `tests/test_robustness_v9.py` (`unittest`).

### Documentation corrigée

- Traitement par lot : suite d'études indépendantes, aucune concaténation entre
  sous-dossiers (le guide annonçait 900 lignes pour trois sous-dossiers de 300).
- L'enveloppe κ porte sur les zones **conservées** par la sélection, non sur les zones
  écartées.
- Les décisions de sélection utilisent les pourcentages arrondis à 0,01 %.
- Rôles des seuils : `threshold_low` filtre, `threshold_high` désigne seulement les
  zones dimensionnantes (la notice et le guide ne parlent plus de bande de sélection).
- Exactitude : l'enveloppe concerne les zones retenues par la sélection, par
  comparaison ligne à ligne ; la borne haute de κ est incluse si elle appartient à la
  grille, à la tolérance d'arrondi près ; une cible incalculable est un refus de
  configuration avec message ; les matrices JSON sont arrondies à 4 décimales et les
  arêtes décidées sur les proportions non arrondies ; section « Limites connues » du
  guide.
- Exemples : le JSON agrégé est à la racine (`./output.json`) ; les trois études ont
  des effectifs et des κ différents (907, 907 et 400 observations ; κ de 1,55, 1,35 et
  1,53).
- Succès de κ : `kappa_found`, non l'égalité à `kappa_max`.
- Notation `A < B` : A précède B dans le classement de dominance (A majore B).
- Commandes de validation et chemins de l'exemple ; ressources non livrées retirées ;
  les références sont l'état de la version 8.0, non les sorties du calcul d'origine.

---

## v8.1.0 — licence libre, et un dépôt qui sert les deux usages

### Licence

Le projet passe sous **GNU Affero General Public License, version 3 ou ultérieure**.

Copyright © 2026 Philippe AMUZUGA. Le fichier [`LICENSE`](LICENSE) porte le texte
officiel, repris verbatim.

Les versions antérieures étaient déclarées « Proprietary », ce qui contredisait
l'objectif de diffusion : un dépôt public sans licence explicite équivaut à « tous
droits réservés », lisible mais inutilisable sans s'exposer.

### Un dépôt, deux façons de l'utiliser

Le dossier [`examples/`](examples/) porte le cas d'application, avec ses données, sa
configuration commentée et ses références de non-régression. On peut donc désormais
soit télécharger le dépôt entier et rejouer le cas, soit installer le paquet et
travailler sur ses propres données. Les deux voies exécutent le même code.

```bash
# en local, depuis la racine du dépôt
python main_OCA.py --config examples/CAS_TEST_00/input.json
python tests/validate_against_reference.py --config examples/CAS_TEST_00/input.json --no-run

# en paquet installé, sur ses propres données
pip install git+https://github.com/Philippe-AMUZUGA/OCA.git
oca --init
```

L'exemple ne porte **pas** de copie du code, contrairement au dossier de cas de la
livraison interne : la divergence entre copies est un défaut qui s'est produit dans ce
projet, et un dépôt public l'exposerait à chaque contributeur. Conséquence
pédagogique assumée : les chemins de la configuration d'exemple sont préfixés par
`examples/CAS_TEST_00/`, puisque `input_dir` et `output_dir` se résolvent au
**répertoire courant** et non au dossier du fichier de configuration.

### Défaut corrigé : la commande de non-régression documentée échouait

Deux endroits de la documentation prescrivaient :

```bash
python tests/validate_against_reference.py --config input_CAS_TEST_00.json
```

Cette commande échouait depuis ses deux implantations, et par deux causes
différentes — configuration introuvable depuis le dossier du cas, dossier de sortie
inexistant depuis le dépôt — en remontant à chaque fois une **trace d'exception** là
où un diagnostic est attendu. C'est la première commande qu'un relecteur essaie.

Le harnais fonctionnait en réalité avec sa configuration par défaut : l'argument
`--config` surnuméraire était la seule cause de l'échec. Trois mesures :

- la commande documentée est ramenée à ce qui fonctionne, et l'endroit d'où la lancer
  est énoncé ;
- l'absence du dossier de sortie est désormais **nommée**, avec le répertoire courant,
  la configuration lue et la commande correcte, en rendant le code de sortie 2 ;
- le dossier de références est **cherché** parmi les emplacements plausibles au lieu
  d'être présumé, ce qui couvre la disposition du dépôt comme celle de la livraison.

> Ce dernier point corrige un défaut plus grave qu'il n'y paraît : le chemin présumé
> n'existait depuis aucune des deux implantations, si bien que chaque comparaison
> retournait « Référence absente » et que le harnais **rendait un succès sans avoir
> rien comparé**.

### Le guide pour agents automatiques est annoncé

Le guide utilisateur ne mentionnait pas `AI_AGENT_GUIDE.md`, alors que le projet frère
lui consacrait une sous-section. C'est corrigé, avec un renvoi vers le chapitre des
deux critères de dominance — le point de méthode que ce document formule de façon
opératoire.

### Mise en page

Trois flottants figés en `[h!]` se disputaient la même page, et LaTeX signalait déjà ne
pouvoir honorer cette contrainte. Leur placement a été libéré : le document revient à
**0 overfull**, 52 pages, aucun renvoi non résolu.

### Validation

```
Non-régression : 5 comparaisons, toutes identiques aux références
Valeurs clés   : κ = 1,55 et β moyen = 1,21 sur 907 observations
Trace          : output.json strictement conforme à la norme JSON
```

---

## v8.0.0 — première version publique

Première mise à disposition publique. Les versions antérieures (v1 à v7) sont internes
internes ; leurs journaux détaillés accompagnent la livraison hors dépôt.

### Ce que contient cette version

**OCA — Ordinal Correlation Algorithm** classe des zones critiques par *dominance
ordinale* : non par la valeur moyenne d'une grandeur, mais par la **proportion des
observations où une zone en majore une autre**.

Quatre modules :

| Module | Rôle |
|---|---|
| `config_manager.py` | Lecture et certification de la configuration et de la table de travail |
| `oca_model.py` | Le calcul : matrice ordinale, recherche de κ, scores de dominance |
| `post_processor.py` | Figures, tableaux LaTeX, trace `output.json` |
| `main_OCA.py` | Orchestration |

### Les deux critères de dominance, et pourquoi ils ne se confondent pas

C'est le point de méthode le plus important, et le plus facile à mal interpréter. OCA
produit **deux** classements, qui ne mesurent pas la même chose.

**La fréquence `p`** est un décompte *un-contre-tous* : la proportion des observations
où une zone majore **toutes** les autres simultanément. Elle **sélectionne** avec
netteté, mais elle **sature** : dès qu'une zone domine largement, toutes les autres
tombent à zéro, quelles que soient leurs relations mutuelles. Elle ne permet donc pas
de départager le peloton.

**Le score matriciel** `s_i = (1/(m−1)) · Σ_{j≠i} p(i,j)` est une moyenne de
comparaisons *par paires*. Il **dispose** : il ordonne l'ensemble du peloton, et ne
sature pas.

> La règle : **la fréquence sélectionne, le score matriciel dispose.** Utiliser l'un à
> la place de l'autre conduit à des conclusions fausses — le premier ne hiérarchise pas
> ce qui est dominé, le second ne désigne pas de vainqueur net.

La matrice ordinale est une **proportion de majoration**, et non une corrélation de
Pearson ou de Spearman. Elle **n'est pas symétrique** : `p(i,j) + p(j,i)` ne vaut pas
nécessairement 1, les égalités n'étant pas comptées comme des majorations. La matrice
linéaire de Pearson, présente dans les premières versions, a été retirée en v3 parce
qu'elle induisait cette confusion.

### Validation

Le harnais `tests/validate_against_reference.py` compare les sorties aux références de
non-régression. Il se lance depuis le dossier du cas de test, qui porte les sorties :

```bash
cd CAS_TEST_00
python tests/validate_against_reference.py --no-run
```

Cinq comparaisons, toutes identiques aux références : κ = 1,55 et β moyen = 1,21 sur
907 observations.

> **Portée exacte de cette garantie.** Les références sont l'état de la **v8.0**, non
> les sorties historiques du calcul d'origine. Le harnais vérifie donc l'invariance
> *depuis la v8.0*, ce qui est plus faible que ce que son nom suggère. Le dossier
> qu'interrogeaient les versions antérieures n'existait pas : le harnais était
> inexécutable tout en étant annoncé comme utilisable à quatre endroits de la
> documentation, et chaque comparaison retournait « Référence absente » — c'est-à-dire
> un succès sans avoir rien comparé.

### Corrections apportées avant cette première publication

- **Trace conforme à la norme JSON.** La diagonale de la matrice ordinale vaut `NaN`,
  que l'écriture sérialisait en littéral `NaN` — que la norme JSON ne connaît pas. Le
  fichier livré en portait 33 occurrences et était **rejeté par tout analyseur strict**
  (`JSON.parse`, Jackson, jsonlite, l'analyseur de Go). Les valeurs non finies sont
  désormais normalisées en `null`, ce qui est exact ici : la diagonale d'une matrice de
  majoration n'est pas un nombre, c'est la comparaison d'une colonne à elle-même, que
  le calcul laisse indéfinie à dessein. L'écriture refuse en outre explicitement toute
  constante hors norme, ce qui rend la conformité vérifiable et non seulement espérée.
- **Commande `--init`.** Le modèle d'entrée est un `input.json` et un dossier `input/`
  pris dans le répertoire courant. Après une installation par `pip`, l'utilisateur
  n'avait ni l'un ni l'autre. `oca --init` écrit le gabarit et crée le dossier, sans
  jamais écraser un fichier existant.
- **Certification de la table de travail.** Une table est refusée si elle porte des
  `NaN` sur les zones comparées, si elle déclare moins de deux zones, ou si elle ne
  contient aucune observation. Ces trois cas produisaient auparavant un classement
  d'apparence valide.
- **Déclarations d'étude obligatoires.** `critical_zones` et `feature_cols_init` n'ont
  plus de valeur par défaut propre à une étude.
- **Recherche de κ.** La valeur retournée en cas d'échec est désormais la dernière
  réellement testée, et non une borne jamais évaluée.
- **Égalités de dominance signalées.** Le critère de désignation automatique de la
  cible reste l'`idxmax` d'origine — le modifier changerait une méthode publiée — mais
  une égalité en tête est maintenant signalée au lieu d'être tranchée en silence.
- **Variables synthétiques non finies refusées** après évaluation de leur expression.
- **Configuration ambiguë écartée du dépôt.** Un fichier nommé `input_CAS_TEST_00.json`
  décrivait en réalité une autre étude, sur des données non distribuées, et la commande
  documentée qui l'invoquait échouait depuis ses deux implantations. La commande a été
  rectifiée et le fichier requalifié hors du dépôt.

### Limite connue, laissée en l'état

Le critère de désignation automatique de la cible reste un `idxmax`, qui tranche
arbitrairement une égalité parfaite. L'égalité est signalée mais non arbitrée : le
choix appartient à l'utilisateur.

### Socle logiciel

Python ≥ 3.10, validé sous 3.12.13. Les dépendances sont déclarées dans
`pyproject.toml` et reprises dans `requirements.txt`.
