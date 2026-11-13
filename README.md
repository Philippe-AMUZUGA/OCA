# OCA, algorithme de corrélation ordinale (Ordinal Correlation Algorithm)

Réduction de dimension d'un ensemble de variables cibles, désignées « zones
critiques », par **ordre partiel de dominance** (GPO, ordre de Pareto), suivie de
la régularisation d'une variable synthétique enveloppe `kappa * c_z` qui majore,
ligne à ligne et au seuil de proportion requis, chacune des zones **retenues par la
sélection** (cible comprise), et non nécessairement toutes les zones initiales : une
zone écartée par le seuil bas n'entre pas dans κ (voir « Seuils de sélection »).

La méthode ne repose sur aucune hypothèse de linéarité entre les variables : elle
procède exclusivement de comparaisons ordinales terme à terme.

Version 9.0.0.

## 1. Installation

```bash
pip install .
```

Le package reste utilisable sans installation, directement depuis son dossier :

```bash
pip install -r requirements.txt
```

### Prérequis

| Composant | Version minimale | Rôle |
|---|---|---|
| Python | 3.12 | interpréteur |
| `pandas` | 3.0.0 | tables, lecture Excel, export LaTeX |
| `numpy` | 2.0 | algèbre et masques booléens |
| `scipy` | 1.16.1 | corrélations |
| `matplotlib` | 3.9.3 | figures |
| `seaborn` | 0.13 | cartes de chaleur et styles |
| `networkx` | 3.3 | graphes de dominance et condensation |
| `openpyxl` | 3.1.5 | classeurs `.xlsx` (lecture et écriture) |
| `Jinja2` | 3.1.5 | requis par pandas pour écrire les tableaux LaTeX |
| `pypdf` | 3.9 | fusion des rapports PDF |

Les minima de `openpyxl`, `Jinja2`, `scipy` et `matplotlib` sont les exigences de
pandas 3.0 pour ses dépendances optionnelles (`pandas.compat._optional`). Si une
dépendance manque, le programme le dit en une ligne qui nomme le paquet et donne la
commande `pip install -r requirements.txt` (code de sortie 3).

`pdflatex` (TeX Live ou MiKTeX) est requis pour la compilation des rapports PDF.
`tectonic` est accepté à sa place (`latex_output.compiler`, voir plus bas). Sans
compilateur LaTeX, le programme avertit **une seule fois**, produit tout le reste
(figures PDF, sources `.tex`, relevés texte, `output.json`) et se termine avec le
code 0 : le rapport consolidé est alors déclaré **partiel**. `PyPDF2 >= 3.0` reste
accepté en repli par le code.

Validation effective : Python 3.12, pandas 3.0, numpy 2.x, scipy 1.1x,
matplotlib 3.11, seaborn 0.13, networkx 3.x, openpyxl 3.1.5, Jinja2 3.1.6, TeX Live 2026.

## 2. Exécution

Premier essai, depuis la racine du dépôt, sur l'exemple livré :

```bash
python main_OCA.py --config examples/CAS_TEST_00/input.json
```

Les résultats sont écrits dans `examples/CAS_TEST_00/output/`. Pour votre propre
étude :

```bash
python main_OCA.py --init             # écrit un input.json gabarit et un dossier input/
# puis : déposez vos classeurs dans input/<sous-dossier>/ et adaptez input.json
python main_OCA.py                    # utilise ./input.json
python main_OCA.py --config autre.json
oca                                   # après "pip install ."
```

`--init` n'écrase rien (un `input.json` existant est laissé tel quel) ; il ne copie
pas les exemples. Entre `--init` et le calcul, il reste à déposer vos données et à
adapter les noms de colonnes et de sous-dossiers du gabarit.

Lancé sans données (dépôt fraîchement cloné, dossier `input/` vide), le programme
n'affiche pas de trace Python : un message de deux lignes renvoie à
`python main_OCA.py --init` et à `examples/`. Toute erreur de configuration ou de
données est rapportée en une ou deux lignes qui nomment la clé fautive, par exemple
`gpo_regularization.kappa_step` ; il en va de même d'un entier JSON démesuré, d'un
`--init` dans un dossier non inscriptible ou quand `input` est un fichier, d'un
`input_dir` qui désigne un fichier. `--debug` affiche la trace complète d'une erreur
imprévue.

## 3. Configuration

Tout est piloté par `input.json`. Aucun nom de colonne, aucun chemin et aucune
cible ne sont codés en dur. Une configuration minimale de quatre clés suffit :

```json
{
  "critical_zones": ["ZoneA", "ZoneB", "ZoneC"],
  "feature_cols_init": ["V1", "V2", "V3"],
  "data_files": "mesures.xlsx",
  "merged_subdir": ["essai-01"]
}
```

Les autres clés prennent les défauts ci-dessous (ceux du fichier `input.json`
livré) : pas de variable synthétique, cible choisie automatiquement, seuil de
régularisation 0,99, κ de 1 à 3 par pas de 0,01, seuil du graphe 0,99.

```json
{
  "critical_zones": ["TRESCA_LIN_Z1_R", "TRESCA_LIN_Z2_R", "TRESCA_LIN_Z3_R",
                     "TRESCA_LIN_Z1_L", "TRESCA_LIN_Z2_L", "TRESCA_LIN_Z3_L"],
  "feature_cols_init": ["e", "D1", "ALPHA", "H", "R1", "R2", "H3", "D3"],
  "input_dir": "./input",
  "output_dir": "./output",
  "output_json": "./output.json",
  "data_files": "dataSet_ALL.xlsx",
  "merged_subdir": ["01-RMR-P1_Fv0_Fext0", "02-RMR-P0_Fv1_Fext0"],
  "user_defined_vars": false,
  "user_defined_vars_cols": [],
  "gpo_selection":      { "threshold_low": 0.5, "threshold_high": 10.0 },
  "gpo_regularization": { "target_zone": "none", "threshold": 0.99,
                          "kappa_min": 1.0, "kappa_max": 3.0, "kappa_step": 0.01 },
  "ordinal_graph":      { "threshold": 0.99 },
  "latex_output":       { "sample_size": 30, "max_columns_per_block": 6,
                          "table_font_size": "scriptsize",
                          "table_column_separation": "2pt",
                          "compiler": "auto" },
  "sample_random_state": 6,
  "color_style":        { "color_map_matrix": "RdBu_r",
                          "color_map_critical_zones": "jet",
                          "saturation": 1.0, "alpha": 1.0,
                          "apply_to": "background" }
}
```

### Listes de colonnes vides ou omises (version 9)

`critical_zones` et `feature_cols_init` peuvent être vides (`[]`), `null`, `""` ou
absentes. La liste manquante se déduit alors des colonnes **numériques** du
classeur :

| `critical_zones` | `feature_cols_init` | Résolution |
|---|---|---|
| déclarée | déclarée | chaque nom doit exister, être numérique et n'appartenir qu'à une liste |
| déclarée | vide | les variables sont les colonnes numériques qui ne sont pas des zones |
| vide | déclarée | les zones sont les colonnes numériques qui ne sont pas des variables |
| vide | vide | **toutes** les colonnes numériques sont des zones, il n'y a aucune variable de conception (avertissement) |

Les colonnes textuelles ou vides ne sont jamais déduites ; une colonne de
commentaires est ignorée. Une colonne presque entièrement numérique qui contient du
texte est refusée, faute de saisie probable, avec la ligne et le contenu fautifs. La
liste résolue est consignée au journal, dans `output.json` (`column_resolution`) et
dans le rapport. Avec la fusion par fichiers par-zone, les zones doivent être
déclarées (voir plus bas). Un nom absent du classeur, un doublon, un nom vide ou un
nom présent dans les deux listes lève une erreur qui liste les colonnes disponibles.

### Champs vides, types et erreurs (version 9)

Pour un champ **optionnel**, `null` ou une chaîne vide reprend le défaut annoncé,
avec une note au journal. Un mauvais type (chaîne là où un nombre est attendu, liste
pour un booléen, nombre non fini) lève une erreur de configuration d'une ou deux
lignes qui nomme la clé par son chemin complet. Sont aussi vérifiés :
`kappa_min <= kappa_max`, `kappa_step > 0`, les seuils bas et haut dans [0, 100], les
seuils de graphe et de régularisation dans ]0, 1], les entiers sans partie
décimale. Une clé inconnue (faute de frappe) est ignorée avec un avertissement.

### Chemins sûrs (version 9)

Le dossier de sortie de chaque sous-dossier est vidé à chaque exécution. Avant tout
nettoyage, le programme refuse : un `merged_subdir` qui contient un séparateur de
chemin, `..` ou qui est absolu ; un `data_files` qui n'est pas un nom de fichier
simple ; un `output_dir` identique à `input_dir` ou qui le contient ; un
`output_json`, un `output_files.*` ou un nom de rapport consolidé qui désigne un
fichier source (classeurs fusionnés ou fichiers par-zone, chemins résolus) ; deux
sorties (`output_files.*`, copie de traçabilité, `output_json`) qui désignent le même
fichier. Ces contrôles précèdent tout nettoyage et tout calcul.

Sont aussi refusés, avec un message, les noms de zone, de variable, de variable
synthétique ou de cible qui commencent par `$` ou qui ont la forme d'un libellé
engendré par OCA (`2.00$\times$A`), ainsi que `config_id` en fusion par-zone : ils
écraseraient une colonne calculée.

### Format du classeur (version 9)

* `.xlsx` (lecture et écriture) et `.csv` (lecture, séparateur `,` ou `;` détecté,
  virgule décimale admise avec `;`). Le format `.xls` est refusé avec une demande de
  conversion.
* **Première feuille uniquement** : les autres feuilles d'un classeur ne sont pas
  lues. La première ligne est l'en-tête.
* Une première colonne sans en-tête (ou `Unnamed`) ou nommée `index`, `id`,
  `config`, `config_id` ou `n` est un index d'observations et est retirée (une ligne
  du journal le dit) ; toute autre première colonne est conservée comme les suivantes.
  **Exception :** une première colonne à nom d'index usuel (`n`, `id`...) qui est
  citée dans `critical_zones`, `feature_cols_init` ou dans une expression de variable
  synthétique (`user_defined_vars_cols`, `target_zone`) est une grandeur : elle n'est
  pas écartée. Seule la première colonne est concernée : un `n` en deuxième position
  est toujours une colonne ordinaire. Les deux formats, avec et sans colonne d'index,
  donnent les mêmes résultats.
* Les en-têtes et les colonnes utiles sont vérifiés avant la conversion en nombres ;
  une erreur nomme le fichier, la colonne et la ligne. Les colonnes textuelles non
  utilisées sont ignorées.
* Chaque zone comparée doit être finie : une valeur manquante ou infinie est refusée
  avec la colonne, le nombre et les observations fautives. Il faut au moins 2
  observations et 2 zones.

### Compilateur LaTeX (version 9)

`latex_output.compiler` vaut `auto` (défaut : `pdflatex`, sinon `tectonic`),
`pdflatex`, `tectonic` ou `none` (sources `.tex` seules). Chaque passe de
compilation est limitée à 300 secondes ; un code retour non nul, un compilateur qui
ne démarre pas ou un délai dépassé ne produisent pas de PDF valide : le `.log` est
conservé, sa première erreur est citée au journal, et le pipeline continue. Quand
des PDF manquent, le rapport consolidé `00-` est déclaré **partiel** (dans le
journal, la notice, `output.json` et le message final) avec la liste des sections
absentes.

### Noms de zones et de variables (version 9)

Les noms sont repris tels quels dans `output.json`, le classeur et les figures. Dans
les tableaux LaTeX (`01-`, `04-`, `07-`) et la notice, les caractères
`& # % { } $ ^ ~ \` sont échappés en une passe (le trait bas est géré par le paquet
`underscore`) et les lettres grecques sont composées en mode mathématique. Dans les
figures, matplotlib n'interprète pas les `$` des noms (un nom comme `Z$x$` est
affiché littéralement). Les relevés `.txt` restent en cp1252 : un caractère que cet
encodage ne représente pas y devient `?` sans faire échouer l'écriture (deux noms
qui ne diffèrent que par de tels caractères y deviennent indiscernables ; le JSON,
le classeur et les tableaux gardent les noms exacts).

### Écriture commentée (version 7)

Le fichier de configuration admet les commentaires, de fin de ligne `//` comme
encadrés `/* ... */`. C'est l'écriture recommandée pour documenter un paramètre
énuméré là même où on le règle.

```json
"color_style": {
  "color_map_matrix": "RdBu_r", // options : "vlag", "RdBu_r", "coolwarm", "viridis", "plasma", "jet"
  "apply_to": "background"      // options : "text", "background"
}
```

Le retrait est **systématique et silencieux**, en amont de toute analyse. Un
commentaire n'est pas une anomalie de ponctuation : sa présence est consignée au
journal au niveau information, jamais en avertissement, et ne figure pas parmi les
réparations. Le rapport d'exécution d'une configuration commentée ne porte donc
aucun avertissement de syntaxe.

Chaque commentaire est remplacé par autant d'espaces qu'il comptait de caractères,
les sauts de ligne étant préservés : les positions que rapporterait le décodeur JSON
désignent ainsi exactement celles du fichier d'origine. Le balayage suit l'état de
chaîne, en sorte qu'une barre oblique ou une astérisque interne à une valeur, ainsi
`"D1/e + 0.5*H3/e"`, n'est jamais altérée.

Les virgules orphelines demeurent des anomalies : elles sont neutralisées, mais
signalées en avertissement.

### Auto-documentation par clés dédiées (version 7)

À l'écriture commentée s'ajoute une seconde voie : une clé homonyme suffixée de
`__options` déclarée à côté du paramètre et énumérant les valeurs admises.

```json
"color_style": {
  "color_map_matrix": "RdBu_r",
  "color_map_matrix__options": ["RdBu_r", "coolwarm", "jet", "viridis", "plasma"],
  "color_map_critical_zones": "jet",
  "color_map_critical_zones__options": ["jet", "tab10", "tab20", "Set1", "Dark2"],
  "saturation": 1.0,
  "alpha": 1.0,
  "apply_to": "text",
  "apply_to__options": ["text", "background"]
}
```

Le suffixe est un **double** tiret bas, choisi pour qu'aucun nom de paramètre ne
puisse le porter par accident. Ces clés relèvent de la seule documentation : elles
sont retirées récursivement, à toute profondeur, dans les objets comme dans les
éléments de tableau, avant toute validation et avant que la configuration ne
parvienne aux modules de calcul. Aucune ne saurait être confondue avec un
paramètre, provoquer une erreur de type, ni figurer dans les hyperparamètres
consignés au fichier de résultats.

Les clés écartées sont consignées au journal en **information**, non en
avertissement : une clé d'aide est un document conforme à la norme JSON, et la
signaler comme un écart induirait à corriger un fichier correct.

La présence d'une clé d'aide ne dispense d'aucune validation : déclarer
`apply_to` à `"foreground"` lève une erreur de configuration, que
`apply_to__options` soit présente ou non.

### Traitement par lot (version 6)

`merged_subdir` accepte une chaîne unique ou une liste de chaînes. Avec une liste,
chaque sous-dossier de `./input/` est traité séquentiellement et de façon
strictement isolée : données relues intégralement, journal réinitialisé, dossier
de sortie propre. L'échec d'un sous-dossier n'interrompt pas le lot.

**Un lot est une suite d'études indépendantes, non une concaténation.** Trois
sous-dossiers de 300 lignes donnent trois études de 300 lignes, avec trois choix de
cible et de κ ; aucune population de 900 lignes n'est analysée. Ne pas confondre
avec la fusion par fichiers par-zone ci-dessous.

Avec plusieurs sous-dossiers, `output.json` porte les clés `merged_subdirs` et
`runs` (indexé par sous-dossier) ; chaque sous-dossier reçoit en outre son propre
`output.json`. Avec un seul, la structure des versions antérieures est conservée.
Les dossiers de sortie des sous-dossiers qui ont échoué (`failed_subdirs`) ne sont
pas modifiés : ils peuvent contenir les résultats d'un lancement antérieur.

### Fusion par fichiers par-zone

Si, pour un seul sous-dossier, tous les fichiers
`<input_dir>/dataSet_<zone>/dataSet_<zone>.xlsx` existent pour les zones **déclarées**,
ils sont fusionnés sur un identifiant de configuration construit à partir des
variables (celles de `feature_cols_init`, sinon toutes les colonnes sauf la
dernière, qui porte la cible). Quand les variables sont déduites, leur ordre et leur
ensemble sont fixés par le **premier** fichier ; un fichier dont l'ensemble de
variables diffère est refusé avec un message qui nomme les fichiers et les colonnes,
et les colonnes des fichiers suivants sont réordonnées sur le premier avant de
construire les clés : permuter des colonnes d'une zone à l'autre ne change rien. Les
doublons retirés, les configurations répétées avec des valeurs contradictoires
(avertissement : la première occurrence est conservée, donc le résultat dépend de
l'ordre des lignes) et les configurations hors de l'intersection sont comptés au
journal.

### Sélection automatique de la cible

`"target_zone": "none"` retient la colonne dont la fréquence de dominance `p` est
la plus élevée à l'issue du classement GPO. Une cible synthétique non déclarée est
calculée et ajoutée automatiquement. Quand plusieurs zones sont ex æquo au maximum
(deux zones identiques, ou toutes à 0 %), le choix est déterministe : la première
selon l'ordre des colonnes du tableau de comptage (nombre de dominations croissant,
puis ordre de déclaration). Les ex æquo et cette règle sont consignés au journal et
dans `output.json` ; fixer `target_zone` lève toute ambiguïté.

### Seuils de sélection, périmètre de κ et lecture des résultats

* **Arrondi.** Les décisions de sélection (seuils bas et haut, cible automatique,
  cumuls `P`) utilisent les pourcentages arrondis à 0,01 %, convention du papier de
  référence. Une zone à 0,4975 % est ainsi arrondie à 0,50 % et n'est pas écartée par
  un seuil bas de 0,5 %.
* **Rôles des seuils.** `threshold_low` est le seul filtre de conservation : les
  zones dont `p` est strictement inférieure sont écartées. `threshold_high` ne
  filtre rien : il désigne comme « dimensionnantes » les zones dont `p` lui est
  strictement supérieure, étiquette portée au tableau et à la notice. Les zones au
  dessus du seuil haut sont conservées.
* **Périmètre de l'enveloppe.** La recherche de κ porte sur les zones **conservées**
  par la sélection (cible comprise) : κ·cz majore ces zones au seuil de
  régularisation, pas celles que le seuil bas a écartées. Une zone écartée (par
  exemple dominée presque partout par la cible sauf sur 0,1 % des lignes) n'entre pas
  dans κ ni dans β.
* **Succès de κ.** Le succès se lit sur `kappa_found` (champ de `output.json`), non
  sur l'égalité de κ à `kappa_max` : un succès exactement à la borne haute est un
  succès, un échec rend le dernier point testé. `output.json` publie en outre le
  nombre de points de la grille, le dernier point testé et la plus faible couverture
  atteinte (`kappa_search`).
* **Grille de κ.** Elle est construite par indice (`kappa_min + i * kappa_step`),
  sans accumulation de flottants ; la borne haute est incluse si elle appartient à la
  grille, à la tolérance d'arrondi près (`[1, 2]` au pas de 0,3 s'arrête à 1,9) ; elle
  compte au plus 100 000 points, tous distincts (une grille dont deux points ne se
  distinguent pas à la précision des flottants est refusée). Le κ publié dans
  `output.json` (`metrics.kappa_optimal`) est le flottant réellement éprouvé, sans
  réarrondi ; seuls les libellés et le journal sont arrondis, à autant de décimales
  que le pas l'exige, au moins 2 (pas de 0,001 : 3 décimales).
* **Débordement.** Si l'enveloppe κ·cible ou un rapport de β sort du domaine des
  flottants (valeurs de l'ordre de 1e308), le calcul est refusé avec un message : un
  produit infini majorerait toutes les zones et ne serait pas un succès.
* **Précision des matrices publiées.** Les matrices de `output.json` sont arrondies à
  4 décimales ; les arêtes des graphes, les cumuls et les degrés sont décidés sur les
  proportions **non arrondies**. Une proportion publiée égale au seuil à 4 décimales
  peut donc ne pas porter d'arête (0,66666 publiée 0,6667 pour un seuil de 0,6667) ;
  les relations (`ordinal_relations`) sont publiées à 2 décimales.
* **Notation `A < B`.** Dans les relevés texte de graphe et les chaînes de
  comparaison, `A < B` se lit « A précède B dans le classement de dominance »
  (A majore B dans la proportion indiquée). Une phrase le rappelle en tête des
  relevés de graphe.
* **β.** Il se lit comme une marge de conservatisme pour des valeurs strictement
  positives. Si des valeurs négatives ou nulles figurent parmi les zones conservées,
  un avertissement est consigné et `output.json` porte `metrics.beta_defined: false`
  (et la raison) ; le calcul n'est pas modifié. Un dénominateur nul donne un rapport
  posé à `+inf`, publié `null`.

### Découpage adaptatif des tableaux LaTeX (version 6)

Un tableau dont le nombre de colonnes excède `max_columns_per_block` (6 par
défaut) est scindé en sous-tableaux de largeurs homogènes, empilés verticalement
sous une légende commune, chacun répétant la colonne d'index.

Toute mise à l'échelle du texte est proscrite. Tous les blocs de **tous** les
tableaux de données (`01-`, `04-` et `07-`) sont composés dans une taille de
caractères fixe et identique, gouvernée par `table_font_size` et
`table_column_separation`. Le tableau `07-` conserve ses deux colonnes terminales
mises en évidence, κ·cz et β, dans le bloc où le découpage les fait tomber.

Le pipeline dépouille le journal de `pdflatex` après chaque compilation et
consigne tout dépassement de marge horizontale, avec son ampleur et la conduite à
tenir.

### Tri horizontal par score de dominance (version 6)

Deux critères coexistent et ne servent pas le même dessein.

1. La **fréquence `p`** du comptage un contre tous gouverne la **sélection** des
   zones retenues pour l'étape 2, et l'ordre du tableau `01-` dont les cumuls
   `N` et `P` en dépendent. Son caractère « tout ou rien » la fait saturer : dès
   qu'une zone domine largement, les autres tombent à zéro et cessent d'être
   départagées.
2. Le **score moyen de majoration**, issu de la matrice ordinale, gouverne la
   **disposition** des colonnes des tableaux `04-` et `07-` :
   `s_i = moyenne sur j != i de p(i, j)`. Fondé sur les comparaisons deux à deux,
   il ne sature pas et laisse rarement des ex aequo.

Le tableau `04-` se règle sur la matrice de l'étape 1 (`02-`), le tableau `07-`
sur celle de l'étape 2 (`05-`). Chaque tableau et la matrice qui le précède
donnent ainsi à lire la même hiérarchie. Le tri est stable, et le tableau `07-`
maintient en fin de ligne ses deux colonnes de régularisation, κ·cz et β.

La réorganisation s'opère par sélection de colonnes entières : aucun décalage de
données n'est structurellement possible. Le contrôle 13 du harnais confronte
chaque cellule aux données sources pour l'établir.

### Livrable `04-` en deux tableaux (version 6)

Le document porte deux tableaux décrivant les mêmes observations et les mêmes
valeurs, et ne différant que par la disposition de leurs colonnes.

1. Zones dans leur **ordre de déclaration** (`input.json`), avant hiérarchisation.
2. Zones par **score ordinal croissant**, après hiérarchisation.

Une teinte est attribuée à chaque zone d'après son **rang**, et non d'après son
nom : la moins dominante reçoit l'extrémité froide de la carte de couleurs, la
plus dominante l'extrémité chaude. La même teinte suit la même zone dans les deux
tableaux, ce qui permet de suivre son déplacement d'un coup d'œil. Elle est
portée par le fond de colonne ou par l'en-tête selon `color_style.apply_to`.
Chacun des deux tableaux reste soumis au découpage par `max_columns_per_block`.

### Périmètre statistique (version 6)

L'intégralité des indicateurs, comptage un contre tous et fréquences `p`,
classement GPO, sélection, comparaisons par paires dont procèdent les matrices
ordinales des deux étapes, graphes, coefficient κ et indice β, est établie sur la
**totalité des observations**. Le paramètre `sample_size` ne gouverne que
l'affichage des tableaux `04-` et `07-` : aucune décision de l'algorithme n'est
prise sur ce sous ensemble.

La notice analytique du rapport consolidé reporte, à sa section 2.1, un tableau à
cinq colonnes : le rang, la zone critique, la fréquence `p` (un contre tous), le
score ordinal de l'étape 1 (celui qui ordonne le second tableau de `04-`) et le
score ordinal de l'étape 2 (celui qui ordonne `07-`). Le lecteur y retrouve donc
le score exact ayant présidé à chaque disposition. La mention `s.o.`, sans objet,
désigne les zones que la sélection a écartées, absentes de la matrice de
l'étape 2.

### Style de couleurs (versions 6 et 7)

Un unique bloc gouverne la couleur de l'ensemble des sorties. La version 7
dissocie la carte du fond des matrices de celle des zones critiques.

```json
"color_style": { "color_map_matrix": "RdBu_r", "color_map_critical_zones": "jet",
                 "saturation": 1.0, "alpha": 1.0, "apply_to": "text" }
```

| Paramètre | Domaine | Effet |
|---|---|---|
| `color_map_matrix` | rampe continue matplotlib ou seaborn | fond des matrices ordinales `02-` et `05-` : `RdBu_r` (défaut), `vlag`, `coolwarm`, `jet`, `viridis`, `plasma` |
| `color_map_critical_zones` | carte matplotlib ou seaborn | palette désignant les zones : `jet` (défaut), `vlag`, `tab10`, `tab20`, `Set1`, `Dark2` |
| `saturation` | [0, 1] | 1 restitue la teinte telle quelle, 0 la mêle à parts égales du pôle neutre |
| `alpha` | [0, 1] | opacité, composée sur ce même pôle |
| `apply_to` | `background`, `text` | destination de la couleur, aires ou identifiants |

#### Pourquoi deux cartes

Les deux usages n'appellent pas la même famille de cartes. Le fond d'une matrice
ordinale encode une **grandeur continue**, une proportion de majoration dans
l'intervalle de zéro à un : il appelle une rampe continue, et de préférence
divergente, `RdBu_r` distinguant nettement les proportions faibles des fortes de
part et d'autre de la demie. La palette des zones désigne au contraire des
**catégories** à distinguer deux à deux, dont le rang de dominance fixe
l'attribution : une rampe telle que `jet` y matérialise la progression du
classement, du froid pour les moins dominantes au chaud pour les plus dominantes.

Jusqu'à la version 6, une carte unique servait les deux usages et la seconde
devait céder à la première par un repli automatique dès qu'elle était
catégorielle : `tab10`, excellente pour désigner des catégories, était
systématiquement remplacée par `coolwarm` au fond des matrices. La dissociation
supprime ce compromis.

Une carte catégorielle déclarée à `color_map_matrix` est signalée puis remplacée
par `coolwarm`. Une carte catégorielle sollicitée par
`color_map_critical_zones` pour plus de variables qu'elle ne porte de teintes est
remplacée par `turbo`.

La clé unique `color_map` des versions antérieures reste acceptée et alimente
alors les deux usages : une configuration écrite pour la version 6 est lue sans
modification, produit les mêmes grandeurs et retient les mêmes cartes. Si elle
coexiste avec l'une des clés nouvelles, ce sont ces dernières qui prévalent, et
une mention est consignée au journal.

#### Garantie de lisibilité

La palette est portée au rapport de contraste **4,5**, seuil d'accessibilité du
texte courant, par un facteur commun à toutes les teintes, journalisé lorsqu'il
s'applique. Les deux modes appellent des corrections de sens opposé : en mode
`text` la teinte porte le libellé et doit se détacher du blanc de la page, elle est
donc assombrie ; en mode `background` elle est un fond, et c'est le texte noir
qu'elle porte qui doit s'en détacher, elle est donc éclaircie.

La nécessité en est manifeste avec `jet` : son bleu nuit et son rouge sombre ne
donnent que 1,31 de contraste avec du texte noir, ce qui rendait la colonne
correspondante illisible. `viridis` descend à 1,38 et `tab10` à 3,55.

Une configuration de la version 6 voit donc ses fonds légèrement éclaircis. L'ordre
des teintes, leur nombre et leur identité chromatique sont préservés ; seule leur
clarté est ajustée.

Colorées : `02-`, `03-`, `04-`, `05-`, `06-`, `08-`, `09-`, `10-`.

#### Destination de la couleur

| Valeur | Destination | Rendu |
|---|---|---|
| `background` (défaut) | présentation, soutenance | fonds de colonne, disques des nœuds, boîtes, barres, points |
| `text` | article scientifique | noms de variables en gras sans empattement, aires neutres |

Le pôle d'atténuation suit la destination. En mode `background` la teinte est
mêlée de **blanc** : elle s'éclaircit et devient pastel, ce qui préserve la
lisibilité du texte noir posé dessus. En mode `text` elle est mêlée de **noir** :
elle s'assombrit et gagne en contraste sur la page. Mêler du blanc à un libellé
le ferait disparaître ; l'atténuation d'un fond et celle d'un texte appellent des
sens opposés. `alpha` compose sur ce même pôle, et devient donc en mode `text` un
second levier d'assombrissement.

L'association zone vers teinte est établie **une seule fois par exécution**, sur
l'ordre du classement de dominance de l'étape 1, et transmise à tous les
livrables : une zone porte donc la même teinte dans les tableaux et dans les
figures, et cette teinte porte une information plutôt qu'un numéro d'ordre de
déclaration.

En mode `text`, la palette est portée au rapport de contraste **4,5 sur fond
blanc**, seuil d'accessibilité du texte courant, par un facteur d'assombrissement
commun à toutes les teintes, journalisé lorsqu'il s'applique. Les rapports entre
teintes sont ainsi préservés tout en garantissant la lisibilité de chaque
libellé.

Une carte catégorielle sollicitée pour plus de variables qu'elle ne porte de
teintes, `tab10` en compte dix, est remplacée par la rampe continue `turbo` afin
que deux variables ne partagent jamais la même teinte. Le repli est signalé.

| Livrable | `background` | `text` |
|---|---|---|
| `02-`, `05-` | rampe continue des cases | rampe inchangée, étiquettes d'axes teintées |
| `03-`, `06-` | disque du nœud teinté | disque blanc, étiquette teintée |
| `04-` | fond de colonne teinté | en-tête teinté et gras, corps noir |
| `07-` | colonnes terminales sur fond sombre | colonnes terminales en gras, filet conservé |
| `08-` | boîte teintée | boîte blanche, étiquette d'abscisse teintée |
| `09-` | barres teintées | barres gris neutre |
| `10-` | points teintés | points gris neutre |

Les arêtes des graphes conservent la teinte de leur source dans les deux modes :
elle impute le faisceau à son nœud émetteur, ce qui relève du codage de
l'information et non de l'ornement. Le trait de médiane du diagramme en boîtes
est composé en noir doublé d'un liseré blanc dans les deux modes : le rouge des
versions antérieures se confondait avec une boîte que la carte de couleurs
rendait elle même rouge, et le noir seul reste insuffisant sur une rampe continue
sombre.

Le style ne gouverne que la composition : les grandeurs calculées sont
rigoureusement identiques quel que soit le mode retenu.

### Arrondi dynamique des valeurs numériques (version 6)

Le format d'affichage est choisi indépendamment pour chaque colonne, d'après la
médiane `x̃` de ses valeurs absolues non nulles, de sorte qu'un jeu de données de
faibles amplitudes ne voie plus ses valeurs tronquées à zéro.

| Distribution | Format | Exemple |
|---|---|---|
| `x̃ >= 1` | `.3g` | `161` |
| `1e-3 <= x̃ < 1` | `.5f` | `0.01606` |
| `x̃ < 1e-3` | `.3e` | `1.606e-07` |

Une colonne entièrement nulle, vide ou non finie conserve le format standard.
S'applique aux tableaux d'échantillonnage `04-` et `07-`.

## 4. Modules

| Fichier | Rôle |
|---|---|
| `main_OCA.py` | Point d'entrée, orchestration du lot |
| `config_manager.py` | Configuration, lecture des classeurs, variables synthétiques, journal |
| `oca_model.py` | GPO, κ, matrices ordinales, graphe de dominance, β |
| `post_processor.py` | Graphiques, LaTeX et PDF, rapport consolidé, `output.json` |
| `latex_support.py` | Détection du compilateur LaTeX et message d'absence |

## 5. Sorties

Produites dans `./output/<sous-dossier>/`, numérotées par ordre d'importance.

| Fichier | Contenu |
|---|---|
| `00-OCA_Report.pdf` | Rapport consolidé : notice, journal, sections 01 à 10 |
| `01-Step1_Dominance_Hierarchy_Result.pdf` | Hiérarchisation un contre tous |
| `02-OCA_CorrMat_Targets.pdf` | Matrice ordinale de l'étape 1 |
| `03-Step1_OrdinalGraph_Targets.pdf` | Graphe de dominance brute, contenu centré et étiquettes intégrales |
| `04-Step1_Critical_Zones_Sample_Result.pdf` | Échantillon des zones critiques, avant et après hiérarchisation |
| `05-OCA_OrdinalCorrMat_Targets.pdf` | Matrice ordinale de l'étape 2 |
| `06-OCA_OrdinalGraph_Targets.pdf` | Graphe de dominance final, contenu centré et étiquettes intégrales |
| `07-CriticalZones_Targets.pdf` | Échantillon avec κ et β |
| `08-OCA_Boxplot_Targets.pdf` | Boîtes à moustaches, en deux panneaux : ordre de déclaration puis ordre du classement de dominance, en regard des deux tableaux du livrable `04-` |
| `09-OCA_Hist_Targets.pdf` | Histogrammes |
| `10-OCA_Scatter_Targets.pdf` | Nuages de points |
| `11-OCA_Report_intro.pdf` | Notice analytique et journal d'exécution |

Quatre relevés textuels assurent la traçabilité numérique des figures, et
`OCA_run.log` consigne les événements d'exécution. Sans compilateur LaTeX, les
PDF `01-`, `04-`, `07-` et `11-` sont absents (leurs sources `.tex` sont conservées)
et `00-` est déclaré partiel.

`output.json` porte la version (`oca_version`), les zones et variables résolues
(`column_resolution`), le statut du rapport (`report`), le domaine de β
(`beta_domain`) et le bilan de la recherche de κ (`kappa_search`). Le champ
`config_file` reproduit le chemin de configuration **tel que fourni** s'il est relatif ;
un chemin absolu est réduit au nom du fichier, pour ne publier aucun nom de dossier local.

#### Statut de la tentative (`status`)

Au début de chaque lancement, l'ancien `output_json` est remplacé par
`{"status": "running", "started_at": ...}` : un consommateur qui surveille ce seul
fichier ne relit jamais le résultat d'un lancement antérieur. En fin de calcul, le
document de résultats porte `"status": "complete"` (`"partial"` si des
sous-dossiers d'un lot ont échoué, voir `failed_subdirs`) ; son contenu est par
ailleurs inchangé. En cas d'échec global (aucune donnée, tous les sous-dossiers en
erreur, erreur de configuration détectée après le chargement), il est remplacé par
`{"status": "failed", "reason": ..., "started_at": ..., "finished_at": ...}`. Les
dossiers de sortie des sous-dossiers en échec ne sont pas touchés.

Quand le rapport consolidé est partiel et que la notice n'a pas pu être compilée
(pas de LaTeX, ou compilation en échec), `00-OCA_Report.pdf` s'ouvre sur une page
« Rapport partiel » produite sans LaTeX, qui liste les sections absentes
(`report.status_page_included`). Un PDF dont la compilation a échoué n'est jamais
fusionné, même s'il subsiste sur le disque.

## 6. Validation

Toutes les commandes se lancent depuis la **racine du dépôt** :

```bash
python -m unittest tests.test_robustness_v9 -v          # robustesse, sans LaTeX requis
python tests/validate_pipeline.py                       # calcule l'exemple puis valide
python tests/validate_pipeline.py --no-run              # valide les sorties existantes
python tests/validate_against_reference.py              # calcule l'exemple puis compare
python tests/validate_against_reference.py --no-run     # compare des sorties déjà calculées
```

Les deux harnais prennent par défaut `examples/CAS_TEST_00/input.json` ; `--config`
désigne une autre configuration. Sans `--no-run`, le calcul est refait avant la
vérification de présence des sorties (un dépôt fraîchement cloné n'en a pas encore).

`validate_pipeline.py` porte sur le **premier** sous-dossier d'un lot et sur le
classeur de traçabilité que le pipeline a exporté (toujours en `.xlsx`, même si la
source est un `.csv`) : il ne rejoue pas la lecture des sources. Il tient compte de
l'absence de LaTeX (PDF facultatifs non exigés, page de statut comptée), d'un β publié
`null` et de l'arrondi à 4 décimales des matrices (une cellule publiée à moins de
5e-5 du seuil est ambiguë). Il recalcule indépendamment chaque grandeur publiée, par
une implémentation naïve distincte de celle du pipeline, et contrôle dix-sept
familles de propriétés : comptage de dominance, matrice ordinale initiale,
exhaustivité des nœuds du graphe de l'étape 1, admissibilité et optimalité de κ,
indice β, ressources et pagination du rapport, conventions rédactionnelles,
cohérence de l'étape 2, épuration des figures, découpage adaptatif des tableaux,
arrondi dynamique avec uniformité typographique, composition sans dépassement de
marge, tri horizontal par dominance avec absence de décalage, résolution des
annotations de type, cohérence du mode de coloration, et schéma de configuration
de la version 7 (filtrage des clés d'auto-documentation, dissociation des cartes,
rétrocompatibilité du schéma à carte unique), et cadrage des graphes orientés
avec intégrité des étiquettes.

`validate_against_reference.py` compare les sorties aux relevés de référence de
`examples/CAS_TEST_00/dataSet_criticalZones`, octet par octet pour les relevés
texte et au niveau du contenu logique pour les tableaux soumis au découpage. **Portée
exacte :** ces références sont l'état de la version 8.0, les sorties du calcul
d'origine n'ayant pas été conservées. Le harnais établit l'invariance depuis la
version 8.0 ; les graphes et figures y sont comparés sans l'en-tête de notation
ajouté en 9.0. `validate_pipeline.py` recalcule chaque grandeur publiée du premier
sous-dossier ; aucun des deux harnais ne prouve à lui seul l'exactitude de tous les
sous-dossiers d'un lot.

`tests/test_robustness_v9.py` (module `unittest`) couvre la configuration minimale,
les champs vides, nuls ou mal typés, les chemins dangereux, les classeurs sans
index, les valeurs infinies, les effectifs limites, la grille de κ, les noms de
zones à caractères spéciaux, l'absence de LaTeX, les formats de fichiers et les
manifestes d'installation.

## 7. Robustesse

Sont traités sans interruption de l'exécution : configuration non conforme à la
norme JSON (virgule orpheline signalée puis neutralisée ; commentaires retirés
silencieusement, l’écriture commentée étant admise),
divisions par zéro dans le calcul de β (statut `beta_defined`), cycles du graphe
(condensation en graphe acyclique orienté par composantes fortement connexes),
absence de relation au seuil (les variables concernées demeurent représentées au
graphe), incohérence entre une matrice et son graphe, absence ou échec du
compilateur LaTeX (rapport partiel), absence de `pypdf`, colonne constante (le
coefficient de Pearson n'est pas calculé, la figure l'annonce) et échec d'un
sous-dossier au sein d'un lot.

Sont refusés avec un message (erreur de configuration, code de sortie 2, aucune
trace) : une cible incalculable ou absente, un nom réservé, une collision de
sorties, des valeurs non finies, un débordement de l'enveloppe κ·cible. Aucun
calcul n'est alors publié (`status: "failed"` dans `output_json`).

Limites connues : voir la section « Limites connues » de `AI_AGENT_GUIDE.md`.

## 8. Référence

Amuzuga, P., Iwaniack, J.-L., « Size the reinforcement rings of all your openings
with 2pDe/E. Envelope formula of mechanical strength through a new synthetic
approach: Ordinal Correlation Algorithm », *International Journal of Pressure
Vessels and Piping*, **222** (2026), 105801.
DOI [10.1016/j.ijpvp.2026.105801](https://doi.org/10.1016/j.ijpvp.2026.105801).

| Objet | Emplacement dans le papier |
|---|---|
| Ordre de Pareto généralisé | section 2.5.1 |
| Formalisation algorithmique | section 2.5.2 |
| Comptage un contre tous, étape 1 | algorithme 1, Table 3 |
| Régularisation, étape 2 | algorithme 2 |
| Matrice de corrélation ordinale | figure 11, annexe B.5 |
| Échantillon tabulé avec colonne régularisée | Table 4 |

La mise en page des tableaux découpés suit celle de la Table 3. Correspondance
des notations : `q` vaut `n`, `f` vaut `p`, `Q` vaut `N`, `F` vaut `P`, `γ` vaut
le seuil de dominance, `τ` et `τ̄` valent `β` et `β̄`, `κ` est commun.

## 9. Licence

### Licence

Copyright © 2026 Philippe AMUZUGA

Ce programme est un logiciel libre : vous pouvez le redistribuer et le modifier selon
les termes de la **GNU Affero General Public License**, version 3 ou toute version
ultérieure, telle que publiée par la Free Software Foundation. Le texte complet
figure dans le fichier [`LICENSE`](LICENSE).

Ce programme est distribué dans l'espoir qu'il sera utile, mais **SANS AUCUNE
GARANTIE**, sans même la garantie implicite de qualité marchande ou d'adéquation à un
usage particulier.

Ce que l'AGPL implique, en pratique : toute personne qui redistribue ce programme, ou
une version modifiée, doit en fournir le code source sous la même licence. La clause
propre à l'AGPL (la section 13) étend cette obligation au cas où le programme est
mis à disposition **par le réseau** : un service en ligne bâti sur ce code doit en
publier les sources. C'est ce qui la distingue de la GPL ordinaire, et ce qui garantit
que les améliorations restent accessibles à la communauté scientifique.

Si vous utilisez ce travail dans une publication, merci de citer l'auteur.
