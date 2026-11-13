# Références de non-régression

Ces fichiers sont l'**état de référence de la version 8.0**, et non les sorties
historiques du calcul d'origine, qui n'ont pas été conservées.

Ils ont été régénérés à partir du sous-dossier `01-RMR-P1_Fv0_Fext0` de l'exemple
`CAS_TEST_00`, dont les résultats coïncident au centième avec les valeurs de
référence documentées : κ = 1,55 et β moyen = 1,21 sur 907 observations.

Le harnais `tests/validate_against_reference.py` les compare aux sorties du calcul :
il vérifie donc l'**invariance depuis la version 8.0**, non l'exactitude du calcul
d'origine. C'est une garantie plus faible que celle que son nom suggère, et elle est
énoncée ici pour qu'elle ne soit pas surestimée. L'exactitude absolue est contrôlée
par `tests/validate_pipeline.py`, qui recalcule chaque grandeur publiée par une
implémentation indépendante.

Depuis la version 9.0, les relevés de graphe portent en tête une phrase de lecture de
la notation `A < B` ; le harnais l'écarte avant la comparaison, ces références restant
celles de la version 8.0.
