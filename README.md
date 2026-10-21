# MiniBERT-CBOW-FromScratch

Implémentation pédagogique, **from scratch**, d'un modèle de prédiction
contextuelle inspiré de **CBOW (Continuous Bag of Words)** et du
**Masked Language Modeling (MLM)** de BERT.

Le projet est réalisé en Python, sans framework de Deep Learning :
**PyTorch, TensorFlow et Keras ne sont pas utilisés**. NumPy pourra être
utilisé uniquement pour faciliter les calculs matriciels de base.

## Objectif général

À partir d'une phrase contenant un mot masqué ou manquant, le modèle devra
utiliser les mots voisins pour prédire le mot correct parmi les mots du
vocabulaire.

Le modèle final reposera sur les composants suivants :

- un corpus de phrases françaises ;
- un vocabulaire indexé de `0` à `V - 1` ;
- une matrice d'embeddings de taille `V x d` ;
- une représentation du contexte obtenue par somme ou moyenne des embeddings ;
- une couche de sortie produisant un score pour chaque mot ;
- une fonction `softmax` transformant les scores en probabilités ;
- une fonction de coût et une rétropropagation codées manuellement.

## Progression du TP

| Étape | Description | Statut | Fichiers prévus |
| :---: | :--- | :---: | :--- |
| **1** | Prétraitement, tokenisation et vocabulaire | **Terminé** | `corpus.py`, `tokenizer.py`, `vocabulary.py`, `main.py` |
| **2** | Génération des exemples `contexte -> cible` | À venir | `dataset.py` |
| **3** | Embeddings et architecture du modèle | À venir | `model.py` |
| **4** | Forward pass, loss et backpropagation | À venir | `train.py` |
| **5** | Inférence, prédiction et évaluation | À venir | `predict.py`, `main.py` |

Chaque nouvelle étape sera ajoutée au projet et marquée comme terminée dans
ce tableau après validation.

## Étape 1 : prétraitement et vocabulaire

Cette étape est terminée. Elle fournit le pipeline de préparation des données :

1. chargement d'un corpus français ;
2. nettoyage et tokenisation ;
3. extraction des mots uniques ;
4. création des mappings mot/indice ;
5. conversion des tokens en indices et inversement ;
6. préparation d'un token `<MASK>` pour les étapes suivantes.

### Modules disponibles

#### `corpus.py`

- charge jusqu'à 500 phrases depuis le Projet Gutenberg ;
- utilise `corpus_gutenberg.txt` comme cache local ;
- propose un mini-corpus français intégré pour les tests rapides ;
- expose `obtenir_corpus(source="gutenberg" | "mini")`.

#### `tokenizer.py`

- convertit le texte en minuscules ;
- conserve les lettres accentuées françaises ;
- remplace la ponctuation par des espaces ;
- transforme une phrase ou un corpus en listes de tokens.

#### `vocabulary.py`

- extrait les mots uniques et les trie ;
- construit `mot_vers_indice` et `indice_vers_mot` ;
- ajoute les tokens spéciaux :
  - `<PAD>` : remplissage des contextes ;
  - `<UNK>` : mot absent du vocabulaire ;
  - `<MASK>` : mot masqué à prédire ;
- encode les tokens en indices et décode les indices en tokens.

#### `main.py`

Orchestre l'étape 1 et affiche :

- le nombre de phrases chargées ;
- le nombre de phrases tokenisées ;
- le nombre total de tokens ;
- la taille du vocabulaire ;
- un exemple d'encodage/décodage ;
- un exemple de remplacement d'un mot par `<MASK>`.

## Vision des prochaines étapes

### Étape 2 - Création des données CBOW/MLM

Pour chaque phrase, une fenêtre glissante sera utilisée afin de créer des
exemples de la forme :

```text
contexte : ["le", "chat", "sur", "tapis"]
cible    : "dort"
```

Le mot cible pourra également être remplacé par `<MASK>` dans la séquence
d'entrée.

### Étape 3 - Embeddings et modèle

Une matrice d'embeddings sera initialisée aléatoirement :

```text
E : V x d
```

Les embeddings des mots du contexte seront ensuite additionnés ou moyennés
pour obtenir un vecteur de contexte de dimension `d`.

### Étape 4 - Forward, loss et backpropagation

Le modèle calculera :

```text
vecteur_contexte = moyenne(embeddings_du_contexte)
scores = vecteur_contexte @ poids_de_sortie
probabilités = softmax(scores)
```

La perte sera calculée à partir du mot cible. Les gradients seront dérivés
et appliqués manuellement avec la descente de gradient :

```text
W_nouveau = W_actuel - taux_apprentissage * gradient
```

### Étape 5 - Prédiction et évaluation

Le modèle sera utilisé pour :

- prédire le mot masqué ;
- afficher les meilleurs candidats et leurs probabilités ;
- mesurer la précision sur des exemples de test ;
- comparer les prédictions du modèle avec les mots réels.

## Installation et exécution

Python 3.10 ou une version plus récente est recommandé.

L'étape actuelle peut être exécutée sans dépendance externe :

```bash
python main.py
```

Pour utiliser le mini-corpus depuis Python :

```python
from corpus import obtenir_corpus

phrases = obtenir_corpus(source="mini")
```

## Arborescence actuelle

```text
MiniBERT-CBOW-FromScratch/
├── corpus.py
├── corpus_gutenberg.txt
├── tokenizer.py
├── vocabulary.py
├── main.py
├── README.md
└── .gitignore
```

Les fichiers `dataset.py`, `model.py`, `train.py` et `predict.py` seront
ajoutés au fur et à mesure de l'avancement.

## Contraintes pédagogiques

- aucune bibliothèque de Deep Learning ;
- implémentation manuelle des opérations importantes ;
- séparation claire entre préparation des données, modèle, entraînement et
  inférence ;
- validation de chaque étape avant de passer à la suivante.

## Nom GitHub proposé

**`MiniBERT-CBOW-FromScratch`** est le nom recommandé, car il décrit :

- le caractère minimal et pédagogique du modèle ;
- l'inspiration CBOW et MLM ;
- l'implémentation sans framework de Deep Learning.

Alternatives possibles :

- `FrenchContextModel-FromScratch`
- `CBOW-MLM-NLP`
- `NLP-Predictive-Model-FromScratch`
