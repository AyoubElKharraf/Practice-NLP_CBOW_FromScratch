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
| **2** | Génération des exemples `contexte -> cible` | **Terminé** | `dataset.py`, `main.py` |
| **3** | Embeddings et architecture du modèle | **Terminé** | `embedding.py`, `model.py`, `main.py` |
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

## Étape 2 : génération des données CBOW / MLM

Cette étape est terminée. Le fichier `dataset.py` transforme chaque phrase
tokenisée en exemples d'entraînement simples :

```text
phrase   : ["le", "chat", "dort", "sur", "le", "tapis"]
contexte : ["le", "chat", "sur", "le"]
cible    : "dort"
```

La taille de la fenêtre est désormais paramétrable par l'utilisateur :
on peut choisir séparément le nombre de mots avant (`fenetre_avant`)
et après (`fenetre_apres`), avec une valeur par défaut de 2.
Quand une position sort de la phrase, elle est remplacée par `<PAD>`, ce qui donne
des contextes de longueur constante (`fenetre_avant + fenetre_apres`).

Le module fournit trois fonctions faciles à suivre :

- `construire_contexte(...)` : récupère les mots voisins d'une cible ;
- `generer_exemples(...)` : crée les couples contexte/cible et la phrase avec
  `<MASK>` ;
- `convertir_exemples_en_indices(...)` : prépare les données numériques pour
  le futur modèle.

Chaque exemple contient :

```python
{
    "contexte": ["le", "chat", "sur", "le"],
    "cible": "dort",
    "phrase_masquee": ["le", "chat", "<MASK>", "sur", "le", "tapis"]
}
```

La démonstration de cette étape est disponible dans `main.py`.

## Étape 3 : Embeddings et architecture du modèle

Cette étape est terminée. Elle met en place la structure du réseau de neurones en séparant la gestion des plongements lexicaux et l'architecture globale du modèle :

### Modules disponibles

#### `embedding.py` (`CoucheEmbedding`)
- initialise la matrice des embeddings $E \in \mathbb{R}^{V \times d}$ avec l'initialisation uniforme de Xavier/Glorot ($[-\frac{1}{\sqrt{d}}, +\frac{1}{\sqrt{d}}]$) ;
- fixe le vecteur du token spécial `<PAD>` à zéro ;
- fournit `obtenir_vecteur(indice)` et `obtenir_vecteurs(indices)` ;
- propose `agreger_contexte(indices_contexte, mode="moyenne")` pour condenser les $K$ vecteurs de mots voisins en un unique vecteur de contexte $h \in \mathbb{R}^d$ (en ignorant le padding pour préserver le sens).

#### `model.py` (`ModeleCBOW`)
- intègre la `CoucheEmbedding` ;
- initialise les poids de la couche linéaire de sortie $W_{\text{sortie}} \in \mathbb{R}^{d \times V}$ et le vecteur de biais $b_{\text{sortie}} \in \mathbb{R}^V$ ;
- implémente la fonction `softmax` avec stabilisation numérique ($z - \max(z)$) pour éviter tout risque de dépassement de capacité (*overflow*) ;
- fournit la méthode `forward(indices_contexte)` calculant :
  1. $h = \text{moyenne}(E[\text{contexte}])$
  2. $z = h \cdot W_{\text{sortie}} + b_{\text{sortie}}$ (scores / logits)
  3. $p = \text{softmax}(z)$ (distribution de probabilités sur les $V$ mots)
- fournit `predire_top_k(indices_contexte, k=5)` pour extraire directement les candidats les plus probables.

#### `main.py` (Section 6)
- instancie le modèle avec la taille réelle du vocabulaire ($V = 1617$) et $d = 32$ ;
- teste et affiche la passe avant (*forward pass*) sur un exemple concret extrait du corpus.

## Vision des prochaines étapes

Les étapes 1, 2 et 3 étant terminées, les prochaines étapes sont :

### Étape 4 - Forward pass, loss et backpropagation (`train.py`)
Le modèle calculera la perte d'entropie croisée (*cross-entropy loss*) entre les probabilités prédites et le mot cible :
```text
perte = -log(p[indice_cible])
```
Les gradients par rapport à $W_{\text{sortie}}$, $b_{\text{sortie}}$ et aux embeddings des mots du contexte seront dérivés manuellement et appliqués par descente de gradient stochastique (SGD).

### Étape 5 - Inférence, prédiction et évaluation (`predict.py`)
- prédiction interactive de mots masqués (`<MASK>`) sur des phrases fournies par l'utilisateur ;
- évaluation quantitative de la précision (Accuracy Top-1 et Top-5) sur le jeu de données.

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
├── dataset.py
├── embedding.py
├── model.py
├── main.py
├── README.md
└── .gitignore
```

Les fichiers `train.py` et `predict.py` seront ajoutés au fur et à
mesure de l'avancement.

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
