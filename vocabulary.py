"""
Module vocabulary.py
--------------------
Rôle : Création du vocabulaire unique et gestion des correspondances mots <-> indices numériques.
- Extraction des mots uniques.
- Création des dictionnaires mot_vers_indice et indice_vers_mot.
- Conversion de séquences de tokens en indices numériques (et inversement).
- Intégration optionnelle de tokens spéciaux (<PAD>, <UNK>, <MASK>) indispensables pour la prédiction de mots masqués (approches type CBOW / BERT).
"""

from typing import Iterable

TOKEN_PAD = "<PAD>" # Token de remplissage pour les séquences de longueur variable (padding).
TOKEN_UNK = "<UNK>" # Token pour les mots inconnus (hors vocabulaire).
TOKEN_MASK = "<MASK>" # Token pour les mots masqués (utilisé dans les approches de type CBOW / BERT).

def extraire_vocabulaire_unique(corpus_tokenise: list[list[str]]) -> list[str]:
    """
    Extrait la liste triée de tous les mots uniques apparaissant
    dans le corpus tokenisé.
    """
    vocabulaire = set()
    for phrase_tokens in corpus_tokenise:
        vocabulaire.update(phrase_tokens)
    return sorted(list(vocabulaire))

def construire_vocabulaire(corpus_tokenise: list[list[str]], inclure_speciaux: bool = True) -> tuple[dict[str, int], dict[int, str]]:
    """
    Construit les dictionnaires de correspondance mot <-> indice.
    Paramètres :
        corpus_tokenise : Liste de listes de tokens.
        inclure_speciaux : Si True, ajoute au début les tokens spéciaux : <PAD> (remplissage), <UNK> (mot inconnu) et <MASK> (mot masqué).
    Retourne :
        (mot_vers_indice, indice_vers_mot)
    """
    mots_uniques = extraire_vocabulaire_unique(corpus_tokenise)
    
    mot_vers_indice = {}
    indice_vers_mot = {}
    
    index_actuel = 0

    if inclure_speciaux:
        tokens_speciaux = [TOKEN_PAD, TOKEN_UNK, TOKEN_MASK]
        for token in tokens_speciaux:
            mot_vers_indice[token] = index_actuel
            indice_vers_mot[index_actuel] = token
            index_actuel += 1

    for mot in mots_uniques:
        if mot not in mot_vers_indice:
            mot_vers_indice[mot] = index_actuel
            indice_vers_mot[index_actuel] = mot
            index_actuel += 1

    return mot_vers_indice, indice_vers_mot

def tokens_vers_indices(tokens: list[str], mot_vers_indice: dict[str, int], token_inconnu: str = TOKEN_UNK) -> list[int]:
    """
    Convertit une liste de tokens en une liste d'indices numériques.
    Si un mot n'est pas dans le vocabulaire, utilise l'indice du token inconnu.
    """
    indice_unk = mot_vers_indice.get(token_inconnu, 0)
    return [mot_vers_indice.get(token, indice_unk) for token in tokens]

def indices_vers_tokens(indices: list[int], indice_vers_mot: dict[int, str]) -> list[str]:
    """
    Convertit une liste d'indices numériques en une liste de tokens (texte).
    """
    return [indice_vers_mot.get(idx, TOKEN_UNK) for idx in indices]

if __name__ == "__main__":
    mini_corpus = [
        ["le", "chat", "dort"],
        ["le", "chien", "joue"]
    ]
    m2i, i2m = construire_vocabulaire(mini_corpus, inclure_speciaux=True)
    print("mot_vers_indice :", m2i)
    print("indice_vers_mot :", i2m)
    indices = tokens_vers_indices(["le", "chat", "inconnu"], m2i)
    print("Tokens vers indices :", indices)
    print("Indices vers tokens :", indices_vers_tokens(indices, i2m))

