"""
Point d'entrée principal : main.py
----------------------------------
Étape 1 : Prétraitement, Tokenisation et Vocabulaire (From Scratch en Python)

Pipeline :
1. Chargement du corpus brut (depuis corpus.py).
2. Nettoyage et tokenisation (depuis tokenizer.py).
3. Construction du vocabulaire et des dictionnaires de mapping (depuis vocabulary.py).
4. Affichage complet et démonstration de l'encodage/décodage numérique.
"""

import sys

# Assure le rendu correct des caractères accentués sous Windows
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from corpus import phrases, obtenir_corpus
from tokenizer import tokeniser_corpus, nettoyer_et_tokeniser
from vocabulary import (
    construire_vocabulaire, 
    tokens_vers_indices, 
    indices_vers_tokens,
    TOKEN_MASK
)

def afficher_separateur(titre: str):
    print("\n" + "=" * 70)
    print(f"  {titre}")
    print("=" * 70)

def main():
    # -------------------------------------------------------------
    # 1. Chargement du corpus brut
    # -------------------------------------------------------------
    afficher_separateur("1. CHARGEMENT DU CORPUS BRUT")
    print(f"Nombre total de phrases récupérées : {len(phrases)}")
    print("\n[Aperçu des 3 premières phrases brutes] :")
    for i, p in enumerate(phrases[:3], start=1):
        print(f"  {i}. {p}")

    # -------------------------------------------------------------
    # 2. Nettoyage et Tokenisation
    # -------------------------------------------------------------
    afficher_separateur("2. NETTOYAGE ET TOKENISATION")
    corpus_tokenise = tokeniser_corpus(phrases)
    nb_tokens_total = sum(len(p) for p in corpus_tokenise)
    print(f"Nombre de phrases après tokenisation : {len(corpus_tokenise)}")
    print(f"Nombre total de mots (tokens) dans le corpus : {nb_tokens_total}")
    
    print("\n[Aperçu des 3 premières phrases tokenisées] :")
    for i, tokens in enumerate(corpus_tokenise[:3], start=1):
        print(f"  {i}. {tokens}")

    # -------------------------------------------------------------
    # 3. Construction du Vocabulaire
    # -------------------------------------------------------------
    afficher_separateur("3. CONSTRUCTION DU VOCABULAIRE ET DES MAPPINGS")
    mot_vers_indice, indice_vers_mot = construire_vocabulaire(corpus_tokenise, inclure_speciaux=True)
    V = len(mot_vers_indice)
    print(f"Taille du vocabulaire unique (V) : {V} mots (incluant tokens spéciaux)")

    # Aperçu des dictionnaires
    print("\n[Échantillon de mot_vers_indice (10 premiers éléments)] :")
    sample_m2i = dict(list(mot_vers_indice.items())[:10])
    print(" ", sample_m2i)

    print("\n[Échantillon de indice_vers_mot (10 premiers éléments)] :")
    sample_i2m = dict(list(indice_vers_mot.items())[:10])
    print(" ", sample_i2m)

    # -------------------------------------------------------------
    # 4. Démonstration de conversion numérique (Encodage / Décodage)
    # -------------------------------------------------------------
    afficher_separateur("4. EXEMPLE DE CONVERSION EN INDICES NUMÉRIQUES")
    # Exemple 1 : Une phrase issue du corpus
    phrase_exemple = phrases[0]
    tokens_exemple = nettoyer_et_tokeniser(phrase_exemple)
    indices_exemple = tokens_vers_indices(tokens_exemple, mot_vers_indice)
    reconstruit = indices_vers_tokens(indices_exemple, indice_vers_mot)

    print("Phrase originale :")
    print(f"  \"{phrase_exemple}\"")
    print("\nTokens extraits :")
    print(f"  {tokens_exemple}")
    print("\nIndices numériques correspondants :")
    print(f"  {indices_exemple}")
    print("\nTokens reconstruits à partir des indices :")
    print(f"  {reconstruit}")

    # Exemple 2 : Préparation pour la prédiction d'un mot masqué (contexte CBOW / MLM)
    print("\n" + "-" * 70)
    print("Cas d'usage pour la suite du TP (Mot masqué / Contexte) :")
    if len(tokens_exemple) >= 3:
        position_a_masquer = 1
        tokens_masques = list(tokens_exemple)
        mot_cible = tokens_masques[position_a_masquer]
        tokens_masques[position_a_masquer] = TOKEN_MASK
        indices_masques = tokens_vers_indices(tokens_masques, mot_vers_indice)

        print(f"Mot original masqué à la position {position_a_masquer} : '{mot_cible}'")
        print(f"Tokens avec masque : {tokens_masques}")
        print(f"Indices numériques avec {TOKEN_MASK} : {indices_masques}")

if __name__ == "__main__":
    main()
