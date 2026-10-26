"""
Point d'entrée principal : main.py
----------------------------------
Étapes 1, 2 et 3 : Prétraitement, données CBOW/MLM et architecture du modèle

Pipeline :
1. Chargement du corpus brut (depuis corpus.py).
2. Nettoyage et tokenisation (depuis tokenizer.py).
3. Construction du vocabulaire et des dictionnaires de mapping (depuis vocabulary.py).
4. Affichage complet et démonstration de l'encodage/décodage numérique.
5. Génération des exemples contexte -> cible pour l'entraînement (depuis dataset.py).
6. Initialisation des embeddings et du modèle CBOW / MiniBERT (depuis embedding.py et model.py).
"""

import sys
import numpy as np

# Assure le rendu correct des caractères accentués sous Windows
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from corpus import phrases
from dataset import convertir_exemples_en_indices, generer_exemples
from model import ModeleCBOW
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

def demander_entier_utilisateur(message: str, valeur_defaut: int = 2) -> int:
    """
    Demande un entier à l'utilisateur (ex: taille de fenêtre).
    Si l'utilisateur appuie directement sur Entrée (ou si non interactif),
    utilise la valeur par défaut.
    """
    try:
        saisie = input(f"{message} [défaut={valeur_defaut}] : ").strip()
        if not saisie:
            return valeur_defaut
        val = int(saisie)
        return val if val >= 0 else valeur_defaut
    except (ValueError, EOFError):
        return valeur_defaut

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

    # -------------------------------------------------------------
    # 5. Génération des exemples CBOW / MLM
    # -------------------------------------------------------------
    afficher_separateur("5. GÉNÉRATION DES DONNÉES CONTEXTE -> CIBLE")
    print("Configuration de la fenêtre de contexte (choix utilisateur) :")
    fenetre_avant = demander_entier_utilisateur("  Nombre de mots AVANT la cible", valeur_defaut=2)
    fenetre_apres = demander_entier_utilisateur("  Nombre de mots APRÈS la cible", valeur_defaut=2)
    print(f"\n-> Fenêtre retenue : {fenetre_avant} mot(s) avant et {fenetre_apres} mot(s) après.")
    print(f"-> Taille totale du contexte pour chaque mot : {fenetre_avant + fenetre_apres} mots.\n")

    exemples = generer_exemples(
        corpus_tokenise, fenetre_avant=fenetre_avant, fenetre_apres=fenetre_apres
    )
    exemples_indices = convertir_exemples_en_indices(
        exemples, mot_vers_indice
    )
    print(f"Nombre total d'exemples créés : {len(exemples)}")
    print("\n[Exemples texte] :")
    for exemple in exemples[:3]:
        print(f"  Contexte : {exemple['contexte']}")
        print(f"  Cible : {exemple['cible']}")
        print(f"  Phrase masquée : {exemple['phrase_masquee']}")
        print()

    print("[Premier exemple converti en indices] :")
    print(f"  {exemples_indices[0]}")

    # -------------------------------------------------------------
    # 6. Étape 3 : Embeddings et Modèle CBOW / MiniBERT
    # -------------------------------------------------------------
    afficher_separateur("6. ÉTAPE 3 : EMBEDDINGS ET ARCHITECTURE DU MODÈLE")
    d = 32  # Dimension des embeddings
    modele = ModeleCBOW(
        taille_vocabulaire=V,
        dimension_embedding=d,
        indice_pad=mot_vers_indice["<PAD>"]
    )

    print("Modèle initialisé avec succès :")
    print(f"  - Taille du vocabulaire (V)   : {V}")
    print(f"  - Dimension des embeddings (d) : {d}")
    print(f"  - Matrice des embeddings (E)  : {modele.couche_embedding.poids.shape} (V x d)")
    print(f"  - Matrice de sortie (W_sortie) : {modele.W_sortie.shape} (d x V)")
    print(f"  - Biais de sortie (b_sortie)   : {modele.b_sortie.shape} (V,)")

    # Test du Forward Pass sur le premier exemple généré
    premier_ex = exemples_indices[0]
    contexte_indices = premier_ex["contexte"]
    cible_indice = premier_ex["cible"]
    mot_cible_texte = indice_vers_mot[cible_indice]
    mots_contexte = [indice_vers_mot[idx] for idx in contexte_indices]

    print("\n[Démonstration du Forward Pass sur un exemple réel] :")
    print(f"  Contexte (mots)    : {mots_contexte}")
    print(f"  Contexte (indices) : {contexte_indices}")
    print(f"  Cible attendue     : '{mot_cible_texte}' (indice {cible_indice})")

    # Passe avant
    sortie = modele.forward(contexte_indices)
    h = sortie["vecteur_contexte"]
    scores = sortie["scores"]
    probabilites = sortie["probabilites"]

    print(f"\n  1. Vecteur contexte agrégé h (taille {len(h)}) :")
    print(f"     {np.round(h[:5], 3)} ... (aperçu des 5 premières dimensions)")
    print(f"  2. Scores bruts z (taille {len(scores)}) :")
    print(f"     {np.round(scores[:5], 3)} ... (aperçu)")
    print(f"  3. Probabilités Softmax (taille {len(probabilites)}) :")
    print(f"     Somme des probabilités = {np.sum(probabilites):.4f}")

    top3 = modele.predire_top_k(contexte_indices, k=3)
    print("\n  Top 3 des prédictions (modèle non encore entraîné) :")
    for rang, (idx, prob) in enumerate(top3, start=1):
        print(f"     {rang}. '{indice_vers_mot[idx]}' (indice {idx}) avec probabilité {prob*100:.2f}%")

if __name__ == "__main__":
    main()
