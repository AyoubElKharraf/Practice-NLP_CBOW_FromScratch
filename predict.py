"""
Prédiction et évaluation du modèle CBOW.

Le module utilise le modèle entraîné pour prédire le mot indiqué par <MASK>
et calculer les précisions Top-1 / Top-5 sur des exemples encodés.
"""

import numpy as np

from model import ModeleCBOW
from tokenizer import nettoyer_et_tokeniser
from vocabulary import TOKEN_MASK, TOKEN_PAD, TOKEN_UNK


def construire_contexte_masque(
    phrase_masquee: str,
    mot_vers_indice: dict[str, int],
    fenetre_avant: int = 2,
    fenetre_apres: int | None = None,
) -> list[int]:
    """Transforme les mots voisins de l'unique <MASK> en indices de contexte."""
    if fenetre_avant < 0:
        raise ValueError("fenetre_avant doit être positive ou nulle.")
    if fenetre_apres is None:
        fenetre_apres = fenetre_avant
    if fenetre_apres < 0:
        raise ValueError("fenetre_apres doit être positive ou nulle.")

    morceaux = phrase_masquee.split(TOKEN_MASK)
    if len(morceaux) != 2:
        raise ValueError("La phrase doit contenir exactement un <MASK>.")

    mots_gauche = nettoyer_et_tokeniser(morceaux[0])
    mots_droite = nettoyer_et_tokeniser(morceaux[1])

    # Garder les mots les plus proches du masque, dans l'ordre attendu par dataset.py.
    mots_avant = mots_gauche[-fenetre_avant:] if fenetre_avant else []
    contexte_avant = [TOKEN_PAD] * (fenetre_avant - len(mots_avant)) + mots_avant
    mots_apres = mots_droite[:fenetre_apres]
    contexte_apres = mots_apres + [TOKEN_PAD] * (fenetre_apres - len(mots_apres))
    mots_contexte = contexte_avant + contexte_apres

    indice_unk = mot_vers_indice.get(TOKEN_UNK, 0)
    return [mot_vers_indice.get(mot, indice_unk) for mot in mots_contexte]


def predire_mot_masque(
    modele: ModeleCBOW,
    phrase_masquee: str,
    mot_vers_indice: dict[str, int],
    indice_vers_mot: dict[int, str],
    k: int = 5,
    fenetre_avant: int = 2,
    fenetre_apres: int | None = None,
) -> list[tuple[str, float]]:
    """Retourne jusqu'à k mots candidats et leurs probabilités pour <MASK>."""
    if k < 1:
        raise ValueError("k doit être supérieur ou égal à 1.")

    contexte = construire_contexte_masque(
        phrase_masquee,
        mot_vers_indice,
        fenetre_avant=fenetre_avant,
        fenetre_apres=fenetre_apres,
    )
    probabilites = modele.forward(contexte)["probabilites"]

    # Les tokens de service ne sont pas des réponses linguistiques à proposer.
    tokens_speciaux = {TOKEN_PAD, TOKEN_UNK, TOKEN_MASK}
    ordre = np.argsort(probabilites)[::-1]
    resultats = []
    for indice in ordre:
        mot = indice_vers_mot.get(int(indice), TOKEN_UNK)
        if mot not in tokens_speciaux:
            resultats.append((mot, float(probabilites[indice])))
            if len(resultats) == k:
                break
    return resultats


def evaluer_modele(
    modele: ModeleCBOW,
    exemples: list[dict[str, list[int] | int]],
    k: int = 5,
) -> dict[str, float]:
    """Calcule les accuracies Top-1 et Top-k sur des exemples encodés."""
    if not exemples:
        raise ValueError("La liste d'exemples d'évaluation est vide.")
    if k < 1:
        raise ValueError("k doit être supérieur ou égal à 1.")

    bonnes_top_1 = 0
    bonnes_top_k = 0

    for exemple in exemples:
        probabilites = modele.forward(exemple["contexte"])["probabilites"]
        meilleurs_indices = np.argsort(probabilites)[::-1][:k]
        cible = exemple["cible"]

        if meilleurs_indices[0] == cible:
            bonnes_top_1 += 1
        if cible in meilleurs_indices:
            bonnes_top_k += 1

    nombre_exemples = len(exemples)
    return {
        "accuracy_top1": bonnes_top_1 / nombre_exemples,
        f"accuracy_top{k}": bonnes_top_k / nombre_exemples,
    }


def mode_interactif(
    modele: ModeleCBOW,
    mot_vers_indice: dict[str, int],
    indice_vers_mot: dict[int, str],
    k: int = 5,
    fenetre_avant: int = 2,
    fenetre_apres: int | None = None,
) -> None:
    """Demande des phrases contenant <MASK> et affiche les mots candidats."""
    print(f"Entrez une phrase avec {TOKEN_MASK} (Entrée vide pour quitter).")
    while True:
        try:
            phrase = input("Phrase : ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not phrase:
            break

        try:
            candidats = predire_mot_masque(
                modele,
                phrase,
                mot_vers_indice,
                indice_vers_mot,
                k=k,
                fenetre_avant=fenetre_avant,
                fenetre_apres=fenetre_apres,
            )
        except ValueError as erreur:
            print(f"Entrée invalide : {erreur}")
            continue

        if not candidats:
            print("Aucun candidat disponible.")
            continue

        print("Candidats :")
        for rang, (mot, probabilite) in enumerate(candidats, start=1):
            print(f"  {rang}. {mot} ({probabilite * 100:.2f} %)")
