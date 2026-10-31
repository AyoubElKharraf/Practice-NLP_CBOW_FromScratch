"""
Entraînement manuel du modèle CBOW avec NumPy.

Ce module calcule la perte d'entropie croisée, dérive les gradients à la main
et met à jour les poids avec une descente de gradient stochastique (SGD).
"""

import numpy as np

from model import ModeleCBOW


def calculer_perte(probabilites: np.ndarray, indice_cible: int) -> float:
    """Calcule -log de la probabilité attribuée au mot cible."""
    probabilite_cible = np.clip(probabilites[indice_cible], 1e-12, 1.0)
    return float(-np.log(probabilite_cible))


def entrainer_exemple(
    modele: ModeleCBOW,
    contexte: list[int],
    cible: int,
    taux_apprentissage: float = 0.05,
) -> float:
    """Entraîne le modèle sur un exemple et retourne sa perte avant mise à jour."""
    sortie = modele.forward(contexte)
    h = sortie["vecteur_contexte"]
    probabilites = sortie["probabilites"]
    perte = calculer_perte(probabilites, cible)

    # Pour softmax + entropie croisée : gradient des scores = probabilités - cible.
    gradient_scores = probabilites.copy()
    gradient_scores[cible] -= 1.0

    # Gradients de la couche de sortie : z = h @ W_sortie + b_sortie.
    gradient_W = np.outer(h, gradient_scores)
    gradient_b = gradient_scores
    gradient_h = modele.W_sortie @ gradient_scores

    # Met à jour la couche de sortie avec SGD.
    modele.W_sortie -= taux_apprentissage * gradient_W
    modele.b_sortie -= taux_apprentissage * gradient_b

    # Le contexte a été moyenné sans les PAD : répartit le gradient également
    # entre ses embeddings non-PAD. Une occurrence répétée reçoit sa part aussi.
    indices_actifs = [idx for idx in contexte if idx != modele.couche_embedding.indice_pad]
    if indices_actifs:
        gradient_par_mot = gradient_h / len(indices_actifs)
        for indice in indices_actifs:
            modele.couche_embedding.poids[indice] -= taux_apprentissage * gradient_par_mot

    # Le padding reste toujours un vecteur nul.
    modele.couche_embedding.poids[modele.couche_embedding.indice_pad] = 0.0

    return perte


def entrainer_modele(
    modele: ModeleCBOW,
    exemples: list[dict[str, list[int] | int]],
    epochs: int = 5,
    taux_apprentissage: float = 0.05,
    melanger: bool = True,
) -> list[float]:
    """Entraîne le modèle sur tous les exemples et retourne la perte moyenne par epoch."""
    if not exemples:
        raise ValueError("La liste d'exemples d'entraînement est vide.")
    if epochs < 1:
        raise ValueError("Le nombre d'epochs doit être supérieur ou égal à 1.")
    if taux_apprentissage <= 0:
        raise ValueError("Le taux d'apprentissage doit être positif.")

    pertes_epochs = []

    for numero_epoch in range(epochs):
        if melanger:
            ordre = np.random.permutation(len(exemples))
        else:
            ordre = range(len(exemples))

        perte_totale = 0.0
        for position in ordre:
            exemple = exemples[int(position)]
            perte_totale += entrainer_exemple(
                modele,
                exemple["contexte"],
                exemple["cible"],
                taux_apprentissage=taux_apprentissage,
            )

        perte_moyenne = perte_totale / len(exemples)
        pertes_epochs.append(perte_moyenne)
        print(f"Epoch {numero_epoch + 1}/{epochs} - perte moyenne : {perte_moyenne:.4f}")

    return pertes_epochs
