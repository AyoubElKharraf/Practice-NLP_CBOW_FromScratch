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
    etat_adam: dict | None = None,
    taux_dropout: float | None = None,
) -> float:
    """Entraîne le modèle sur un exemple et retourne sa perte avant mise à jour."""
    if taux_dropout is not None:
        modele.taux_dropout = taux_dropout

    sortie = modele.forward(contexte, mode_agregation="concat", entrainement=True)
    h = sortie["vecteur_contexte"]
    activations_reglees = sortie["activations_reglees"]
    scores_hidden = sortie["scores_hidden"]
    probabilites = sortie["probabilites"]
    perte = calculer_perte(probabilites, cible)

    # Pour softmax + entropie croisée : gradient des scores = probabilités - cible.
    gradient_scores = probabilites.copy()
    gradient_scores[cible] -= 1.0

    # Gradients de la couche de sortie : z = h @ W_sortie + b_sortie.
    gradient_W_sortie = np.outer(activations_reglees, gradient_scores)
    gradient_b_sortie = gradient_scores

    gradient_activations = modele.W_sortie @ gradient_scores
    masque_dropout = sortie["masque_dropout"]
    if masque_dropout is not None:
        gradient_activations = (
            gradient_activations * masque_dropout
        ) / (1.0 - modele.taux_dropout)

    gradient_scores_hidden = gradient_activations * (scores_hidden > 0)
    gradient_W_hidden = np.outer(h, gradient_scores_hidden)
    gradient_b_hidden = gradient_scores_hidden
    gradient_h = modele.W_hidden @ gradient_scores_hidden

    gradients_embeddings = np.zeros_like(modele.couche_embedding.poids)
    for position, indice in enumerate(contexte):
        if indice != modele.couche_embedding.indice_pad:
            debut = position * modele.d
            gradients_embeddings[indice] += gradient_h[debut:debut + modele.d]

    gradients = {
        "poids_embeddings": gradients_embeddings,
        "W_hidden": gradient_W_hidden,
        "b_hidden": gradient_b_hidden,
        "W_sortie": gradient_W_sortie,
        "b_sortie": gradient_b_sortie,
    }

    if etat_adam is None:
        modele.W_hidden -= taux_apprentissage * gradient_W_hidden
        modele.b_hidden -= taux_apprentissage * gradient_b_hidden
        modele.W_sortie -= taux_apprentissage * gradient_W_sortie
        modele.b_sortie -= taux_apprentissage * gradient_b_sortie
        modele.couche_embedding.poids -= taux_apprentissage * gradients_embeddings
    else:
        appliquer_adam(modele, etat_adam, gradients, taux_apprentissage)

    # Le padding reste toujours un vecteur nul.
    modele.couche_embedding.poids[modele.couche_embedding.indice_pad] = 0.0

    return perte


def appliquer_adam(
    modele: ModeleCBOW,
    etat: dict,
    gradients: dict[str, np.ndarray],
    taux_apprentissage: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> None:
    """Applique Adam aux paramètres du modèle."""
    etat["pas"] += 1
    pas = etat["pas"]

    parametres = {
        "poids_embeddings": modele.couche_embedding.poids,
        "W_hidden": modele.W_hidden,
        "b_hidden": modele.b_hidden,
        "W_sortie": modele.W_sortie,
        "b_sortie": modele.b_sortie,
    }

    for nom, parametre in parametres.items():
        gradient = gradients[nom]
        if nom not in etat["moment1"]:
            etat["moment1"][nom] = np.zeros_like(parametre)
            etat["moment2"][nom] = np.zeros_like(parametre)

        moment1 = etat["moment1"][nom]
        moment2 = etat["moment2"][nom]
        moment1 *= beta1
        moment1 += (1.0 - beta1) * gradient
        moment2 *= beta2
        moment2 += (1.0 - beta2) * (gradient * gradient)

        moment1_corrige = moment1 / (1.0 - beta1 ** pas)
        moment2_corrige = moment2 / (1.0 - beta2 ** pas)
        parametre -= taux_apprentissage * moment1_corrige / (
            np.sqrt(moment2_corrige) + epsilon
        )


def entrainer_modele(
    modele: ModeleCBOW,
    exemples: list[dict[str, list[int] | int]],
    epochs: int = 5,
    taux_apprentissage: float = 0.001,
    melanger: bool = True,
    exemples_validation: list[dict[str, list[int] | int]] | None = None,
    patience: int = 3,
) -> list[float]:
    """Entraîne le modèle sur tous les exemples et retourne la perte moyenne par epoch."""
    if not exemples:
        raise ValueError("La liste d'exemples d'entraînement est vide.")
    if epochs < 1:
        raise ValueError("Le nombre d'epochs doit être supérieur ou égal à 1.")
    if taux_apprentissage <= 0:
        raise ValueError("Le taux d'apprentissage doit être positif.")

    pertes_epochs = []
    etat_adam = {"pas": 0, "moment1": {}, "moment2": {}}
    meilleure_perte_validation = float("inf")
    epochs_sans_amélioration = 0
    meilleurs_parametres = None

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
                etat_adam=etat_adam,
            )

        perte_moyenne = perte_totale / len(exemples)
        pertes_epochs.append(perte_moyenne)
        print(f"Epoch {numero_epoch + 1}/{epochs} - perte moyenne : {perte_moyenne:.4f}")

        if exemples_validation:
            perte_validation = sum(
                calculer_perte(
                    modele.forward(
                        exemple["contexte"],
                        mode_agregation="concat",
                    )["probabilites"],
                    exemple["cible"],
                )
                for exemple in exemples_validation
                if exemple["cible"] != modele.indice_unk
            ) / max(
                1,
                sum(
                    exemple["cible"] != modele.indice_unk
                    for exemple in exemples_validation
                ),
            )
            print(f"  perte validation : {perte_validation:.4f}")

            if perte_validation < meilleure_perte_validation:
                meilleure_perte_validation = perte_validation
                epochs_sans_amélioration = 0
                meilleurs_parametres = {
                    "poids_embeddings": modele.couche_embedding.poids.copy(),
                    "W_hidden": modele.W_hidden.copy(),
                    "b_hidden": modele.b_hidden.copy(),
                    "W_sortie": modele.W_sortie.copy(),
                    "b_sortie": modele.b_sortie.copy(),
                }
            else:
                epochs_sans_amélioration += 1
                if epochs_sans_amélioration >= patience:
                    print(
                        f"Arrêt anticipé : aucune amélioration pendant "
                        f"{patience} epochs."
                    )
                    if meilleurs_parametres is not None:
                        modele.couche_embedding.poids = meilleurs_parametres["poids_embeddings"]
                        modele.W_hidden = meilleurs_parametres["W_hidden"]
                        modele.b_hidden = meilleurs_parametres["b_hidden"]
                        modele.W_sortie = meilleurs_parametres["W_sortie"]
                        modele.b_sortie = meilleurs_parametres["b_sortie"]
                    break

    return pertes_epochs
