"""
Module model.py
---------------
Rôle : Architecture du modèle neuronal contextuel (MiniBERT-CBOW).
Étape 3 (Partie 2) du projet MiniBERT-CBOW-FromScratch.

Ce module assemble :
1. La couche d'embedding (importée depuis embedding.py).
2. La couche linéaire de projection vers le vocabulaire (poids W_sortie et biais).
3. La fonction d'activation Softmax codée manuellement (avec stabilité numérique).
4. La passe avant (Forward pass) produisant la distribution de probabilités sur les mots cibles.
"""

import numpy as np
from embedding import CoucheEmbedding


class ModeleCBOW:
    """
    Architecture du modèle CBOW / MiniBERT codée from scratch avec NumPy.
    
    Structure :
        Indices contexte (K mots)
                 │
                 ▼
        Couche Embedding (E : V x d)
                 │
                 ▼
        Agrégation (Moyenne) ──> Vecteur de contexte h (taille d)
                 │
                 ▼
        Couche Linéaire de Sortie (W_sortie : d x V, b_sortie : V)
                 │
                 ▼
        Scores / Logits z = h @ W_sortie + b_sortie (taille V)
                 │
                 ▼
        Activation Softmax ──> Probabilités p (taille V)
    """

    def __init__(
        self,
        taille_vocabulaire: int,
        dimension_embedding: int = 32,
        indice_pad: int = 0,
        taille_contexte: int = 4,
        dimension_cachee: int = 128,
        taux_dropout: float = 0.3,
        indice_unk: int = 1,
    ):
        self.V = taille_vocabulaire
        self.d = dimension_embedding
        self.taille_contexte = taille_contexte
        self.dimension_cachee = dimension_cachee
        self.taux_dropout = taux_dropout
        self.indice_unk = indice_unk
        self.dimension_concatenee = taille_contexte * dimension_embedding

        # 1. Couche d'embedding
        self.couche_embedding = CoucheEmbedding(
            taille_vocabulaire=self.V,
            dimension_embedding=self.d,
            indice_pad=indice_pad
        )

        # 2. Couche cachée ReLU : (K × d) → dimension_cachee
        limite_hidden = 1.0 / np.sqrt(self.dimension_concatenee)
        self.W_hidden = np.random.uniform(
            -limite_hidden,
            limite_hidden,
            (self.dimension_concatenee, self.dimension_cachee),
        )
        self.b_hidden = np.zeros(self.dimension_cachee)

        # 3. Couche de projection vers le vocabulaire
        limite_sortie = 1.0 / np.sqrt(self.dimension_cachee)
        self.W_sortie = np.random.uniform(
            -limite_sortie,
            limite_sortie,
            (self.dimension_cachee, self.V),
        )

        # Biais de sortie (taille V), initialisé à zéro
        self.b_sortie = np.zeros(self.V)

    @staticmethod
    def softmax(scores: np.ndarray) -> np.ndarray:
        """
        Calcule la fonction Softmax de manière numériquement stable.
        
        Pour éviter les dépassements de capacité (overflow avec exp),
        on soustrait le maximum du vecteur avant d'appliquer l'exponentielle :
            softmax(z)_i = exp(z_i - max(z)) / sum(exp(z_j - max(z)))
        """
        # Soustraction du max pour stabilité numérique
        scores_stables = scores - np.max(scores)
        exponentielles = np.exp(scores_stables)
        probabilites = exponentielles / np.sum(exponentielles)
        return probabilites

    def forward(
        self,
        indices_contexte: list[int],
        mode_agregation: str = "concat",
        entrainement: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Effectue la passe avant (Forward Pass) pour un exemple de contexte.
        
        Paramètres :
            indices_contexte (list[int]) : Liste des indices des mots voisins.
            mode_agregation (str) : 'moyenne' ou 'somme'.
        
        Retourne :
            dict contenant :
                - 'vecteur_contexte' : h (forme d,)
                - 'scores'           : z (forme V,)
                - 'probabilites'     : p (forme V,)
        """
        if len(indices_contexte) != self.taille_contexte:
            raise ValueError(
                f"La taille du contexte doit être {self.taille_contexte}, "
                f"mais {len(indices_contexte)} indices ont été fournis."
            )

        # 1. Concaténation des embeddings dans l'ordre du contexte
        h = self.couche_embedding.agreger_contexte(
            indices_contexte,
            mode=mode_agregation
        )

        if h.shape[0] != self.dimension_concatenee:
            raise ValueError("La dimension concaténée du contexte ne correspond pas au modèle.")

        # 2. Couche cachée ReLU
        scores_hidden = h @ self.W_hidden + self.b_hidden
        activations_hidden = np.maximum(0.0, scores_hidden)

        # 3. Dropout pendant l'entraînement uniquement
        masque_dropout = None
        activations_reglees = activations_hidden
        if entrainement and self.taux_dropout > 0:
            masque_dropout = (
                np.random.rand(self.dimension_cachee) >= self.taux_dropout
            ).astype(np.float64)
            activations_reglees = (
                activations_hidden * masque_dropout
            ) / (1.0 - self.taux_dropout)

        # 4. Projection vers le vocabulaire puis Softmax
        scores = np.dot(activations_reglees, self.W_sortie) + self.b_sortie
        probabilites = self.softmax(scores)

        return {
            "vecteur_contexte": h,
            "scores_hidden": scores_hidden,
            "activations_hidden": activations_hidden,
            "masque_dropout": masque_dropout,
            "activations_reglees": activations_reglees,
            "scores": scores,
            "probabilites": probabilites
        }

    def predire_top_k(
        self,
        indices_contexte: list[int],
        k: int = 5
    ) -> list[tuple[int, float]]:
        """
        Retourne les k indices les plus probables avec leurs probabilités associées.
        """
        sortie = self.forward(indices_contexte)
        probabilites = sortie["probabilites"]

        # Indices triés par probabilité décroissante
        indices_tries = np.argsort(probabilites)[::-1][:k]

        return [(int(idx), float(probabilites[idx])) for idx in indices_tries]


if __name__ == "__main__":
    print("--- Test unitaire de l'architecture du modèle (ModeleCBOW) ---")
    V_test = 20
    d_test = 8
    modele = ModeleCBOW(taille_vocabulaire=V_test, dimension_embedding=d_test)

    print(f"Modèle initialisé avec V={V_test} mots et d={d_test} dimensions.")
    print(f"Forme de E (Embeddings)  : {modele.couche_embedding.poids.shape}")
    print(f"Forme de W_sortie         : {modele.W_sortie.shape}")
    print(f"Forme de b_sortie         : {modele.b_sortie.shape}")

    # Simulation d'un contexte de 4 mots
    contexte_test = [0, 5, 12, 18]
    resultats = modele.forward(contexte_test)

    h = resultats["vecteur_contexte"]
    z = resultats["scores"]
    p = resultats["probabilites"]

    print("\n--- Résultats de la passe avant (Forward Pass) ---")
    print(f"Vecteur contexte h (taille {len(h)}) :", np.round(h[:4], 3), "...")
    print(f"Scores z (taille {len(z)})           :", np.round(z[:4], 3), "...")
    print(f"Probabilités p (somme={np.sum(p):.4f}) :", np.round(p[:4], 3), "...")

    top3 = modele.predire_top_k(contexte_test, k=3)
    print("\nTop 3 des prédictions (indice, probabilité) :", top3)

