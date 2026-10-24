"""
Module embedding.py
-------------------
Rôle : Gestion de la couche de plongement lexical (Word Embeddings).
Étape 3 (Partie 1) du projet MiniBERT-CBOW-FromScratch.

Ce module gère :
1. L'initialisation de la matrice des embeddings E de taille (V x d).
2. L'extraction du vecteur associé à un token ou à une liste de tokens.
3. L'agrégation des embeddings du contexte (moyenne ou somme) pour former
   un vecteur unique représentant le contexte d'un mot.
"""

import numpy as np


class CoucheEmbedding:
    """
    Couche d'embeddings implémentée manuellement avec NumPy.
    
    Paramètres :
        taille_vocabulaire (int) : Nombre de mots dans le vocabulaire (V).
        dimension_embedding (int) : Dimension des vecteurs d'embeddings (d).
        indice_pad (int) : Indice du token <PAD> (par défaut 0).
    """

    def __init__(
        self,
        taille_vocabulaire: int,
        dimension_embedding: int = 64,
        indice_pad: int = 0
    ):
        self.V = taille_vocabulaire
        self.d = dimension_embedding
        self.indice_pad = indice_pad

        # Initialisation uniforme de Xavier / Glorot :
        # Les poids sont initialisés dans l'intervalle [-limite, +limite]
        # avec limite = 1 / sqrt(d), ce qui assure une variance stable.
        limite = 1.0 / np.sqrt(self.d)
        self.poids = np.random.uniform(-limite, limite, (self.V, self.d))

        # Le vecteur pour le token de padding <PAD> est initialisé à 0
        self.poids[self.indice_pad] = np.zeros(self.d)

    def obtenir_vecteur(self, indice: int) -> np.ndarray:
        """
        Retourne le vecteur d'embedding correspondant à un indice donné.
        Forme de sortie : (d,)
        """
        if not (0 <= indice < self.V):
            raise IndexError(f"Indice {indice} hors du vocabulaire (taille={self.V})")
        return self.poids[indice]

    def obtenir_vecteurs(self, indices: list[int]) -> np.ndarray:
        """
        Retourne la matrice des vecteurs pour une liste d'indices.
        Forme de sortie : (len(indices), d)
        """
        return self.poids[indices]

    def agreger_contexte(
        self,
        indices_contexte: list[int],
        mode: str = "moyenne",
        ignorer_pad: bool = True
    ) -> np.ndarray:
        """
        Agrège les embeddings des mots du contexte pour produire un vecteur unique h de taille (d,).
        
        Paramètres :
            indices_contexte (list[int]) : Indices des mots du contexte.
            mode (str) : 'moyenne' (recommandé pour CBOW) ou 'somme'.
            ignorer_pad (bool) : Si True, ne prend pas en compte le token <PAD>
                                 dans le calcul de la moyenne pour ne pas diluer le sens.
        
        Retourne :
            vecteur_contexte (np.ndarray) : Vecteur agrégé de dimension (d,).
        """
        if ignorer_pad:
            # On conserve uniquement les tokens utiles (non PAD)
            indices_actifs = [idx for idx in indices_contexte if idx != self.indice_pad]
        else:
            indices_actifs = indices_contexte

        # Si tous les mots étaient du PAD (ex. phrase vide), on retourne un vecteur nul
        if not indices_actifs:
            return np.zeros(self.d)

        # Récupération des vecteurs des mots du contexte (taille : K x d)
        vecteurs = self.obtenir_vecteurs(indices_actifs)

        if mode == "moyenne":
            # Moyenne le long de l'axe des mots (axe 0) -> résultat de taille (d,)
            vecteur_contexte = np.mean(vecteurs, axis=0)
        elif mode == "somme":
            vecteur_contexte = np.sum(vecteurs, axis=0)
        else:
            raise ValueError(f"Mode d'agrégation inconnu : '{mode}'. Choisir 'moyenne' ou 'somme'.")

        return vecteur_contexte


if __name__ == "__main__":
    print("--- Test unitaire de la CoucheEmbedding ---")
    V_test = 10
    d_test = 4
    couche_emb = CoucheEmbedding(taille_vocabulaire=V_test, dimension_embedding=d_test)

    print(f"Forme de la matrice d'embeddings E : {couche_emb.poids.shape} (V={V_test}, d={d_test})")
    
    # Test d'extraction
    vecteur_mot_3 = couche_emb.obtenir_vecteur(3)
    print("Vecteur pour le mot indice 3 :", np.round(vecteur_mot_3, 3))

    # Test d'agrégation d'un contexte avec padding [0, 0, 3, 5]
    contexte_indices = [0, 0, 3, 5]
    contexte_agrégé = couche_emb.agreger_contexte(contexte_indices, mode="moyenne")
    print(f"Vecteur agrégé (moyenne sans PAD) pour contexte {contexte_indices} :")
    print(" ", np.round(contexte_agrégé, 3))

