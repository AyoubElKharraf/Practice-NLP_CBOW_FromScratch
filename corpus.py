"""
Module corpus.py
----------------
Rôle : Téléchargement et mise à disposition du corpus de texte brut en français.
Fournit le dataset Gutenberg (500 phrases) ainsi qu'un mini-corpus d'exemple simple.
"""

import os
import urllib.request

# Mini-corpus statique de secours (utile pour les tests rapides et le débogage)
MINI_CORPUS_TEST = [
    "Le chat noir dort sur le canapé dans le salon.",
    "Le chien fidèle garde la maison et aboie joyeusement.",
    "Un petit oiseau chante une belle mélodie dans l'arbre.",
    "L'enfant lit un livre passionnant à la bibliothèque.",
    "Le professeur explique la leçon de français aux étudiants.",
    "La lune éclaire la forêt sombre pendant la nuit.",
    "Le boulanger prépare du bon pain chaud chaque matin.",
    "Les étoiles brillent dans le ciel calme et dégagé."
]

def charger_corpus_gutenberg(n_phrases: int = 500, fichier_cache: str = "corpus_gutenberg.txt") -> list[str]:
    """
    Télécharge ou charge depuis le cache le texte du Projet Gutenberg
    (roman en français) et le découpe en phrases simples.
    """
    texte_brut = ""
    
    # Vérification du cache local pour éviter de télécharger à chaque exécution
    if os.path.exists(fichier_cache):
        with open(fichier_cache, "r", encoding="utf-8") as f:
            texte_brut = f.read()
    else:
        url = "https://www.gutenberg.org/cache/epub/4791/pg4791.txt"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                texte_brut = response.read().decode("utf-8")
            # Sauvegarde dans le cache local
            with open(fichier_cache, "w", encoding="utf-8") as f:
                f.write(texte_brut)
        except Exception as e:
            print(f"[Attention] Impossible de télécharger depuis Gutenberg ({e}). Utilisation du mini-corpus de test.")
            return MINI_CORPUS_TEST

    # Découpage en phrases / lignes non vides (> 10 caractères)
    phrases = [
        p.strip() for p in texte_brut.split("\n") if len(p.strip()) > 10
    ][:n_phrases]

    return phrases

def obtenir_corpus(source: str = "gutenberg", n_phrases: int = 500) -> list[str]:
    """
    Retourne le corpus sélectionné ('gutenberg' ou 'mini').
    """
    if source == "mini":
        return MINI_CORPUS_TEST.copy()
    return charger_corpus_gutenberg(n_phrases=n_phrases)

# Variable exportée par défaut contenant le corpus
phrases = obtenir_corpus(source="gutenberg", n_phrases=500)

if __name__ == "__main__":
    print(f"Nombre de phrases chargées : {len(phrases)}")
    print("Exemple de phrase :", phrases[0] if phrases else "Aucune")

