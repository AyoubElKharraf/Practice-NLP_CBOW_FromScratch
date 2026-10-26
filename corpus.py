"""
Module corpus.py
----------------
Rôle : Téléchargement et mise à disposition du corpus de texte brut en français.
Fournit le dataset Gutenberg (500 phrases complètes découpées par ponctuation)
ainsi qu'un mini-corpus d'exemple simple.
"""

import os
import re
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


def decouper_en_phrases(texte_brut: str, n_phrases: int = 500) -> list[str]:
    """
    Découpe un texte brut en véritables phrases grammaticales complètes.
    
    1. Élimine l'en-tête Gutenberg en anglais pour démarrer au début du roman en français.
    2. Remplace les retours à la ligne (\n) et tabulations par des espaces pour reconstituer
       les phrases réparties sur plusieurs lignes.
    3. Découpe selon les signes de ponctuation forte de fin de phrase : '.', '!' et '?'.
    4. Filtre les fragments trop courts (< 15 caractères) pour ne garder que des phrases significatives.
    """
    texte = texte_brut

    # 1. Élimination des métadonnées anglaises de Gutenberg si présentes
    if "*** START OF THE PROJECT" in texte:
        parties = texte.split("*** START OF THE PROJECT", 1)[1]
        # Début effectif du roman de Jules Verne
        if "Le 24 mai 1863" in parties:
            texte = "Le 24 mai 1863" + parties.split("Le 24 mai 1863", 1)[1]
        else:
            texte = parties

    if "*** END OF THE PROJECT" in texte:
        texte = texte.split("*** END OF THE PROJECT", 1)[0]

    # 2. Reconstitution du texte continu : remplacement des sauts de ligne par un espace
    texte_continu = re.sub(r"\s+", " ", texte)

    # 3. Découpage sur la ponctuation de fin de phrase (. ! ?)
    phrases_brutes = re.split(r"[.!?]+", texte_continu)

    # 4. Nettoyage des espaces et filtrage des phrases courtes
    phrases_valides = [
        p.strip() for p in phrases_brutes if len(p.strip()) > 15
    ][:n_phrases]

    return phrases_valides


def charger_corpus_gutenberg(n_phrases: int = 500, fichier_cache: str = "corpus_gutenberg.txt") -> list[str]:
    """
    Télécharge ou charge depuis le cache le texte du Projet Gutenberg
    (roman en français de Jules Verne) et le découpe en phrases complètes.
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
            return MINI_CORPUS_TEST.copy()

    return decouper_en_phrases(texte_brut, n_phrases=n_phrases)


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
    print("\n[Aperçu des 3 premières phrases] :")
    for i, p in enumerate(phrases[:3], start=1):
        print(f"  {i}. {p}")
