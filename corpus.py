"""
Module corpus.py
----------------
Rôle : Téléchargement et mise à disposition du corpus de texte brut en français.
Fournit :
1. Le dataset Gutenberg sous forme de phrases complètes (pour l'entraînement CBOW/MLM).
2. Le roman structuré en paragraphes cohérents et lisibles (fonction obtenir_paragraphes).
3. Un mini-corpus d'exemple simple pour tests rapides.
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


def extraire_texte_roman(texte_brut: str) -> str:
    """
    Extrait uniquement le corps du roman en français en retirant
    les en-têtes et pieds de page de licence du Projet Gutenberg.
    """
    texte = texte_brut
    if "*** START OF THE PROJECT" in texte:
        parties = texte.split("*** START OF THE PROJECT", 1)[1]
        if "Le 24 mai 1863" in parties:
            texte = "Le 24 mai 1863" + parties.split("Le 24 mai 1863", 1)[1]
        else:
            texte = parties

    if "*** END OF THE PROJECT" in texte:
        texte = texte.split("*** END OF THE PROJECT", 1)[0]

    return texte


def decouper_en_paragraphes(texte_brut: str, n_paragraphes: int | None = None) -> list[str]:
    """
    Extrait et regroupe le roman en paragraphes cohérents et naturels.
    
    1. Élimine les métadonnées Gutenberg.
    2. Détecte les sauts de paragraphe (séparés par au moins 3 sauts de ligne consécutifs).
    3. Reconstitue chaque paragraphe en remplaçant les retours à la ligne internes par des espaces.
    4. Filtre les numéros de chapitres isolés (ex: 'I', 'II', etc.) et les lignes trop courtes.
    """
    roman = extraire_texte_roman(texte_brut)

    # Séparation par 3 sauts de ligne ou plus
    blocs = re.split(r"(?:\r?\n\s*){3,}", roman)

    paragraphes = []
    for b in blocs:
        p = re.sub(r"\s+", " ", b).strip()
        # On filtre les numéros de chapitre romains et les fragments très courts
        if len(p) > 20 and not re.match(r"^[IVXLCDM]+\s*$", p):
            paragraphes.append(p)

    if n_paragraphes is not None:
        return paragraphes[:n_paragraphes]
    return paragraphes


def decouper_en_phrases(texte_brut: str, n_phrases: int = 500) -> list[str]:
    """
    Découpe le texte brut en véritables phrases grammaticales complètes.
    
    1. Élimine l'en-tête Gutenberg pour commencer au début du roman de Jules Verne.
    2. Remplace les retours à la ligne par des espaces.
    3. Découpe selon les signes de ponctuation forte de fin de phrase : '.', '!' et '?'.
    4. Filtre les phrases trop courtes (< 15 caractères).
    """
    roman = extraire_texte_roman(texte_brut)

    # Reconstitution du texte continu
    texte_continu = re.sub(r"\s+", " ", roman)

    # Découpage sur la ponctuation (. ! ?)
    phrases_brutes = re.split(r"[.!?]+", texte_continu)

    # Filtrage et nettoyage
    phrases_valides = [
        p.strip() for p in phrases_brutes if len(p.strip()) > 15
    ][:n_phrases]

    return phrases_valides


def charger_texte_brut(fichier_cache: str = "corpus_gutenberg.txt") -> str:
    """Charge le texte depuis le cache local ou le télécharge si absent."""
    if os.path.exists(fichier_cache):
        with open(fichier_cache, "r", encoding="utf-8") as f:
            return f.read()

    url = "https://www.gutenberg.org/cache/epub/4791/pg4791.txt"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            texte_brut = response.read().decode("utf-8")
        with open(fichier_cache, "w", encoding="utf-8") as f:
            f.write(texte_brut)
        return texte_brut
    except Exception as e:
        print(f"[Attention] Téléchargement impossible ({e}). Utilisation du mini-corpus de test.")
        return "\n\n\n".join(MINI_CORPUS_TEST)


def charger_corpus_gutenberg(n_phrases: int = 500, fichier_cache: str = "corpus_gutenberg.txt") -> list[str]:
    """Retourne les phrases découpées pour l'entraînement CBOW/MLM."""
    texte_brut = charger_texte_brut(fichier_cache=fichier_cache)
    return decouper_en_phrases(texte_brut, n_phrases=n_phrases)


def obtenir_corpus(source: str = "gutenberg", n_phrases: int = 500) -> list[str]:
    """Retourne les phrases du corpus sélectionné ('gutenberg' ou 'mini')."""
    if source == "mini":
        return MINI_CORPUS_TEST.copy()
    return charger_corpus_gutenberg(n_phrases=n_phrases)


def obtenir_paragraphes(source: str = "gutenberg", n_paragraphes: int | None = None) -> list[str]:
    """Retourne les paragraphes complets et lisibles du corpus."""
    if source == "mini":
        return MINI_CORPUS_TEST.copy()
    texte_brut = charger_texte_brut()
    return decouper_en_paragraphes(texte_brut, n_paragraphes=n_paragraphes)


# Variable exportée par défaut contenant les phrases pour le modèle CBOW
phrases = obtenir_corpus(source="gutenberg", n_phrases=500)


if __name__ == "__main__":
    paragraphes = obtenir_paragraphes()
    print(f"Nombre total de paragraphes extraits : {len(paragraphes)}")
    print(f"Nombre de phrases pour le modèle : {len(phrases)}")

    print("\n--- Exemple du premier paragraphe complet ---")
    print(paragraphes[0])

    print("\n--- Exemple du deuxième paragraphe complet ---")
    print(paragraphes[1])
