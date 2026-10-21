"""
Module tokenizer.py
-------------------
Rôle : Nettoyage et tokenisation du texte brut en français.
- Met en minuscules.
- Supprime la ponctuation de base et les caractères spéciaux tout en préservant les caractères accentués de la langue française.
- Découpe en une liste de tokens (mots).
- Applique ce traitement à l'ensemble du corpus.
"""

import re

def nettoyer_et_tokeniser(phrase: str) -> list[str]:
    """
    Nettoie une phrase individuelle et la découpe en tokens.
    
    1. Mise en minuscules.
    2. Remplacement de la ponctuation et caractères spéciaux par des espaces (en conservant l'alphabet latin et les lettres accentuées françaises).
    3. Découpage en mots (split) et filtrage des tokens vides.
    """
    # 1. Minuscules
    texte = phrase.lower()

    # 2. Remplacement de la ponctuation courante et symboles par des espaces
    # On préserve : a-z, chiffres 0-9 et caractères accentués (à, â, ä, é, è, ê, ë, î, ï, ô, ö, ù, û, ü, ç)
    texte = re.sub(r"[^a-z0-9àâäéèêëîïôöùûüç]+", " ", texte)

    # 3. Découpage par espace
    tokens = texte.split()

    return tokens

def tokeniser_corpus(corpus: list[str]) -> list[list[str]]:
    """
    Applique la tokenisation à une liste de phrases (tout le corpus).
    Retourne une liste de phrases tokenisées (chaque phrase étant une liste de tokens).
    Filtre les phrases qui se retrouvent vides après nettoyage.
    """
    corpus_tokenise = []
    for phrase in corpus:
        tokens = nettoyer_et_tokeniser(phrase)
        if tokens:  # On ne garde que les phrases non vides
            corpus_tokenise.append(tokens)
    return corpus_tokenise

if __name__ == "__main__":
    phrase_test = "L'enfant mange une pomme rouge... C'est délicieux !"
    tokens = nettoyer_et_tokeniser(phrase_test)
    print("Phrase test :", phrase_test)
    print("Tokens :", tokens)

