"""
Création des exemples d'entraînement CBOW.

Pour chaque mot d'une phrase, le mot devient la cible et les mots voisins
forment le contexte. Les positions qui dépassent la phrase sont remplies
avec <PAD> afin que tous les contextes aient la même taille.
"""

from vocabulary import TOKEN_MASK, TOKEN_PAD, tokens_vers_indices


def construire_contexte(
    tokens: list[str], position_cible: int, taille_fenetre: int = 2
) -> list[str]:
    """Retourne les mots autour de la cible, de gauche à droite."""
    contexte = []

    for decalage in range(-taille_fenetre, taille_fenetre + 1):
        if decalage == 0:
            continue

        position = position_cible + decalage
        if 0 <= position < len(tokens):
            contexte.append(tokens[position])
        else:
            contexte.append(TOKEN_PAD)

    return contexte


def generer_exemples(
    corpus_tokenise: list[list[str]], taille_fenetre: int = 2
) -> list[dict[str, list[str] | str]]:
    """
    Crée un exemple pour chaque mot du corpus.

    Chaque exemple contient :
    - contexte : mots voisins, avec une longueur fixe ;
    - cible : mot à prédire ;
    - phrase_masquee : phrase où la cible est remplacée par <MASK>.
    """
    exemples = []

    for tokens in corpus_tokenise:
        for position_cible in range(len(tokens)):
            cible = tokens[position_cible]
            contexte = construire_contexte(tokens, position_cible, taille_fenetre)

            phrase_masquee = list(tokens)
            phrase_masquee[position_cible] = TOKEN_MASK

            exemples.append(
                {
                    "contexte": contexte,
                    "cible": cible,
                    "phrase_masquee": phrase_masquee,
                }
            )

    return exemples


def convertir_exemples_en_indices(
    exemples: list[dict[str, list[str] | str]],
    mot_vers_indice: dict[str, int],
) -> list[dict[str, list[int] | int]]:
    """Convertit les mots des exemples en indices du vocabulaire."""
    exemples_indices = []

    for exemple in exemples:
        contexte = exemple["contexte"]
        cible = exemple["cible"]
        phrase_masquee = exemple["phrase_masquee"]

        exemples_indices.append(
            {
                "contexte": tokens_vers_indices(contexte, mot_vers_indice),
                "cible": mot_vers_indice[cible],
                "phrase_masquee": tokens_vers_indices(
                    phrase_masquee, mot_vers_indice
                ),
            }
        )

    return exemples_indices
