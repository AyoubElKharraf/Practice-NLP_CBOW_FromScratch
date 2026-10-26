"""
Création des exemples d'entraînement CBOW.

Pour chaque mot d'une phrase, le mot devient la cible et les mots voisins
(avant et après) forment le contexte. La taille de la fenêtre avant et
après est entièrement paramétrable par l'utilisateur.
Les positions qui dépassent la phrase sont remplies avec <PAD> afin que
tous les contextes aient une taille constante.
"""

from vocabulary import TOKEN_MASK, TOKEN_PAD, tokens_vers_indices


def construire_contexte(
    tokens: list[str],
    position_cible: int,
    fenetre_avant: int = 2,
    fenetre_apres: int | None = None,
    taille_fenetre: int | None = None
) -> list[str]:
    """
    Retourne les mots autour de la cible :
    - fenetre_avant : nombre de mots à gauche (avant la cible)
    - fenetre_apres : nombre de mots à droite (après la cible)
    
    Si taille_fenetre est spécifié, il sert de valeur par défaut pour les deux côtés.
    """
    if taille_fenetre is not None:
        fenetre_avant = taille_fenetre
        fenetre_apres = taille_fenetre
    elif fenetre_apres is None:
        fenetre_apres = fenetre_avant

    contexte = []

    # 1. Mots avant la cible (de gauche à droite)
    for decalage in range(-fenetre_avant, 0):
        position = position_cible + decalage
        if 0 <= position < len(tokens):
            contexte.append(tokens[position])
        else:
            contexte.append(TOKEN_PAD)

    # 2. Mots après la cible (de la cible vers la droite)
    for decalage in range(1, fenetre_apres + 1):
        position = position_cible + decalage
        if 0 <= position < len(tokens):
            contexte.append(tokens[position])
        else:
            contexte.append(TOKEN_PAD)

    return contexte


def generer_exemples(
    corpus_tokenise: list[list[str]],
    fenetre_avant: int = 2,
    fenetre_apres: int | None = None,
    taille_fenetre: int | None = None
) -> list[dict[str, list[str] | str]]:
    """
    Crée un exemple pour chaque mot du corpus.

    Chaque exemple contient :
    - contexte : mots voisins (avant + après), de longueur fixe ;
    - cible : mot à prédire ;
    - phrase_masquee : phrase où la cible est remplacée par <MASK>.
    """
    if taille_fenetre is not None:
        fenetre_avant = taille_fenetre
        fenetre_apres = taille_fenetre
    elif fenetre_apres is None:
        fenetre_apres = fenetre_avant

    exemples = []

    for tokens in corpus_tokenise:
        for position_cible in range(len(tokens)):
            cible = tokens[position_cible]
            contexte = construire_contexte(
                tokens=tokens,
                position_cible=position_cible,
                fenetre_avant=fenetre_avant,
                fenetre_apres=fenetre_apres
            )

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


if __name__ == "__main__":
    phrase_test = ["le", "chat", "noir", "dort", "sur", "le", "tapis"]
    print("Phrase test :", phrase_test)

    # Test avec 3 mots avant et 1 mot après
    ctx = construire_contexte(phrase_test, position_cible=3, fenetre_avant=3, fenetre_apres=1)
    print("Cible : 'dort' (position 3)")
    print("Contexte avec 3 avant et 1 après :", ctx)
