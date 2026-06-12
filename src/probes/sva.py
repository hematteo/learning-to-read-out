"""SVA Across-PP loader (Marks 2025, "Sparse feature circuits", arxiv:2403.19647).

Subject-verb agreement across a prepositional phrase. Each item is a (clean,
corrupted) pair that differ only in subject number. Both sentences are
truncated immediately before the verb; the task is to evaluate
logit(verb_singular) - logit(verb_plural) at the final position.

We bundle a templated 1,000-sample generator. Marks's published lists at
https://github.com/saprmarks/feature-circuits use a similar template family
(subject-noun + PP modifier + verb); when the published JSONs are not
available locally we synthesize on the fly with a fixed seed for
reproducibility. Document this fallback in any paper writeup.
"""

import random
from dataclasses import dataclass

SUBJECT_NOUNS_SG = [
    "boy",
    "girl",
    "man",
    "woman",
    "child",
    "doctor",
    "teacher",
    "student",
    "author",
    "artist",
    "scientist",
    "neighbor",
    "engineer",
    "lawyer",
    "farmer",
    "pilot",
    "driver",
    "musician",
    "writer",
    "officer",
    "guard",
    "manager",
    "dancer",
    "athlete",
    "chef",
    "actor",
    "actress",
    "judge",
    "minister",
    "captain",
]
SUBJECT_NOUNS_PL = [
    "boys",
    "girls",
    "men",
    "women",
    "children",
    "doctors",
    "teachers",
    "students",
    "authors",
    "artists",
    "scientists",
    "neighbors",
    "engineers",
    "lawyers",
    "farmers",
    "pilots",
    "drivers",
    "musicians",
    "writers",
    "officers",
    "guards",
    "managers",
    "dancers",
    "athletes",
    "chefs",
    "actors",
    "actresses",
    "judges",
    "ministers",
    "captains",
]
assert len(SUBJECT_NOUNS_SG) == len(SUBJECT_NOUNS_PL)

PP_PREP = ["near", "behind", "beside", "by", "next to", "in front of", "across from"]
PP_NOUN_PHRASES = [
    "the table",
    "the window",
    "the door",
    "the garden",
    "the river",
    "the school",
    "the office",
    "the street",
    "the building",
    "the kitchen",
    "the museum",
    "the library",
    "the bridge",
    "the entrance",
    "the fountain",
    "the children",
    "the parents",
    "the teachers",
    "the lawyers",
    "the actors",
    "the engineers",
    "the dancers",
    "the artists",
]
# Verbs in present tense, singular (verb_s) vs plural (verb_p).
VERB_PAIRS = [
    ("is", "are"),
    ("was", "were"),
    ("has", "have"),
    ("does", "do"),
    ("seems", "seem"),
    ("looks", "look"),
    ("appears", "appear"),
    ("knows", "know"),
    ("walks", "walk"),
    ("smiles", "smile"),
    ("laughs", "laugh"),
    ("speaks", "speak"),
]


@dataclass
class SvaItem:
    clean_text: str  # singular subject prompt; "correct" verb is singular form
    corrupted_text: str  # plural subject prompt; "correct" verb is plural form
    verb_sg: str  # e.g. " is" (with leading space — tokenizer convention)
    verb_pl: str  # e.g. " are"


def generate_sva_across_pp(n: int = 1000, seed: int = 0) -> list[SvaItem]:
    rng = random.Random(seed)
    items: list[SvaItem] = []
    while len(items) < n:
        i = rng.randrange(len(SUBJECT_NOUNS_SG))
        prep = rng.choice(PP_PREP)
        # Random PP noun (PP_NOUN_PHRASES mixes singular and plural); its
        # grammatical number is NOT matched against the subject. The "across-PP"
        # framing refers only to the prepositional-phrase position separating
        # subject and verb, not to a number-disagreement attractor.
        pp = rng.choice(PP_NOUN_PHRASES)
        verb_sg, verb_pl = rng.choice(VERB_PAIRS)
        sg_subj = SUBJECT_NOUNS_SG[i]
        pl_subj = SUBJECT_NOUNS_PL[i]
        # Prompt is truncated right before the verb, no period.
        clean = f"The {sg_subj} {prep} {pp}"
        corrupted = f"The {pl_subj} {prep} {pp}"
        items.append(
            SvaItem(
                clean_text=clean,
                corrupted_text=corrupted,
                verb_sg=" " + verb_sg,
                verb_pl=" " + verb_pl,
            )
        )
    return items
