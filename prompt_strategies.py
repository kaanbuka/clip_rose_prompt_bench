"""Prompt strategies for zero-shot CLIP rose classification.

Each strategy returns a dict mapping rose class key -> list of prompt strings.
The CLIP text encoder embeds each prompt; embeddings are averaged per class
(and across strategies for the ensemble).
"""

def simple_prompts() -> dict[str, list[str]]:
    """Strategy 1 - minimal, direct label prompts."""
    return {
        "rosa_canina": [
            "a photo of a rosa canina",
            "a photo of a dog rose",
            "a photo of a kuşburnu",
        ],
        "rosa_damascena": [
            "a photo of a rosa damascena",
            "a photo of a damask rose",
            "a photo of a damask rose flower",
        ],
        "rosa_multiflora": [
            "a photo of a rosa multiflora",
            "a photo of a multiflora rose",
            "a photo of a japanese rose",
        ],
    }


def descriptive_prompts() -> dict[str, list[str]]:
    """Strategy 2 - visual attribute prompts (color, petal shape, structure)."""
    return {
        "rosa_canina": [
            "a pale pink wild rose with five petals and yellow stamens",
            "a delicate light pink five-petaled rose with a simple open bloom",
            "a wild rose with oval pink petals and prominent yellow center",
        ],
        "rosa_damascena": [
            "a richly fragrant deep pink rose with densely packed petals",
            "a full double rose with many layers of deep pink petals",
            "a lush rose bloom with velvety deep pink petals densely arranged",
        ],
        "rosa_multiflora": [
            "a small white or pale pink rose growing in large clusters",
            "tiny white five-petaled roses in dense arching clusters",
            "a cluster of many small white roses on a long arching stem",
        ],
    }


def scientific_prompts() -> dict[str, list[str]]:
    """Strategy 3 - taxonomy and botanical language."""
    return {
        "rosa_canina": [
            "rosa canina, a deciduous shrub of the Rosaceae family with simple pink flowers",
            "a botanical specimen of rosa canina showing its characteristic pink petals and hip fruit",
            "the species rosa canina, known as dog rose, native to Europe and Asia",
        ],
        "rosa_damascena": [
            "rosa damascena, an ornamental rose cultivar with highly fragrant multipetaled blooms",
            "a botanical photograph of rosa damascena, the damask rose, used in perfumery",
            "rosa damascena mill., a hybrid rose species with dense petal arrangement",
        ],
        "rosa_multiflora": [
            "rosa multiflora, an invasive shrub rose producing corymbs of small white flowers",
            "a botanical image of rosa multiflora showing its characteristic flower clusters",
            "rosa multiflora thunb., the multiflora rose, with many small flowers per stem",
        ],
    }


def contextual_prompts() -> dict[str, list[str]]:
    """Strategy 4 - scene, habitat, and usage context hints."""
    return {
        "rosa_canina": [
            "a wild rose growing in a hedgerow or woodland edge",
            "a dog rose blooming in the countryside, its hips will become red berries in autumn",
            "a rose flower photographed in a natural meadow or forest path",
        ],
        "rosa_damascena": [
            "a damask rose cultivated in a garden or rose farm, used for rose oil extraction",
            "a rose flower in a formal garden, traditionally used in Middle Eastern perfumes",
            "a cultivated rose bloom often seen in Bulgarian or Turkish rose fields",
        ],
        "rosa_multiflora": [
            "a rose with many small flowers used as a rootstock in horticulture",
            "an invasive rose species forming dense thickets along roadsides and fields",
            "a rose shrub with sprays of tiny flowers commonly found along fences",
        ],
    }


def all_strategies() -> dict[str, dict[str, list[str]]]:
    """Return all named strategies (the ensemble is built at runtime)."""
    return {
        "simple": simple_prompts(),
        "descriptive": descriptive_prompts(),
        "scientific": scientific_prompts(),
        "contextual": contextual_prompts(),
    }


STRATEGY_DESCRIPTIONS = {
    "simple": "Minimal label prompts",
    "descriptive": "Visual attribute prompts",
    "scientific": "Botanical / taxonomic prompts",
    "contextual": "Scene & habitat prompts",
    "ensemble": "Average embedding of all strategies",
}
