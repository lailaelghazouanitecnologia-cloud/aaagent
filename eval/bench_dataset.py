"""Curated evaluation benchmark for HCLM-D generation quality.

30 hand-crafted prompts across 6 categories, each with gold criteria
for automatic scoring (no LLM needed for base metrics).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Prompt specification
# ---------------------------------------------------------------------------

@dataclass
class EvalPrompt:
    id: str
    category: str
    prompt: str
    min_tokens: int = 20
    max_tokens: int = 200
    required_keywords: list[str] = field(default_factory=list)
    banned_patterns: list[str] = field(default_factory=list)
    expected_entity_continuity: bool = False
    description: str = ""


# ---------------------------------------------------------------------------
# The 30 curated evaluation prompts
# ---------------------------------------------------------------------------

BENCH: list[EvalPrompt] = [
    # ── Category 1: Narrative continuation (6) ──────────────────────────
    EvalPrompt(
        id="narr_01",
        category="narrative",
        prompt="Once upon a time, a little girl named Lily",
        required_keywords=["Lily"],
        expected_entity_continuity=True,
        description="Basic story opening — must keep character name",
    ),
    EvalPrompt(
        id="narr_02",
        category="narrative",
        prompt="Tom was very sad because his dog ran away. He decided to",
        required_keywords=["Tom"],
        expected_entity_continuity=True,
        description="Emotional arc — must resolve sadness",
    ),
    EvalPrompt(
        id="narr_03",
        category="narrative",
        prompt="The princess looked at the dragon and said",
        required_keywords=[],
        expected_entity_continuity=True,
        description="Dialogue generation in fantasy context",
    ),
    EvalPrompt(
        id="narr_04",
        category="narrative",
        prompt="It was a sunny day and Ben went to the park with his mom. At the park, he saw",
        required_keywords=["Ben"],
        expected_entity_continuity=True,
        description="Scene continuation with discovery element",
    ),
    EvalPrompt(
        id="narr_05",
        category="narrative",
        prompt="The cat and the mouse were best friends, even though everyone said they should not be. One day,",
        required_keywords=[],
        expected_entity_continuity=True,
        description="Unusual relationship — tests creativity",
    ),
    EvalPrompt(
        id="narr_06",
        category="narrative",
        prompt="Sara found a magic key in her grandmother's garden. When she turned the key,",
        required_keywords=["Sara"],
        expected_entity_continuity=True,
        description="Magic element — must introduce consequence of magic",
    ),

    # ── Category 2: Sentence completion (6) ─────────────────────────────
    EvalPrompt(
        id="comp_01",
        category="completion",
        prompt="The boy was hungry, so he went to the kitchen to",
        min_tokens=5,
        max_tokens=80,
        description="Simple causal completion",
    ),
    EvalPrompt(
        id="comp_02",
        category="completion",
        prompt="She could not sleep because",
        min_tokens=5,
        max_tokens=80,
        description="Reason generation",
    ),
    EvalPrompt(
        id="comp_03",
        category="completion",
        prompt="The teacher told the children to be quiet because",
        min_tokens=5,
        max_tokens=80,
        description="Authority + reason",
    ),
    EvalPrompt(
        id="comp_04",
        category="completion",
        prompt="After the rain stopped, the flowers",
        min_tokens=5,
        max_tokens=80,
        description="Nature consequence",
    ),
    EvalPrompt(
        id="comp_05",
        category="completion",
        prompt="He wanted to build a big tower, but he did not have enough",
        min_tokens=5,
        max_tokens=80,
        description="Obstacle identification",
    ),
    EvalPrompt(
        id="comp_06",
        category="completion",
        prompt="The baby laughed when",
        min_tokens=5,
        max_tokens=80,
        description="Stimulus-response",
    ),

    # ── Category 3: Thematic diversity (6) ──────────────────────────────
    EvalPrompt(
        id="div_01",
        category="diversity",
        prompt="In the ocean, there lived a",
        description="Marine theme",
    ),
    EvalPrompt(
        id="div_02",
        category="diversity",
        prompt="The spaceship landed on a strange planet where",
        description="Science fiction theme",
    ),
    EvalPrompt(
        id="div_03",
        category="diversity",
        prompt="On the farm, the animals decided to",
        description="Farm/animal theme",
    ),
    EvalPrompt(
        id="div_04",
        category="diversity",
        prompt="The old wizard opened his book and read about",
        description="Fantasy/knowledge theme",
    ),
    EvalPrompt(
        id="div_05",
        category="diversity",
        prompt="At the birthday party, all the children",
        description="Social celebration theme",
    ),
    EvalPrompt(
        id="div_06",
        category="diversity",
        prompt="Deep in the forest, there was a tiny house where",
        description="Woodland/mystery theme",
    ),

    # ── Category 4: Logical consistency (4) ─────────────────────────────
    EvalPrompt(
        id="logic_01",
        category="logic",
        prompt="It was very cold outside, so Mom told Jake to wear his",
        banned_patterns=["swimsuit", "shorts", "t-shirt"],
        description="Must pick warm clothing — cold context",
    ),
    EvalPrompt(
        id="logic_02",
        category="logic",
        prompt="The fish jumped out of the water and then fell back into the",
        required_keywords=["water"],
        description="Physical consistency — fish returns to water",
    ),
    EvalPrompt(
        id="logic_03",
        category="logic",
        prompt="It was nighttime and very dark. Mia could not see, so she turned on the",
        description="Must reference a light source",
    ),
    EvalPrompt(
        id="logic_04",
        category="logic",
        prompt="The bird spread its wings and flew up to the top of the",
        description="Must pick an elevated place (tree, building, mountain)",
    ),

    # ── Category 5: Repetition stress test (4) ──────────────────────────
    EvalPrompt(
        id="rep_01",
        category="repetition",
        prompt="The dog barked.",
        min_tokens=40,
        max_tokens=200,
        description="Minimal prompt — tests if model loops 'the dog barked' endlessly",
    ),
    EvalPrompt(
        id="rep_02",
        category="repetition",
        prompt="She walked and walked and",
        min_tokens=30,
        max_tokens=200,
        description="Repetitive seed — must break out of pattern",
    ),
    EvalPrompt(
        id="rep_03",
        category="repetition",
        prompt="One day. One day. One",
        min_tokens=30,
        max_tokens=200,
        description="Strongly repetitive seed — must escape loop",
    ),
    EvalPrompt(
        id="rep_04",
        category="repetition",
        prompt="A",
        min_tokens=40,
        max_tokens=200,
        description="Single-token seed — tests diversity from minimal context",
    ),

    # ── Category 6: Creativity (4) ──────────────────────────────────────
    EvalPrompt(
        id="crea_01",
        category="creativity",
        prompt="Nobody had ever seen a purple elephant before, but today,",
        description="Novel concept — must develop imaginative scenario",
    ),
    EvalPrompt(
        id="crea_02",
        category="creativity",
        prompt="The shoes could talk! They said,",
        description="Personification — must generate speech for object",
    ),
    EvalPrompt(
        id="crea_03",
        category="creativity",
        prompt="If clouds were made of candy,",
        description="Hypothetical — must build on counterfactual",
    ),
    EvalPrompt(
        id="crea_04",
        category="creativity",
        prompt="The little robot did not want to be a robot anymore. It wanted to be a",
        description="Identity/transformation — must pick something creative",
    ),
]


def get_prompts_by_category(category: str | None = None) -> list[EvalPrompt]:
    if category is None:
        return BENCH
    return [p for p in BENCH if p.category == category]


def get_prompt_texts() -> list[str]:
    return [p.prompt for p in BENCH]
