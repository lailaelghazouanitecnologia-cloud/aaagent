"""Centralized special token registry for Za v7.

All token IDs defined here. No magic numbers elsewhere.
100 special tokens: IDs 0-99 reserved, BPE tokens start at 100.

Layout:
  0-4:    Core tokens (MASK, PAD, BOS, EOS, UNK)
  5-14:   Structure delimiters (SLOT, BLOCK, PLAN, RANK)
  15-24:  Language / domain tags
  25-39:  VM opcodes
  40-99:  Reserved for future use
"""

from __future__ import annotations


# ─── Core tokens (0-4) ───────────────────────────────────────────────
MASK_TOKEN = "[MASK]"
PAD_TOKEN = "[PAD]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
UNK_TOKEN = "[UNK]"

MASK_TOKEN_ID = 0
PAD_TOKEN_ID = 1
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3
UNK_TOKEN_ID = 4

# ─── Structure delimiters (5-14) ─────────────────────────────────────
SLOT_START_TOKEN = "[SLOT_START]"
SLOT_END_TOKEN = "[SLOT_END]"
BLOCK_START_TOKEN = "[BLOCK_START]"
BLOCK_END_TOKEN = "[BLOCK_END]"
PLAN_START_TOKEN = "[PLAN_START]"
PLAN_END_TOKEN = "[PLAN_END]"
RANK_QUERY_TOKEN = "[RANK_QUERY]"
RANK_RESULT_TOKEN = "[RANK_RESULT]"
SEP_TOKEN = "[SEP]"
NEWBLOCK_TOKEN = "[NEWBLOCK]"

SLOT_START_ID = 5
SLOT_END_ID = 6
BLOCK_START_ID = 7
BLOCK_END_ID = 8
PLAN_START_ID = 9
PLAN_END_ID = 10
RANK_QUERY_ID = 11
RANK_RESULT_ID = 12
SEP_ID = 13
NEWBLOCK_ID = 14

# ─── Language / domain tags (15-24) ──────────────────────────────────
LANG_EN_TOKEN = "[LANG_EN]"
LANG_ES_TOKEN = "[LANG_ES]"
LANG_PY_TOKEN = "[LANG_PY]"
LANG_FR_TOKEN = "[LANG_FR]"
LANG_DE_TOKEN = "[LANG_DE]"
DOMAIN_STORY_TOKEN = "[DOMAIN_STORY]"
DOMAIN_WIKI_TOKEN = "[DOMAIN_WIKI]"
DOMAIN_QA_TOKEN = "[DOMAIN_QA]"
DOMAIN_CODE_TOKEN = "[DOMAIN_CODE]"
DOMAIN_MATH_TOKEN = "[DOMAIN_MATH]"

LANG_EN_ID = 15
LANG_ES_ID = 16
LANG_PY_ID = 17
LANG_FR_ID = 18
LANG_DE_ID = 19
DOMAIN_STORY_ID = 20
DOMAIN_WIKI_ID = 21
DOMAIN_QA_ID = 22
DOMAIN_CODE_ID = 23
DOMAIN_MATH_ID = 24

# ─── VM opcodes (25-39) ──────────────────────────────────────────────
VM_EXEC_TOKEN = "[VM_EXEC]"
VM_HALT_TOKEN = "[VM_HALT]"
VM_PUSH_TOKEN = "[VM_PUSH]"
VM_POP_TOKEN = "[VM_POP]"
VM_CALL_TOKEN = "[VM_CALL]"
VM_RET_TOKEN = "[VM_RET]"
VM_LOAD_TOKEN = "[VM_LOAD]"
VM_STORE_TOKEN = "[VM_STORE]"
VM_CMP_TOKEN = "[VM_CMP]"
VM_JMP_TOKEN = "[VM_JMP]"
VM_LOOP_TOKEN = "[VM_LOOP]"
VM_EMIT_TOKEN = "[VM_EMIT]"
VM_SEARCH_TOKEN = "[VM_SEARCH]"
VM_RANK_TOKEN = "[VM_RANK]"
VM_COMPOSE_TOKEN = "[VM_COMPOSE]"

VM_EXEC_ID = 25
VM_HALT_ID = 26
VM_PUSH_ID = 27
VM_POP_ID = 28
VM_CALL_ID = 29
VM_RET_ID = 30
VM_LOAD_ID = 31
VM_STORE_ID = 32
VM_CMP_ID = 33
VM_JMP_ID = 34
VM_LOOP_ID = 35
VM_EMIT_ID = 36
VM_SEARCH_ID = 37
VM_RANK_ID = 38
VM_COMPOSE_ID = 39

# ─── Reserved (40-99) ────────────────────────────────────────────────
NUM_RESERVED = 60
RESERVED_START_ID = 40
RESERVED_END_ID = 99

# ─── Registry ────────────────────────────────────────────────────────
NUM_SPECIAL_TOKENS = 100
BPE_OFFSET = NUM_SPECIAL_TOKENS  # BPE tokens start at ID 100

# Ordered list: index = token ID
SPECIAL_TOKEN_LIST: list[str] = [
    # Core (0-4)
    MASK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
    # Structure (5-14)
    SLOT_START_TOKEN, SLOT_END_TOKEN,
    BLOCK_START_TOKEN, BLOCK_END_TOKEN,
    PLAN_START_TOKEN, PLAN_END_TOKEN,
    RANK_QUERY_TOKEN, RANK_RESULT_TOKEN,
    SEP_TOKEN, NEWBLOCK_TOKEN,
    # Language (15-24)
    LANG_EN_TOKEN, LANG_ES_TOKEN, LANG_PY_TOKEN,
    LANG_FR_TOKEN, LANG_DE_TOKEN,
    DOMAIN_STORY_TOKEN, DOMAIN_WIKI_TOKEN, DOMAIN_QA_TOKEN,
    DOMAIN_CODE_TOKEN, DOMAIN_MATH_TOKEN,
    # VM opcodes (25-39)
    VM_EXEC_TOKEN, VM_HALT_TOKEN, VM_PUSH_TOKEN, VM_POP_TOKEN,
    VM_CALL_TOKEN, VM_RET_TOKEN, VM_LOAD_TOKEN, VM_STORE_TOKEN,
    VM_CMP_TOKEN, VM_JMP_TOKEN, VM_LOOP_TOKEN, VM_EMIT_TOKEN,
    VM_SEARCH_TOKEN, VM_RANK_TOKEN, VM_COMPOSE_TOKEN,
]

# Pad reserved slots with placeholder tokens
for _i in range(RESERVED_START_ID, RESERVED_END_ID + 1):
    SPECIAL_TOKEN_LIST.append(f"[RESERVED_{_i}]")

assert len(SPECIAL_TOKEN_LIST) == NUM_SPECIAL_TOKENS, (
    f"Expected {NUM_SPECIAL_TOKENS} special tokens, got {len(SPECIAL_TOKEN_LIST)}"
)

# Name → ID mapping
SPECIAL_TOKEN_MAP: dict[str, int] = {
    tok: idx for idx, tok in enumerate(SPECIAL_TOKEN_LIST)
}

# ID → Name mapping
ID_TO_SPECIAL: dict[int, str] = {
    idx: tok for idx, tok in enumerate(SPECIAL_TOKEN_LIST)
}

# ─── Structure token sets (for detection in sequences) ───────────────
STRUCTURE_TOKENS = {
    SLOT_START_ID, SLOT_END_ID,
    BLOCK_START_ID, BLOCK_END_ID,
}

LANG_TOKENS = {
    LANG_EN_ID, LANG_ES_ID, LANG_PY_ID, LANG_FR_ID, LANG_DE_ID,
}

DOMAIN_TOKENS = {
    DOMAIN_STORY_ID, DOMAIN_WIKI_ID, DOMAIN_QA_ID,
    DOMAIN_CODE_ID, DOMAIN_MATH_ID,
}

VM_TOKENS = {
    VM_EXEC_ID, VM_HALT_ID, VM_PUSH_ID, VM_POP_ID,
    VM_CALL_ID, VM_RET_ID, VM_LOAD_ID, VM_STORE_ID,
    VM_CMP_ID, VM_JMP_ID, VM_LOOP_ID, VM_EMIT_ID,
    VM_SEARCH_ID, VM_RANK_ID, VM_COMPOSE_ID,
}


def is_special_token(token_id: int) -> bool:
    """Check if a token ID is a special token."""
    return 0 <= token_id < NUM_SPECIAL_TOKENS


def get_vocab_size(bpe_vocab_size: int = 32768) -> int:
    """Total vocab = BPE tokens + special tokens."""
    return bpe_vocab_size + NUM_SPECIAL_TOKENS
