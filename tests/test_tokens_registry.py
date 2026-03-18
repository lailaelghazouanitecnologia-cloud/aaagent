"""Tests for the special token registry."""

from data.tokens import (
    SPECIAL_TOKEN_LIST,
    SPECIAL_TOKEN_MAP,
    ID_TO_SPECIAL,
    NUM_SPECIAL_TOKENS,
    MASK_TOKEN_ID,
    PAD_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    UNK_TOKEN_ID,
    SLOT_START_ID,
    SLOT_END_ID,
    BLOCK_START_ID,
    BLOCK_END_ID,
    STRUCTURE_TOKENS,
    LANG_TOKENS,
    VM_TOKENS,
    is_special_token,
    get_vocab_size,
)


class TestSpecialTokenRegistry:
    def test_total_count(self):
        assert len(SPECIAL_TOKEN_LIST) == NUM_SPECIAL_TOKENS == 100

    def test_core_ids(self):
        assert MASK_TOKEN_ID == 0
        assert PAD_TOKEN_ID == 1
        assert BOS_TOKEN_ID == 2
        assert EOS_TOKEN_ID == 3
        assert UNK_TOKEN_ID == 4

    def test_structure_ids(self):
        assert SLOT_START_ID == 5
        assert SLOT_END_ID == 6
        assert BLOCK_START_ID == 7
        assert BLOCK_END_ID == 8

    def test_map_consistency(self):
        for idx, tok in enumerate(SPECIAL_TOKEN_LIST):
            assert SPECIAL_TOKEN_MAP[tok] == idx
            assert ID_TO_SPECIAL[idx] == tok

    def test_no_duplicates(self):
        assert len(set(SPECIAL_TOKEN_LIST)) == len(SPECIAL_TOKEN_LIST)

    def test_structure_token_set(self):
        assert SLOT_START_ID in STRUCTURE_TOKENS
        assert BLOCK_END_ID in STRUCTURE_TOKENS

    def test_lang_tokens(self):
        assert len(LANG_TOKENS) == 5

    def test_vm_tokens(self):
        assert len(VM_TOKENS) == 15

    def test_is_special(self):
        assert is_special_token(0)
        assert is_special_token(99)
        assert not is_special_token(100)
        assert not is_special_token(-1)

    def test_vocab_size(self):
        assert get_vocab_size(32768) == 32868
        assert get_vocab_size(8192) == 8292
