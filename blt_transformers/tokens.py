"""BLT byte vocabulary: token id = UTF-8 byte + 4, with four special ids below."""

BOE_ID, BOS_ID, EOS_ID, PAD_ID = 0, 1, 2, 3
BYTE_OFFSET = 4
VOCAB_SIZE = 260


def encode(text: str, *, bos: bool = True, eos: bool = False) -> list[int]:
    ids = [byte + BYTE_OFFSET for byte in text.encode("utf-8")]
    return [BOS_ID] * bos + ids + [EOS_ID] * eos


def decode(ids, *, errors: str = "replace") -> str:
    """Drop a leading BOS, stop at the first EOS, and reject any other special id."""
    ids = list(ids)
    if ids[:1] == [BOS_ID]:
        ids = ids[1:]
    if EOS_ID in ids:
        ids = ids[:ids.index(EOS_ID)]
    invalid = [i for i in ids if not BYTE_OFFSET <= i < VOCAB_SIZE]
    if invalid:
        raise ValueError(f"Non-byte token ids in sequence: {invalid}")
    return bytes(i - BYTE_OFFSET for i in ids).decode("utf-8", errors=errors)
