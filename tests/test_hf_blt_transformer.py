import unittest

from blt_transformer.tokens import BOS_ID, EOS_ID, decode, encode


class BltTokensTest(unittest.TestCase):
    def test_encode_is_utf8_byte_plus_four(self):
        self.assertEqual(encode("가a"), [BOS_ID, 0xEA + 4, 0xB0 + 4, 0x80 + 4, ord("a") + 4])
        self.assertEqual(encode("a", bos=False, eos=True), [ord("a") + 4, EOS_ID])

    def test_decode_round_trip_and_stops_at_eos(self):
        text = "오늘 날씨가 좋아요."
        self.assertEqual(decode(encode(text, eos=True) + [ord("x") + 4]), text)

    def test_decode_rejects_special_ids_and_marks_broken_utf8(self):
        with self.assertRaises(ValueError):
            decode([ord("a") + 4, 3])
        self.assertEqual(decode([0xEA + 4]), "\ufffd")


if __name__ == "__main__":
    unittest.main()
