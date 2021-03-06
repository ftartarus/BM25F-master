#!/usr/bin/env python
# coding: utf-8

from BM25F.ja import Normalizer
from BM25F.ja import PosFilter
from BM25F.ja import StemFilter
from BM25F.ja import Tokenizer
import unittest


class TestJapanese(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stem_filter = StemFilter()
        cls.pos_filter = PosFilter()

    def test_normalzier(self):
        n = Normalizer()
        self.assertEqual('abc', n.normalize('ＡＢＣ'))
        self.assertEqual('カラー', n.normalize('カラー'))
        self.assertEqual('モニタ', n.normalize('モニター'))
        self.assertEqual('モニター', n.normalize('モニターー'))
        self.assertEqual('モニタの', n.normalize('モニターの'))
        self.assertEqual('イーメール', n.normalize('イーメール'))

    def test_stem_filter(self):
        self.assertTrue('ある' in self.stem_filter)
        self.assertFalse('テスト' in self.stem_filter)

    def test_pos_filter(self):
        self.assertTrue('助詞-格助詞-一般' in self.pos_filter)
        self.assertFalse('名詞-一般' in self.pos_filter)

    def test_tokenizer(self):
        m = Tokenizer()
        self.assertEqual([
            ('テスト', '名詞-サ変接続'),
            ('データ', '名詞-一般'),
        ], m.tokenize_smartly('テストデータ'))

    def test_tokenizer_with_stem_filter(self):
        m = Tokenizer(stem_filter=self.stem_filter)
        self.assertEqual([
            ('テスト', '名詞-サ変接続'),
            ('データ', '名詞-一般'),
        ], m.tokenize_smartly('その他テストデータ'))

    def test_tokenizer_with_pos_filter(self):
        m = Tokenizer(pos_filter=self.pos_filter)
        self.assertEqual([
            ('テスト', '名詞-サ変接続'),
            ('データ', '名詞-一般'),
        ], m.tokenize_smartly('テストのデータ'))


if __name__ == '__main__':
    unittest.main()
