"""Unit tests for name normalization."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from normalize import normalize_publisher, normalize_author, PUBLISHER_NORM, AUTHOR_NORM


class TestPublisherNorm:
    def test_sanlian_short_to_full(self):
        assert normalize_publisher('三联书店') == '生活·读书·新知三联书店'

    def test_shanghai_sanlian_independent(self):
        assert normalize_publisher('上海三联书店') == '上海三联书店'

    def test_combined_entry_split(self):
        result = normalize_publisher('生活·读书·新知三联书店 上海三联书店')
        assert result == '生活·读书·新知三联书店'

    def test_gz_hc_split(self):
        result = normalize_publisher('广州出版社 花城出版社')
        assert result == '广州出版社'

    def test_dongli_variants(self):
        for v in ['东立', '东立出版', '东立出版社有限公司']:
            assert normalize_publisher(v) == '东立出版社', f'Failed for {repr(v)}'

    def test_huangguan_variants(self):
        for v in ['皇冠', '皇冠出版社', '皇冠文化出版公司']:
            assert normalize_publisher(v) == '皇冠文化出版有限公司'

    def test_jianduan_weixiang(self):
        assert normalize_publisher('尖端') == '尖端出版'
        assert normalize_publisher('威向') == '威向文化'

    def test_whitespace_handling(self):
        assert normalize_publisher('  中华书局  ') == '中华书局'

    def test_slash_split(self):
        assert normalize_publisher('广州出版社 / 花城出版社') == '广州出版社'


class TestAuthorNorm:
    def test_qian_zhongshu_merge(self):
        assert normalize_author('钱钟书') == '钱锺书'

    def test_qian_mu_unchanged(self):
        assert normalize_author('钱穆') == '钱穆'

    def test_bracket_removal(self):
        assert normalize_author('[美] 雷蒙德·钱德勒') == '雷蒙德·钱德勒'

    def test_coauthor_split(self):
        assert normalize_author('钱理群 / 温儒敏 / 吴福辉') == '钱理群'
