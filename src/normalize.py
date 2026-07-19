"""Publisher and author name normalization."""
import re

# Publisher normalization map: variant -> canonical
PUBLISHER_NORM = {
    # 三联 variants -> 生活·读书·新知三联书店
    '三联书店': '生活·读书·新知三联书店',
    # 东立 variants -> 东立出版社
    '东立': '东立出版社',
    '東立': '东立出版社',
    '东立出版': '东立出版社',
    '東立出版社': '东立出版社',
    '东立出版社有限公司': '东立出版社',
    '東立出版社有限公司': '东立出版社',
    # 皇冠 variants -> 皇冠文化出版有限公司
    '皇冠': '皇冠文化出版有限公司',
    '皇冠出版社': '皇冠文化出版有限公司',
    '皇冠文化出版公司': '皇冠文化出版有限公司',
    # 尖端 variants -> 尖端出版
    '尖端': '尖端出版',
    # 威向 variants -> 威向文化
    '威向': '威向文化',
    # 新经典 variants -> 新经典文化
    '新经典图文传播有限公司': '新经典文化',
}

# Author normalization map: variant -> canonical
AUTHOR_NORM = {
    '钱钟书': '钱锺书',  # 简体钟 -> 正体锺
}


def normalize_publisher(name):
    if not isinstance(name, str) or not name.strip():
        return '未知'
    text = name.strip()
    if ' ' in text or '　' in text or ' / ' in text:
        parts = re.split(r'[ 　/]+', text)
        for part in parts:
            part = part.strip()
            if len(part) >= 4:
                text = part
                break
    # Traditional->Simplified (common Taiwan/HK publishers)
    text = text.replace('東', '东')  # 東->东
    # Strip suffix
    for sfx in ['有限公司', '股份有限公司']:
        if text.endswith(sfx):
            text = text[:-len(sfx)]
            break
    return PUBLISHER_NORM.get(text, text)
def normalize_author(name):
    """Normalize author name: trim brackets, resolve variants."""
    if not isinstance(name, str) or not name.strip():
        return '未知'
    text = name.strip()
    # Remove nationality brackets for lookup
    # Remove [XX] nationality/role brackets
    text = re.sub(r'\[[^]]*\]', '', text).strip()
    # Remove co-authors after / or space
    if ' / ' in text:
        text = text.split(' / ')[0].strip()
    return AUTHOR_NORM.get(text, text)
