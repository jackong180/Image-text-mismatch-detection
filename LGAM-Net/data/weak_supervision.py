"""Explicit synthetic proxies, not human OOC annotations."""
import random
import re
from collections import defaultdict

ALLOWED_TYPES = {'PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'EVENT'}

def build_entity_pool(records):
    pools = defaultdict(set)
    for r in records:
        for a in r.get('articles', []):
            for item in a.get('entity_list', []):
                if len(item) == 2:
                    entity, kind = item
                    if kind in ALLOWED_TYPES and isinstance(entity, str) and len(entity.strip()) > 1:
                        pools[kind].add(entity)
    return {k: sorted(v) for k, v in pools.items()}

def corrupt_caption(article, pools, rng):
    caption = article['caption']
    entities = list(article.get('entity_list', []))
    rng.shuffle(entities)
    for entity, kind in entities:
        if kind not in ALLOWED_TYPES or not entity.strip():
            continue
        candidates = [v for v in pools.get(kind, []) if v.casefold() not in caption.casefold()]
        pattern = re.compile(r'(?<!\w)' + re.escape(entity) + r'(?!\w)', re.IGNORECASE)
        if candidates and pattern.search(caption):
            replacement = rng.choice(candidates)
            return pattern.sub(lambda _: replacement, caption), f'entity_swap:{kind}'
    return None, None

def make_weak_triple(record, pools, seed, negative, same_image_weight=0.35):
    rng = random.Random(seed)
    articles = [a for a in record['articles'] if a.get('caption', '').strip()]
    if not articles:
        raise ValueError('No nonempty original captions')
    first = rng.choice(articles)
    others = [a for a in articles if a['caption'] != first['caption']]
    second = rng.choice(others) if others else first
    a, b = first['caption'], second['caption']
    if negative:
        changed, origin = corrupt_caption(second, pools, rng)
        if changed is not None:
            return a, changed, 1., 0.7, origin
        # Cross-image swaps are image-text mismatch proxies, not verified OOC.
        return a, b, 0., 0., 'no_valid_corruption:alignment_only'
    weight = same_image_weight if others else 0.25
    return a, b, 0., weight, 'same_image_weak_pair' if others else 'duplicate_caption_weak_pair'
