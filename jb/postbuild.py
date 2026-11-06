"""Fix up the static site after `jupyter book build --html`.

    python postbuild.py _build/html

The book theme takes the site's address from the local server that renders the
static export, so the sitemap, robots.txt, and page data say http://localhost:PORT,
and the theme writes no canonical link. This script:

- replaces the local server's address with the real one;
- rewrites the sitemap with the published URLs (with a trailing slash, which is
  what GitHub Pages serves without a redirect), leaving out pages whose canonical
  version is elsewhere;
- adds a canonical link and og:url to each page. Pages in CANONICAL_ELSEWHERE
  point to the original (none here);
- adds descriptions to the notebook pages (DESCRIPTIONS);
- makes og:image an absolute URL, which link previews need;
- adds the page-view counter (greenteapress.com/gh.gif, counted from the
  greenteapress.com access logs; no cookies, no third party).
"""

import re
import sys
from pathlib import Path

ORIGIN = 'https://allendowney.github.io'
BASE = '/SelectionSAT'
SITE = ORIGIN + BASE

CANONICAL_ELSEWHERE = {}

# Descriptions for the notebook pages, which have no frontmatter (adding a
# frontmatter cell to the published notebook would show up on Colab).
DESCRIPTIONS = {
    'selection-bias3': 'The Bayesian model from the PyData Boston 2025 talk: latent math ability plus a selection function, fitted to SAT math scores by gender, 2011-2015.',
    'selection-bias3-loo': 'Leave-one-out analysis of the SAT selection model: which score bins most influence the estimated difference between male and female test takers.',
}

COUNTER = ('<script>new Image().src="https://greenteapress.com/gh.gif?u="+encodeURIComponent(location.href)'
           '+"&r="+encodeURIComponent(document.referrer)</script>')

LOCALHOST = re.compile(r'http://localhost:\d+')


def page_url(slug):
    return f'{SITE}/{slug}/' if slug else f'{SITE}/'


def fix_page(path, slug):
    s = path.read_text(encoding='utf-8')
    s = LOCALHOST.sub(ORIGIN, s)
    s = re.sub(r'(<meta (?:property="og:image"|name="image") content=")' + re.escape(BASE) + r'/+',
               r'\g<1>' + SITE + '/', s)
    if slug in DESCRIPTIONS:
        # The theme gives these pages the project's description; replace it.
        d = DESCRIPTIONS[slug]
        s, n = re.subn(r'(<meta (?:name="description"|property="og:description"|name="twitter:description") content=")[^"]*',
                       lambda m: m.group(1) + d, s)
        if n == 0:
            s = s.replace('</head>', f'<meta name="description" content="{d}"/></head>', 1)
    canonical = CANONICAL_ELSEWHERE.get(slug, page_url(slug))
    if 'rel="canonical"' not in s:
        tags = f'<link rel="canonical" href="{canonical}"/><meta property="og:url" content="{canonical}"/>'
        s = s.replace('</head>', tags + '</head>', 1)
    if 'greenteapress.com/gh.gif' not in s:
        s = s.replace('</body>', COUNTER + '</body>', 1)
    path.write_text(s, encoding='utf-8')


def write_sitemap(html, slugs):
    urls = [page_url(slug) for slug in slugs if slug not in CANONICAL_ELSEWHERE]
    entries = ''.join(f'<url><loc>{u}</loc></url>' for u in urls)
    (html / 'sitemap.xml').write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{entries}</urlset>\n',
        encoding='utf-8')
    return len(urls)


def main(html):
    html = Path(html)
    slugs = [''] + sorted(p.parent.name for p in html.glob('*/index.html') if p.parent.name != 'build')
    for slug in slugs:
        fix_page(html / slug / 'index.html', slug)

    for path in html.glob('*.json'):
        s = path.read_text(encoding='utf-8')
        if LOCALHOST.search(s):
            path.write_text(LOCALHOST.sub(ORIGIN, s), encoding='utf-8')

    robots = html / 'robots.txt'
    robots.write_text(LOCALHOST.sub(SITE, robots.read_text(encoding='utf-8')), encoding='utf-8')

    n = write_sitemap(html, slugs)
    print(f'postbuild: {len(slugs)} pages fixed, {n} URLs in sitemap.xml')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '_build/html')
