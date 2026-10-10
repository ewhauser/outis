// Publish an explicit list of repository documents. Never glob the repository:
// add each new public source here.
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const site = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = path.resolve(site, '..');
const out = path.join(site, 'src/content/docs/docs');
const repo = 'https://github.com/ewhauser/outis';
const pages = [
  { source: 'README.md', section: 'Quickstart', slug: 'quickstart', title: 'Quickstart', description: 'Install Outis, download the model weights, and de-identify text from Python, HTTP or the CLI.' },
  { source: 'README.md', section: 'Results', slug: 'results', title: 'Results', description: 'Public benchmark comparisons with MedDeID and GLiNER-PII, and held-out synthetic results.' },
  { source: 'README.md', section: 'Limitations', slug: 'limitations', title: 'Limitations', description: 'What Outis has not been measured on, and what it does not guarantee.' },
  { source: 'docs/api.md', slug: 'api', description: 'Endpoints, request policies, masks, structured data, files and WebSockets.' },
  { source: 'docs/replacement.md', slug: 'replacement', description: 'Secrets, patient anchors, date shifting and identity linking in synthetic mode.' },
  { source: 'docs/deployment.md', slug: 'deployment', description: 'The local launcher, the default model, batch limits and security controls.' },
  { source: 'MODEL_CARD.md', slug: 'model-card', title: 'Model card', description: 'Architecture, training data, evaluation, intended use and risks of the DeBERTa-v3-base detector.' },
  { source: 'docs/training.md', slug: 'training', description: 'Training detector candidates and generating synthetic corpora.' },
  { source: 'docs/annotation-policy-v2.md', slug: 'annotation-policy', title: 'Annotation policy', description: 'Span conventions for every PHI type.' },
  { source: 'docs/annotation-policy-v2-places.md', slug: 'annotation-policy-places', title: 'Places addendum', description: 'Real localities, care-site abbreviations and public references to places.' },
  { source: 'docs/evaluation.md', slug: 'evaluation', description: 'Targets, any-label masking and how a new model is compared with the current one.' },
  { source: 'docs/external-verification.md', slug: 'external-verification', description: 'The public benchmark protocol, and the rules that keep that data out of training.' },
  { source: 'benchmarks/results/COMPARISON.md', slug: 'benchmark-comparison', title: 'Benchmark comparison', description: 'Per-system, per-identifier results on each public benchmark.' },
  { source: 'docs/html-reports.md', slug: 'html-reports', description: 'Building and reading the self-contained evaluation report page.' },
  { source: 'CONTRIBUTING.md', slug: 'contributing', description: 'Development setup, checks and the rules for contributions.' },
];

const slugify = heading => heading.toLowerCase().trim().replace(/[^\p{L}\p{N}\s_-]/gu, '').replace(/\s/g, '-');
// Whole documents route by path; README sections route by path#anchor.
const routes = new Map(pages.map(p => [p.section ? `${p.source}#${slugify(p.section)}` : p.source, p.slug]));
const asides = { NOTE: 'note', TIP: 'tip', IMPORTANT: 'caution[Important]', WARNING: 'caution', CAUTION: 'danger' };

function destination(page, href) {
  if (/^(?:[a-z][a-z\d+.-]*:|\/)/i.test(href)) return href;
  const [target, fragment] = href.split('#');
  const source = target ? path.posix.normalize(path.posix.join(path.posix.dirname(page.source), target)) : page.source;
  const section = routes.get(`${source}#${fragment}`);
  if (section) return section === page.slug ? '#_top' : `../${section}/`;
  if (!target) return href;
  const route = routes.get(source);
  if (route) return `../${route}/${fragment ? `#${fragment}` : ''}`;
  if (source === 'README.md' && !fragment) return '../../';
  const kind = target.endsWith('/') ? 'tree' : 'blob';
  return `${repo}/${kind}/main/${source}${fragment ? `#${fragment}` : ''}`;
}

rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });
for (const page of pages) {
  let body = readFileSync(path.join(root, page.source), 'utf8').replace(/^---\n[\s\S]*?\n---\n/, '');
  let title = page.title;
  if (page.section) {
    const match = body.match(new RegExp(`^## ${page.section}\\n([\\s\\S]*?)(?=^## |$(?![\\s\\S]))`, 'm'));
    if (!match) throw new Error(`Missing ${page.section} section in ${page.source}`);
    body = match[1];
  } else {
    const heading = body.match(/^# (.+)\n/m);
    if (!heading) throw new Error(`Missing title in ${page.source}`);
    title ??= heading[1];
    body = body.slice(heading.index + heading[0].length);
  }
  // Leave code blocks alone. Promote section subheadings, turn GitHub alerts into
  // Starlight asides, and rewrite Markdown destinations relative to their source.
  let fence = null;
  let aside = null;
  const lines = [];
  for (const line of body.split('\n')) {
    const marker = line.match(/^\s*(`{3,}|~{3,})/);
    if (marker) {
      if (!fence) fence = marker[1];
      else if (marker[1][0] === fence[0] && marker[1].length >= fence.length) fence = null;
      lines.push(line);
      continue;
    }
    if (fence) { lines.push(line); continue; }
    const alert = line.match(/^> \[!(\w+)\]\s*$/);
    if (alert) {
      if (!asides[alert[1]]) throw new Error(`Unknown alert ${alert[1]} in ${page.source}`);
      aside = true;
      lines.push(`:::${asides[alert[1]]}`);
      continue;
    }
    let text = line;
    if (aside) {
      if (/^>/.test(line)) text = line.replace(/^> ?/, '');
      else { lines.push(':::'); aside = null; }
    }
    if (page.section) text = text.replace(/^(#{3,6}) /, heading => heading.slice(1));
    lines.push(text.replace(/\]\(([^\s)]+)\)/g, (_, href) => `](${destination(page, href)})`));
  }
  if (aside) lines.push(':::');
  const metadata = { title, description: page.description, editUrl: `${repo}/edit/main/${page.source}` };
  const frontmatter = Object.entries(metadata).map(([key, value]) => `${key}: ${JSON.stringify(value)}`).join('\n');
  writeFileSync(path.join(out, `${page.slug}.md`), `---\n${frontmatter}\n---\n\n${lines.join('\n').trim()}\n`);
}
console.log(`Synced ${pages.length} pages.`);
