// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLinksValidator from 'starlight-links-validator';

const repo = 'https://github.com/ewhauser/outis';
const huggingFace = 'https://huggingface.co/ewhauser';
export default defineConfig({
  site: process.env.SITE_URL ?? 'https://ewhauser.github.io',
  base: process.env.SITE_BASE ?? '/outis',
  trailingSlash: 'always',
  integrations: [starlight({
    title: 'Outis',
    description: 'Local PHI detection, typed redaction and patient-consistent synthetic replacement for English clinical text.',
    favicon: '/favicon.svg',
    social: [{ icon: 'github', label: 'GitHub', href: repo }],
    editLink: { baseUrl: `${repo}/edit/main/site/` },
    customCss: ['./src/styles/custom.css'],
    components: {
      SiteTitle: './src/components/SiteTitle.astro',
      Footer: './src/components/Footer.astro',
    },
    plugins: [starlightLinksValidator({ errorOnRelativeLinks: false, errorOnLocalLinks: false })],
    sidebar: [
      { label: 'Start here', items: [
        { label: 'Overview', slug: '' },
        { label: 'Quickstart', slug: 'docs/quickstart' },
        { label: 'Results', slug: 'docs/results' },
        { label: 'Limitations', slug: 'docs/limitations' },
      ] },
      { label: 'Using Outis', items: [
        { label: 'Clinical text API', slug: 'docs/api' },
        { label: 'Synthetic replacement', slug: 'docs/replacement' },
        { label: 'Deployment', slug: 'docs/deployment' },
      ] },
      { label: 'Model', items: [
        { label: 'Model card', slug: 'docs/model-card' },
        { label: 'Training', slug: 'docs/training' },
        { label: 'Annotation policy', slug: 'docs/annotation-policy' },
        { label: 'Places addendum', slug: 'docs/annotation-policy-places' },
      ] },
      { label: 'Evaluation', items: [
        { label: 'Methodology', slug: 'docs/evaluation' },
        { label: 'External verification', slug: 'docs/external-verification' },
        { label: 'Benchmark comparison', slug: 'docs/benchmark-comparison' },
        { label: 'HTML reports', slug: 'docs/html-reports' },
      ] },
      { label: 'Downloads', items: [
        { label: 'Model weights ↗', link: `${huggingFace}/outis-deberta-v3-base` },
        { label: 'Synthetic training data ↗', link: 'https://huggingface.co/datasets/ewhauser/outis-synthetic-phi' },
        { label: 'Source code ↗', link: repo },
      ] },
      { label: 'Contributing', slug: 'docs/contributing' },
    ],
  })],
});
