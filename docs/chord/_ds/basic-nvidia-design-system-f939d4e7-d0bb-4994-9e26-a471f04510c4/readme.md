# NVIDIA Design System

A brand-accurate design system for building NVIDIA-styled interfaces, slides,
and marketing assets — for production or throwaway prototypes. Built from
**NVIDIA's official 2025 PowerPoint template** (the only source provided).

> **Provenance & caveats.** The only source material was
> `uploads/NVIDIA-PPT-Template-Light-2025.pptx` (NVIDIA's official "Light"
> deck template). Colors, the type system, the logo lockups, and the slide
> layouts are extracted directly from that file and are faithful. No product
> codebase, Figma, or website source was provided, so the **web UI kit is an
> original brand-applied composition** (built from this system's own tokens
> and components), **not** a pixel recreation of a specific nvidia.com screen.
> The brand typeface **NVIDIA Sans** is proprietary and not redistributable —
> this system ships **Hanken Grotesk** as a close free substitute (see Fonts).
>
> **Update — Kaizen Figma added.** NVIDIA's real **Kaizen Design System v10**
> Figma was later provided and is now a second source of record. It confirmed
> the brand foundations (Green `#76B900`, the gray ramp, the secondary palette)
> and supplied the genuine **product-UI** type stack, Material-style grays, and
> categorical accent palette — shipped as the opt-in **Kaizen mode** (see
> below). The full **NVIDIA GUI Icons** Figma was later provided and **154
> icons are now folded in** as real vectors (see Iconography). The **KUI
> Foundations** Figma is also folded in: the authentic **947-token KUI variable
> system** is captured at `tokens/kui/fig-tokens.css` (opt-in reference — see
> "Kaizen mode") and Kaizen mode's accent ramps now use its exact values.

---

## Brand context

**NVIDIA** is the accelerated-computing company: GPUs and a full software stack
(CUDA, frameworks, microservices) spanning data center, cloud, gaming,
professional visualization, robotics, and edge AI. The brand voice is
**confident, technical, and forward-looking** — it talks about platforms,
architectures, and performance multipliers.

Visually the brand is **disciplined and near-monochrome**: white surfaces,
near-black ink, a precise gray scale, and a single signature **NVIDIA Green
(#76B900)** used sparingly as the accent of record. Black backgrounds with
green type are the iconic "keynote" expression. Everything reads engineered:
tight corners, even strokes, generous structural whitespace.

### Source files
- `uploads/NVIDIA-PPT-Template-Light-2025.pptx` — official brand/marketing
  template (60 slides, theme, logos, palette). Extracted media is in
  `assets/_source/`. **Drives the default brand system.**
- **Kaizen Design System v10** (Figma) — NVIDIA's real web product-UI system.
  Drives the opt-in **Kaizen mode** (`tokens/kaizen-mode.css`). Verified values:
  Brand Green 700 `#59A700`, Accent Teal 500 `#008471`, Material gray ramp,
  type = Roboto / Open Sans / JetBrains Mono.
- *(Attached but not reachable at build time:* **NVIDIA GUI Icons.fig**,
  **KUI Foundations - Components.fig** — re-import individually to fold in the
  full ~2,300-icon library and KUI specs.)

---

## CONTENT FUNDAMENTALS

How NVIDIA writes (rules taken straight from the template's own guidance slides):

- **Titles are Title Case; subtitles are sentence case.** This is an explicit
  rule in the template ("Titles Are 'Title Case' / Subtitles are sentence
  case"). Title Case = all major words capitalized, minor words lowercase (AP
  style). Sentence case = only the first word and proper nouns capitalized.
- **Voice: third-person / product-forward, not chatty.** Copy is about the
  technology and the customer outcome ("The engine of accelerated computing"),
  not "we" or "you" heavy. Avoid first-person singular. Light, occasional
  second person is fine in CTAs ("Start building with NVIDIA").
- **Lead with the outcome, then the detail.** Bullets are short and
  informative — "Keep points short and informative" is literally the template's
  advice. One idea per bullet.
- **Quantify everything.** Performance is expressed as multipliers and hard
  figures: "4× faster time-to-train", "30× throughput", "208B transistors",
  "20 PFLOPS". Numbers are a core part of the voice.
- **Casing of product names is exact:** NVIDIA (all caps), CUDA, GeForce RTX,
  DGX, Jetson, Blackwell, Hopper, TensorRT. "NVIDIA" is always uppercase.
- **No emoji.** The brand never uses emoji in product or marketing UI.
- **Tone words:** accelerated, full-stack, platform, architecture, scale,
  engine, real-time, end-to-end. Avoid hype adjectives without a number behind
  them.
- **Quotes** use sentence case, never all caps, with an em-dash attribution
  ("— Source Name, Title").

Examples (from the template / in-voice):
- Title: *"The Engine of Accelerated Computing"*
- Subtitle: *"One full-stack platform — from GPU silicon to AI frameworks"*
- Stat: *"208B transistors · 2.6×"*
- CTA: *"Explore the Platform"*, *"Join the Developer Program"*

---

## VISUAL FOUNDATIONS

- **Color.** Overwhelmingly monochrome. White (`#FFFFFF`) surface, black
  (`#000000`) ink, a 10-step gray ramp, and **NVIDIA Green `#76B900`** as the
  single accent. Secondary hues (Garnet, Fluorite, Emerald, CPU Blue, Amethyst)
  exist **only for data viz / categorical use — never UI chrome**. Green is
  bright, so **ink on green is always black, never white**.
- **The signature motif** is bright green on deep black (keynote slides, hero
  bands). On light surfaces green appears as small accents: a 3–5px bar, an
  underline, an active-tab indicator, a bullet square, a CTA fill.
- **Type.** NVIDIA Sans (substitute: Hanken Grotesk). Medium (500) for
  headings/titles, Regular (400) for body, Light (300) for large display
  numbers and long quotes. Mono is **Roboto Mono** (the template's actual code
  spec). Display type is set tight (`-0.02em`) with line-height ~1.04. Eyebrows
  are semibold, uppercase, wide-tracked, green.
- **Spacing** is a 4px grid; layouts are airy and structural with strong left
  alignment and clear column gutters.
- **Backgrounds** are flat color — white, black, or full-bleed green. **No
  gradients** in UI (one subtle green radial glow is acceptable as hero
  atmosphere only). No textures, no patterns, no hand illustration. Imagery,
  when used, is high-contrast product/tech photography (often dark, cool-toned).
- **Corners are tight and technical:** 2px inputs, 4px buttons, 6px cards,
  10px modals. The logo's green block is a hard square — radii never get
  pillowy except for avatars/toggles/status dots (`999px`).
- **Borders** are hairline grays (`#E6E6E6` / `#CDCDCD`). The green accent bar
  is 3–5px. Cards = white + 1px subtle border + a soft neutral shadow.
- **Shadows** are soft, neutral, and **never colored** — `0 1px 3px rgba(0,0,0,.08)`
  at rest, lifting to `0 4px 12px` on hover. Elevation is restrained.
- **Motion** is quick and functional: 120–200ms, standard ease
  `cubic-bezier(.4,0,.2,1)`. Fades and small translateY lifts. **No bounce, no
  decorative loops.** Hover = lighter green / gray wash; press = darker green +
  ~0.5px nudge down.
- **Focus** is a green ring: `0 0 0 3px rgba(118,185,0,.40)`.
- **Transparency / blur** is used sparingly — only the sticky nav uses a dark
  translucent backdrop blur. Surfaces are otherwise opaque.

---

## TWO MODES — brand (default) vs Kaizen (product UI)

NVIDIA runs **two** distinct visual systems, and this design system carries both:

| | **Brand / marketing** (default) | **Kaizen product UI** (`data-mode="kaizen"`) |
|---|---|---|
| Use for | decks, hero/landing pages, keynote visuals | app screens, dashboards, **prototypes** |
| Headings | NVIDIA Sans → Hanken Grotesk | **Roboto** Medium |
| Body / UI | NVIDIA Sans → Hanken Grotesk | **Open Sans** |
| Mono | Roboto Mono | **JetBrains Mono** |
| Grays | custom ink ramp | **Material** ramp (`#FAFAFA`…`#212121`) |
| Accents | Emerald / CPU-Blue / Amethyst / Garnet / Fluorite (deep, print) | Blue `#7191D5` · Teal `#008471` · Purple `#DF78EF` · Orange `#F08039` · Red `#F03232` (bright) |
| Green | `#76B900`, pressed `#5C9000` | `#76B900`, pressed **`#59A700`** |

**NVIDIA Green `#76B900` is the signature accent in BOTH modes.**

**How it works.** Every component resolves *semantic* tokens (`--font-sans`,
`--accent`, `--surface-*`, `--text-*`, `--nv-gray-*`) through `var()`.
`tokens/kaizen-mode.css` re-points the **base** tokens inside a single
`[data-mode="kaizen"]` scope, so the whole system + every component re-themes
with **zero component edits**. Just set the attribute on a wrapper:

```html
<html data-mode="kaizen"> … </html>   <!-- or any wrapping element -->
```

**Convention for THIS project: build product prototypes in Kaizen mode; keep
slides, decks, and marketing pages in the default brand mode.** (Also recorded
in `CLAUDE.md`.) Kaizen-only categorical colors are exposed as `--kz-*`
(`--kz-blue`, `--kz-teal`, `--kz-purple`, `--kz-orange`, `--kz-red`, …) using
authentic KUI ramp values.

**Full KUI token set.** The complete authentic KUI variable system (947 tokens —
primitive ramps, `--color-accent-*` semantic roles, sizes, breakpoints, plus
Light/Dark/density/responsive modes) is captured verbatim at
`tokens/kui/fig-tokens.css`. It is **not** imported by `styles.css` by default
(its length tokens are unitless floats needing `calc(var(--x) * 1px)`, and its
`--radius-*` names would shadow this system's). Import it explicitly when you
need full-fidelity KUI tokens for a production-accurate Kaizen build.

---

## ICONOGRAPHY

- NVIDIA ships an official **GUI icon library** (the "NVIDIA GUI Icons" Figma
  file) — a large set (~2,300 icons) of **thin, single-color line icons on a
  16px grid**, organized by category: `common-*`, `hardware-*` (incl. GPU/CPU),
  `editor-*`, `av-*`, `communication-*`, `files-*`, `3d-*`, `social-*`, etc.
  They are filled-outline vectors (not stroked), so they paint with **`fill:
  currentColor`** and scale cleanly.
- Use the **`Icon`** component: `<Icon name="hardware-gpu" size={20} />`.
  Names follow the source `category-slug` convention (the `-line` suffix is
  dropped). This build ships **154 icons** extracted **verbatim** from the
  official NVIDIA GUI Icons Figma — real compound-outline geometry, not
  redrawn — spanning `common-*`, `editor-*`, `hardware-*`, `av-*`,
  `communication-*`, `files-*`, and `cursor-*`. Raw SVGs also live in
  `assets/icons/<name>.svg` for non-React contexts. `iconNames` lists all of
  them; the remaining ~2,150 from the full library can be folded in on request.
- Tint green only for emphasis; default to ink/secondary text color. Keep one
  weight (line) for UI consistency. Do not mix in third-party icon sets.
- **The marketing web UI kit** (`ui_kits/web/`) still uses a small Lucide helper
  (`UiIcon`) for a few glyphs not yet in the curated subset (arrows, social).
  Migrate it to `Icon` once those slugs are added.
- **Emoji and unicode glyphs are not used** as icons. The one decorative glyph
  in slides is the bullet square (a green box drawn in CSS, not a character).
- **Logos** — use the vector **`Logo`** component (`components/brand/`), built
  from the real Kaizen `Core / Logo` geometry: `variant="horizontal"` (lockup)
  or `"eye"`, with `tone` green/black/white. The eye is always NVIDIA green;
  only the wordmark recolors. Never place the green eye on a green field. Raster
  fallbacks remain in `assets/logos/` for non-React contexts.

---

## Index / manifest

**Foundations**
- `styles.css` — root entry point (imports only). Consumers link this.
- `tokens/fonts.css` · `colors.css` · `typography.css` · `spacing.css` · `base.css`
- `guidelines/cards/*.html` — foundation specimen cards (Colors, Type, Spacing, Brand)

**Components** (`window.DesignSystem_6d5263.*`)
- `components/brand/` — **Logo** (vector lockup + eye)
- `components/icons/` — **Icon** (official NVIDIA GUI line icons by name)
- `components/buttons/` — **Button**, **IconButton**
- `components/forms/` — **Input**, **Select**, **Checkbox**, **Switch**
- `components/data-display/` — **Badge**, **Tag**, **Card**, **Stat**, **Avatar**
- `components/navigation/` — **Tabs**, **Breadcrumb**
- `components/feedback/` — **Spinner**, **Banner** (inline status alert)
- `components/overlay/` — **Tooltip**
- Each dir has `<Name>.jsx`, `<Name>.d.ts`, a `.prompt.md`, and a `*.card.html`.

**UI kits**
- `ui_kits/web/` — NVIDIA-style product/developer landing page (NavBar, Hero,
  Showcase, Footer). Brand-applied composition; see provenance note above.

**Slides**
- `slides/` — sample deck (`index.html`, scaled with `deck-stage.js`) plus
  standalone slide-type cards: Title, Section Divider, Agenda, Content,
  Two Column, Quote, Color Palette. Shared layout in `slides/slide.css`.

**Assets**
- `assets/logos/` — logo lockups + eye mark.
- `assets/_source/` — raw media extracted from the PPTX (reference).

**Other**
- `SKILL.md` — Agent Skills manifest (for use in Claude Code).
- `readme.md` — this file.
