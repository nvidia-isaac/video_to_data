# Kaizen Design System

**KUI v11 (Beta)** — the design language and component library that powers NVIDIA's product surfaces. Reconstructed in this project from the canonical KUI v11 Figma file ("KUI v11.fig"), it covers tokens (color, type, spacing, elevation), a complete component set (~50 component families), and reference UI assemblies.

Kaizen is NVIDIA's internal design system. "Kaizen" (改善) is the Japanese practice of continuous, incremental improvement — fitting for a system whose calling card is sharp restraint and methodical iteration.

> **Source of truth.** Everything here is rebuilt from the Figma file `KUI v11.fig` (56 pages, 258 frames). Where the .fig and a screenshot disagreed, the .fig values won.
> If you have access to the live Kaizen Figma library or the upstream KUI codebase, prefer those for any new component work; this project ships a faithful but pseudocode-derived recreation.

---

## Index

| File | What's in it |
|---|---|
| `README.md` | This file — context, voice, visual foundations, iconography, manifest |
| `SKILL.md` | Claude Skills entry point — read first if invoked as a skill |
| `colors_and_type.css` | Every token as a CSS custom property (`--kui-*`) |
| `assets/logos/` | NVIDIA wordmark + eye-mark, SVG, copied from Figma |
| `assets/icons/kui-icons.js` | 30+ KUI icons (`common/*`, `shapes/*`) reconstructed as inline SVG paths |
| `assets/icons/cog-fill.svg` | Sample raw SVG icon copied straight from Figma |
| `preview/` | Card specimens that populate the Design System tab |
| `ui_kits/kaizen-app/` | A reference Kaizen application screen built from the components |
| `fonts/` | Drop real NVIDIA Sans `.woff2` files here to override the substitute |

---

## What "Kaizen" looks like (at a glance)

- **Sharp, almost printlike.** 2 px / 4 px corner radii. Black wordmark, NVIDIA-green eye-mark.
- **Functional, not decorative.** No gradients. No drop shadows on layout. Surfaces sit on a `#F2F2F2` canvas with thin `#D8D8D8` borders. Color carries meaning — green = brand & primary affirmation, red = danger, blue = info.
- **Information-dense.** Body text is `14 / 1.5` NVIDIA Sans. Headings are the same family at `16 / 1.25, -0.25 tracking`. Sizes step in narrow increments — Kaizen is built for dashboards and documentation, not marketing pages.
- **Engineered.** Spacing is named on a fixed scale (`xxs / xs / ss / sm / ms / md / ml / lg / xl / xxl` = `2 4 6 8 12 16 24 32 48 64`). Layout is rectangular and grid-aligned.

---

## CONTENT FUNDAMENTALS

**Voice.** Direct, neutral, engineer-to-engineer. Kaizen reads like documentation, not marketing. It states what a thing is and what to do with it. There is no exclamation. There are no metaphors.

**Tone.** Plain present tense. Active voice. Second person ("you") when speaking to the user; product/component names are referred to by their literal name ("Button", "Toast", "Notification"), not characters or personas.

**Casing.**
- **Sentence case** everywhere — UI labels, button text, dialog titles, table headers. *Not* Title Case.
- **ALL CAPS** is reserved for the NVIDIA wordmark and section dividers (e.g. "OVERVIEW", "ON THIS PAGE") in docs nav, never for body copy.
- Names of NVIDIA products keep their official casing (`GeForce`, `CUDA`, `Omniverse`, `RTX`).

**Punctuation.**
- No trailing punctuation on labels, headings, or button text.
- Sentences in body copy end in a period.
- Quotes are straight (`"`) inside code and in compact UI; curly quotes are fine in longer prose.

**Examples (from the library):**
- Card heading: `Card Heading`  ·  Subheading: `Card Subheading`
- Button labels: `Button` (default), `Save`, `Cancel`, `Add user`
- Empty/placeholder: `Placeholder value`
- Inline help: short, factual: `Hint text` — no smileys, no "we recommend".
- Status copy: `5 of 32 selected` · `Last updated 2 minutes ago` · `Connection lost`

**"I" vs "you".** Always "you" for the reader; system messages refer to themselves in the third person ("The server returned…") rather than "I". Never use "we" in product copy.

**Emoji.** Not used in product UI. Some Figma component names contain emoji (💎 🔄 ✏️) as Kaizen's *internal* tagging convention for slot kinds (`💎 Kind=Primary`, `🔄 Slot`, `✏️ Text`). These are author-facing markers; they never appear in rendered product surfaces. Don't ship emoji.

**Unicode glyphs as icons.** Avoid. Kaizen has a full `common/*` and `shapes/*` SVG set — use that.

**Vibe.** Sober. Operational. The aesthetic equivalent of a multimeter: precise, neutral, useful.

---

## VISUAL FOUNDATIONS

### Color

A neutral grayscale spine carries layout; NVIDIA Green (`#76B900`) is the single brand accent. Status colors are reserved for status — never for emphasis.

| Role | Token | Hex |
|---|---|---|
| Brand / Primary action | `--kui-brand-green` | `#76B900` |
| Background canvas (light) | `--kui-bg-canvas` (`N050`) | `#F2F2F2` |
| Surface (light) | `--kui-bg-mid` (`N000`) | `#FFFFFF` |
| Border default | `--kui-border-default` | `#D8D8D8` |
| Foreground strong | `--kui-fg-strong` | `#202020` |
| Foreground muted | `--kui-fg-muted` | `#5E5E5E` |
| Success | `--kui-success` | `#007D00` |
| Info | `--kui-info` | `#0D69D4` |
| Warning | `--kui-warning` | `#C54600` |
| Danger | `--kui-danger` | `#DC3528` |

A dark theme is shipped (`[data-theme="dark"]`) with `#0D0D0D` canvas and `#202020` surface — used in the Kaizen "Layers-Dark" elevation system as `.canvas / .low / .mid / .high / .higher` running from `N900 → N700`.

### Type

- **NVIDIA Sans** is the brand sans (Regular / Medium / Bold). It's proprietary, so this system substitutes **DM Sans** from Google Fonts as the closest open-source match. Drop real `.woff2` files into `fonts/` and the @font-face block in `colors_and_type.css` to swap back.
- **Inter Bold** is used for *documentation* display headings (the big `48px` titles on Kaizen's spec pages — "Anatomy", "Layout", "Behavior"). Inter is the workhorse of Kaizen *docs*, not product UI.
- **Roboto Mono** for code and tabular numerals.

Kaizen's "Text" component exposes exactly three semantic kinds:
| Kind | Size / line-height | Weight | Used for |
|---|---|---|---|
| `Heading` | 16 / 1.25 (-0.25 tracking) | 500 | Card titles, section heads |
| `Body` | 14 / 1.5 | 400 | Paragraphs, descriptions |
| `Text` | 14 / 1 | 400 | One-line UI labels, button text |

### Spacing

A named scale, identical to the Kaizen `Spacer` component family:

`xxs=2 · xs=4 · ss=6 · sm=8 · ms=12 · md=16 · ml=24 · lg=32 · xl=48 · xxl=64`

Component internals use the small end (`xs/sm/ms`); page layout uses the large end (`md/ml/lg/xl`).

### Backgrounds

- **Flat fills only.** No gradients, no noise, no patterns in the core library. The Cover Page renders on plain `#050505` with a single `#1FA18D` teal "Accent" sidebar — a recurring KUI motif for "library type" surfaces.
- **No full-bleed photography in the system.** Photography lives at the product layer (marketing, GeForce, etc.) and is always loaded via the `Image / Aspect Ratio` component, never as a background.
- **Color of imagery (when present): cool, technical, hardware-forward.** Studio-lit GPUs, datacenter shots, simulation renders. Never lifestyle, never grainy, never warm.

### Animation

- **Minimal.** Kaizen does not animate layout. State changes are color/border swaps, not motion.
- Hover/focus transitions are `120ms ease-out` on `background` and `border-color` — that's it.
- Toasts and popovers fade/slide ≤ 200 ms.
- No bounces, no spring physics, no decorative parallax.

### Hover / Focus / Press

- **Hover** on borders: `border-default` → `border-strong` (`#D8D8D8` → `#767676`).
- **Hover** on tertiary buttons / nav items: background fills with `--kui-n050` (`#F2F2F2`).
- **Focus**: 2 px solid `--kui-brand-green` outline at 2 px offset (rectangular outline, sharp corners — matches Kaizen's geometry).
- **Press** ("Active"): solid fill turns to its `-darker` variant (e.g. `#76B900` → `#5E9600`); outlined controls darken their border by one step. No scale transform.
- **Selected**: 2 px solid `--kui-brand-green` border (cards, radios, segmented controls).
- **Disabled**: foreground → `--kui-fg-disabled` (`#8F8F8F`), 60% opacity on icons, no border darkening on hover.

### Borders

1 px solid `--kui-border-default` is the workhorse. Selected = 2 px green. Focused = 2 px green offset outline. Tags can be borderless (Solid) or 1 px (Outline).

### Shadows / Elevation

Kaizen prefers **elevation by value, not by shadow**. The Elevations page demonstrates five surface levels with no shadows at all — the value of the surface alone signals depth:

| Level | Light | Dark |
|---|---|---|
| `.canvas` | N050 `#F2F2F2` | N900 `#0D0D0D` |
| `.low` | N050 | N950 `#050505` |
| `.mid` | N000 `#FFFFFF` | N900 |
| `.high` | N000 | N800 `#202020` |
| `.higher` | N000 | N700 `#343434` |

Shadows are reserved for **floating ephemeral surfaces**: tooltips, popovers, toasts, carousel buttons. Three tokens cover this: `--kui-shadow-1/2/3` from `0 1px 2px rgba(0,0,0,.08)` up to `0 4px 12px rgba(0,0,0,.12)`. Modals use a separate `--kui-overlay` (`rgba(0,0,0,.5)`) scrim called the **Blanket**.

### Protection gradients vs. capsules

Not used. Kaizen rejects the "marketing" pattern of a dark gradient under hero text. When text sits on imagery, the design uses an opaque capsule (`.bg-mid` rounded `--kui-radius-sm`) at full opacity, not a protection gradient.

### Transparency & blur

Sparingly. The only systemic uses:
- Modal scrim: `rgba(0,0,0,0.5)` (no blur).
- Carousel arrows: 80% opacity white `rgba(255,255,255,0.8)` with `--kui-shadow-2`.
- Tag backgrounds: solid pastel tints (`#FBEEFE`, `#E9F4FB`) — opaque, not translucent.

No `backdrop-filter: blur()` is used in the library.

### Corner radii

Sharp. The system intentionally avoids "soft" UI.

| Token | px | Used by |
|---|---|---|
| `--kui-radius-xs` | `2` | Inputs, datepickers |
| `--kui-radius-sm` | `4` | Buttons, cards, banners |
| `--kui-radius-md` | `5` | Demo wrappers in Figma |
| `--kui-radius-lg` | `8` | Some popovers |
| `--kui-radius-pill` | `999` | Avatars, tags-as-pills (rare) |

### Cards

- 1 px solid `#D8D8D8` border, 4 px radius, `#FFFFFF` background.
- **No shadow.** Elevation comes from the border + the canvas behind it.
- Default → Hover: border to `#8F8F8F`. Default → Selected: border to 2 px `#76B900`.
- Padding: 16 px content gutter; image fills the top half edge-to-edge.

### Layout rules

- **Fixed vs Fluid:** Top-level shells (`Page-Header`, `App-Bar`) ship in `Fluid=false` (centered max-width) and `Fluid=true` (full-bleed) variants. Pick one and stay there.
- **Breakpoints:** Kaizen library demonstrates `XS=320, SM=576, MD=768, LG=992+`. Components reflow at these breakpoints.
- **No floating UI.** No floating action buttons. No drawer over content. Side panels push content; modals scrim it.

---

## ICONOGRAPHY

Kaizen ships its own icon set, organized into namespaces:

- `common/*` — utility (`cog-fill`, `clock-fill`, `info-circle-fill`, `check-fill`, `error-fill`, `close-line`, `close-fill`, `menu-line`, `bell-line`, `home-line`, `pencil-fill`, `warning-fill`).
- `shapes/*` — geometry primitives (`chevron-down/up/left/right`, in both `-line` and `-fill` weights).
- `hardware/*` — NVIDIA-specific glyphs (`gpu-line`, `dpu-chip`).
- `maps/*` — `world-fill` (sparingly used in i18n controls).
- `social/*` — `profile-line/fill` for avatars.
- `editor/*` — `pencil-fill` and friends for inline-edit affordances.

**Style:** 16×16 viewBox, monochrome, single path per icon, `currentColor` fill, ~1.5 px effective stroke weight for the `*-line` family, solid geometry for the `*-fill` family. **Never two-tone, never multi-color.** No emoji, ever. No unicode-glyph icons.

**Sizes:** Components consume icons at 12 (small buttons), 16 (default), 20 (page header), or 24 (avatars). The library ships separate "GUI" and "Marketing" symbol families — GUI is the one you want for product UI.

**This project:** The three NVIDIA brand SVGs are copied 1:1 from Figma into `assets/logos/`. `assets/icons/cog-fill.svg` is a literal Figma copy as a reference. The full set lives in `assets/icons/kui-icons.js` as inline SVG paths — call `KUI.hydrateIcons(root)` to swap any `<i data-kui-icon="cog-fill">` placeholder with the rendered SVG, or grab the path string from `KUI.icons[name]` directly.

**Substitutions flagged:** none — every icon used in the UI kit is a Kaizen icon. If you need an icon Kaizen doesn't have, prefer **Phosphor Regular** as the closest match (same 16×16 grid, 1.5 px line, geometric construction). Flag the substitution wherever you ship it.

---

## Quick start

```html
<link rel="stylesheet" href="colors_and_type.css">
<script src="assets/icons/kui-icons.js"></script>
<div class="kui">
  <h2>Card Heading</h2>
  <p>Body copy goes here.</p>
  <button class="kui-btn kui-btn--primary">
    <i data-kui-icon="cog-fill"></i> Settings
  </button>
</div>
<script>KUI.hydrateIcons(document);</script>
```

See `ui_kits/kaizen-app/index.html` for a worked example.

---

## Caveats

- **NVIDIA Sans is substituted with DM Sans.** Replace the @font-face block in `colors_and_type.css` and drop the real `.woff2` files into `fonts/` to restore brand fidelity. DM Sans is metrically close but not identical — capital height and rhythm differ slightly.
- The icon set in `kui-icons.js` is a faithful reconstruction of the Kaizen `common/*` and `shapes/*` symbols but each path is hand-redrawn against the 16×16 grid (the originals are encoded inside Figma's binary). Visual cadence matches; sub-pixel alignment may differ.
- This project covers tokens + the most-used components (Button, Input, Card, Banner, Tag, App-Bar, Vertical-Nav, Modal, Toast, Notification, Avatar). Lower-frequency Kaizen components (Carousel, Slider, Datepicker, Pagination, Table) are not yet built — the tokens cover them.
