# Cohere Design System

A faithful recreation of Cohere's web design language — a sober enterprise-AI command center with editorial restraint. Austere black-and-white UI punctuated by bursts of tactile brand imagery: dark product environments, coral editorial taxonomy, blue research links, and monumental display type with a research-lab cadence.

> **Source:** built from a written brand brief (colors, typography, layout, components, do's/don'ts). **No codebase or Figma was attached** — there are no upstream source links to record. If a Cohere codebase or Figma becomes available, re-derive component internals and imagery from it; this system encodes the documented spec, not proprietary source.

---

## Brand context

Cohere builds frontier AI models and the secure enterprise platform to deploy them. The web presence spans three tonal surfaces:

- **Marketing / home** — huge typographic declarations over white canvas, photography + dark product mockups, monochrome trust logos, generous empty space.
- **Product / solutions** — deep green-black and dark navy full-width bands; agent-console mockups with status chips and integration badges.
- **Editorial (blog / research)** — publishing-system clarity: oversized coral taxonomy chips, search fields, rule-separated dense lists, pale technical washes, blue links.

The signature is the mix of restrained UI chrome with media-led color. The interface shell stays black/white/stone; color arrives through imagery and a few editorial accents — never as decorative surface fills.

---

## CONTENT FUNDAMENTALS

**Voice.** Measured, technical, enterprise-credible. Confidence through restraint — no hype, no exclamation, no emoji. Reads like a research lab that also ships product.

**Person.** Addresses the reader as **you** ("deployed wherever your data lives"), refers to the company as **we / Cohere**. Customer outcomes are stated plainly, not sold.

**Casing.** Sentence case for headlines and body ("The AI platform built for enterprise"). UPPERCASE reserved for mono system/category labels ("COMMAND", "RESEARCH", "AI MOVES FAST"). Product names are proper-cased (Command, Embed, Rerank, North).

**Headlines.** Short, declarative, often a complete claim: "Security at the foundation", "The era of AI agents", "Build agents that act securely across your enterprise". One oversized headline per page; everything else settles into 16–24px copy.

**Body.** Concrete and specific — capabilities, deployment, security, citations. Avoids adjectives-as-argument. Example lead: "Secure, private, and adaptable models and agents — deployed wherever your data lives."

**CTAs.** Verb-first and plain: "Request a demo", "Explore products", "Get started", "Try the Playground". Primary CTA is the single highest-priority action; the companion is an underlined text link.

**Editorial micro-labels.** Mono uppercase kickers sit above headings ("THE COHERE BLOG", "GET IN TOUCH"). The newsletter uses the recurring coral label "AI moves fast".

**No emoji.** Iconography is thin-line geometric, never emoji or decorative unicode.

---

## VISUAL FOUNDATIONS

**Color.** White is the default canvas. Dark **deep green (#003c33)** and **navy (#071829)** arrive as full-width product/solution bands; **near-black (#17171c)** for primary CTAs, agent consoles, and the footer. Editorial accents are **coral (#ff7759)** for blog taxonomy and **action blue (#1863dc)** for research/editorial links. Warm neutrals — **stone (#eeece7)** and pale green/blue washes — back media and CTA sections. *Never* turn coral or blue into broad surface fills.

**Type.** Display/body/mono split. **CohereText** (display) for monumental hero/product headlines — tight (line-height 1.0) with negative tracking (−1.2 to −1.92px), carved not airy. **Unica77 Cohere Web** (body) for section headings down to 16px UI copy. **CohereMono** for uppercase technical labels (+0.28px tracking). Weights stay light — 400 with occasional 500; size, spacing, and surface contrast do the hierarchy work, not bold.

**Spacing.** 8px base with documented one-offs (2, 6, 10, 22, 28, 36, 56, 60, 64, 80…). Whitespace is a *trust signal* — dramatic vertical intervals (~120px) separate brand claim, customer proof, product proof, and CTA. Dense content appears only where IA needs it (research rows, blog grids, form fields).

**Backgrounds.** Flat color fields only. No gradient UI fills. Gradients/color fields are **media-led** — abstract 3D hero imagery, particle fields, product video posters, dark green-to-black environments. Imagery sits as *rounded cards with visible corners* (8px and 22px dominate), not full-bleed text backdrops except in CTA bands.

**Imagery vibe.** Tactile, premium, enterprise. Warm photography, dark product environments, abstract 3D renders. Color is carried by media while the UI shell stays restrained.

**Corner radii.** xs 4 (search, thumbnails) · sm 8 (chips, cards, dialogs) · md 16 (medium cards) · lg 22 (signature media cards) · xl 30 (filter pills) · pill 32 (CTAs) · full (round status). Never round major media below 8px.

**Cards.** Rounded but not cute. Flat — no drop shadows. Containment via surface tone (stone/white) and a thin 1px hairline border (`#d9d9dd` / `#f2f2f2`), or a top rule only. Don't box every section; Cohere uses unframed rows and rules liberally.

**Elevation.** Mostly flat. Depth = surface alternation + media contrast + rounded corners + thin borders. Four levels: Flat → Bordered → Media lift (rounded media over contrasting field) → Dark product field.

**Borders & rules.** Thin 1px hairlines separate research rows and divide sections; translucent white rules (`rgba(255,255,255,0.16)`) on dark bands.

**Buttons.** Primary = near-black pill (white pill on dark), 12×24px padding, 32px radius. Secondary = underlined text link, no fill. Outline = transparent pill with 1px border, 30px radius (taxonomy/filters).

**Animation.** Restrained. Standard ease `cubic-bezier(0.4,0,0.2,1)`, ~120–200ms. Hover dims to ~78% opacity; links underline. No bounces, no infinite decorative loops. Respect reduced-motion.

**Transparency & blur.** Used sparingly — translucent white surfaces (`rgba(255,255,255,0.03–0.06)`) inside dark bands for nested console panels. No heavy glassmorphism.

---

## ICONOGRAPHY

Cohere uses **thin-line geometric** icons and illustrations — a custom icon font plus line illustrations on research and capability surfaces. The proprietary set is not bundled here.

- **Substitute:** [Lucide](https://lucide.dev) via CDN (`lucide@0.460.0`), rendered at **stroke-width 1.25–1.5** to match the thin geometric line weight. **Flag:** swap for Cohere's licensed icon set in production.
- **Usage:** capability/feature markers, agent-console glyphs (bot, sparkles, plug, shield-check, workflow), search affordance, footer newsletter arrow. The `ui_kits/marketing/Icon.jsx` helper wraps Lucide for React surfaces.
- **No emoji, no decorative unicode** as icons. The only symbolic glyphs used inline are the checkmark (✓) in product bullets and a small chevron (▾) on selects / arrow (→) on links — kept minimal.
- **Logo:** the `cohere` wordmark is rendered as a lowercase text treatment in the display face — a **placeholder** for the licensed logo asset. Replace with the official SVG/PNG for production.

---

## Index / manifest

**Root**
- `styles.css` — global entry (consumers link this); `@import`s all tokens + fonts.
- `readme.md` — this guide.
- `SKILL.md` — Agent Skill manifest for download/Claude Code use.
- `tokens/` — `fonts.css`, `colors.css`, `typography.css`, `spacing.css` (spacing + radius + elevation + motion).

**Foundations** (`foundations/*.card.html`) — Design System tab specimen cards: Colors (Brand, Surface, Text, Semantic), Type (Display, Headings, Body, Mono), Spacing (Scale, Radius), Brand (Elevation, Dark Bands, Wordmark).

**Components** (`components/<group>/`)
- `core/` — **Button**, **Chip**, **Badge**, **MonoLabel**, **Card** + **CardBullet**
- `forms/` — **Input**, **Select**
- `navigation/` — **AnnouncementBar**, **TopNav**

Each is `<Name>.jsx` + `<Name>.d.ts` + `<Name>.prompt.md`, with one `*.card.html` per group.

**UI kits** (`ui_kits/`)
- `marketing/` — interactive marketing website: `Home` (hero + trust strip + dark agent-console band), `Blog` (coral taxonomy + article grid), `Research` (filter pills + rule-separated publication table), `Contact` (form card on green + footer newsletter). Entry: `index.html`.

**Generated (do not edit):** `_ds_bundle.js`, `_ds_manifest.json`, `_adherence.oxlintrc.json`.

---

## Known gaps & substitutions

- **Fonts:** CohereText / Unica77 Cohere Web / CohereMono are proprietary and not bundled. Substituted with Space Grotesk / Inter / Space Mono (Google Fonts), aliased to the real family names. **Swap in licensed files when available.**
- **Icons:** Lucide stands in for Cohere's custom icon font. **Flag for replacement.**
- **Logo:** text wordmark placeholder — no licensed logo asset.
- **Imagery:** photography, abstract 3D renders, and product video posters are represented as stone/wash color-field placeholders. Drop in real assets for production.
