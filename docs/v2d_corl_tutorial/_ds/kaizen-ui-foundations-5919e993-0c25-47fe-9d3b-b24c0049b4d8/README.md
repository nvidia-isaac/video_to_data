# KuiFoundationsReact (@kui/foundations-react@1.1.2)

This design system is the published @kui/foundations-react React library, bundled as a single
browser global. All 66 components are the real upstream code.

## Where things are

- `_ds_bundle.js` — the whole-DS bundle at the project root; loads every component to `window.KuiFoundationsReact`. First line is a `/* @ds-bundle: … */` metadata header.
- `styles.css` — the single stylesheet entry: it `@import`s the tokens, fonts, and component styles (`_ds_bundle.css`). Link this one file.
- `components/<group>/<Name>/<Name>.prompt.md` (example JSX + variants), `<Name>.d.ts` (types), `<Name>.html` (variant grid).
- `tokens/*.css` — CSS custom properties, names verbatim from upstream.
- `fonts/` — `@font-face` files + `fonts.css` (when the package ships fonts).

For a specific component, `read_file("components/<group>/<Name>/<Name>.prompt.md")`.

## Loading

Add these two lines to your page once (React must be on the page first):

```html
<link rel="stylesheet" href="styles.css">
<script src="_ds_bundle.js"></script>
```

Components are then available at `window.KuiFoundationsReact.*`. Mount into a dedicated child node (e.g. `<div id="ds-root">`), not the host page's own React root, so the two trees don't collide:

```jsx
const { Accordion } = window.KuiFoundationsReact;
ReactDOM.createRoot(document.getElementById('ds-root')).render(<Accordion />);
```

Wrap the tree in the provider — most components read theme/i18n from context:

```jsx
<ThemeProvider theme={"dark"} className={"nv-dark"} style={{"backgroundColor":"var(--background-color-surface-raised)","minHeight":"100vh"}}>{children}</ThemeProvider>
```

## Tokens

453 CSS custom properties from @kui/foundations-react. Names are
preserved verbatim from upstream. They are declared inside `_ds_bundle.css` (this DS ships one compiled stylesheet rather than separate token files).

- **color** (319): `--bg-color`, `--text-color`, `--border-color`, …
- **spacing** (11): `--nv-button-icon-margin`, `--nv-input-padding`, `--input-gap`, …
- **typography** (8): `--nv-accordion-label-font-size`, `--icon-font-size`, `--font-sans`, …
- **radius** (9): `--radius-none`, `--radius-sm`, `--radius-md`, …
- **shadow** (4): `--spinner-shadow-size`, `--shadow-lg`, `--shadow-md`, …
- **other** (102): `--nv-button-icon-size`, `--nv-code-snippet-custom-background`, `--menu-translate-start`, …

## Components

### kui-react
- `Accordion` — A vertically stacked group of sections the user can expand and collapse to show or hide their contents.
- `Anchor` — Interactive text that navigates the user to another page, section, or resource.
- `AnimatedChevron` — A chevron icon that animates between two orientations to indicate the open or closed state of a disclosure. Use inside expand and collapse t
- `AppBar` — The top-level masthead for an application, anchoring branding, primary navigation, and global actions. Every application has exactly one.
- `Avatar` — A small visual identifier for a user, account, or entity. Prioritizes an image when available and falls back to initials or an icon otherwis
- `Badge` — A small color-coded label that displays status, category, or metadata at a glance. Badges are
- `Banner` — A prominent, full-width inline message embedded in the page layout for system-wide announcements, page-level alerts, or important context.
- `Block` — A general-purpose block-level container for arbitrary content. Use it as a low-level layout primitive when consistent spacing and overflow b
- `Breadcrumbs` — A navigation trail that shows the user's location within the app hierarchy and links back to parent pages. Use concise labels for each level
- `Button` — A clickable element that triggers an action. Use specific verb + noun labels instead of vague text.
- `ButtonGroup` — A row of related buttons presented as a single set of actions with consistent styling.
- `Card` — A container that groups content, actions, and optional media about a single subject. Cards are fluid and grow to fill their container.
- `Checkbox` — A form control that lets users toggle a single option on or off. Supports a mixed state for representing partially-selected groups of relate
- `CodeSnippet` — Displays source code with syntax highlighting and a copy button. Pick kindinline for code samples within prose and kindblock for standalone,
- `Collapsible` — An unstyled disclosure that toggles a single block of content between expanded and collapsed states. Use to hide secondary detail behind a c
- `Combobox` — A text input paired with a filterable listbox for selecting one or more values from a list. Use when users need to search through a large se
- `DatePicker` — An input field paired with a calendar popup for selecting a single date or a date range. Use when users need to enter or browse to a specifi
- `Divider` — A horizontal or vertical line that visually separates groups of content. Use to create breaks between sections, list items, or toolbar regio
- `Dropdown` — A button that opens a menu of actions, links, or selectable options. Use for contextual actions, filters, and overflow menus.
- `Flex` — A layout primitive that arranges its children with CSS flexbox using design-token spacing.
- `FormField` — Groups a label, input, and helper text to keep form elements consistent and accessible.
- `Grid` — A two-dimensional layout primitive that arranges children into rows and columns. Use for dashboards, card listings, and any layout that need
- `Group` — A container that joins related items into a single visual unit. Reach for it when adjacent elements should read as one cohesive cluster, suc
- `Hero` — A large page banner that combines headline messaging and calls to action with imagery.
- `HorizontalNav` — A horizontal navigation bar that organizes links to top-level sections of an application or site.
- `Inline` — An inline layout primitive that renders a span with the system's spacing tokens exposed as props.
- `InputShell` — A low-level wrapper that styles any element as a KUI input. Reach for the higher-level inputs first use this only when composing a custom co
- `Label` — A text label for a form control. Prefer FormField for end-user forms this primitive is intended for composing custom field layouts.
- `List` — Groups related items into a sequence, supporting numbered, bulleted, and icon-marked formats.
- `Menu` — A list of actions, toggles, or selections. Use when the list should always be visible (sidebars, settings panels).
- `Modal` — A dialog that interrupts the page to capture the user's attention until they act on or dismiss its content. Use for confirmations, focused t
- `Notification` — A non-blocking overlay message that delivers detailed feedback in the user's peripheral vision, optionally with actions. Reach for it when t
- `PageHeader` — Top-of-page banner that establishes context with a title, breadcrumbs, and page-level actions.
- `Pagination` — Navigation control that splits a long list of items across multiple pages. Pair with tables and grids that exceed a single page of content.
- `Panel` — Container that groups related content and actions with a background, border, and consistent padding. Use for general-purpose surfaces on das
- `Popover` — A floating panel that displays rich content, options, or actions anchored to a trigger.
- `ProgressBar` — Visual indicator for long-running operations like uploads, downloads, and other loading states.
- `RadioGroup` — A set of mutually exclusive options where the user picks exactly one. Use when the available options can comfortably fit on screen at once.
- `RangeSlider` — A two-thumb slider for selecting a minimum and maximum value within a range. Use for filtering by price, time window, or any min/max selecti
- `SegmentedControl` — A horizontal group of mutually exclusive options where selection takes effect immediately. Use for switching between alternate views or mode
- `Select` — An input that lets the user pick one or more values from a predefined list.
- `SidePanel` — A dialog panel that slides in from the left or right edge to show supplementary content without leaving the current view.
- `Skeleton` — A placeholder block that mimics the shape of content while it loads. Match each skeleton's size and shape to the real content as closely as 
- `Slider` — A single-value range input rendered along a bar. Use for continuous settings like volume, brightness, or threshold values where the relative
- `Spinner` — An animated indicator for indeterminate work. Use when an operation is expected to take roughly one to five seconds reach for a progress bar
- `Stack` — A flex container that arranges its children in a column by default, with an optional divider between items.
- `StatusIndicator` — A small colored dot that signals a new change or short-lived status.
- `StatusMessage` — A full-area centered message displayed when there is no content to show  empty states, error pages, permission denied, not found. Replaces t
- `Stepper` — A progress tracker for moving through a fixed, ordered set of steps.
- `Switch` — A two-state toggle for binary settings whose change takes effect immediately. Commonly used for preferences and feature toggles.
- `Table` — Organizes sets of structured data into columns and rows so users can scan, compare, and analyze information.
- `Tabs` — Organizes related content into sections that share the same context and switches between them in place. Use for managing settings, comparing
- `Tag` — A small label that categorizes or groups an item with text and optional icons. Use for classification and filtering reach for Badge to commu
- `Text` — Renders text with the design system's typography tokens applied.
- `TextArea` — A multi-line text input for free-form responses such as descriptions, comments, or notes.
- `TextInput` — A single-line input for text, numbers, or special symbols. Commonly used in forms switch to TextArea when the response is likely to wrap.
- `ThemeProvider` — Establishes a theme context for descendant components, controlling color theme and density. Nest providers to scope different settings to sp
- `Toast` — A small, temporary overlay that gives the user non-critical feedback after an action. Reach for it when the message is informational and can
- `Tooltip` — Reveals a short, supplemental label when the user hovers or focuses an element. Use to clarify icon-only controls or surface non-essential c
- `TreeNav` — A hierarchical navigation list that lets users move through links at different levels, similar to a directory tree
- `Upload` — A file picker with drag-and-drop support that renders previews and basic management controls for the selected files.
- `VerticalNav` — A vertical navigation menu of single links and nested link groups, typically rendered in a sidebar. Use as section-level navigation paired w

### brandlogo
- `AppBarLogo` — A responsive NVIDIA lockup for use inside AppBar's slotStart. Renders the logo mark on its own at small viewports and switches to the full h
- `HorizontalLogo` — The NVIDIA eye-mark paired with the wordmark, laid out horizontally. Use as the primary lockup in AppBars, page headers, and most branding c
- `LogoOnly` — The NVIDIA eye-mark rendered on its own without the wordmark. Use in space-constrained contexts such as compact AppBars, favicons, and mobil
- `VerticalLogo` — The NVIDIA eye-mark stacked above the wordmark. Use for splash screens, marketing surfaces, and other portrait-oriented branding where the h
