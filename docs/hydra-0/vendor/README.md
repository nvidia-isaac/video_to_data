# Vendored runtime dependencies

These are the exact React UMD bundles that `../support.js` pins by URL and
SHA-384 (`REACT_URL` / `REACT_SRI`, `REACT_DOM_URL` / `REACT_DOM_SRI`). They are
served locally so the page does not depend on `unpkg.com` being reachable:
`support.js` hides all page content synchronously and only restores it once
React loads, so a failed CDN fetch left a blank page with no fallback.

`../index.html` redirects the runtime to these copies via `window.__resources`,
a lookup `support.js` performs in `cdnScriptFor()` before falling back to the CDN.

| File | Upstream | SHA-384 |
|---|---|---|
| `react.production.min.js` | https://unpkg.com/react@18.3.1/umd/react.production.min.js | `sha384-DGyLxAyjq0f9SPpVevD6IgztCFlnMF6oW/XQGmfe+IsZ8TqEiDrcHkMLKI6fiB/Z` |
| `react-dom.production.min.js` | https://unpkg.com/react-dom@18.3.1/umd/react-dom.production.min.js | `sha384-gTGxhz21lVGYNMcdJOyq01Edg0jhn/c22nsx0kyqP0TxaV5WVdsSH1fSDUf5YJj1` |

Both digests were verified against the values already pinned in `support.js` at
the time of vendoring. React is MIT-licensed (Copyright (c) Meta Platforms, Inc.
and affiliates); the upstream licence text ships inside each bundle.

To update: bump the versions in `support.js`, re-download, re-verify the digests
against the new `*_SRI` constants, and update the URLs in `index.html`.
