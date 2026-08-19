/* @ds-bundle: {"format":3,"namespace":"NVIDIAKaizenDesignSystem_b84c2d","components":[],"sourceHashes":{"assets/icons/kui-icons.js":"e836cdcab084","kaizen-react/console-app.jsx":"120f061c653b","kaizen-react/console-parts.jsx":"688444cc6724","kaizen-react/icons.js":"531f99e35367","kaizen-react/kui-controls.js":"a5caeaf58f19","kaizen-react/kui-core.js":"c1d234952ba2","ui_kits/kaizen-app/AppBar.jsx":"9d16d7c8da25","ui_kits/kaizen-app/ClusterDetails.jsx":"add612029e56","ui_kits/kaizen-app/ClusterTable.jsx":"0190882bda98","ui_kits/kaizen-app/Modal.jsx":"65f19b398fe0","ui_kits/kaizen-app/Sidebar.jsx":"a1ffcf31480d"},"inlinedExternals":[],"unexposedExports":[]} */

(() => {

const __ds_ns = (window.NVIDIAKaizenDesignSystem_b84c2d = window.NVIDIAKaizenDesignSystem_b84c2d || {});

const __ds_scope = {};

(__ds_ns.__errors = __ds_ns.__errors || []);

// assets/icons/kui-icons.js
try { (() => {
/* Kaizen — KUI icon set
   ----------------------
   Reconstructed from the Kaizen "common/" and "shapes/" icon symbols in
   the KUI v11 Figma library. All icons are 16×16, monochrome, with a
   consistent visual weight that matches the library.
   Usage:
     <i data-kui-icon="cog-fill"></i>
     KUI.hydrateIcons(document);  // replaces <i data-kui-icon> with SVG
   Or import the raw path strings (KUI.icons) and inline yourself.
*/
(function (global) {
  // Each path uses the 16×16 viewBox and currentColor for fills.
  const ICONS = {
    /* --- common (filled, ~14×14 inside 16×16) --- */
    "cog-fill": '<path fill="currentColor" fill-rule="evenodd" d="M6.5 1.5h3l.4 1.7a5 5 0 0 1 1.1.66l1.66-.55 1.5 2.6-1.25 1.2c.06.32.09.65.09.99s-.03.67-.09.99l1.25 1.2-1.5 2.6-1.66-.55a5 5 0 0 1-1.1.66L9.5 14.5h-3l-.4-1.7a5 5 0 0 1-1.1-.66l-1.66.55-1.5-2.6 1.25-1.2A5 5 0 0 1 3 8c0-.34.03-.67.09-.99L1.84 5.81l1.5-2.6 1.66.55a5 5 0 0 1 1.1-.66L6.5 1.5ZM8 10.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z"/>',
    "info-circle-fill": '<path fill="currentColor" fill-rule="evenodd" d="M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13Zm.75 3.5h-1.5v1.5h1.5V5Zm0 2.5h-1.5V12h1.5V7.5Z"/>',
    "check-fill": '<path fill="currentColor" fill-rule="evenodd" d="M14.354 4.354 6 12.707 1.646 8.354l.708-.708L6 11.293l7.646-7.647.708.708Z"/>',
    "check-circle-fill": '<path fill="currentColor" fill-rule="evenodd" d="M8 1.5a6.5 6.5 0 1 1 0 13 6.5 6.5 0 0 1 0-13Zm3.4 4.2L7.1 10 4.6 7.5l-.7.7 3.2 3.2 5-5-.7-.7Z"/>',
    "error-fill": '<path fill="currentColor" fill-rule="evenodd" d="M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13ZM7.25 4v5h1.5V4h-1.5Zm0 6v1.5h1.5V10h-1.5Z"/>',
    "warning-fill": '<path fill="currentColor" fill-rule="evenodd" d="M8 1.5 14.928 13.5H1.072L8 1.5Zm-.75 4v4h1.5v-4h-1.5Zm0 5V12h1.5v-1.5h-1.5Z"/>',
    "close-line": '<path fill="currentColor" fill-rule="evenodd" d="M8 7.293 12.646 2.646l.708.708L8.707 8l4.647 4.646-.708.708L8 8.707l-4.646 4.647-.708-.708L7.293 8 2.646 3.354l.708-.708L8 7.293Z"/>',
    "close-fill": '<path fill="currentColor" fill-rule="evenodd" d="M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13ZM5.5 5l-.5.5L7.5 8 5 10.5l.5.5L8 8.5 10.5 11l.5-.5L8.5 8 11 5.5l-.5-.5L8 7.5 5.5 5Z"/>',
    "menu-line": '<path fill="currentColor" fill-rule="evenodd" d="M2 3.5h12v1H2v-1Zm0 4h12v1H2v-1Zm0 4h12v1H2v-1Z"/>',
    "search-line": '<path fill="currentColor" fill-rule="evenodd" d="M7 1.5a5.5 5.5 0 1 0 3.4 9.81l3.15 3.14.7-.7-3.14-3.15A5.5 5.5 0 0 0 7 1.5Zm0 1a4.5 4.5 0 1 1 0 9 4.5 4.5 0 0 1 0-9Z"/>',
    "bell-line": '<path fill="currentColor" fill-rule="evenodd" d="M8 1.5a4 4 0 0 0-4 4v3l-1.5 2.5h11L12 8.5v-3a4 4 0 0 0-4-4Zm0 1a3 3 0 0 1 3 3v3.27L11.85 10H4.15L5 8.77V5.5a3 3 0 0 1 3-3ZM6.5 12.5a1.5 1.5 0 0 0 3 0h-3Z"/>',
    "home-line": '<path fill="currentColor" fill-rule="evenodd" d="m8 1.5 6.5 5.5h-2v6h-3v-4h-3v4h-3v-6h-2L8 1.5Zm0 1.31L4.5 6V12h1v-4h5v4h1V6L8 2.81Z"/>',
    "user-line": '<path fill="currentColor" fill-rule="evenodd" d="M8 2.5a2.75 2.75 0 1 0 0 5.5 2.75 2.75 0 0 0 0-5.5Zm0 1a1.75 1.75 0 1 1 0 3.5 1.75 1.75 0 0 1 0-3.5ZM3.5 14a4.5 4.5 0 0 1 9 0h-1a3.5 3.5 0 0 0-7 0h-1Z"/>',
    "user-fill": '<path fill="currentColor" fill-rule="evenodd" d="M8 2.5a2.75 2.75 0 1 1 0 5.5 2.75 2.75 0 0 1 0-5.5ZM12.5 14a4.5 4.5 0 0 0-9 0h9Z"/>',
    "clock-fill": '<path fill="currentColor" fill-rule="evenodd" d="M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13ZM7.5 4v4.5h4v-1H8.5V4h-1Z"/>',
    "pencil-fill": '<path fill="currentColor" fill-rule="evenodd" d="m11.293 1.793 2.914 2.914-9 9H2.293v-2.914l9-9Zm-1.207 2.621L3 11.5V13h1.5l7.086-7.086-1.5-1.5Z"/>',
    "plus-line": '<path fill="currentColor" fill-rule="evenodd" d="M7.5 2h1v5.5H14v1H8.5V14h-1V8.5H2v-1h5.5V2Z"/>',
    "minus-line": '<path fill="currentColor" fill-rule="evenodd" d="M2 7.5h12v1H2v-1Z"/>',
    "external-link": '<path fill="currentColor" fill-rule="evenodd" d="M9 2h5v5h-1V3.707L8.354 8.354l-.708-.708L12.293 3H9V2ZM3 4h4v1H4v7h7V9h1v4H3V4Z"/>',
    "more-horizontal": '<path fill="currentColor" fill-rule="evenodd" d="M3 8a1.25 1.25 0 1 1 2.5 0A1.25 1.25 0 0 1 3 8Zm3.75 0a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0Zm3.75 0a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0Z"/>',
    "download-line": '<path fill="currentColor" fill-rule="evenodd" d="M7.5 2h1v6.293l2.146-2.147.708.708L8 10.207 4.646 6.854l.708-.708L7.5 8.293V2ZM3 12h10v1H3v-1Z"/>',
    "filter-line": '<path fill="currentColor" fill-rule="evenodd" d="M2 3h12v1L9.5 9v4l-3-1.5V9L2 4V3Z"/>',
    "gpu-line": '<path fill="currentColor" fill-rule="evenodd" d="M1.5 4h13v8h-13V4Zm1 1v6h11V5h-11Zm1.5 1h3v4H4V6Zm4 0h3v4H8V6Z"/>',
    "world-fill": '<path fill="currentColor" fill-rule="evenodd" d="M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13Zm0 1c1.1 0 2.1.9 2.8 2.3-1.7.4-3.9.4-5.6 0C5.9 3.4 6.9 2.5 8 2.5Zm-4 4.6c.6.2 1.3.3 2 .4v1c-.7.1-1.4.2-2 .4-.1-.4-.2-.9-.2-1.4s.1-.9.2-1.4Zm6 0c.1.5.2.9.2 1.4 0 .5-.1 1-.2 1.4-.6-.2-1.3-.3-2-.4v-1c.7-.1 1.4-.2 2-.4Z"/>',
    /* --- shapes --- */
    "chevron-down-line": '<path fill="currentColor" fill-rule="evenodd" d="M3.646 5.646 8 10l4.354-4.354.708.708L8 11.414 2.939 6.354l.707-.708Z"/>',
    "chevron-up-line": '<path fill="currentColor" fill-rule="evenodd" d="M12.354 10.354 8 6l-4.354 4.354-.708-.708L8 4.586l5.061 5.061-.707.707Z"/>',
    "chevron-left-line": '<path fill="currentColor" fill-rule="evenodd" d="M10.354 3.646 6 8l4.354 4.354-.708.708L4.586 8l5.061-5.061.707.707Z"/>',
    "chevron-right-line": '<path fill="currentColor" fill-rule="evenodd" d="M5.646 12.354 10 8 5.646 3.646l.708-.708L11.414 8l-5.061 5.061-.707-.707Z"/>',
    "chevron-down-fill": '<path fill="currentColor" fill-rule="evenodd" d="M3 5.5h10L8 11 3 5.5Z"/>',
    "chevron-right-fill": '<path fill="currentColor" fill-rule="evenodd" d="M5.5 3v10L11 8 5.5 3Z"/>'
  };
  function renderIcon(name, opts = {}) {
    const path = ICONS[name];
    if (!path) return "";
    const size = opts.size || 16;
    const cls = opts.class ? ` class="${opts.class}"` : "";
    const fill = opts.color ? ` style="color:${opts.color}"` : "";
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 16 16" aria-hidden="true"${cls}${fill}>${path}</svg>`;
  }
  function hydrateIcons(root = document) {
    root.querySelectorAll("[data-kui-icon]").forEach(el => {
      const name = el.getAttribute("data-kui-icon");
      const size = el.getAttribute("data-size") || 16;
      const svg = renderIcon(name, {
        size
      });
      if (svg) el.innerHTML = svg;
    });
  }
  const api = {
    icons: ICONS,
    renderIcon,
    hydrateIcons
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  global.KUI = Object.assign(global.KUI || {}, api);
})(typeof window !== "undefined" ? window : globalThis);
})(); } catch (e) { __ds_ns.__errors.push({ path: "assets/icons/kui-icons.js", error: String((e && e.message) || e) }); }

// kaizen-react/console-app.jsx
try { (() => {
/* ==========================================================================
   Kaizen Console — App shell (composed entirely from window.KUI components)
   ========================================================================== */
(function () {
  const {
    useState,
    useMemo
  } = React;
  const I = window.KIcons;
  const {
    ThemeProvider,
    Text,
    Flex,
    Stack,
    Grid,
    Badge,
    Tag,
    Avatar,
    ProgressBar,
    StatusIndicator,
    Table,
    TableToolbar,
    Button,
    TextInput,
    Switch,
    AppBar,
    AppBarLogo,
    AppBarExpanderButton,
    HorizontalNav,
    Anchor,
    VerticalNav,
    Banner,
    Breadcrumbs,
    PageHeader,
    Modal,
    ModalCloseButton
  } = window.KUI;
  const {
    CLUSTERS,
    STATUS_COLOR,
    STATUS_DOT,
    fmt,
    NAV_ITEMS,
    HNAV_ITEMS,
    Kpi,
    ClusterDetails
  } = window.ConsoleParts;
  function App() {
    const [theme, setTheme] = useState(() => localStorage.getItem("kui-theme") || "light");
    const [selectedId, setSelectedId] = useState("kp01");
    const [checked, setChecked] = useState({}); // bulk-select set
    const [toDelete, setToDelete] = useState(null); // modal target
    const [query, setQuery] = useState("");
    const [bannerOpen, setBannerOpen] = useState(true); // info banner dismiss state

    function setThemeP(t) {
      setTheme(t);
      localStorage.setItem("kui-theme", t);
    }
    const selected = CLUSTERS.find(c => c.id === selectedId);
    const checkedIds = Object.keys(checked).filter(k => checked[k]);
    const filtered = useMemo(() => CLUSTERS.filter(c => (c.name + c.region + c.type).toLowerCase().includes(query.toLowerCase())), [query]);

    /* ----- table rows (rich cells, KUI components inside) ----- */
    const columns = ["Cluster", "Region", "Utilization", "Jobs", {
      children: "Spend MTD",
      sortDir: "desc"
    }, "Status", ""];
    const rows = filtered.map(c => {
      const util = Math.round(c.used / c.total * 100);
      return {
        id: c.id,
        selected: !!checked[c.id],
        onRowSelect: () => setChecked(s => ({
          ...s,
          [c.id]: !s[c.id]
        })),
        cells: [{
          children: /*#__PURE__*/React.createElement(Flex, {
            gap: 2,
            align: "center"
          }, /*#__PURE__*/React.createElement(Avatar, {
            size: "sm",
            fallback: /*#__PURE__*/React.createElement(I.Gpu, {
              size: 14
            })
          }), /*#__PURE__*/React.createElement("button", {
            onClick: e => {
              e.stopPropagation();
              setSelectedId(c.id);
            },
            style: {
              all: "unset",
              cursor: "pointer",
              whiteSpace: "nowrap",
              fontFamily: "var(--font-mono)",
              fontWeight: 500,
              color: c.id === selectedId ? "var(--text-color-accent-green)" : "var(--text-color-primary)"
            }
          }, c.name))
        }, c.region, {
          children: /*#__PURE__*/React.createElement(Flex, {
            gap: 2,
            align: "center",
            style: {
              minWidth: 150
            }
          }, /*#__PURE__*/React.createElement("div", {
            style: {
              flex: 1
            }
          }, /*#__PURE__*/React.createElement(ProgressBar, {
            value: c.used,
            max: c.total
          })), /*#__PURE__*/React.createElement(Text, {
            kind: "mono/sm",
            color: "secondary",
            style: {
              width: 34,
              textAlign: "right"
            }
          }, util, "%"))
        }, String(c.jobs), {
          children: /*#__PURE__*/React.createElement("span", {
            style: {
              fontFamily: "var(--font-mono)"
            }
          }, fmt(c.cost))
        }, {
          children: /*#__PURE__*/React.createElement(Badge, {
            color: STATUS_COLOR[c.status],
            slotLeft: /*#__PURE__*/React.createElement(StatusIndicator, {
              status: STATUS_DOT[c.status]
            })
          }, c.status)
        }, {
          children: /*#__PURE__*/React.createElement(Flex, {
            gap: 1,
            justify: "flex-end"
          }, /*#__PURE__*/React.createElement(Button, {
            kind: "tertiary",
            color: "neutral",
            size: "small",
            iconOnly: true,
            "aria-label": "Edit",
            onClick: e => {
              e.stopPropagation();
              setSelectedId(c.id);
            }
          }, /*#__PURE__*/React.createElement(I.Pencil, {
            size: 14
          })), /*#__PURE__*/React.createElement(Button, {
            kind: "tertiary",
            color: "danger",
            size: "small",
            iconOnly: true,
            "aria-label": "Terminate",
            onClick: e => {
              e.stopPropagation();
              setToDelete(c);
            }
          }, /*#__PURE__*/React.createElement(I.Trash, {
            size: 14
          })))
        }]
      };
    });
    const totalGpus = CLUSTERS.reduce((a, c) => a + c.total, 0);
    const usedGpus = CLUSTERS.reduce((a, c) => a + c.used, 0);
    const totalJobs = CLUSTERS.reduce((a, c) => a + c.jobs, 0);
    const spend = CLUSTERS.reduce((a, c) => a + c.cost, 0);
    return /*#__PURE__*/React.createElement(ThemeProvider, {
      theme: theme,
      density: "standard",
      style: {
        height: "100%"
      }
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        display: "grid",
        gridTemplateRows: "56px 1fr",
        height: "100%"
      }
    }, /*#__PURE__*/React.createElement(AppBar, {
      slotStart: /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(AppBarExpanderButton, null), /*#__PURE__*/React.createElement(AppBarLogo, null), /*#__PURE__*/React.createElement("span", {
        className: "kui-appbar__divider"
      }), /*#__PURE__*/React.createElement(Anchor, {
        kind: "standalone",
        href: "#",
        textKind: "inherit",
        className: "kui-appbar__product"
      }, "Kaizen Console")),
      slotEnd: /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
        style: {
          width: 240
        }
      }, /*#__PURE__*/React.createElement(TextInput, {
        placeholder: "Search clusters, jobs\u2026",
        value: query,
        onChange: e => setQuery(e.target.value),
        slotLeft: /*#__PURE__*/React.createElement(I.MagnifyingGlass, {
          size: 16
        }),
        slotRight: /*#__PURE__*/React.createElement("kbd", {
          className: "kui-kbd"
        }, "\u2318K")
      })), /*#__PURE__*/React.createElement(Button, {
        kind: "tertiary",
        color: "neutral",
        size: "medium",
        iconOnly: true,
        "aria-label": "Notifications",
        style: {
          position: "relative"
        }
      }, /*#__PURE__*/React.createElement(I.Bell, null), /*#__PURE__*/React.createElement("span", {
        className: "kui-badge kui-badge--count",
        style: {
          position: "absolute",
          top: 2,
          right: 2
        }
      }, "3")), /*#__PURE__*/React.createElement(Switch, {
        checked: theme === "dark",
        onChange: v => setThemeP(v ? "dark" : "light"),
        "aria-label": "Toggle dark theme"
      }), /*#__PURE__*/React.createElement("span", {
        style: {
          display: "inline-flex",
          color: "var(--text-color-secondary)",
          marginRight: 4
        }
      }, theme === "dark" ? /*#__PURE__*/React.createElement(I.Moon, {
        size: 16
      }) : /*#__PURE__*/React.createElement(I.SunHigh, {
        size: 16
      })), /*#__PURE__*/React.createElement(Avatar, {
        fallback: "AN",
        brand: true
      }))
    }, /*#__PURE__*/React.createElement(HorizontalNav, {
      defaultValue: "clusters",
      items: HNAV_ITEMS
    })), /*#__PURE__*/React.createElement("div", {
      style: {
        display: "grid",
        gridTemplateColumns: "248px minmax(0,1fr) 360px",
        minHeight: 0
      }
    }, /*#__PURE__*/React.createElement(VerticalNav, {
      items: NAV_ITEMS,
      footer: /*#__PURE__*/React.createElement(React.Fragment, null, "Kaizen Console \xB7 KUI v11", /*#__PURE__*/React.createElement("br", null), "v1.0.1")
    }), /*#__PURE__*/React.createElement("main", {
      style: {
        overflow: "auto",
        padding: "var(--spacing-density-2xl)",
        display: "flex",
        flexDirection: "column",
        gap: "var(--spacing-density-xl)"
      }
    }, /*#__PURE__*/React.createElement(PageHeader, {
      slotBreadcrumbs: /*#__PURE__*/React.createElement(Breadcrumbs, {
        items: [{
          children: /*#__PURE__*/React.createElement("a", {
            href: "#"
          }, "Workspaces")
        }, {
          children: /*#__PURE__*/React.createElement("a", {
            href: "#"
          }, "acme-research")
        }, "Clusters"]
      }),
      slotSubheading: "Workspace \xB7 acme-research",
      slotHeading: /*#__PURE__*/React.createElement(Flex, {
        gap: 2,
        align: "center"
      }, /*#__PURE__*/React.createElement(Text, {
        as: "h1",
        kind: "title/lg"
      }, "Clusters"), /*#__PURE__*/React.createElement(Badge, {
        color: "gray"
      }, CLUSTERS.length)),
      slotDescription: "Manage GPU compute clusters across regions. Select a cluster to view utilization and launch workloads.",
      slotActions: /*#__PURE__*/React.createElement(Flex, {
        gap: 2,
        align: "end"
      }, /*#__PURE__*/React.createElement(Button, {
        kind: "secondary",
        color: "neutral",
        size: "medium"
      }, /*#__PURE__*/React.createElement(I.Download, {
        size: 14
      }), " Export"), /*#__PURE__*/React.createElement(Button, {
        color: "brand",
        kind: "primary",
        size: "medium"
      }, /*#__PURE__*/React.createElement(I.Add, {
        size: 14
      }), " New cluster"))
    }), bannerOpen && /*#__PURE__*/React.createElement(Banner, {
      status: "info",
      kind: "inline",
      slotActions: /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Button, {
        kind: "secondary",
        size: "tiny"
      }, "View plan")),
      onClose: () => setBannerOpen(false),
      slotSubheading: "Migrate workloads from the cluster details panel."
    }, "H200 instances are now available in us-east-1"), /*#__PURE__*/React.createElement(Grid, {
      columns: 4,
      gap: 3
    }, /*#__PURE__*/React.createElement(Kpi, {
      label: "Total clusters",
      value: String(CLUSTERS.length),
      delta: "+1 this week",
      deltaColor: "success",
      icon: /*#__PURE__*/React.createElement(I.Datacenter, {
        size: 16
      })
    }), /*#__PURE__*/React.createElement(Kpi, {
      label: "GPUs allocated",
      value: `${usedGpus} / ${totalGpus}`,
      delta: `${Math.round(usedGpus / totalGpus * 100)}% capacity`,
      icon: /*#__PURE__*/React.createElement(I.Gpu, {
        size: 16
      })
    }), /*#__PURE__*/React.createElement(Kpi, {
      label: "Active jobs",
      value: String(totalJobs),
      delta: "6 queued",
      icon: /*#__PURE__*/React.createElement(I.Workspace, {
        size: 16
      })
    }), /*#__PURE__*/React.createElement(Kpi, {
      label: "Spend MTD",
      value: fmt(spend),
      delta: "\u22124.2% vs last month",
      deltaColor: "success",
      icon: /*#__PURE__*/React.createElement(I.ArrowDown, {
        size: 16
      })
    })), checkedIds.length > 0 ? /*#__PURE__*/React.createElement(TableToolbar, {
      showBulkActionsToolbar: true,
      slotBulkActions: /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Text, {
        kind: "body/semibold/md"
      }, checkedIds.length, " selected"), /*#__PURE__*/React.createElement("div", {
        style: {
          flex: 1
        }
      }), /*#__PURE__*/React.createElement(Button, {
        kind: "tertiary",
        color: "neutral",
        size: "small"
      }, /*#__PURE__*/React.createElement(I.Refresh, {
        size: 14
      }), " Restart"), /*#__PURE__*/React.createElement(Button, {
        kind: "tertiary",
        color: "danger",
        size: "small"
      }, /*#__PURE__*/React.createElement(I.Trash, {
        size: 14
      }), " Terminate"), /*#__PURE__*/React.createElement(Button, {
        kind: "tertiary",
        color: "neutral",
        size: "small",
        onClick: () => setChecked({})
      }, "Clear"))
    }) : /*#__PURE__*/React.createElement(TableToolbar, null, /*#__PURE__*/React.createElement(Text, {
      kind: "body/regular/md",
      color: "secondary"
    }, filtered.length, " clusters"), /*#__PURE__*/React.createElement(Flex, {
      gap: 2
    }, /*#__PURE__*/React.createElement(Button, {
      kind: "secondary",
      color: "neutral",
      size: "small"
    }, /*#__PURE__*/React.createElement(I.Filter, {
      size: 14
    }), " Filter"), /*#__PURE__*/React.createElement(Button, {
      kind: "secondary",
      color: "neutral",
      size: "small"
    }, /*#__PURE__*/React.createElement(I.Sort, {
      size: 14
    }), " Sort"))), /*#__PURE__*/React.createElement(Table, {
      hoverableRows: true,
      layout: "auto",
      columns: columns,
      rows: rows
    })), /*#__PURE__*/React.createElement(ClusterDetails, {
      cluster: selected,
      onDelete: setToDelete
    }))), /*#__PURE__*/React.createElement(Modal, {
      open: !!toDelete,
      onOpenChange: o => {
        if (!o) setToDelete(null);
      },
      slotHeading: "Terminate cluster",
      slotFooter: /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(ModalCloseButton, {
        kind: "tertiary",
        color: "neutral"
      }, "Cancel"), /*#__PURE__*/React.createElement(Button, {
        color: "danger",
        kind: "primary",
        onClick: () => setToDelete(null)
      }, "Terminate cluster"))
    }, /*#__PURE__*/React.createElement(Stack, {
      gap: 3
    }, /*#__PURE__*/React.createElement(Text, {
      kind: "body/regular/md"
    }, "This permanently terminates ", /*#__PURE__*/React.createElement("strong", {
      style: {
        fontFamily: "var(--font-mono)",
        color: "var(--text-color-primary)"
      }
    }, toDelete && toDelete.name), " and stops all ", toDelete && toDelete.jobs, " running jobs. This action cannot be undone."), toDelete && /*#__PURE__*/React.createElement(Banner, {
      status: "error",
      kind: "inline",
      slotIcon: /*#__PURE__*/React.createElement(I.Warning, {
        variant: "fill"
      })
    }, toDelete.jobs, " active job", toDelete.jobs === 1 ? "" : "s", " will be lost."))));
  }
  ReactDOM.createRoot(document.getElementById("app")).render(/*#__PURE__*/React.createElement(App, null));
})();
})(); } catch (e) { __ds_ns.__errors.push({ path: "kaizen-react/console-app.jsx", error: String((e && e.message) || e) }); }

// kaizen-react/console-parts.jsx
try { (() => {
/* ==========================================================================
   Kaizen Console — application, built ONLY from @kui/foundations-react
   components (ported into window.KUI). No raw styled divs for UI chrome.
   ========================================================================== */
(function () {
  const {
    useState,
    useMemo
  } = React;
  const I = window.KIcons;
  const {
    ThemeProvider,
    Text,
    Flex,
    Stack,
    Grid,
    Divider,
    Badge,
    Tag,
    Avatar,
    ProgressBar,
    StatusIndicator,
    Panel,
    Table,
    TableToolbar,
    Button,
    Anchor,
    TextInput,
    Switch,
    AppBar,
    AppBarLogo,
    AppBarExpanderButton,
    HorizontalNav,
    VerticalNav,
    Banner,
    Breadcrumbs,
    PageHeader,
    Modal,
    ModalCloseButton
  } = window.KUI;

  /* ---------------- data ---------------- */
  const CLUSTERS = [{
    id: "kp01",
    name: "kaizen-prod-01",
    region: "us-east-1",
    type: "H200 SXM",
    used: 78,
    total: 80,
    jobs: 14,
    status: "Active",
    cost: 18420
  }, {
    id: "kp02",
    name: "kaizen-prod-02",
    region: "us-east-1",
    type: "H100 SXM",
    used: 60,
    total: 64,
    jobs: 11,
    status: "Active",
    cost: 12960
  }, {
    id: "kt01",
    name: "kaizen-train-01",
    region: "us-west-2",
    type: "H100 SXM",
    used: 31,
    total: 32,
    jobs: 6,
    status: "Active",
    cost: 7110
  }, {
    id: "ke01",
    name: "kaizen-eval-01",
    region: "eu-central-1",
    type: "L40S",
    used: 4,
    total: 16,
    jobs: 2,
    status: "Idle",
    cost: 1240
  }, {
    id: "kd01",
    name: "kaizen-dev-01",
    region: "us-west-2",
    type: "A100 SXM",
    used: 12,
    total: 16,
    jobs: 3,
    status: "Degraded",
    cost: 2180
  }, {
    id: "kn01",
    name: "kaizen-infer-01",
    region: "ap-south-1",
    type: "L40S",
    used: 0,
    total: 24,
    jobs: 0,
    status: "Provisioning",
    cost: 320
  }];
  const STATUS_COLOR = {
    Active: "green",
    Idle: "gray",
    Degraded: "yellow",
    Provisioning: "blue",
    Failed: "red"
  };
  const STATUS_DOT = {
    Active: "success",
    Idle: "neutral",
    Degraded: "warning",
    Provisioning: "info",
    Failed: "danger"
  };
  const fmt = n => "$" + n.toLocaleString("en-US");

  /* ---------------- nav config ---------------- */
  const NAV_ITEMS = [{
    id: "overview",
    children: "Overview",
    href: "#",
    slotIcon: /*#__PURE__*/React.createElement(I.Home, null)
  }, {
    id: "compute",
    children: "Compute",
    defaultOpen: true,
    slotIcon: /*#__PURE__*/React.createElement(I.Gpu, null),
    subItems: [{
      id: "clusters",
      children: "Clusters",
      href: "#",
      active: true
    }, {
      id: "workloads",
      children: "Workloads",
      href: "#"
    }, {
      id: "jobs",
      children: "Jobs",
      href: "#"
    }]
  }, {
    id: "storage",
    children: "Storage",
    defaultOpen: false,
    slotIcon: /*#__PURE__*/React.createElement(I.Datacenter, null),
    subItems: [{
      id: "volumes",
      children: "Volumes",
      href: "#"
    }, {
      id: "datasets",
      children: "Datasets",
      href: "#"
    }]
  }, {
    id: "monitoring",
    children: "Monitoring",
    href: "#",
    slotIcon: /*#__PURE__*/React.createElement(I.Clock, null)
  }, {
    id: "settings",
    children: "Settings",
    href: "#",
    slotIcon: /*#__PURE__*/React.createElement(I.Cog, null)
  }];
  const HNAV_ITEMS = [{
    value: "overview",
    href: "#",
    children: "Overview"
  }, {
    value: "clusters",
    href: "#",
    children: "Clusters"
  }, {
    value: "workloads",
    href: "#",
    children: "Workloads"
  }, {
    value: "storage",
    href: "#",
    children: "Storage"
  }, {
    value: "settings",
    href: "#",
    children: "Settings"
  }];

  /* ---------------- KPI tile (Panel) ---------------- */
  function Kpi({
    label,
    value,
    delta,
    deltaColor,
    icon
  }) {
    return /*#__PURE__*/React.createElement(Panel, null, /*#__PURE__*/React.createElement(Stack, {
      gap: 2
    }, /*#__PURE__*/React.createElement(Flex, {
      justify: "space-between",
      align: "center"
    }, /*#__PURE__*/React.createElement(Text, {
      kind: "label/semibold/sm",
      color: "secondary",
      style: {
        textTransform: "uppercase",
        letterSpacing: ".04em"
      }
    }, label), /*#__PURE__*/React.createElement("span", {
      style: {
        color: "var(--text-color-subtle)"
      }
    }, icon)), /*#__PURE__*/React.createElement(Text, {
      kind: "title/lg",
      as: "div"
    }, value), /*#__PURE__*/React.createElement(Text, {
      kind: "body/regular/sm",
      color: deltaColor || "secondary"
    }, delta)));
  }

  /* ---------------- cluster details rail (Panel) ---------------- */
  function ClusterDetails({
    cluster,
    onDelete
  }) {
    if (!cluster) return null;
    const util = Math.round(cluster.used / cluster.total * 100);
    return /*#__PURE__*/React.createElement("aside", {
      style: {
        display: "flex",
        flexDirection: "column",
        gap: "var(--spacing-density-xl)",
        padding: "var(--spacing-density-2xl)",
        borderLeft: "1px solid var(--border-color-base)",
        background: "var(--background-color-surface-sunken)",
        overflow: "auto"
      }
    }, /*#__PURE__*/React.createElement(Stack, {
      gap: 2
    }, /*#__PURE__*/React.createElement(Text, {
      kind: "label/semibold/sm",
      color: "secondary",
      style: {
        textTransform: "uppercase",
        letterSpacing: ".04em"
      }
    }, "Cluster"), /*#__PURE__*/React.createElement(Flex, {
      gap: 2,
      align: "center",
      wrap: true
    }, /*#__PURE__*/React.createElement(Text, {
      kind: "title/sm",
      as: "h2",
      style: {
        fontFamily: "var(--font-mono)",
        whiteSpace: "nowrap"
      }
    }, cluster.name), /*#__PURE__*/React.createElement(Badge, {
      color: STATUS_COLOR[cluster.status],
      slotLeft: /*#__PURE__*/React.createElement(StatusIndicator, {
        status: STATUS_DOT[cluster.status]
      })
    }, cluster.status)), /*#__PURE__*/React.createElement(Flex, {
      gap: 2,
      wrap: true
    }, /*#__PURE__*/React.createElement(Tag, {
      kind: "outline",
      slotLeft: /*#__PURE__*/React.createElement(I.World, {
        size: 12
      })
    }, cluster.region), /*#__PURE__*/React.createElement(Tag, {
      kind: "outline",
      slotLeft: /*#__PURE__*/React.createElement(I.Gpu, {
        size: 12
      })
    }, cluster.type))), /*#__PURE__*/React.createElement(Panel, null, /*#__PURE__*/React.createElement(Stack, {
      gap: 3
    }, /*#__PURE__*/React.createElement(Flex, {
      justify: "space-between",
      align: "baseline"
    }, /*#__PURE__*/React.createElement(Text, {
      kind: "label/semibold/md"
    }, "GPU utilization"), /*#__PURE__*/React.createElement(Text, {
      kind: "body/semibold/md"
    }, util, "%")), /*#__PURE__*/React.createElement(ProgressBar, {
      value: cluster.used,
      max: cluster.total
    }), /*#__PURE__*/React.createElement(Text, {
      kind: "body/regular/sm",
      color: "secondary"
    }, cluster.used, " of ", cluster.total, " GPUs allocated"), /*#__PURE__*/React.createElement(Divider, null), /*#__PURE__*/React.createElement(Grid, {
      columns: 2,
      gap: 3
    }, /*#__PURE__*/React.createElement(Stack, {
      gap: 1
    }, /*#__PURE__*/React.createElement(Text, {
      kind: "body/regular/sm",
      color: "secondary"
    }, "Active jobs"), /*#__PURE__*/React.createElement(Text, {
      kind: "title/sm",
      as: "div"
    }, cluster.jobs)), /*#__PURE__*/React.createElement(Stack, {
      gap: 1
    }, /*#__PURE__*/React.createElement(Text, {
      kind: "body/regular/sm",
      color: "secondary"
    }, "Spend MTD"), /*#__PURE__*/React.createElement(Text, {
      kind: "title/sm",
      as: "div"
    }, fmt(cluster.cost)))))), cluster.status === "Degraded" && /*#__PURE__*/React.createElement(Banner, {
      status: "warning",
      kind: "inline",
      slotActions: /*#__PURE__*/React.createElement(Button, {
        kind: "secondary",
        size: "tiny"
      }, "View logs")
    }, "2 nodes reporting ECC errors"), /*#__PURE__*/React.createElement(Stack, {
      gap: 2
    }, /*#__PURE__*/React.createElement(Button, {
      color: "brand",
      kind: "primary",
      size: "medium"
    }, /*#__PURE__*/React.createElement(I.Add, {
      size: 14
    }), " Launch workload"), /*#__PURE__*/React.createElement(Flex, {
      gap: 2
    }, /*#__PURE__*/React.createElement(Button, {
      kind: "secondary",
      color: "neutral",
      size: "medium",
      style: {
        flex: 1
      }
    }, /*#__PURE__*/React.createElement(I.Cog, {
      size: 14
    }), " Configure"), /*#__PURE__*/React.createElement(Button, {
      kind: "secondary",
      color: "danger",
      size: "medium",
      style: {
        flex: 1
      },
      onClick: () => onDelete(cluster)
    }, /*#__PURE__*/React.createElement(I.Trash, {
      size: 14
    }), " Terminate"))));
  }
  window.ConsoleParts = {
    CLUSTERS,
    STATUS_COLOR,
    STATUS_DOT,
    fmt,
    NAV_ITEMS,
    HNAV_ITEMS,
    Kpi,
    ClusterDetails
  };
})();
})(); } catch (e) { __ds_ns.__errors.push({ path: "kaizen-react/console-parts.jsx", error: String((e && e.message) || e) }); }

// kaizen-react/icons.js
try { (() => {
/* ==========================================================================
   Kaizen UI — Icons  (port of @nv-brand-assets/react-icons-inline)
   --------------------------------------------------------------------------
   Real API:
     import { ChevronDown } from "@nv-brand-assets/react-icons-inline";
     <ChevronDown />                // outline (default)
     <ChevronDown variant="fill" /> // solid

   This port renders 16x16 currentColor SVGs. Paths are a faithful
   reconstruction of the NVIDIA Brand GUI icon set (geometry approximated
   against the 16x16 grid; sub-pixel alignment may differ from production).
   ========================================================================== */
(function (global) {
  const R = global.React;

  // name -> { outline, fill? }   (fill falls back to outline when absent)
  const PATHS = {
    Menu: {
      outline: 'M2 3.5h12v1H2v-1Zm0 4h12v1H2v-1Zm0 4h12v1H2v-1Z'
    },
    Bell: {
      outline: 'M8 1.5a4 4 0 0 0-4 4v3l-1.5 2.5h11L12 8.5v-3a4 4 0 0 0-4-4Zm0 1a3 3 0 0 1 3 3v3.27L11.85 10H4.15L5 8.77V5.5a3 3 0 0 1 3-3ZM6.5 12.5a1.5 1.5 0 0 0 3 0h-3Z'
    },
    MagnifyingGlass: {
      outline: 'M7 1.5a5.5 5.5 0 1 0 3.4 9.81l3.15 3.14.7-.7-3.14-3.15A5.5 5.5 0 0 0 7 1.5Zm0 1a4.5 4.5 0 1 1 0 9 4.5 4.5 0 0 1 0-9Z'
    },
    Cog: {
      outline: 'M6.5 1.5h3l.4 1.7a5 5 0 0 1 1.1.66l1.66-.55 1.5 2.6-1.25 1.2c.06.32.09.65.09.99s-.03.67-.09.99l1.25 1.2-1.5 2.6-1.66-.55a5 5 0 0 1-1.1.66L9.5 14.5h-3l-.4-1.7a5 5 0 0 1-1.1-.66l-1.66.55-1.5-2.6 1.25-1.2A5 5 0 0 1 3 8c0-.34.03-.67.09-.99L1.84 5.81l1.5-2.6 1.66.55a5 5 0 0 1 1.1-.66L6.5 1.5ZM8 10.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z'
    },
    ChevronDown: {
      outline: 'M3.646 5.646 8 10l4.354-4.354.708.708L8 11.414 2.939 6.354l.707-.708Z',
      fill: 'M3 5.5h10L8 11 3 5.5Z'
    },
    ChevronUp: {
      outline: 'M12.354 10.354 8 6l-4.354 4.354-.708-.708L8 4.586l5.061 5.061-.707.707Z'
    },
    ChevronLeft: {
      outline: 'M10.354 3.646 6 8l4.354 4.354-.708.708L4.586 8l5.061-5.061.707.707Z'
    },
    ChevronRight: {
      outline: 'M5.646 12.354 10 8 5.646 3.646l.708-.708L11.414 8l-5.061 5.061-.707-.707Z',
      fill: 'M5.5 3v10L11 8 5.5 3Z'
    },
    InfoCircle: {
      outline: 'M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13Zm0 1a5.5 5.5 0 1 1 0 11 5.5 5.5 0 0 1 0-11Zm.75 3.5h-1.5v1.5h1.5V6Zm0 2.5h-1.5V12h1.5V8.5Z',
      fill: 'M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13Zm.75 3.5h-1.5v1.5h1.5V5Zm0 2.5h-1.5V12h1.5V7.5Z'
    },
    CheckCircle: {
      outline: 'M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13Zm0 1a5.5 5.5 0 1 1 0 11 5.5 5.5 0 0 1 0-11Zm3.4 3.2L7.1 10 4.6 7.5l-.7.7 3.2 3.2 5-5-.7-.7Z',
      fill: 'M8 1.5a6.5 6.5 0 1 1 0 13 6.5 6.5 0 0 1 0-13Zm3.4 4.2L7.1 10 4.6 7.5l-.7.7 3.2 3.2 5-5-.7-.7Z'
    },
    Check: {
      outline: 'M14.354 4.354 6 12.707 1.646 8.354l.708-.708L6 11.293l7.646-7.647.708.708Z'
    },
    Error: {
      outline: 'M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13Zm0 1a5.5 5.5 0 1 1 0 11 5.5 5.5 0 0 1 0-11ZM7.25 4v5h1.5V4h-1.5Zm0 6v1.5h1.5V10h-1.5Z',
      fill: 'M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13ZM7.25 4v5h1.5V4h-1.5Zm0 6v1.5h1.5V10h-1.5Z'
    },
    Warning: {
      outline: 'M8 1.5 14.928 13.5H1.072L8 1.5Zm0 2.01L2.804 12.5h10.392L8 3.51Zm-.75 2.49v4h1.5v-4h-1.5Zm0 5V12h1.5v-1.5h-1.5Z',
      fill: 'M8 1.5 14.928 13.5H1.072L8 1.5Zm-.75 4v4h1.5v-4h-1.5Zm0 5V12h1.5v-1.5h-1.5Z'
    },
    Close: {
      outline: 'M8 7.293 12.646 2.646l.708.708L8.707 8l4.647 4.646-.708.708L8 8.707l-4.646 4.647-.708-.708L7.293 8 2.646 3.354l.708-.708L8 7.293Z'
    },
    Home: {
      outline: 'm8 1.5 6.5 5.5h-2v6h-3v-4h-3v4h-3v-6h-2L8 1.5Zm0 1.31L4.5 6V12h1v-4h5v4h1V6L8 2.81Z'
    },
    Profile: {
      outline: 'M8 2.5a2.75 2.75 0 1 0 0 5.5 2.75 2.75 0 0 0 0-5.5Zm0 1a1.75 1.75 0 1 1 0 3.5 1.75 1.75 0 0 1 0-3.5ZM3.5 14a4.5 4.5 0 0 1 9 0h-1a3.5 3.5 0 0 0-7 0h-1Z',
      fill: 'M8 2.5a2.75 2.75 0 1 1 0 5.5 2.75 2.75 0 0 1 0-5.5ZM12.5 14a4.5 4.5 0 0 0-9 0h9Z'
    },
    Clock: {
      outline: 'M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13Zm0 1a5.5 5.5 0 1 1 0 11 5.5 5.5 0 0 1 0-11ZM7.5 4v4.5h4v-1H8.5V4h-1Z',
      fill: 'M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13ZM7.5 4v4.5h4v-1H8.5V4h-1Z'
    },
    Pencil: {
      outline: 'm11.293 1.793 2.914 2.914-9 9H2.293v-2.914l9-9Zm-1.207 2.621L3 11.5V13h1.5l7.086-7.086-1.5-1.5Z',
      fill: 'm11.293 1.793 2.914 2.914-9 9H2.293v-2.914l9-9Z'
    },
    Add: {
      outline: 'M7.5 2h1v5.5H14v1H8.5V14h-1V8.5H2v-1h5.5V2Z'
    },
    Subtract: {
      outline: 'M2 7.5h12v1H2v-1Z'
    },
    OpenExternal: {
      outline: 'M9 2h5v5h-1V3.707L8.354 8.354l-.708-.708L12.293 3H9V2ZM3 4h4v1H4v7h7V9h1v4H3V4Z'
    },
    MoreHoriz: {
      outline: 'M3 8a1.25 1.25 0 1 1 2.5 0A1.25 1.25 0 0 1 3 8Zm3.75 0a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0Zm3.75 0a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0Z'
    },
    MoreVert: {
      outline: 'M8 3a1.25 1.25 0 1 1 0 2.5A1.25 1.25 0 0 1 8 3Zm0 3.75a1.25 1.25 0 1 1 0 2.5 1.25 1.25 0 0 1 0-2.5Zm0 3.75a1.25 1.25 0 1 1 0 2.5 1.25 1.25 0 0 1 0-2.5Z'
    },
    Download: {
      outline: 'M7.5 2h1v6.293l2.146-2.147.708.708L8 10.207 4.646 6.854l.708-.708L7.5 8.293V2ZM3 12h10v1H3v-1Z'
    },
    Filter: {
      outline: 'M2 3h12v1L9.5 9v4l-3-1.5V9L2 4V3Z'
    },
    Gpu: {
      outline: 'M1.5 4h13v8h-13V4Zm1 1v6h11V5h-11Zm1.5 1h3v4H4V6Zm4 0h3v4H8V6Z'
    },
    World: {
      outline: 'M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13Zm0 1c1.1 0 2.1.9 2.8 2.3-1.7.4-3.9.4-5.6 0C5.9 3.4 6.9 2.5 8 2.5Zm-4 4.6c.6.2 1.3.3 2 .4v1c-.7.1-1.4.2-2 .4-.1-.4-.2-.9-.2-1.4s.1-.9.2-1.4Zm6 0c.1.5.2.9.2 1.4 0 .5-.1 1-.2 1.4-.6-.2-1.3-.3-2-.4v-1c.7-.1 1.4-.2 2-.4Z'
    },
    Trash: {
      outline: 'M6 1.5h4l.5 1.5H14v1h-1l-.8 9.5H3.8L3 5H2V4h3.5L6 1.5Zm.9 1.5-.3 1h2.8l-.3-1H6.9ZM4.8 5l.7 8.5h5l.7-8.5H4.8Zm1.7 1.5h1V12h-1V6.5Zm2 0h1V12h-1V6.5Z'
    },
    Sort: {
      outline: 'M5 2 7.5 5h-5L5 2Zm0 12L2.5 11h5L5 14Zm4-9h5v1H9V5Zm0 3h4v1H9V8Zm0 3h3v1H9v-1Z'
    },
    Refresh: {
      outline: 'M8 2.5a5.5 5.5 0 0 1 5.3 4H12.2A4.5 4.5 0 0 0 4 6.4V5h-1v3.5h3.5v-1H4.6A4.5 4.5 0 0 1 8 2.5Zm5 5.5h-1v1.1A4.5 4.5 0 0 1 4.7 9.6H3.6A5.5 5.5 0 0 0 12 10.3V11.5h1V8Z'
    },
    Datacenter: {
      outline: 'M2.5 2h11v4h-11V2Zm0 5h11v4h-11V7Zm1-4v2h9V3h-9Zm0 5v2h9V8h-9ZM4 4h3v.8H4V4Zm0 5h3v.8H4V9Zm7.4-5.4a.6.6 0 1 1 0 1.2.6.6 0 0 1 0-1.2Zm0 5a.6.6 0 1 1 0 1.2.6.6 0 0 1 0-1.2Z'
    },
    Moon: {
      outline: 'M6 1.6A6.5 6.5 0 1 0 14.4 10 5 5 0 0 1 6 1.6Zm-.8 1.7A5 5 0 1 0 12.7 11 6 6 0 0 1 5.2 3.3Z',
      fill: 'M6 1.6A6.5 6.5 0 1 0 14.4 10 5 5 0 0 1 6 1.6Z'
    },
    SunHigh: {
      outline: 'M8 4.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7Zm0 1a2.5 2.5 0 1 1 0 5 2.5 2.5 0 0 1 0-5ZM7.5 1h1v2h-1V1Zm0 12h1v2h-1v-2ZM1 7.5h2v1H1v-1Zm12 0h2v1h-2v-1ZM3.3 2.6l1.4 1.4-.7.7-1.4-1.4.7-.7Zm8 8 1.4 1.4-.7.7-1.4-1.4.7-.7Zm1.4-8 .7.7-1.4 1.4-.7-.7 1.4-1.4ZM3.3 13.4l-.7-.7 1.4-1.4.7.7-1.4 1.4Z'
    },
    ArrowUp: {
      outline: 'M8 2.5 12 6.5l-.7.7L8.5 4.9V13.5h-1V4.9L4.7 7.2 4 6.5 8 2.5Z'
    },
    ArrowDown: {
      outline: 'M8 13.5 4 9.5l.7-.7 2.8 2.3V2.5h1v8.6l2.8-2.3.7.7L8 13.5Z'
    },
    Workspace: {
      outline: 'M2 3h12v8H2V3Zm1 1v6h10V4H3Zm2 7h6v1H5v-1Z'
    }
  };
  function Icon(props) {
    const {
      name,
      variant = "outline",
      size = 16,
      color,
      className,
      style,
      ...rest
    } = props;
    const entry = PATHS[name];
    const d = entry ? variant === "fill" && entry.fill ? entry.fill : entry.outline : "";
    return R.createElement("svg", Object.assign({
      xmlns: "http://www.w3.org/2000/svg",
      width: size,
      height: size,
      viewBox: "0 0 16 16",
      fill: "currentColor",
      "aria-hidden": "true",
      className,
      style: Object.assign({
        flex: "none",
        display: "block",
        color
      }, style)
    }, rest), d ? R.createElement("path", {
      d,
      fillRule: "evenodd",
      clipRule: "evenodd"
    }) : null);
  }

  // Generate a named component per icon, matching the real package shape:
  //   <ChevronDown variant="fill" />
  const named = {};
  Object.keys(PATHS).forEach(name => {
    const C = props => Icon(Object.assign({
      name
    }, props));
    C.displayName = name;
    named[name] = C;
  });
  global.KIcons = Object.assign({
    Icon,
    PATHS
  }, named);
})(window);
})(); } catch (e) { __ds_ns.__errors.push({ path: "kaizen-react/icons.js", error: String((e && e.message) || e) }); }

// kaizen-react/kui-controls.js
try { (() => {
/* ==========================================================================
   Kaizen UI — Controls & layout (port of @kui/foundations-react)
   Button, Anchor, inputs, Switch, AppBar, HorizontalNav, VerticalNav,
   Banner, PageHeader, Modal, Notification.
   Exports onto window.KUI.*
   ========================================================================== */
(function (global) {
  const R = global.React;
  const h = R.createElement;
  const {
    Icon
  } = global.KIcons;
  const cx = global.KUI.cx;

  /* ---------- asChild helper ---------- */
  function renderAsChild(child, ownProps) {
    const merged = Object.assign({}, ownProps, child.props, {
      className: cx(ownProps.className, child.props.className),
      style: Object.assign({}, ownProps.style, child.props.style)
    });
    return R.cloneElement(child, merged);
  }

  /* ---------- Button ---------- */
  function Button(props) {
    const {
      color = "neutral",
      kind = "primary",
      size = "medium",
      disabled,
      asChild,
      iconOnly,
      className,
      style,
      children,
      ...rest
    } = props;
    // icon-only is opt-in via the `iconOnly` prop. (We deliberately do NOT
    // auto-detect from a single element child: editor tooling and i18n wrappers
    // wrap visible label text in a <span>, which would false-positive and
    // collapse a text button into a square.)
    const kids = R.Children.toArray(children);
    const cls = cx("kui-btn", "kui-btn--" + kind, "kui-btn--" + color, "kui-btn--" + size, iconOnly && "kui-btn--icon", className);
    if (asChild && R.isValidElement(kids[0])) {
      return renderAsChild(kids[0], {
        className: cls,
        style,
        ...rest
      });
    }
    return h("button", {
      className: cls,
      style,
      disabled,
      ...rest
    }, children);
  }

  /* ---------- Anchor ---------- */
  function Anchor(props) {
    const {
      kind = "inline",
      textKind,
      className,
      style,
      children,
      ...rest
    } = props;
    const cls = cx("kui-anchor", "kui-anchor--" + kind, textKind === "inherit" && "kui-anchor--inherit", className);
    return h("a", {
      className: cls,
      style,
      ...rest
    }, children);
  }

  /* ---------- InputShell / TextInput ---------- */
  function InputShell(props) {
    const {
      status,
      slotLeft,
      slotRight,
      className,
      style,
      children,
      ...rest
    } = props;
    return h("div", {
      className: cx("kui-input-shell", status && "kui-input-shell--" + status, className),
      style,
      ...rest
    }, slotLeft ? h("span", {
      className: "kui-input-shell__slot"
    }, slotLeft) : null, children, slotRight ? h("span", {
      className: "kui-input-shell__slot"
    }, slotRight) : null);
  }
  function TextInput(props) {
    const {
      status,
      slotLeft,
      slotRight,
      className,
      style,
      shellClassName,
      ...rest
    } = props;
    return h(InputShell, {
      status,
      slotLeft,
      slotRight,
      className: shellClassName,
      style
    }, h("input", {
      className: cx("kui-input", className),
      ...rest
    }));
  }

  /* ---------- FormField ---------- */
  function FormField(props) {
    const {
      label,
      description,
      message,
      htmlFor,
      className,
      style,
      children,
      ...rest
    } = props;
    return h("div", {
      className: cx("kui-formfield", className),
      style,
      ...rest
    }, label ? h("label", {
      className: "kui-formfield__label",
      htmlFor
    }, label) : null, description ? h("div", {
      className: "kui-formfield__desc"
    }, description) : null, children, message ? h("div", {
      className: "kui-formfield__msg"
    }, message) : null);
  }

  /* ---------- Switch ---------- */
  function Switch(props) {
    const {
      checked,
      onChange,
      label,
      className,
      style,
      ...rest
    } = props;
    return h("button", {
      type: "button",
      role: "switch",
      "aria-checked": !!checked,
      className: cx("kui-switch", className),
      style,
      onClick: () => onChange && onChange(!checked),
      ...rest
    }, h("span", {
      className: "kui-switch__track"
    }, h("span", {
      className: "kui-switch__thumb"
    })), label ? h("span", {
      className: "kui-text-label-regular-md"
    }, label) : null);
  }

  /* ---------- AppBar ---------- */
  function AppBarLogo(props) {
    const {
      className,
      style,
      ...rest
    } = props;
    return h("span", {
      className: cx("kui-appbar-logo", className),
      style,
      "aria-label": "NVIDIA",
      ...rest
    }, h("svg", {
      width: 105,
      height: 19,
      viewBox: "0 0 150 19.687",
      fill: "none",
      role: "img"
    },
    // eye-mark (brand green)
    h("g", {
      transform: "scale(0.6575)",
      fill: "var(--color-brand)"
    }, h("path", {
      fillRule: "nonzero",
      d: "M 16.715 8.938 L 16.715 6.233 C 16.976 6.215 17.238 6.2 17.505 6.192 C 24.83 5.958 29.635 12.55 29.635 12.55 C 29.635 12.55 24.445 19.833 18.881 19.833 C 18.08 19.833 17.362 19.703 16.715 19.483 L 16.715 11.278 C 19.567 11.627 20.14 12.898 21.854 15.785 L 25.666 12.537 C 25.666 12.537 22.883 8.848 18.192 8.848 C 17.682 8.848 17.194 8.885 16.714 8.937 L 16.715 8.938 Z M 16.714 0 L 16.714 4.042 C 16.976 4.02 17.24 4.003 17.504 3.993 C 27.688 3.647 34.325 12.433 34.325 12.433 C 34.325 12.433 26.704 21.797 18.762 21.797 C 18.035 21.797 17.354 21.728 16.714 21.615 L 16.714 24.113 C 17.261 24.183 17.829 24.225 18.419 24.225 C 25.808 24.225 31.153 20.413 36.327 15.9 C 37.185 16.593 40.697 18.282 41.42 19.022 C 36.499 23.183 25.034 26.537 18.533 26.537 C 17.906 26.537 17.304 26.498 16.714 26.442 L 16.714 29.952 L 44.798 29.952 L 44.798 0.002 L 16.715 0.002 L 16.714 0 Z M 16.714 19.482 L 16.714 21.615 C 9.879 20.383 7.982 13.207 7.982 13.207 C 7.982 13.207 11.263 9.533 16.714 8.938 L 16.714 11.278 C 16.714 11.278 16.707 11.278 16.704 11.278 C 13.843 10.932 11.61 13.632 11.61 13.632 C 11.61 13.632 12.862 18.177 16.715 19.485 L 16.714 19.482 Z M 4.578 12.898 C 4.578 12.898 8.627 6.86 16.715 6.235 L 16.715 4.045 C 7.758 4.772 0 12.437 0 12.437 C 0 12.437 4.393 25.27 16.715 26.445 L 16.715 24.117 C 7.672 22.967 4.578 12.898 4.578 12.898 L 4.578 12.898 Z"
    })),
    // wordmark (theme color)
    h("g", {
      transform: "translate(46, 0)",
      fill: "currentColor"
    }, h("path", {
      fillRule: "nonzero",
      d: "M 43.229 0.027 L 43.229 19.687 L 48.726 19.687 L 48.726 0.027 L 43.229 0.027 L 43.229 0.027 Z M 0 0 L 0 19.687 L 5.544 19.687 L 5.544 4.405 L 9.87 4.42 C 11.291 4.42 12.276 4.765 12.961 5.503 C 13.83 6.44 14.185 7.948 14.185 10.708 L 14.185 19.687 L 19.556 19.687 L 19.556 8.81 C 19.556 1.047 14.658 0 9.868 0 L 0 0 Z M 52.078 0.027 L 52.078 19.685 L 60.99 19.685 C 65.739 19.685 67.288 18.887 68.966 17.098 C 70.15 15.842 70.916 13.085 70.916 10.072 C 70.916 7.308 70.268 4.843 69.138 3.308 C 67.102 0.563 64.169 0.027 59.789 0.027 L 52.078 0.027 L 52.078 0.027 Z M 57.528 4.307 L 59.89 4.307 C 63.318 4.307 65.535 5.862 65.535 9.897 C 65.535 13.932 63.318 15.488 59.89 15.488 L 57.528 15.488 L 57.528 4.307 Z M 35.306 0.027 L 30.72 15.607 L 26.326 0.027 L 20.394 0.027 L 26.671 19.685 L 34.59 19.685 L 40.916 0.027 L 35.308 0.027 L 35.306 0.027 Z M 73.473 19.685 L 78.969 19.685 L 78.969 0.027 L 73.471 0.027 L 73.471 19.685 L 73.473 19.685 Z M 88.878 0.033 L 81.204 19.678 L 86.623 19.678 L 87.837 16.207 L 96.918 16.207 L 98.068 19.678 L 103.951 19.678 L 96.219 0.032 L 88.878 0.032 L 88.878 0.033 Z M 92.445 3.618 L 95.774 12.822 L 89.01 12.822 L 92.445 3.618 Z"
    }))));
  }
  function AppBarExpanderButton(props) {
    return h(Button, Object.assign({
      kind: "tertiary",
      size: "medium",
      "aria-label": "Toggle navigation"
    }, props), h(Icon, {
      name: "Menu"
    }));
  }
  function AppBar(props) {
    const {
      slotStart,
      slotEnd,
      className,
      style,
      children,
      ...rest
    } = props;
    return h("header", {
      className: cx("kui-appbar", className),
      style,
      ...rest
    }, slotStart ? h("div", {
      className: "kui-appbar__start"
    }, slotStart) : null, h("div", {
      className: "kui-appbar__center"
    }, children), slotEnd ? h("div", {
      className: "kui-appbar__end"
    }, slotEnd) : null);
  }

  /* ---------- HorizontalNav ---------- */
  function HorizontalNav(props) {
    const {
      items = [],
      value,
      defaultValue,
      onValueChange,
      className,
      style,
      ...rest
    } = props;
    const [internal, setInternal] = R.useState(defaultValue);
    const current = value !== undefined ? value : internal;
    return h("nav", {
      className: cx("kui-hnav", className),
      style,
      "aria-label": "Primary",
      ...rest
    }, items.map(it => h("a", {
      key: it.value,
      href: it.href || "#",
      className: cx("kui-hnav__link", current === it.value && "is-active"),
      "aria-current": current === it.value ? "page" : undefined,
      onClick: e => {
        if (!it.href || it.href === "#") e.preventDefault();
        if (value === undefined) setInternal(it.value);
        onValueChange && onValueChange(it.value);
        it.onClick && it.onClick(e);
      }
    }, it.slotIcon ? h("span", {
      className: "kui-vnav__item-slot"
    }, it.slotIcon) : null, it.children)));
  }

  /* ---------- VerticalNav (composed primitives) ---------- */
  const VerticalNavRoot = p => h("nav", {
    className: cx("kui-vnav", p.className),
    style: p.style,
    "aria-label": "Sidebar"
  }, p.children);
  const VerticalNavList = p => h("ul", {
    className: cx("kui-vnav__list", p.className),
    style: p.style
  }, p.children);
  const VerticalNavListItem = p => h("li", {
    className: p.className
  }, p.children);
  const VerticalNavSubList = p => h("ul", {
    className: cx("kui-vnav__list", "kui-vnav__list--sub", p.className)
  }, p.children);
  const VerticalNavSubListItem = p => h("li", null, p.children);
  function VerticalNavItem(props) {
    const {
      kind,
      active,
      slotIcon,
      slotEnd,
      asChild,
      className,
      children,
      ...rest
    } = props;
    const cls = cx("kui-vnav__item", kind === "secondary" && "kui-vnav__item--secondary", active && "is-active", className);
    const inner = [slotIcon ? h("span", {
      key: "i",
      className: "kui-vnav__item-slot"
    }, slotIcon) : null, h("span", {
      key: "t",
      style: {
        flex: 1,
        minWidth: 0
      }
    }, children), slotEnd ? h("span", {
      key: "e",
      className: "kui-vnav__item-end"
    }, slotEnd) : null];
    if (asChild && R.isValidElement(children)) return renderAsChild(children, {
      className: cls,
      ...rest
    });
    const tag = rest.href ? "a" : "button";
    return h(tag, Object.assign({
      className: cls,
      "aria-current": active ? "page" : undefined
    }, tag === "button" ? {
      type: "button"
    } : {}, rest), inner);
  }

  // high-level declarative VerticalNav
  function VerticalNav(props) {
    const {
      items = [],
      renderLink,
      footer,
      className,
      style
    } = props;
    function leaf(it) {
      const content = it.children;
      if (renderLink) return h(VerticalNavItem, {
        asChild: true,
        active: it.active,
        kind: it.kind,
        slotIcon: it.slotIcon
      }, renderLink(it));
      return h(VerticalNavItem, {
        href: it.href,
        active: it.active,
        slotIcon: it.slotIcon,
        kind: it.kind
      }, content);
    }
    return h(VerticalNavRoot, {
      className,
      style
    }, h(VerticalNavList, {
      className: "kui-vnav__list",
      style: {
        flex: 1
      }
    }, items.map(it => {
      if (it.subItems) {
        return h("li", {
          key: it.id
        }, h("details", {
          className: "kui-vnav__section",
          open: it.defaultOpen !== false
        }, h("summary", {
          style: {
            display: "block"
          }
        }, h(VerticalNavItem, {
          slotIcon: it.slotIcon,
          slotEnd: h(Icon, {
            name: "ChevronDown",
            size: 14,
            className: "kui-vnav__chevron"
          })
        }, it.children)), h(VerticalNavSubList, null, it.subItems.map(s => h(VerticalNavSubListItem, {
          key: s.id
        }, leaf(Object.assign({
          kind: "secondary"
        }, s)))))));
      }
      return h(VerticalNavListItem, {
        key: it.id
      }, leaf(it));
    })), footer ? h("div", {
      className: "kui-vnav__footer"
    }, footer) : null);
  }

  /* ---------- Banner ---------- */
  const BANNER_ICON = {
    info: "InfoCircle",
    success: "CheckCircle",
    warning: "Warning",
    error: "Error",
    danger: "Error"
  };
  function Banner(props) {
    const {
      status = "info",
      kind = "inline",
      slotIcon,
      slotSubheading,
      slotActions,
      actionsPosition,
      onClose,
      className,
      style,
      children,
      ...rest
    } = props;
    const showIcon = slotIcon !== null;
    return h("div", {
      className: cx("kui-banner", "kui-banner--" + status, "kui-banner--" + kind, className),
      style,
      role: "status",
      ...rest
    }, showIcon ? h("span", {
      className: "kui-banner__icon"
    }, slotIcon || h(Icon, {
      name: BANNER_ICON[status],
      variant: "fill"
    })) : null, h("div", {
      className: "kui-banner__body"
    }, h("div", {
      className: "kui-banner__title"
    }, children), slotSubheading ? h("div", {
      className: "kui-text-body-regular-sm",
      style: {
        color: "var(--text-color-secondary)",
        marginTop: 2
      }
    }, slotSubheading) : null), slotActions ? h("div", {
      className: "kui-banner__actions",
      style: actionsPosition === "bottom" ? {
        width: "100%",
        justifyContent: "flex-end"
      } : undefined
    }, slotActions) : null, onClose ? h(Button, {
      kind: "tertiary",
      color: "neutral",
      size: "tiny",
      iconOnly: true,
      "aria-label": "Close",
      onClick: onClose
    }, h(Icon, {
      name: "Close",
      size: 14
    })) : null);
  }

  /* ---------- Breadcrumbs ---------- */
  function Breadcrumbs(props) {
    const {
      items = [],
      className,
      style,
      ...rest
    } = props;
    return h("nav", {
      className: cx("kui-breadcrumbs", className),
      style,
      "aria-label": "Breadcrumb",
      ...rest
    }, items.map((it, i) => {
      const node = it && typeof it === "object" && !R.isValidElement(it) ? it.children : it;
      const isLast = i === items.length - 1;
      return [i > 0 ? h("span", {
        key: "s" + i,
        className: "kui-breadcrumbs__sep"
      }, h(Icon, {
        name: "ChevronRight",
        size: 12
      })) : null, h("span", {
        key: "c" + i,
        className: isLast ? "kui-breadcrumbs__current" : undefined
      }, node)];
    }));
  }

  /* ---------- PageHeader ---------- */
  function PageHeader(props) {
    const {
      kind = "flat",
      slotBreadcrumbs,
      slotSubheading,
      slotHeading,
      slotDescription,
      slotActions,
      titleKind = "title/lg",
      className,
      style,
      children,
      ...rest
    } = props;
    const Text = global.KUI.Text;
    return h("div", {
      className: cx("kui-pageheader", kind === "floating" && "kui-pageheader--floating", className),
      style,
      ...rest
    }, h("div", {
      className: "kui-pageheader__main"
    }, slotBreadcrumbs ? h("div", null, slotBreadcrumbs) : null, slotSubheading ? h("div", {
      className: "kui-pageheader__subheading"
    }, slotSubheading) : null, R.isValidElement(slotHeading) ? slotHeading : h(Text, {
      as: "h1",
      kind: titleKind,
      className: "kui-pageheader__title"
    }, slotHeading), slotDescription ? h("div", {
      className: "kui-pageheader__description"
    }, slotDescription) : null, children), slotActions ? h("div", {
      className: "kui-pageheader__actions"
    }, slotActions) : null);
  }

  /* ---------- Modal ---------- */
  function ModalCloseButton(props) {
    const {
      children,
      kind = "tertiary",
      color = "neutral",
      size = "medium",
      className,
      onClick,
      ...rest
    } = props;
    // __kuiClose injected by Modal via context-free clone
    return h(Button, Object.assign({
      kind,
      color,
      size,
      className,
      "data-modal-close": "true",
      onClick
    }, rest), children);
  }
  function Modal(props) {
    const {
      slotTrigger,
      slotAnchorTrigger,
      slotHeading,
      slotFooter,
      dismissible = true,
      hideCloseButton,
      closeOnClickOutside = true,
      density = "standard",
      open: controlledOpen,
      defaultOpen = false,
      onOpenChange,
      renderContent,
      className,
      style,
      children
    } = props;
    const [internal, setInternal] = R.useState(defaultOpen);
    const isControlled = controlledOpen !== undefined;
    const open = isControlled ? controlledOpen : internal;
    const setOpen = v => {
      if (!isControlled) setInternal(v);
      onOpenChange && onOpenChange(v);
    };
    const close = () => {
      if (dismissible) setOpen(false);
    };
    const trigger = slotTrigger ? R.cloneElement(slotTrigger, {
      onClick: e => {
        slotTrigger.props.onClick && slotTrigger.props.onClick(e);
        setOpen(true);
      }
    }) : slotAnchorTrigger ? R.cloneElement(slotAnchorTrigger, {
      onClick: e => {
        e.preventDefault();
        setOpen(true);
      }
    }) : null;

    // intercept clicks on ModalCloseButton inside footer
    function wireFooter(node) {
      if (!node) return node;
      return R.Children.map(node, child => {
        if (R.isValidElement(child) && child.props && child.props["data-modal-close"]) {
          return R.cloneElement(child, {
            onClick: e => {
              child.props.onClick && child.props.onClick(e);
              setOpen(false);
            }
          });
        }
        return child;
      });
    }
    const body = renderContent ? renderContent({
      children
    }) : children;
    return h(R.Fragment, null, trigger, open ? h("div", {
      className: "kui-blanket",
      onClick: closeOnClickOutside ? close : undefined
    }, h("div", {
      className: cx("kui-modal", "kui-modal--" + density, className),
      style,
      role: "dialog",
      "aria-modal": "true",
      onClick: e => e.stopPropagation()
    }, h("div", {
      className: "kui-modal__header"
    }, h(global.KUI.Text, {
      as: "h2",
      kind: "title/sm",
      className: "kui-modal__title"
    }, slotHeading), dismissible && !hideCloseButton ? h(Button, {
      kind: "tertiary",
      color: "neutral",
      size: "small",
      iconOnly: true,
      "aria-label": "Close",
      onClick: close
    }, h(Icon, {
      name: "Close",
      size: 16
    })) : null), h("div", {
      className: "kui-modal__body"
    }, body), slotFooter ? h("div", {
      className: "kui-modal__footer"
    }, wireFooter(slotFooter)) : null)) : null);
  }

  /* ---------- Notification ---------- */
  function Notification(props) {
    const {
      color = "info",
      heading,
      subheading,
      slotIcon,
      onClose,
      footer,
      className,
      style,
      children
    } = props;
    return h("div", {
      className: cx("kui-notification", "kui-notification--" + color, className),
      style
    }, h("span", {
      className: "kui-banner__icon",
      style: {
        color: "var(--text-color-feedback-" + (color === "info" ? "info" : color) + ")"
      }
    }, slotIcon || h(Icon, {
      name: BANNER_ICON[color] || "InfoCircle",
      variant: "fill"
    })), h("div", null, heading ? h("div", {
      className: "kui-notification__heading"
    }, heading) : null, subheading ? h("div", {
      className: "kui-notification__sub"
    }, subheading) : null, children, footer ? h("div", {
      style: {
        marginTop: 8
      }
    }, footer) : null), onClose ? h(Button, {
      kind: "tertiary",
      size: "tiny",
      iconOnly: true,
      "aria-label": "Dismiss",
      onClick: onClose
    }, h(Icon, {
      name: "Close",
      size: 14
    })) : null);
  }

  /* ---------- export ---------- */
  global.KUI = Object.assign(global.KUI || {}, {
    Button,
    Anchor,
    InputShell,
    TextInput,
    FormField,
    Switch,
    AppBar,
    AppBarLogo,
    AppBarExpanderButton,
    HorizontalNav,
    VerticalNav,
    VerticalNavRoot,
    VerticalNavList,
    VerticalNavListItem,
    VerticalNavSubList,
    VerticalNavSubListItem,
    VerticalNavItem,
    Banner,
    Breadcrumbs,
    PageHeader,
    Modal,
    ModalCloseButton,
    Notification
  });
})(window);
})(); } catch (e) { __ds_ns.__errors.push({ path: "kaizen-react/kui-controls.js", error: String((e && e.message) || e) }); }

// kaizen-react/kui-core.js
try { (() => {
/* ==========================================================================
   Kaizen UI — Core components (port of @kui/foundations-react)
   Primitives + display/data components.
   Controls (Button, inputs, overlays, nav) live in kui-controls.jsx.
   Exports onto window.KUI.*
   ========================================================================== */
(function (global) {
  const R = global.React;
  const h = R.createElement;
  const {
    Icon
  } = global.KIcons;

  /* ---------- helpers ---------- */
  function cx() {
    return Array.prototype.filter.call(arguments, Boolean).join(" ");
  }
  // gap/spacing: number -> n*4px ; "density-md" -> var(--spacing-density-md) ; raw string passthrough
  function space(v) {
    if (v == null) return undefined;
    if (typeof v === "number") return v * 4 + "px";
    if (/^density-/.test(v)) return "var(--spacing-" + v + ")";
    return v;
  }

  /* ---------- ThemeProvider ---------- */
  const ThemeCtx = R.createContext({
    theme: "light",
    density: "standard"
  });
  function ThemeProvider(props) {
    const {
      theme = "light",
      density = "standard",
      as = "div",
      className,
      style,
      children,
      ...rest
    } = props;
    return h(ThemeCtx.Provider, {
      value: {
        theme,
        density
      }
    }, h(as, {
      "data-theme": theme,
      "data-density": density,
      className: cx("kui-root", className),
      style,
      ...rest
    }, children));
  }

  /* ---------- Text ---------- */
  function Text(props) {
    const {
      kind = "body/regular/md",
      color,
      as = "span",
      className,
      style,
      children,
      ...rest
    } = props;
    const kindClass = "kui-text-" + kind.replace(/\//g, "-");
    const colorClass = color ? "kui-text--" + color : null;
    return h(as, {
      className: cx("kui-text", kindClass, colorClass, className),
      style,
      ...rest
    }, children);
  }

  /* ---------- Flex / Stack / Grid / Inline / Block / Group ---------- */
  function Flex(props) {
    const {
      gap,
      align,
      justify,
      direction = "row",
      wrap,
      inline,
      as = "div",
      className,
      style,
      children,
      ...rest
    } = props;
    const s = Object.assign({
      display: inline ? "inline-flex" : "flex",
      flexDirection: direction,
      alignItems: align,
      justifyContent: justify,
      flexWrap: wrap ? "wrap" : undefined,
      gap: space(gap)
    }, style);
    return h(as, {
      className: cx("kui-flex", className),
      style: s,
      ...rest
    }, children);
  }
  function Stack(props) {
    const {
      direction = "column",
      ...rest
    } = props;
    return Flex(Object.assign({
      direction
    }, rest));
  }
  function Grid(props) {
    const {
      gap,
      columns,
      colMinWidth,
      align,
      justify,
      as = "div",
      className,
      style,
      children,
      ...rest
    } = props;
    const template = colMinWidth ? "repeat(auto-fill, minmax(" + (typeof colMinWidth === "number" ? colMinWidth + "px" : colMinWidth) + ", 1fr))" : columns ? typeof columns === "number" ? "repeat(" + columns + ", 1fr)" : columns : undefined;
    const s = Object.assign({
      display: "grid",
      gridTemplateColumns: template,
      alignItems: align,
      justifyContent: justify,
      gap: space(gap)
    }, style);
    return h(as, {
      className: cx("kui-grid", className),
      style: s,
      ...rest
    }, children);
  }
  function Inline(props) {
    const {
      as = "span",
      className,
      style,
      children,
      ...rest
    } = props;
    return h(as, {
      className: cx("kui-inline", className),
      style,
      ...rest
    }, children);
  }
  function Block(props) {
    const {
      as = "div",
      className,
      style,
      children,
      ...rest
    } = props;
    return h(as, {
      className: cx("kui-block", className),
      style,
      ...rest
    }, children);
  }
  function Group(props) {
    const {
      gap = 2,
      align = "center",
      as = "div",
      className,
      style,
      children,
      ...rest
    } = props;
    const s = Object.assign({
      display: "inline-flex",
      alignItems: align,
      gap: space(gap)
    }, style);
    return h(as, {
      className: cx("kui-group", className),
      style: s,
      ...rest
    }, children);
  }

  /* ---------- Divider ---------- */
  function Divider(props) {
    const {
      orientation = "horizontal",
      className,
      style,
      ...rest
    } = props;
    return h("hr", {
      className: cx("kui-divider", "kui-divider--" + orientation, className),
      style,
      ...rest
    });
  }

  /* ---------- Badge ---------- */
  function Badge(props) {
    const {
      color = "gray",
      kind = "subtle",
      slotLeft,
      className,
      style,
      children,
      ...rest
    } = props;
    return h("span", {
      className: cx("kui-badge", "kui-badge--" + color, kind === "solid" && "kui-badge--solid", className),
      style,
      ...rest
    }, slotLeft ? h("span", {
      className: "kui-icon-slot"
    }, slotLeft) : null, children);
  }

  /* ---------- Tag ---------- */
  function Tag(props) {
    const {
      kind = "solid",
      slotLeft,
      onRemove,
      className,
      style,
      children,
      ...rest
    } = props;
    return h("span", {
      className: cx("kui-tag", kind === "outline" && "kui-tag--outline", className),
      style,
      ...rest
    }, slotLeft ? h("span", {
      className: "kui-icon-slot"
    }, slotLeft) : null, h("span", null, children), onRemove ? h("button", {
      className: "kui-tag__remove",
      onClick: onRemove,
      "aria-label": "Remove"
    }, h(Icon, {
      name: "Close",
      size: 12
    })) : null);
  }

  /* ---------- Avatar ---------- */
  function Avatar(props) {
    const {
      fallback,
      src,
      alt,
      size = "md",
      brand,
      className,
      style,
      ...rest
    } = props;
    const cls = cx("kui-avatar", size === "sm" && "kui-avatar--sm", size === "lg" && "kui-avatar--lg", brand && "kui-avatar--brand", className);
    return h("span", {
      className: cls,
      style,
      ...rest
    }, src ? h("img", {
      src,
      alt: alt || fallback || ""
    }) : fallback || h(Icon, {
      name: "Profile",
      size: 16
    }));
  }

  /* ---------- Spinner ---------- */
  function Spinner(props) {
    const {
      size = "md",
      className,
      style,
      ...rest
    } = props;
    return h("span", {
      className: cx("kui-spinner", size === "lg" && "kui-spinner--lg", className),
      style,
      role: "status",
      "aria-label": "Loading",
      ...rest
    });
  }

  /* ---------- ProgressBar ---------- */
  function ProgressBar(props) {
    const {
      value = 0,
      max = 100,
      color,
      className,
      style,
      ...rest
    } = props;
    const pct = Math.max(0, Math.min(100, value / max * 100));
    return h("div", {
      className: cx("kui-progress", color && "kui-progress--" + color, className),
      style,
      role: "progressbar",
      "aria-valuenow": value,
      "aria-valuemax": max,
      ...rest
    }, h("div", {
      className: "kui-progress__fill",
      style: {
        width: pct + "%"
      }
    }));
  }

  /* ---------- StatusIndicator ---------- */
  function StatusIndicator(props) {
    const {
      status = "neutral",
      className,
      style,
      ...rest
    } = props;
    return h("span", {
      className: cx("kui-status-indicator", "kui-status-indicator--" + status, className),
      style,
      ...rest
    });
  }

  /* ---------- Skeleton ---------- */
  function Skeleton(props) {
    const {
      width,
      height = 16,
      radius = "var(--radius-sm)",
      className,
      style,
      ...rest
    } = props;
    const s = Object.assign({
      width: typeof width === "number" ? width + "px" : width || "100%",
      height: typeof height === "number" ? height + "px" : height,
      borderRadius: radius,
      background: "var(--background-color-component-skeleton)",
      animation: "var(--animate-pulse)"
    }, style);
    return h("div", {
      className: cx("kui-skeleton", className),
      style: s,
      ...rest
    });
  }

  /* ---------- Panel ---------- */
  function Panel(props) {
    const {
      className,
      style,
      children,
      ...rest
    } = props;
    return h("div", {
      className: cx("kui-panel", className),
      style,
      ...rest
    }, children);
  }

  /* ---------- Card ---------- */
  function Card(props) {
    const {
      interactive,
      selected,
      className,
      style,
      children,
      ...rest
    } = props;
    return h("div", {
      className: cx("kui-card", interactive && "kui-card--interactive", selected && "kui-card--selected", className),
      style,
      ...rest
    }, children);
  }
  function CardMedia(props) {
    const {
      src,
      alt,
      height,
      className,
      style,
      children,
      ...rest
    } = props;
    const s = Object.assign({
      height: typeof height === "number" ? height + "px" : height
    }, style);
    return h("div", {
      className: cx("kui-card__media", className),
      style: s,
      ...rest
    }, src ? h("img", {
      src,
      alt: alt || ""
    }) : children);
  }
  function CardContent(props) {
    const {
      className,
      style,
      children,
      ...rest
    } = props;
    return h("div", {
      className: cx("kui-card__content", className),
      style,
      ...rest
    }, children);
  }
  function CardFooter(props) {
    const {
      className,
      style,
      children,
      ...rest
    } = props;
    return h("div", {
      className: cx("kui-card__footer", className),
      style,
      ...rest
    }, children);
  }
  Card.Media = CardMedia;
  Card.Content = CardContent;
  Card.Footer = CardFooter;

  /* ---------- Table (composed primitives) ---------- */
  function TableRoot(props) {
    const {
      hoverableRows,
      density = "standard",
      layout = "fixed",
      align = "left",
      className,
      style,
      children,
      ...rest
    } = props;
    return h("div", {
      className: "kui-table-wrap"
    }, h("table", {
      className: cx("kui-table", "kui-table--" + layout, "kui-table--" + density, hoverableRows && "kui-table--hover", align !== "left" && "kui-table--align-" + align, className),
      style,
      ...rest
    }, children));
  }
  const TableHead = p => h("thead", p, p.children);
  const TableBody = p => h("tbody", p, p.children);
  function TableRow(props) {
    const {
      selected,
      onClick,
      className,
      children,
      ...rest
    } = props;
    return h("tr", {
      className: cx(selected && "is-selected", className),
      onClick,
      ...rest
    }, children);
  }
  function TableHeaderCell(props) {
    const {
      onSelect,
      sortDir,
      className,
      children,
      ...rest
    } = props;
    const inner = h("span", {
      className: "kui-table__th-inner"
    }, children, sortDir ? h(Icon, {
      name: sortDir === "asc" ? "ChevronUp" : "ChevronDown",
      size: 12
    }) : null);
    return h("th", {
      className: cx(onSelect && "kui-table__sortable", className),
      onClick: onSelect,
      ...rest
    }, inner);
  }
  function TableDataCell(props) {
    const {
      select,
      className,
      children,
      ...rest
    } = props;
    return h("td", {
      className: cx(select && "kui-table__select", className),
      ...rest
    }, children);
  }

  /* high-level declarative Table (columns / rows) */
  function Table(props) {
    const {
      columns = [],
      rows = [],
      hoverableRows,
      density,
      layout,
      align,
      ...rest
    } = props;
    const anySelectable = rows.some(r => r.onRowSelect);
    return h(TableRoot, {
      hoverableRows,
      density,
      layout,
      align,
      ...rest
    }, h(TableHead, null, h(TableRow, null, anySelectable ? h(TableHeaderCell, {
      className: "kui-table__select"
    }) : null, columns.map((col, i) => {
      const isObj = col && typeof col === "object";
      return h(TableHeaderCell, {
        key: i,
        onSelect: isObj && col.onColumnSelect ? () => col.onColumnSelect({
          columnIndex: i
        }) : undefined,
        sortDir: isObj ? col.sortDir : undefined
      }, isObj ? col.children : col);
    }))), h(TableBody, null, rows.map(row => h(TableRow, {
      key: row.id,
      selected: row.selected,
      onClick: row.onRowSelect ? () => row.onRowSelect({
        rowId: row.id
      }) : row.onClick,
      style: row.onRowSelect || row.onClick ? {
        cursor: "pointer"
      } : undefined
    }, row.onRowSelect ? h(TableDataCell, {
      select: true
    }, h("input", {
      type: "checkbox",
      checked: !!row.selected,
      onChange: () => row.onRowSelect({
        rowId: row.id
      }),
      "aria-label": "Select row"
    })) : null, (row.cells || []).map((cell, ci) => {
      const isObj = cell && typeof cell === "object" && !R.isValidElement(cell);
      return h(TableDataCell, {
        key: ci,
        onClick: isObj && cell.onCellSelect ? () => cell.onCellSelect({
          rowId: row.id,
          columnIndex: ci
        }) : undefined
      }, isObj ? cell.children : cell);
    })))));
  }
  function TableToolbar(props) {
    const {
      showBulkActionsToolbar,
      slotBulkActions,
      className,
      style,
      children,
      ...rest
    } = props;
    if (showBulkActionsToolbar) {
      return h("div", {
        className: cx("kui-table-toolbar", "kui-table-toolbar--bulk", className),
        style,
        ...rest
      }, slotBulkActions);
    }
    // place first child left, rest right
    const kids = R.Children.toArray(children);
    return h("div", {
      className: cx("kui-table-toolbar", className),
      style,
      ...rest
    }, kids[0] || null, h("div", {
      className: "kui-table-toolbar__spacer"
    }), kids.slice(1));
  }

  /* ---------- export ---------- */
  global.KUI = Object.assign(global.KUI || {}, {
    cx,
    space,
    ThemeProvider,
    ThemeCtx,
    Text,
    Flex,
    Stack,
    Grid,
    Inline,
    Block,
    Group,
    Divider,
    Badge,
    Tag,
    Avatar,
    Spinner,
    ProgressBar,
    StatusIndicator,
    Skeleton,
    Panel,
    Card,
    CardMedia,
    CardContent,
    CardFooter,
    Table,
    TableRoot,
    TableHead,
    TableBody,
    TableRow,
    TableHeaderCell,
    TableDataCell,
    TableToolbar
  });
})(window);
})(); } catch (e) { __ds_ns.__errors.push({ path: "kaizen-react/kui-core.js", error: String((e && e.message) || e) }); }

// ui_kits/kaizen-app/AppBar.jsx
try { (() => {
// Top App Bar component for the Kaizen Console.
// Hosts the NVIDIA wordmark, product name, primary nav, and the user menu.
function AppBar({
  onMenu,
  productName = "Kaizen Console",
  current = "Clusters",
  user = "AN"
}) {
  const navItems = ["Overview", "Clusters", "Workloads", "Storage", "Settings"];
  return /*#__PURE__*/React.createElement("header", {
    className: "kui-appbar"
  }, /*#__PURE__*/React.createElement("button", {
    className: "kui-btn kui-btn--tertiary kui-btn--icon",
    onClick: onMenu,
    "aria-label": "Menu"
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "menu-line"
  })), /*#__PURE__*/React.createElement("div", {
    className: "kui-appbar__brand"
  }, /*#__PURE__*/React.createElement("img", {
    className: "wm",
    src: "../../assets/logos/NVIDIA-wordmark.svg",
    alt: "NVIDIA"
  }), /*#__PURE__*/React.createElement("span", {
    className: "pname"
  }, productName), /*#__PURE__*/React.createElement(Icon, {
    name: "chevron-down-line",
    color: "var(--kui-fg-muted)"
  })), /*#__PURE__*/React.createElement("nav", {
    className: "kui-appbar__nav",
    "aria-label": "Primary"
  }, navItems.map(item => /*#__PURE__*/React.createElement("a", {
    key: item,
    href: "#",
    className: item === current ? "is-current" : ""
  }, item, item === "Storage" && /*#__PURE__*/React.createElement(Icon, {
    name: "chevron-down-line"
  })))), /*#__PURE__*/React.createElement("div", {
    className: "kui-appbar__spacer"
  }), /*#__PURE__*/React.createElement("div", {
    className: "kui-input-shell",
    style: {
      width: 220
    }
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "search-line",
    color: "var(--kui-fg-muted)"
  }), /*#__PURE__*/React.createElement("input", {
    className: "kui-input",
    placeholder: "Search clusters, jobs\u2026"
  }), /*#__PURE__*/React.createElement("span", {
    className: "right-icons"
  }, /*#__PURE__*/React.createElement("kbd", {
    style: {
      font: "500 11px/1 var(--kui-font-mono)",
      color: "var(--kui-fg-muted)",
      padding: "2px 5px",
      border: "1px solid var(--kui-border-default)",
      borderRadius: 2
    }
  }, "\u2318K"))), /*#__PURE__*/React.createElement("button", {
    className: "kui-btn kui-btn--tertiary kui-btn--icon",
    style: {
      position: "relative"
    },
    "aria-label": "Notifications"
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "bell-line"
  }), /*#__PURE__*/React.createElement("span", {
    className: "kui-badge",
    style: {
      position: "absolute",
      top: 3,
      right: 3,
      transform: "scale(.75)",
      transformOrigin: "top right"
    }
  }, "3")), /*#__PURE__*/React.createElement("button", {
    className: "kui-btn kui-btn--tertiary",
    style: {
      paddingLeft: 4,
      paddingRight: 6,
      gap: 6
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "kui-avatar kui-avatar--small",
    style: {
      background: "#76B900",
      color: "#000"
    }
  }, user), /*#__PURE__*/React.createElement(Icon, {
    name: "chevron-down-line",
    color: "var(--kui-fg-muted)"
  })));
}
window.AppBar = AppBar;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/kaizen-app/AppBar.jsx", error: String((e && e.message) || e) }); }

// ui_kits/kaizen-app/ClusterDetails.jsx
try { (() => {
// Right-hand details rail. Mix of metric cards, a banner, and actions.
function ClusterDetails({
  name = "kaizen-prod-01"
}) {
  return /*#__PURE__*/React.createElement("aside", {
    style: {
      width: 360,
      flex: "none",
      padding: 20,
      background: "#fff",
      borderLeft: "1px solid var(--kui-border-default)",
      display: "flex",
      flexDirection: "column",
      gap: 20,
      overflowY: "auto"
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      alignItems: "center",
      gap: 10
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      width: 36,
      height: 36,
      borderRadius: 4,
      background: "#0D0D0D",
      display: "grid",
      placeItems: "center",
      color: "#76B900"
    }
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "gpu-line",
    size: 20
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      flexDirection: "column",
      gap: 2
    }
  }, /*#__PURE__*/React.createElement("h3", {
    style: {
      font: "var(--kui-h4)",
      letterSpacing: "-0.25px",
      margin: 0
    }
  }, name), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      alignItems: "center",
      gap: 6,
      font: "var(--kui-text-sm)",
      color: "var(--kui-fg-muted)"
    }
  }, /*#__PURE__*/React.createElement("span", {
    className: "kui-badge kui-badge--dot",
    style: {
      background: "#007D00"
    }
  }), "Online \xB7 us-west-2 \xB7 32 \xD7 H100")), /*#__PURE__*/React.createElement("button", {
    className: "kui-btn kui-btn--tertiary kui-btn--icon",
    style: {
      marginLeft: "auto"
    }
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "external-link"
  }))), /*#__PURE__*/React.createElement("div", {
    className: "kui-banner kui-banner--info",
    style: {
      padding: 10,
      gap: 10
    }
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "info-circle-fill"
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      flex: 1
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      font: "500 13px/1.25 var(--kui-font-sans)"
    }
  }, "Maintenance scheduled"), /*#__PURE__*/React.createElement("div", {
    style: {
      font: "var(--kui-text-sm)",
      color: "var(--kui-fg-muted)"
    }
  }, "Sun, May 25 \xB7 02:00\u201303:30 UTC"))), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "grid",
      gridTemplateColumns: "1fr 1fr",
      gap: 12
    }
  }, /*#__PURE__*/React.createElement(Metric, {
    label: "GPUs in use",
    value: "25 / 32",
    sub: "78% utilization"
  }), /*#__PURE__*/React.createElement(Metric, {
    label: "Active jobs",
    value: "14",
    sub: "3 queued"
  }), /*#__PURE__*/React.createElement(Metric, {
    label: "Throughput",
    value: "412 GB/s",
    sub: "last 1 min"
  }), /*#__PURE__*/React.createElement(Metric, {
    label: "Memory",
    value: "68%",
    sub: "2.1 TB / 3.1 TB"
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      flexDirection: "column",
      gap: 8
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      font: "500 10px/1 var(--kui-font-sans)",
      letterSpacing: "0.06em",
      textTransform: "uppercase",
      color: "var(--kui-fg-muted)"
    }
  }, "Members"), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      alignItems: "center",
      gap: 8
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: "inline-flex"
    }
  }, [{
    i: "AN",
    c: "#76B900",
    t: "#000"
  }, {
    i: "JM",
    c: "#1FA18D",
    t: "#fff"
  }, {
    i: "DR",
    c: "#851F41",
    t: "#fff"
  }, {
    i: "KO",
    c: "#A05AB4",
    t: "#fff"
  }].map((u, k) => /*#__PURE__*/React.createElement("div", {
    key: k,
    className: "kui-avatar kui-avatar--small",
    style: {
      background: u.c,
      color: u.t,
      boxShadow: "0 0 0 2px #fff",
      marginLeft: k ? -6 : 0
    }
  }, u.i)), /*#__PURE__*/React.createElement("div", {
    className: "kui-avatar kui-avatar--small",
    style: {
      background: "var(--kui-n050)",
      color: "var(--kui-fg-strong)",
      boxShadow: "0 0 0 2px #fff",
      marginLeft: -6
    }
  }, "+8")), /*#__PURE__*/React.createElement("button", {
    className: "kui-btn kui-btn--tertiary kui-btn--small",
    style: {
      marginLeft: "auto"
    }
  }, "Manage"))), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      flexDirection: "column",
      gap: 8
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      font: "500 10px/1 var(--kui-font-sans)",
      letterSpacing: "0.06em",
      textTransform: "uppercase",
      color: "var(--kui-fg-muted)"
    }
  }, "Recent activity"), [{
    t: "Job qwen-2-finetune started",
    w: "AN",
    a: "2 min ago"
  }, {
    t: "Quota raised to 32 GPUs",
    w: "system",
    a: "14 min ago"
  }, {
    t: "Driver 565.77 installed",
    w: "JM",
    a: "1 h ago"
  }].map((r, i) => /*#__PURE__*/React.createElement("div", {
    key: i,
    style: {
      display: "flex",
      alignItems: "center",
      gap: 8,
      padding: "8px 4px",
      borderBottom: i < 2 ? "1px solid var(--kui-border-subtle)" : "none"
    }
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "clock-fill",
    color: "var(--kui-fg-muted)"
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      font: "var(--kui-text)"
    }
  }, r.t), /*#__PURE__*/React.createElement("div", {
    style: {
      marginLeft: "auto",
      font: "var(--kui-text-sm)",
      color: "var(--kui-fg-muted)"
    }
  }, r.a)))), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      gap: 8,
      marginTop: "auto"
    }
  }, /*#__PURE__*/React.createElement("button", {
    className: "kui-btn kui-btn--secondary kui-btn--medium",
    style: {
      flex: 1
    }
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "download-line"
  }), " Export logs"), /*#__PURE__*/React.createElement("button", {
    className: "kui-btn kui-btn--primary kui-btn--medium",
    style: {
      flex: 1
    }
  }, "Open console")));
}
function Metric({
  label,
  value,
  sub
}) {
  return /*#__PURE__*/React.createElement("div", {
    style: {
      padding: 12,
      border: "1px solid var(--kui-border-default)",
      borderRadius: 4,
      background: "#fff",
      display: "flex",
      flexDirection: "column",
      gap: 2
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      font: "500 10px/1 var(--kui-font-sans)",
      letterSpacing: "0.06em",
      textTransform: "uppercase",
      color: "var(--kui-fg-muted)"
    }
  }, label), /*#__PURE__*/React.createElement("div", {
    style: {
      font: "500 20px/1.25 var(--kui-font-sans)",
      letterSpacing: "-0.25px",
      color: "var(--kui-fg-strong)"
    }
  }, value), /*#__PURE__*/React.createElement("div", {
    style: {
      font: "var(--kui-text-sm)",
      color: "var(--kui-fg-muted)"
    }
  }, sub));
}
window.ClusterDetails = ClusterDetails;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/kaizen-app/ClusterDetails.jsx", error: String((e && e.message) || e) }); }

// ui_kits/kaizen-app/ClusterTable.jsx
try { (() => {
// Sortable-looking table of GPU clusters.
function ClusterTable({
  selectedId,
  onSelect,
  onDelete
}) {
  const rows = [{
    id: "kp01",
    name: "kaizen-prod-01",
    region: "us-west-2",
    gpus: "32× H100",
    status: "Online",
    statusKind: "green",
    updated: "2 min ago",
    util: 78
  }, {
    id: "kst",
    name: "kaizen-staging",
    region: "us-east-1",
    gpus: "16× A100",
    status: "Updating",
    statusKind: "blue",
    updated: "14 min ago",
    util: 12
  }, {
    id: "keu",
    name: "kaizen-eu",
    region: "eu-west-1",
    gpus: "8× L40S",
    status: "Throttled",
    statusKind: "orange",
    updated: "1 h ago",
    util: 94
  }, {
    id: "kdev",
    name: "kaizen-dev",
    region: "us-east-2",
    gpus: "4× L4",
    status: "Offline",
    statusKind: "red",
    updated: "3 h ago",
    util: 0
  }, {
    id: "kap",
    name: "kaizen-apac",
    region: "ap-south-1",
    gpus: "24× H100",
    status: "Online",
    statusKind: "green",
    updated: "5 min ago",
    util: 55
  }];
  return /*#__PURE__*/React.createElement("div", {
    className: "kui-card",
    style: {
      background: "#fff"
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      alignItems: "center",
      padding: "12px 16px",
      gap: 8,
      borderBottom: "1px solid var(--kui-border-subtle)"
    }
  }, /*#__PURE__*/React.createElement("h3", {
    style: {
      font: "var(--kui-h4)",
      letterSpacing: "-0.25px",
      margin: 0
    }
  }, "Clusters"), /*#__PURE__*/React.createElement("span", {
    className: "kui-tag kui-tag--green",
    style: {
      marginLeft: 4
    }
  }, "5 active"), /*#__PURE__*/React.createElement("div", {
    style: {
      flex: 1
    }
  }), /*#__PURE__*/React.createElement("div", {
    className: "kui-input-shell kui-input-shell--small",
    style: {
      width: 220
    }
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "search-line",
    color: "var(--kui-fg-muted)"
  }), /*#__PURE__*/React.createElement("input", {
    className: "kui-input",
    placeholder: "Filter by name or region"
  })), /*#__PURE__*/React.createElement("button", {
    className: "kui-btn kui-btn--secondary kui-btn--small"
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "filter-line"
  }), " Filter"), /*#__PURE__*/React.createElement("button", {
    className: "kui-btn kui-btn--primary kui-btn--small"
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "plus-line"
  }), " New cluster")), /*#__PURE__*/React.createElement("table", {
    className: "kui-table"
  }, /*#__PURE__*/React.createElement("thead", null, /*#__PURE__*/React.createElement("tr", null, /*#__PURE__*/React.createElement("th", {
    style: {
      width: 32
    }
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    className: "kui-check"
  })), /*#__PURE__*/React.createElement("th", null, "Cluster"), /*#__PURE__*/React.createElement("th", null, "Region"), /*#__PURE__*/React.createElement("th", null, "GPUs"), /*#__PURE__*/React.createElement("th", null, "Utilization"), /*#__PURE__*/React.createElement("th", null, "Status"), /*#__PURE__*/React.createElement("th", null, "Updated"), /*#__PURE__*/React.createElement("th", {
    style: {
      width: 80
    }
  }))), /*#__PURE__*/React.createElement("tbody", null, rows.map(r => /*#__PURE__*/React.createElement("tr", {
    key: r.id,
    style: {
      background: selectedId === r.id ? "var(--kui-n050)" : undefined,
      cursor: "pointer"
    },
    onClick: () => onSelect && onSelect(r.id)
  }, /*#__PURE__*/React.createElement("td", null, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    className: "kui-check",
    onClick: e => e.stopPropagation()
  })), /*#__PURE__*/React.createElement("td", null, /*#__PURE__*/React.createElement("span", {
    style: {
      font: "500 14px/1 var(--kui-font-sans)"
    }
  }, r.name)), /*#__PURE__*/React.createElement("td", null, /*#__PURE__*/React.createElement("span", {
    style: {
      color: "var(--kui-fg-muted)"
    }
  }, r.region)), /*#__PURE__*/React.createElement("td", null, r.gpus), /*#__PURE__*/React.createElement("td", {
    style: {
      minWidth: 120
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      alignItems: "center",
      gap: 8
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "kui-progress",
    style: {
      width: 80
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "kui-progress__bar",
    style: {
      width: `${r.util}%`,
      background: r.util > 90 ? "#C54600" : r.util > 60 ? "#76B900" : "var(--kui-n400)"
    }
  })), /*#__PURE__*/React.createElement("span", {
    style: {
      font: "var(--kui-text-sm)",
      color: "var(--kui-fg-muted)",
      minWidth: 32
    }
  }, r.util, "%"))), /*#__PURE__*/React.createElement("td", null, /*#__PURE__*/React.createElement("span", {
    className: `kui-tag kui-tag--${r.statusKind}`
  }, r.status)), /*#__PURE__*/React.createElement("td", null, /*#__PURE__*/React.createElement("span", {
    style: {
      color: "var(--kui-fg-muted)"
    }
  }, r.updated)), /*#__PURE__*/React.createElement("td", {
    onClick: e => e.stopPropagation()
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      gap: 4,
      justifyContent: "flex-end"
    }
  }, /*#__PURE__*/React.createElement("button", {
    className: "kui-btn kui-btn--tertiary kui-btn--icon kui-btn--small"
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "pencil-fill"
  })), /*#__PURE__*/React.createElement("button", {
    className: "kui-btn kui-btn--tertiary kui-btn--icon kui-btn--small",
    onClick: () => onDelete && onDelete(r.id, r.name)
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "more-horizontal"
  })))))))));
}
window.ClusterTable = ClusterTable;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/kaizen-app/ClusterTable.jsx", error: String((e && e.message) || e) }); }

// ui_kits/kaizen-app/Modal.jsx
try { (() => {
// Destructive confirmation modal.
function ConfirmDeleteModal({
  open,
  name,
  onCancel,
  onConfirm
}) {
  if (!open) return null;
  return /*#__PURE__*/React.createElement("div", {
    className: "kui-blanket",
    onClick: onCancel
  }, /*#__PURE__*/React.createElement("div", {
    className: "kui-modal",
    onClick: e => e.stopPropagation(),
    role: "dialog",
    "aria-labelledby": "cdm-title"
  }, /*#__PURE__*/React.createElement("div", {
    className: "kui-modal__head"
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "warning-fill",
    color: "#C54600"
  }), /*#__PURE__*/React.createElement("div", {
    id: "cdm-title",
    className: "kui-modal__title"
  }, "Delete cluster \"", name, "\"?"), /*#__PURE__*/React.createElement("button", {
    className: "kui-modal__close kui-btn kui-btn--tertiary kui-btn--icon kui-btn--small",
    onClick: onCancel
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "close-line"
  }))), /*#__PURE__*/React.createElement("div", {
    className: "kui-modal__body"
  }, "This stops all running nodes and detaches attached volumes. Active workloads will fail. This action cannot be undone.", /*#__PURE__*/React.createElement("div", {
    style: {
      marginTop: 12,
      padding: 10,
      background: "var(--kui-n050)",
      borderRadius: 2,
      font: "var(--kui-code)"
    }
  }, "Type ", /*#__PURE__*/React.createElement("strong", {
    style: {
      color: "var(--kui-fg-strong)"
    }
  }, name), " to confirm."), /*#__PURE__*/React.createElement("div", {
    className: "kui-input-shell",
    style: {
      width: "100%",
      marginTop: 8
    }
  }, /*#__PURE__*/React.createElement("input", {
    className: "kui-input",
    placeholder: name
  }))), /*#__PURE__*/React.createElement("div", {
    className: "kui-modal__foot"
  }, /*#__PURE__*/React.createElement("button", {
    className: "kui-btn kui-btn--tertiary kui-btn--medium",
    onClick: onCancel
  }, "Cancel"), /*#__PURE__*/React.createElement("button", {
    className: "kui-btn kui-btn--primary kui-btn--danger kui-btn--medium",
    onClick: onConfirm
  }, "Delete cluster"))));
}
window.ConfirmDeleteModal = ConfirmDeleteModal;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/kaizen-app/Modal.jsx", error: String((e && e.message) || e) }); }

// ui_kits/kaizen-app/Sidebar.jsx
try { (() => {
// Vertical navigation. Sections + current item.
function Sidebar({
  current = "clusters",
  onNavigate
}) {
  const items = [{
    section: "Workspace"
  }, {
    id: "overview",
    label: "Overview",
    icon: "home-line"
  }, {
    id: "clusters",
    label: "Clusters",
    icon: "gpu-line"
  }, {
    id: "workloads",
    label: "Workloads",
    icon: "cog-fill"
  }, {
    id: "activity",
    label: "Activity",
    icon: "clock-fill"
  }, {
    section: "Admin"
  }, {
    id: "members",
    label: "Members",
    icon: "user-line"
  }, {
    id: "settings",
    label: "Settings",
    icon: "cog-fill"
  }];
  return /*#__PURE__*/React.createElement("aside", {
    className: "kui-vnav",
    style: {
      height: "100%"
    }
  }, items.map((it, i) => it.section ? /*#__PURE__*/React.createElement("div", {
    key: i,
    className: "kui-vnav__section"
  }, it.section) : /*#__PURE__*/React.createElement("a", {
    key: it.id,
    href: "#",
    className: current === it.id ? "is-current" : "",
    onClick: e => {
      e.preventDefault();
      onNavigate && onNavigate(it.id);
    }
  }, /*#__PURE__*/React.createElement(Icon, {
    name: it.icon
  }), it.label)));
}
window.Sidebar = Sidebar;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/kaizen-app/Sidebar.jsx", error: String((e && e.message) || e) }); }

})();
