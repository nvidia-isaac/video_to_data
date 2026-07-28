/* @ds-bundle: {"format":3,"namespace":"DesignSystem_6d5263","components":[{"name":"Logo","sourcePath":"components/brand/Logo.jsx"},{"name":"Button","sourcePath":"components/buttons/Button.jsx"},{"name":"IconButton","sourcePath":"components/buttons/IconButton.jsx"},{"name":"Avatar","sourcePath":"components/data-display/Avatar.jsx"},{"name":"Badge","sourcePath":"components/data-display/Badge.jsx"},{"name":"Card","sourcePath":"components/data-display/Card.jsx"},{"name":"Stat","sourcePath":"components/data-display/Stat.jsx"},{"name":"Tag","sourcePath":"components/data-display/Tag.jsx"},{"name":"Banner","sourcePath":"components/feedback/Banner.jsx"},{"name":"Spinner","sourcePath":"components/feedback/Spinner.jsx"},{"name":"Checkbox","sourcePath":"components/forms/Checkbox.jsx"},{"name":"Input","sourcePath":"components/forms/Input.jsx"},{"name":"Select","sourcePath":"components/forms/Select.jsx"},{"name":"Switch","sourcePath":"components/forms/Switch.jsx"},{"name":"Icon","sourcePath":"components/icons/Icon.jsx"},{"name":"Breadcrumb","sourcePath":"components/navigation/Breadcrumb.jsx"},{"name":"Tabs","sourcePath":"components/navigation/Tabs.jsx"},{"name":"Tooltip","sourcePath":"components/overlay/Tooltip.jsx"}],"sourceHashes":{"components/brand/Logo.jsx":"7aabafb9c08a","components/buttons/Button.jsx":"68b8a8cb1245","components/buttons/IconButton.jsx":"59e2ac640194","components/data-display/Avatar.jsx":"7e3d89413cb8","components/data-display/Badge.jsx":"d0d63ec34139","components/data-display/Card.jsx":"0c013c95e920","components/data-display/Stat.jsx":"9030947c177f","components/data-display/Tag.jsx":"0a4fb2febe50","components/feedback/Banner.jsx":"74236de69d83","components/feedback/Spinner.jsx":"4378fbd3bd36","components/forms/Checkbox.jsx":"6a2d997c0c7a","components/forms/Input.jsx":"9c3487de226c","components/forms/Select.jsx":"2b162a2887fd","components/forms/Switch.jsx":"11e9abba7c1d","components/icons/Icon.jsx":"049cd1502b8c","components/navigation/Breadcrumb.jsx":"5728e1d1e99c","components/navigation/Tabs.jsx":"79244e59fb5b","components/overlay/Tooltip.jsx":"af192f6f48b4","decks/edm-stack/deck-stage.js":"9436a2deeb46","decks/pdmt-boards-systems/deck-stage.js":"9436a2deeb46","slides/deck-stage.js":"9436a2deeb46","ui_kits/web/App.jsx":"167f5443e61d","ui_kits/web/Footer.jsx":"be28ba851d25","ui_kits/web/Hero.jsx":"579f5b009da1","ui_kits/web/NavBar.jsx":"d762d2cadedc","ui_kits/web/Showcase.jsx":"a0bf713ebf82","ui_kits/web/UiIcon.jsx":"d241debd9f5b"},"inlinedExternals":[],"unexposedExports":[{"name":"iconNames","sourcePath":"components/icons/Icon.jsx"}]} */

(() => {

const __ds_ns = (window.DesignSystem_6d5263 = window.DesignSystem_6d5263 || {});

const __ds_scope = {};

(__ds_ns.__errors = __ds_ns.__errors || []);

// components/brand/Logo.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
// Real NVIDIA logo geometry, extracted verbatim from the Kaizen / KUI Figma
// ("Core / Logo / …"). Exact vectors — prefer these over raster logos.

// Standalone eye mark — viewBox 0 0 71 47
const EYE_PATH = "M 7.255 20.234 C 7.255 20.234 13.674 10.758 26.491 9.779 L 26.491 6.341 C 12.294 7.481 0 19.509 0 19.509 C 0 19.509 6.963 39.647 26.491 41.491 L 26.491 37.836 C 12.161 36.032 7.255 20.234 7.255 20.234 Z M 26.491 30.57 L 26.491 33.917 C 15.66 31.985 12.654 20.722 12.654 20.722 C 12.654 20.722 17.855 14.958 26.491 14.024 L 26.491 17.696 C 26.484 17.696 26.481 17.694 26.475 17.694 C 21.943 17.15 18.401 21.386 18.401 21.386 C 18.401 21.386 20.385 28.517 26.491 30.57 Z M 26.491 0 L 26.491 6.341 C 26.908 6.308 27.325 6.282 27.744 6.267 C 43.885 5.723 54.402 19.509 54.402 19.509 C 54.402 19.509 42.322 34.203 29.739 34.203 C 28.586 34.203 27.505 34.096 26.491 33.917 L 26.491 37.836 C 27.359 37.947 28.257 38.012 29.195 38.012 C 40.906 38.012 49.375 32.03 57.575 24.949 C 58.934 26.038 64.5 28.687 65.645 29.848 C 57.847 36.378 39.677 41.641 29.376 41.641 C 28.382 41.641 27.428 41.581 26.491 41.491 L 26.491 47 L 71 47 L 71 0 L 26.491 0 Z M 26.491 14.024 L 26.491 9.779 C 26.903 9.749 27.32 9.727 27.744 9.714 C 39.351 9.349 46.966 19.691 46.966 19.691 C 46.966 19.691 38.742 31.119 29.923 31.119 C 28.653 31.119 27.516 30.914 26.491 30.57 L 26.491 17.696 C 31.01 18.242 31.919 20.239 34.636 24.769 L 40.678 19.673 C 40.678 19.673 36.268 13.886 28.833 13.886 C 28.024 13.886 27.25 13.943 26.491 14.024 Z";

// Lockup eye — viewBox 0 0 45.264 29.952
const LOCKUP_EYE_PATH = "M 16.889 8.938 L 16.889 6.232 C 17.151 6.213 17.417 6.199 17.687 6.191 C 25.087 5.958 29.941 12.549 29.941 12.549 C 29.941 12.549 24.698 19.832 19.076 19.832 C 18.267 19.832 17.542 19.701 16.889 19.482 L 16.889 11.278 C 19.769 11.626 20.349 12.898 22.081 15.785 L 25.933 12.537 C 25.933 12.537 23.121 8.849 18.381 8.849 C 17.865 8.849 17.373 8.886 16.889 8.938 Z M 16.889 0 L 16.889 4.041 C 17.154 4.02 17.42 4.003 17.687 3.994 C 27.978 3.647 34.682 12.433 34.682 12.433 C 34.682 12.433 26.981 21.797 18.959 21.797 C 18.224 21.797 17.535 21.729 16.889 21.615 L 16.889 24.113 C 17.442 24.183 18.015 24.225 18.613 24.225 C 26.078 24.225 31.477 20.412 36.705 15.9 C 37.572 16.594 41.12 18.282 41.85 19.022 C 36.879 23.183 25.295 26.537 18.727 26.537 C 18.094 26.537 17.486 26.499 16.889 26.442 L 16.889 29.952 L 45.264 29.952 L 45.264 0 L 16.889 0 Z M 16.889 19.482 L 16.889 21.615 C 9.984 20.384 8.067 13.206 8.067 13.206 C 8.067 13.206 11.382 9.533 16.889 8.938 L 16.889 11.278 C 16.884 11.278 16.882 11.276 16.878 11.276 C 13.989 10.929 11.731 13.629 11.731 13.629 C 11.731 13.629 12.996 18.174 16.889 19.482 Z M 4.625 12.895 C 4.625 12.895 8.717 6.856 16.889 6.232 L 16.889 4.041 C 7.838 4.768 0 12.433 0 12.433 C 0 12.433 4.439 25.267 16.889 26.442 L 16.889 24.113 C 7.753 22.963 4.625 12.895 4.625 12.895 Z";
const WORDMARK_PATH = "M 43.677 0.027 L 43.679 19.686 L 49.231 19.686 L 49.231 0.027 L 43.677 0.027 Z M 0 0 L 0 19.686 L 5.602 19.686 L 5.602 4.405 L 9.971 4.419 C 11.409 4.419 12.402 4.764 13.095 5.503 C 13.973 6.439 14.332 7.948 14.332 10.709 L 14.332 19.686 L 19.759 19.686 L 19.759 8.809 C 19.759 1.047 14.811 0 9.97 0 L 0 0 Z M 52.617 0.027 L 52.617 19.686 L 61.622 19.686 C 66.421 19.686 67.986 18.888 69.68 17.099 C 70.877 15.843 71.651 13.086 71.651 10.073 C 71.651 7.309 70.996 4.844 69.854 3.309 C 67.797 0.564 64.833 0.027 60.409 0.027 L 52.617 0.027 Z M 58.124 4.308 L 60.512 4.308 C 63.975 4.308 66.215 5.863 66.215 9.899 C 66.215 13.935 63.975 15.49 60.512 15.49 L 58.124 15.49 L 58.124 4.308 Z M 35.672 0.027 L 31.038 15.608 L 26.598 0.028 L 20.604 0.027 L 26.945 19.686 L 34.948 19.686 L 41.339 0.027 L 35.672 0.027 Z M 74.234 19.686 L 79.787 19.686 L 79.787 0.028 L 74.233 0.027 L 74.234 19.686 Z M 89.799 0.034 L 82.046 19.679 L 87.521 19.679 L 88.747 16.207 L 97.923 16.207 L 99.084 19.679 L 105.028 19.679 L 97.215 0.033 L 89.799 0.034 Z M 93.403 3.618 L 96.766 12.822 L 89.934 12.822 L 93.403 3.618 Z";
const REG_PATH = "M 1.507 1.562 L 1.507 1.113 L 1.795 1.113 C 1.953 1.113 2.167 1.125 2.167 1.317 C 2.167 1.525 2.056 1.562 1.871 1.562 L 1.507 1.562 Z M 1.507 1.877 L 1.7 1.877 L 2.146 2.661 L 2.636 2.661 L 2.142 1.845 C 2.398 1.826 2.608 1.705 2.608 1.361 C 2.608 0.933 2.313 0.796 1.814 0.796 L 1.093 0.796 L 1.093 2.661 L 1.507 2.661 L 1.507 1.877 Z M 3.605 1.731 C 3.605 0.635 2.754 0 1.806 0 C 0.851 0 0 0.635 0 1.731 C 0 2.826 0.851 3.464 1.806 3.464 C 2.754 3.464 3.605 2.826 3.605 1.731 Z M 3.086 1.731 C 3.086 2.529 2.499 3.065 1.806 3.065 L 1.806 3.059 C 1.093 3.065 0.517 2.529 0.517 1.731 C 0.517 0.933 1.093 0.4 1.806 0.4 C 2.499 0.4 3.086 0.933 3.086 1.731 Z";
const GREEN = "#76B900";

/**
 * NVIDIA Logo. `variant="eye"` renders the eye mark alone (tone green/black/
 * white); `variant="horizontal"` renders the full lockup (green eye + NVIDIA
 * wordmark; wordmark uses `color`/tone). Set `height`; width scales.
 */
function Logo({
  variant = "horizontal",
  tone = "green",
  height,
  color,
  style,
  ...rest
}) {
  if (variant === "eye") {
    const fill = tone === "white" ? "#FFFFFF" : tone === "black" ? "#0A0A0A" : GREEN;
    const h = height || 32;
    return /*#__PURE__*/React.createElement("svg", _extends({
      viewBox: "0 0 71 47",
      height: h,
      width: h * 71 / 47,
      fill: "none",
      role: "img",
      "aria-label": "NVIDIA",
      style: style
    }, rest), /*#__PURE__*/React.createElement("path", {
      d: EYE_PATH,
      fill: fill,
      fillRule: "evenodd"
    }));
  }

  // Horizontal lockup — exact 164×30 composite from Figma, scaled by height.
  const h = height || 28;
  const ink = color || (tone === "white" ? "#FFFFFF" : "#0A0A0A");
  return /*#__PURE__*/React.createElement("svg", _extends({
    viewBox: "0 0 164 30",
    height: h,
    width: h * 164 / 30,
    fill: "none",
    role: "img",
    "aria-label": "NVIDIA",
    style: style
  }, rest), /*#__PURE__*/React.createElement("g", {
    transform: "translate(0,0.047)"
  }, /*#__PURE__*/React.createElement("g", {
    transform: "translate(0,0) scale(1)"
  }, /*#__PURE__*/React.createElement("path", {
    transform: "translate(0,0)",
    d: LOCKUP_EYE_PATH,
    fill: GREEN,
    fillRule: "evenodd"
  }))), /*#__PURE__*/React.createElement("path", {
    transform: "translate(52.687,5.633)",
    d: WORDMARK_PATH,
    fill: ink,
    fillRule: "evenodd"
  }), /*#__PURE__*/React.createElement("path", {
    transform: "translate(158.844,22.507)",
    d: REG_PATH,
    fill: ink,
    fillRule: "evenodd"
  }));
}
Object.assign(__ds_scope, { Logo });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/brand/Logo.jsx", error: String((e && e.message) || e) }); }

// components/buttons/Button.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * NVIDIA Button — the primary call to action.
 * Square-ish corners, NVIDIA green fill with black ink (green is too bright
 * for white text). Secondary = bordered, ghost = text-only.
 */
function Button({
  children,
  variant = "primary",
  size = "md",
  disabled = false,
  iconLeft = null,
  iconRight = null,
  fullWidth = false,
  type = "button",
  onClick,
  style,
  ...rest
}) {
  const sizes = {
    sm: {
      fontSize: "var(--fs-xs)",
      padding: "0 14px",
      height: 32,
      gap: 6
    },
    md: {
      fontSize: "var(--fs-sm)",
      padding: "0 20px",
      height: 40,
      gap: 8
    },
    lg: {
      fontSize: "var(--fs-base)",
      padding: "0 28px",
      height: 48,
      gap: 10
    }
  };
  const s = sizes[size] || sizes.md;
  const variants = {
    primary: {
      background: "var(--nv-green)",
      color: "var(--nv-black)",
      border: "1px solid var(--nv-green)"
    },
    secondary: {
      background: "transparent",
      color: "var(--text-primary)",
      border: "1px solid var(--border-strong)"
    },
    ghost: {
      background: "transparent",
      color: "var(--text-primary)",
      border: "1px solid transparent"
    },
    inverse: {
      background: "var(--nv-white)",
      color: "var(--nv-black)",
      border: "1px solid var(--nv-white)"
    }
  };
  const v = variants[variant] || variants.primary;
  const [hover, setHover] = React.useState(false);
  const [press, setPress] = React.useState(false);
  const hoverStyle = !disabled && hover ? {
    primary: {
      background: press ? "var(--nv-green-700)" : "var(--nv-green-300)",
      borderColor: "transparent"
    },
    secondary: {
      background: "var(--nv-gray-50)",
      borderColor: "var(--nv-green)"
    },
    ghost: {
      background: "var(--nv-gray-50)"
    },
    inverse: {
      background: "var(--nv-gray-100)"
    }
  }[variant] : null;
  return /*#__PURE__*/React.createElement("button", _extends({
    type: type,
    disabled: disabled,
    onClick: onClick,
    onMouseEnter: () => setHover(true),
    onMouseLeave: () => {
      setHover(false);
      setPress(false);
    },
    onMouseDown: () => setPress(true),
    onMouseUp: () => setPress(false),
    style: {
      display: fullWidth ? "flex" : "inline-flex",
      width: fullWidth ? "100%" : undefined,
      alignItems: "center",
      justifyContent: "center",
      gap: s.gap,
      height: s.height,
      padding: s.padding,
      fontFamily: "var(--font-sans)",
      fontSize: s.fontSize,
      fontWeight: "var(--fw-semibold)",
      lineHeight: 1,
      letterSpacing: "0.01em",
      borderRadius: "var(--radius-sm)",
      cursor: disabled ? "not-allowed" : "pointer",
      opacity: disabled ? 0.45 : 1,
      transform: press && !disabled ? "translateY(0.5px)" : "none",
      transition: "background var(--dur-fast) var(--ease-standard), border-color var(--dur-fast) var(--ease-standard)",
      whiteSpace: "nowrap",
      ...v,
      ...hoverStyle,
      ...style
    }
  }, rest), iconLeft, children, iconRight);
}
Object.assign(__ds_scope, { Button });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/buttons/Button.jsx", error: String((e && e.message) || e) }); }

// components/buttons/IconButton.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * NVIDIA IconButton — a square, icon-only action (toolbar, close, kebab).
 * Pass a single icon node as children. 44px hit target at md by default.
 */
function IconButton({
  children,
  variant = "ghost",
  size = "md",
  disabled = false,
  "aria-label": ariaLabel,
  onClick,
  style,
  ...rest
}) {
  const dims = {
    sm: 32,
    md: 40,
    lg: 48
  };
  const d = dims[size] || dims.md;
  const [hover, setHover] = React.useState(false);
  const variants = {
    ghost: {
      background: hover && !disabled ? "var(--nv-gray-100)" : "transparent",
      color: "var(--text-secondary)",
      border: "1px solid transparent"
    },
    outline: {
      background: hover && !disabled ? "var(--nv-gray-50)" : "transparent",
      color: "var(--text-primary)",
      border: "1px solid var(--border-default)"
    },
    solid: {
      background: hover && !disabled ? "var(--nv-green-300)" : "var(--nv-green)",
      color: "var(--nv-black)",
      border: "1px solid transparent"
    }
  };
  const v = variants[variant] || variants.ghost;
  return /*#__PURE__*/React.createElement("button", _extends({
    type: "button",
    "aria-label": ariaLabel,
    disabled: disabled,
    onClick: onClick,
    onMouseEnter: () => setHover(true),
    onMouseLeave: () => setHover(false),
    style: {
      display: "inline-flex",
      alignItems: "center",
      justifyContent: "center",
      width: d,
      height: d,
      borderRadius: "var(--radius-sm)",
      cursor: disabled ? "not-allowed" : "pointer",
      opacity: disabled ? 0.45 : 1,
      transition: "background var(--dur-fast) var(--ease-standard)",
      ...v,
      ...style
    }
  }, rest), children);
}
Object.assign(__ds_scope, { IconButton });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/buttons/IconButton.jsx", error: String((e && e.message) || e) }); }

// components/data-display/Avatar.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * NVIDIA Avatar — circular user mark. Renders an image when `src` is given,
 * otherwise initials on a neutral fill. Optional green presence dot.
 */
function Avatar({
  src,
  name = "",
  size = 36,
  online = false,
  style,
  ...rest
}) {
  const initials = name.split(" ").map(w => w[0]).filter(Boolean).slice(0, 2).join("").toUpperCase();
  return /*#__PURE__*/React.createElement("span", _extends({
    style: {
      position: "relative",
      display: "inline-flex",
      width: size,
      height: size,
      ...style
    }
  }, rest), /*#__PURE__*/React.createElement("span", {
    style: {
      width: size,
      height: size,
      borderRadius: "var(--radius-pill)",
      overflow: "hidden",
      display: "inline-flex",
      alignItems: "center",
      justifyContent: "center",
      background: "var(--nv-gray-200)",
      color: "var(--nv-gray-700)",
      fontFamily: "var(--font-sans)",
      fontWeight: "var(--fw-semibold)",
      fontSize: Math.round(size * 0.38)
    }
  }, src ? /*#__PURE__*/React.createElement("img", {
    src: src,
    alt: name,
    style: {
      width: "100%",
      height: "100%",
      objectFit: "cover"
    }
  }) : initials), online && /*#__PURE__*/React.createElement("span", {
    style: {
      position: "absolute",
      right: 0,
      bottom: 0,
      width: Math.max(8, size * 0.26),
      height: Math.max(8, size * 0.26),
      borderRadius: "var(--radius-pill)",
      background: "var(--nv-green)",
      border: "2px solid var(--nv-white)"
    }
  }));
}
Object.assign(__ds_scope, { Avatar });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/data-display/Avatar.jsx", error: String((e && e.message) || e) }); }

// components/data-display/Badge.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * NVIDIA Badge — small status/label pill. `solid` green for emphasis,
 * `soft` tints for status, `outline` for neutral metadata.
 */
function Badge({
  children,
  variant = "soft",
  tone = "green",
  style,
  ...rest
}) {
  const tones = {
    green: {
      solid: "var(--nv-green)",
      soft: "var(--nv-green-100)",
      ink: "var(--nv-green-700)"
    },
    neutral: {
      solid: "var(--nv-gray-700)",
      soft: "var(--nv-gray-100)",
      ink: "var(--nv-gray-700)"
    },
    success: {
      solid: "var(--nv-emerald)",
      soft: "#D8F0E8",
      ink: "var(--nv-emerald)"
    },
    info: {
      solid: "var(--nv-cpu-blue)",
      soft: "#D6EAF8",
      ink: "var(--nv-cpu-blue)"
    },
    warning: {
      solid: "var(--nv-fluorite)",
      soft: "#FCF0CC",
      ink: "#8A6D00"
    },
    danger: {
      solid: "var(--nv-garnet)",
      soft: "#F4D9E8",
      ink: "var(--nv-garnet)"
    }
  };
  const t = tones[tone] || tones.green;
  const styles = {
    solid: {
      background: t.solid,
      color: tone === "green" || tone === "warning" ? "var(--nv-black)" : "var(--nv-white)",
      border: "1px solid transparent"
    },
    soft: {
      background: t.soft,
      color: t.ink,
      border: "1px solid transparent"
    },
    outline: {
      background: "transparent",
      color: t.ink,
      border: `1px solid ${t.solid}`
    }
  };
  return /*#__PURE__*/React.createElement("span", _extends({
    style: {
      display: "inline-flex",
      alignItems: "center",
      gap: 5,
      height: 22,
      padding: "0 9px",
      fontFamily: "var(--font-sans)",
      fontSize: "var(--fs-2xs)",
      fontWeight: "var(--fw-semibold)",
      lineHeight: 1,
      letterSpacing: "0.01em",
      borderRadius: "var(--radius-xs)",
      whiteSpace: "nowrap",
      ...(styles[variant] || styles.soft),
      ...style
    }
  }, rest), children);
}
Object.assign(__ds_scope, { Badge });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/data-display/Badge.jsx", error: String((e && e.message) || e) }); }

// components/data-display/Card.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * NVIDIA Card — neutral surface container. White background, hairline border,
 * subtle shadow. `interactive` adds hover lift; `accent` adds a green top bar.
 */
function Card({
  children,
  interactive = false,
  accent = false,
  padding = "20px",
  style,
  onClick,
  ...rest
}) {
  const [hover, setHover] = React.useState(false);
  return /*#__PURE__*/React.createElement("div", _extends({
    onClick: onClick,
    onMouseEnter: () => setHover(true),
    onMouseLeave: () => setHover(false),
    style: {
      position: "relative",
      background: "var(--surface-card)",
      border: "1px solid var(--border-subtle)",
      borderRadius: "var(--radius-md)",
      boxShadow: interactive && hover ? "var(--shadow-md)" : "var(--shadow-sm)",
      transform: interactive && hover ? "translateY(-2px)" : "none",
      transition: "box-shadow var(--dur-base) var(--ease-standard), transform var(--dur-base) var(--ease-standard)",
      cursor: interactive ? "pointer" : "default",
      padding,
      overflow: "hidden",
      ...style
    }
  }, rest), accent && /*#__PURE__*/React.createElement("span", {
    style: {
      position: "absolute",
      top: 0,
      left: 0,
      right: 0,
      height: 3,
      background: "var(--nv-green)"
    }
  }), children);
}
Object.assign(__ds_scope, { Card });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/data-display/Card.jsx", error: String((e && e.message) || e) }); }

// components/data-display/Stat.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * NVIDIA Stat — a big metric figure with label and optional delta. The hero
 * number uses the light display weight; positive deltas read green.
 */
function Stat({
  value,
  label,
  delta,
  deltaDirection = "up",
  style,
  ...rest
}) {
  const positive = deltaDirection === "up";
  return /*#__PURE__*/React.createElement("div", _extends({
    style: {
      display: "flex",
      flexDirection: "column",
      gap: 4,
      ...style
    }
  }, rest), /*#__PURE__*/React.createElement("div", {
    style: {
      fontFamily: "var(--font-display)",
      fontWeight: "var(--fw-light)",
      fontSize: "var(--fs-3xl)",
      lineHeight: 1,
      letterSpacing: "var(--ls-tight)",
      color: "var(--text-primary)"
    }
  }, value), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      alignItems: "center",
      gap: 8
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: "var(--fs-xs)",
      color: "var(--text-secondary)"
    }
  }, label), delta != null && /*#__PURE__*/React.createElement("span", {
    style: {
      display: "inline-flex",
      alignItems: "center",
      gap: 2,
      fontSize: "var(--fs-2xs)",
      fontWeight: "var(--fw-semibold)",
      color: positive ? "var(--nv-green-700)" : "var(--nv-garnet)"
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: 11
    }
  }, positive ? "▲" : "▼"), delta)));
}
Object.assign(__ds_scope, { Stat });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/data-display/Stat.jsx", error: String((e && e.message) || e) }); }

// components/data-display/Tag.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * NVIDIA Tag — removable/selectable chip for filters and multi-select.
 * Pass `onRemove` to show a close affordance; `selected` for the active state.
 */
function Tag({
  children,
  selected = false,
  onRemove,
  onClick,
  style,
  ...rest
}) {
  const [hover, setHover] = React.useState(false);
  return /*#__PURE__*/React.createElement("span", _extends({
    onClick: onClick,
    onMouseEnter: () => setHover(true),
    onMouseLeave: () => setHover(false),
    style: {
      display: "inline-flex",
      alignItems: "center",
      gap: 6,
      height: 28,
      padding: onRemove ? "0 6px 0 12px" : "0 12px",
      fontFamily: "var(--font-sans)",
      fontSize: "var(--fs-xs)",
      fontWeight: "var(--fw-medium)",
      borderRadius: "var(--radius-sm)",
      cursor: onClick ? "pointer" : "default",
      background: selected ? "var(--nv-green-100)" : hover && onClick ? "var(--nv-gray-50)" : "var(--nv-white)",
      color: selected ? "var(--nv-green-700)" : "var(--text-primary)",
      border: `1px solid ${selected ? "var(--nv-green)" : "var(--border-default)"}`,
      transition: "background var(--dur-fast), border-color var(--dur-fast)",
      whiteSpace: "nowrap",
      ...style
    }
  }, rest), children, onRemove && /*#__PURE__*/React.createElement("button", {
    type: "button",
    "aria-label": "Remove",
    onClick: e => {
      e.stopPropagation();
      onRemove(e);
    },
    style: {
      display: "inline-flex",
      alignItems: "center",
      justifyContent: "center",
      width: 16,
      height: 16,
      padding: 0,
      border: "none",
      background: "transparent",
      color: "var(--text-tertiary)",
      cursor: "pointer",
      borderRadius: "var(--radius-xs)"
    }
  }, /*#__PURE__*/React.createElement("svg", {
    width: "11",
    height: "11",
    viewBox: "0 0 12 12",
    fill: "none"
  }, /*#__PURE__*/React.createElement("path", {
    d: "M3 3l6 6M9 3l-6 6",
    stroke: "currentColor",
    strokeWidth: "1.5",
    strokeLinecap: "round"
  }))));
}
Object.assign(__ds_scope, { Tag });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/data-display/Tag.jsx", error: String((e && e.message) || e) }); }

// components/feedback/Spinner.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
// NVIDIA / KUI Spinner — indeterminate circular loader. NVIDIA-green arc on a
// faint neutral track. Sizes sm/md/lg or an explicit pixel number.
const SIZES = {
  sm: 16,
  md: 24,
  lg: 40
};
function Spinner({
  size = "md",
  color = "var(--nv-green)",
  track = "var(--nv-gray-200)",
  thickness,
  label = "Loading",
  style,
  ...rest
}) {
  const px = typeof size === "number" ? size : SIZES[size] || 24;
  const bw = thickness || Math.max(2, Math.round(px / 9));
  return /*#__PURE__*/React.createElement("span", _extends({
    role: "status",
    "aria-label": label,
    style: {
      display: "inline-block",
      width: px,
      height: px,
      lineHeight: 0,
      ...style
    }
  }, rest), /*#__PURE__*/React.createElement("span", {
    style: {
      display: "block",
      width: "100%",
      height: "100%",
      boxSizing: "border-box",
      borderRadius: "50%",
      border: `${bw}px solid ${track}`,
      borderTopColor: color,
      animation: "nv-spin 0.7s linear infinite"
    }
  }), /*#__PURE__*/React.createElement("style", null, "@keyframes nv-spin{to{transform:rotate(360deg)}}"));
}
Object.assign(__ds_scope, { Spinner });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/feedback/Spinner.jsx", error: String((e && e.message) || e) }); }

// components/forms/Checkbox.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * NVIDIA Checkbox — square control, green fill when checked with a black check.
 * Controlled via `checked` + `onChange`, or uncontrolled via `defaultChecked`.
 */
function Checkbox({
  label,
  checked,
  defaultChecked,
  disabled = false,
  onChange,
  style,
  ...rest
}) {
  const [internal, setInternal] = React.useState(defaultChecked || false);
  const isControlled = checked !== undefined;
  const on = isControlled ? checked : internal;
  const toggle = e => {
    if (disabled) return;
    if (!isControlled) setInternal(!on);
    onChange && onChange(!on, e);
  };
  return /*#__PURE__*/React.createElement("label", _extends({
    style: {
      display: "inline-flex",
      alignItems: "center",
      gap: 10,
      cursor: disabled ? "not-allowed" : "pointer",
      opacity: disabled ? 0.5 : 1,
      ...style
    }
  }, rest), /*#__PURE__*/React.createElement("span", {
    onClick: toggle,
    style: {
      width: 18,
      height: 18,
      flex: "none",
      display: "inline-flex",
      alignItems: "center",
      justifyContent: "center",
      borderRadius: "var(--radius-xs)",
      background: on ? "var(--nv-green)" : "var(--nv-white)",
      border: `1.5px solid ${on ? "var(--nv-green)" : "var(--border-strong)"}`,
      transition: "background var(--dur-fast), border-color var(--dur-fast)"
    }
  }, on && /*#__PURE__*/React.createElement("svg", {
    width: "12",
    height: "12",
    viewBox: "0 0 12 12",
    fill: "none"
  }, /*#__PURE__*/React.createElement("path", {
    d: "M2.5 6.2 5 8.5 9.5 3.5",
    stroke: "#000",
    strokeWidth: "1.8",
    strokeLinecap: "round",
    strokeLinejoin: "round"
  }))), label && /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: "var(--fs-sm)",
      color: "var(--text-primary)"
    }
  }, label));
}
Object.assign(__ds_scope, { Checkbox });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/Checkbox.jsx", error: String((e && e.message) || e) }); }

// components/forms/Input.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * NVIDIA text Input. Hairline border, green focus ring, optional label and
 * leading/trailing icon. Square-ish corners consistent with the brand.
 */
function Input({
  label,
  value,
  defaultValue,
  placeholder,
  type = "text",
  size = "md",
  disabled = false,
  error = false,
  hint,
  iconLeft = null,
  iconRight = null,
  onChange,
  style,
  ...rest
}) {
  const [focus, setFocus] = React.useState(false);
  const heights = {
    sm: 32,
    md: 40,
    lg: 48
  };
  const h = heights[size] || heights.md;
  const borderColor = error ? "var(--nv-garnet)" : focus ? "var(--nv-green)" : "var(--border-default)";
  return /*#__PURE__*/React.createElement("label", {
    style: {
      display: "flex",
      flexDirection: "column",
      gap: 6,
      ...style
    }
  }, label && /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: "var(--fs-xs)",
      fontWeight: "var(--fw-medium)",
      color: "var(--text-primary)"
    }
  }, label), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      alignItems: "center",
      gap: 8,
      height: h,
      padding: "0 12px",
      background: disabled ? "var(--nv-gray-50)" : "var(--nv-white)",
      border: `1px solid ${borderColor}`,
      borderRadius: "var(--radius-sm)",
      boxShadow: focus && !error ? "var(--focus-ring)" : "none",
      transition: "border-color var(--dur-fast), box-shadow var(--dur-fast)",
      opacity: disabled ? 0.6 : 1
    }
  }, iconLeft && /*#__PURE__*/React.createElement("span", {
    style: {
      display: "inline-flex",
      color: "var(--text-tertiary)"
    }
  }, iconLeft), /*#__PURE__*/React.createElement("input", _extends({
    type: type,
    value: value,
    defaultValue: defaultValue,
    placeholder: placeholder,
    disabled: disabled,
    onChange: onChange,
    onFocus: () => setFocus(true),
    onBlur: () => setFocus(false),
    style: {
      flex: 1,
      minWidth: 0,
      border: "none",
      outline: "none",
      background: "transparent",
      fontFamily: "var(--font-sans)",
      fontSize: "var(--fs-sm)",
      color: "var(--text-primary)"
    }
  }, rest)), iconRight && /*#__PURE__*/React.createElement("span", {
    style: {
      display: "inline-flex",
      color: "var(--text-tertiary)"
    }
  }, iconRight)), hint && /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: "var(--fs-2xs)",
      color: error ? "var(--nv-garnet)" : "var(--text-tertiary)"
    }
  }, hint));
}
Object.assign(__ds_scope, { Input });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/Input.jsx", error: String((e && e.message) || e) }); }

// components/forms/Select.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * NVIDIA Select — native-backed dropdown styled to match Input. Pass options
 * as [{value, label}] or an array of strings.
 */
function Select({
  label,
  value,
  defaultValue,
  options = [],
  placeholder = "Select…",
  size = "md",
  disabled = false,
  onChange,
  style,
  ...rest
}) {
  const [focus, setFocus] = React.useState(false);
  const heights = {
    sm: 32,
    md: 40,
    lg: 48
  };
  const h = heights[size] || heights.md;
  const opts = options.map(o => typeof o === "string" ? {
    value: o,
    label: o
  } : o);
  return /*#__PURE__*/React.createElement("label", {
    style: {
      display: "flex",
      flexDirection: "column",
      gap: 6,
      ...style
    }
  }, label && /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: "var(--fs-xs)",
      fontWeight: "var(--fw-medium)",
      color: "var(--text-primary)"
    }
  }, label), /*#__PURE__*/React.createElement("div", {
    style: {
      position: "relative",
      display: "flex",
      alignItems: "center",
      height: h,
      background: disabled ? "var(--nv-gray-50)" : "var(--nv-white)",
      border: `1px solid ${focus ? "var(--nv-green)" : "var(--border-default)"}`,
      borderRadius: "var(--radius-sm)",
      boxShadow: focus ? "var(--focus-ring)" : "none",
      opacity: disabled ? 0.6 : 1
    }
  }, /*#__PURE__*/React.createElement("select", _extends({
    value: value,
    defaultValue: defaultValue,
    disabled: disabled,
    onChange: onChange,
    onFocus: () => setFocus(true),
    onBlur: () => setFocus(false),
    style: {
      appearance: "none",
      WebkitAppearance: "none",
      flex: 1,
      height: "100%",
      border: "none",
      outline: "none",
      background: "transparent",
      padding: "0 36px 0 12px",
      fontFamily: "var(--font-sans)",
      fontSize: "var(--fs-sm)",
      color: "var(--text-primary)",
      cursor: disabled ? "not-allowed" : "pointer"
    }
  }, rest), placeholder && /*#__PURE__*/React.createElement("option", {
    value: "",
    disabled: true
  }, placeholder), opts.map(o => /*#__PURE__*/React.createElement("option", {
    key: o.value,
    value: o.value
  }, o.label))), /*#__PURE__*/React.createElement("svg", {
    width: "14",
    height: "14",
    viewBox: "0 0 16 16",
    fill: "none",
    style: {
      position: "absolute",
      right: 12,
      pointerEvents: "none"
    }
  }, /*#__PURE__*/React.createElement("path", {
    d: "M4 6l4 4 4-4",
    stroke: "var(--text-tertiary)",
    strokeWidth: "1.6",
    strokeLinecap: "round",
    strokeLinejoin: "round"
  }))));
}
Object.assign(__ds_scope, { Select });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/Select.jsx", error: String((e && e.message) || e) }); }

// components/forms/Switch.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * NVIDIA Switch — pill toggle, green track when on. For instant on/off
 * settings (not form submission). Controlled or uncontrolled.
 */
function Switch({
  checked,
  defaultChecked,
  disabled = false,
  label,
  onChange,
  style,
  ...rest
}) {
  const [internal, setInternal] = React.useState(defaultChecked || false);
  const isControlled = checked !== undefined;
  const on = isControlled ? checked : internal;
  const toggle = e => {
    if (disabled) return;
    if (!isControlled) setInternal(!on);
    onChange && onChange(!on, e);
  };
  return /*#__PURE__*/React.createElement("label", _extends({
    style: {
      display: "inline-flex",
      alignItems: "center",
      gap: 10,
      cursor: disabled ? "not-allowed" : "pointer",
      opacity: disabled ? 0.5 : 1,
      ...style
    }
  }, rest), /*#__PURE__*/React.createElement("span", {
    role: "switch",
    "aria-checked": on,
    onClick: toggle,
    style: {
      width: 40,
      height: 22,
      flex: "none",
      borderRadius: "var(--radius-pill)",
      background: on ? "var(--nv-green)" : "var(--nv-gray-300)",
      position: "relative",
      transition: "background var(--dur-base) var(--ease-standard)"
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      position: "absolute",
      top: 2,
      left: on ? 20 : 2,
      width: 18,
      height: 18,
      borderRadius: "var(--radius-pill)",
      background: "var(--nv-white)",
      boxShadow: "var(--shadow-sm)",
      transition: "left var(--dur-base) var(--ease-standard)"
    }
  })), label && /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: "var(--fs-sm)",
      color: "var(--text-primary)"
    }
  }, label));
}
Object.assign(__ds_scope, { Switch });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/Switch.jsx", error: String((e && e.message) || e) }); }

// components/icons/Icon.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
// NVIDIA GUI icon set — real geometry extracted verbatim from the official
// "NVIDIA GUI Icons" Figma library (line variants, 16×16 grid). Single-color
// filled-outline vectors painted with currentColor. Names follow the source
// "category-slug" convention (the "-line" suffix is dropped).
const ICONS = {
  "av-stop": "<rect x=\"4.5\" y=\"4.5\" width=\"7\" height=\"7\" fill=\"currentColor\"/>",
  "av-camera": "<g transform=\"translate(2.5,2.5)\"><path d=\"M 0 2 L 0 1.5 L -0.5 1.5 L -0.5 2 L 0 2 Z M 11 2 L 11.5 2 L 11.5 1.5 L 11 1.5 L 11 2 Z M 11 10 L 11 10.5 L 11.5 10.5 L 11.5 10 L 11 10 Z M 0 10 L -0.5 10 L -0.5 10.5 L 0 10.5 L 0 10 Z M 3 2 L 3 2.5 L 3.309 2.5 L 3.447 2.224 L 3 2 Z M 4 0 L 4 -0.5 L 3.691 -0.5 L 3.553 -0.224 L 4 0 Z M 8 2 L 7.553 2.224 L 7.691 2.5 L 8 2.5 L 8 2 Z M 7 0 L 7.447 -0.224 L 7.309 -0.5 L 7 -0.5 L 7 0 Z M 10.5 2 L 10.5 10 L 11.5 10 L 11.5 2 L 10.5 2 Z M 11 9.5 L 0 9.5 L 0 10.5 L 11 10.5 L 11 9.5 Z M 0.5 10 L 0.5 2 L -0.5 2 L -0.5 10 L 0.5 10 Z M 0 2.5 L 3 2.5 L 3 1.5 L 0 1.5 L 0 2.5 Z M 3.447 2.224 L 4.447 0.224 L 3.553 -0.224 L 2.553 1.776 L 3.447 2.224 Z M 8 2.5 L 11 2.5 L 11 1.5 L 8 1.5 L 8 2.5 Z M 4 0.5 L 7 0.5 L 7 -0.5 L 4 -0.5 L 4 0.5 Z M 6.553 0.224 L 7.553 2.224 L 8.447 1.776 L 7.447 -0.224 L 6.553 0.224 Z M 7.5 5.5 C 7.5 6.605 6.605 7.5 5.5 7.5 L 5.5 8.5 C 7.157 8.5 8.5 7.157 8.5 5.5 L 7.5 5.5 Z M 5.5 7.5 C 4.395 7.5 3.5 6.605 3.5 5.5 L 2.5 5.5 C 2.5 7.157 3.843 8.5 5.5 8.5 L 5.5 7.5 Z M 3.5 5.5 C 3.5 4.395 4.395 3.5 5.5 3.5 L 5.5 2.5 C 3.843 2.5 2.5 3.843 2.5 5.5 L 3.5 5.5 Z M 5.5 3.5 C 6.605 3.5 7.5 4.395 7.5 5.5 L 8.5 5.5 C 8.5 3.843 7.157 2.5 5.5 2.5 L 5.5 3.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "av-closed-caption": "<g transform=\"translate(1.5,3.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 13 0 L 13.5 0 L 13.5 -0.5 L 13 -0.5 L 13 0 Z M 0 10 L -0.5 10 L -0.5 10.5 L 0 10.5 L 0 10 Z M 13 10 L 13 10.5 L 13.5 10.5 L 13.5 10 L 13 10 Z M 0 0 L 0 0.5 L 13 0.5 L 13 0 L 13 -0.5 L 0 -0.5 L 0 0 Z M 0 10 L 0.5 10 L 0.5 0 L 0 0 L -0.5 0 L -0.5 10 L 0 10 Z M 13 0 L 12.5 0 L 12.5 10 L 13 10 L 13.5 10 L 13.5 0 L 13 0 Z M 13 10 L 13 9.5 L 0 9.5 L 0 10 L 0 10.5 L 13 10.5 L 13 10 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(9.5,6.5)\"><path d=\"M 0.5 0 L 0.5 0.5 L 1.5 0.5 L 1.5 0 L 1.5 -0.5 L 0.5 -0.5 L 0.5 0 Z M 1.5 4 L 1.5 3.5 L 0.5 3.5 L 0.5 4 L 0.5 4.5 L 1.5 4.5 L 1.5 4 Z M 0 3.5 L 0.5 3.5 L 0.5 0.5 L 0 0.5 L -0.5 0.5 L -0.5 3.5 L 0 3.5 Z M 0.5 4 L 0.5 3.5 L 0 3.5 L -0.5 3.5 C -0.5 4.052 -0.052 4.5 0.5 4.5 L 0.5 4 Z M 2 3.5 L 1.5 3.5 L 1.5 4 L 1.5 4.5 C 2.052 4.5 2.5 4.052 2.5 3.5 L 2 3.5 Z M 1.5 0 L 1.5 0.5 L 2 0.5 L 2.5 0.5 C 2.5 -0.052 2.052 -0.5 1.5 -0.5 L 1.5 0 Z M 0.5 0 L 0.5 -0.5 C -0.052 -0.5 -0.5 -0.052 -0.5 0.5 L 0 0.5 L 0.5 0.5 L 0.5 0 Z M 2 3 L 1.5 3 L 1.5 3.5 L 2 3.5 L 2.5 3.5 L 2.5 3 L 2 3 Z M 2 0.5 L 1.5 0.5 L 1.5 1 L 2 1 L 2.5 1 L 2.5 0.5 L 2 0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(4.5,6.5)\"><path d=\"M 0.5 0 L 0.5 0.5 L 1.5 0.5 L 1.5 0 L 1.5 -0.5 L 0.5 -0.5 L 0.5 0 Z M 1.5 4 L 1.5 3.5 L 0.5 3.5 L 0.5 4 L 0.5 4.5 L 1.5 4.5 L 1.5 4 Z M 0 3.5 L 0.5 3.5 L 0.5 0.5 L 0 0.5 L -0.5 0.5 L -0.5 3.5 L 0 3.5 Z M 0.5 4 L 0.5 3.5 L 0 3.5 L -0.5 3.5 C -0.5 4.052 -0.052 4.5 0.5 4.5 L 0.5 4 Z M 2 3.5 L 1.5 3.5 L 1.5 4 L 1.5 4.5 C 2.052 4.5 2.5 4.052 2.5 3.5 L 2 3.5 Z M 1.5 0 L 1.5 0.5 L 2 0.5 L 2.5 0.5 C 2.5 -0.052 2.052 -0.5 1.5 -0.5 L 1.5 0 Z M 0.5 0 L 0.5 -0.5 C -0.052 -0.5 -0.5 -0.052 -0.5 0.5 L 0 0.5 L 0.5 0.5 L 0.5 0 Z M 2 3 L 1.5 3 L 1.5 3.5 L 2 3.5 L 2.5 3.5 L 2.5 3 L 2 3 Z M 2 0.5 L 1.5 0.5 L 1.5 1 L 2 1 L 2.5 1 L 2.5 0.5 L 2 0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "av-equalizer": "<g transform=\"translate(1,3.5)\"><path d=\"M 0 10.5 L 4 10.5 L 4 9.5 L 0 9.5 L 0 10.5 Z M 0 8.5 L 4 8.5 L 4 7.5 L 0 7.5 L 0 8.5 Z M 0 6.5 L 4 6.5 L 4 5.5 L 0 5.5 L 0 6.5 Z M 5 10.5 L 9 10.5 L 9 9.5 L 5 9.5 L 5 10.5 Z M 5 8.5 L 9 8.5 L 9 7.5 L 5 7.5 L 5 8.5 Z M 5 6.5 L 9 6.5 L 9 5.5 L 5 5.5 L 5 6.5 Z M 5 4.5 L 9 4.5 L 9 3.5 L 5 3.5 L 5 4.5 Z M 5 2.5 L 9 2.5 L 9 1.5 L 5 1.5 L 5 2.5 Z M 5 0.5 L 9 0.5 L 9 -0.5 L 5 -0.5 L 5 0.5 Z M 10 10.5 L 14 10.5 L 14 9.5 L 10 9.5 L 10 10.5 Z M 10 8.5 L 14 8.5 L 14 7.5 L 10 7.5 L 10 8.5 Z M 10 6.5 L 14 6.5 L 14 5.5 L 10 5.5 L 10 6.5 Z M 10 4.5 L 14 4.5 L 14 3.5 L 10 3.5 L 10 4.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "av-fast-forward": "<g transform=\"translate(2.5,4.5)\"><path d=\"M 6 3.49 L 6.5 3.49 L 6.5 3.202 L 6.251 3.058 L 6 3.49 Z M 0 7 L -0.5 7 L -0.5 7.871 L 0.252 7.432 L 0 7 Z M 0 0 L 0.251 -0.432 L -0.5 -0.869 L -0.5 0 L 0 0 Z M 6 3.5 L 6.252 3.932 L 6.5 3.787 L 6.5 3.5 L 6 3.5 Z M 12 3.5 L 12.252 3.932 L 12.992 3.5 L 12.252 3.068 L 12 3.5 Z M 6 7 L 5.5 7 L 5.5 7.871 L 6.252 7.432 L 6 7 Z M 6 0 L 6.252 -0.432 L 5.5 -0.871 L 5.5 0 L 6 0 Z M 0.5 7 L 0.5 0 L -0.5 0 L -0.5 7 L 0.5 7 Z M -0.251 0.432 L 5.749 3.922 L 6.251 3.058 L 0.251 -0.432 L -0.251 0.432 Z M 5.5 3.49 L 5.5 3.5 L 6.5 3.5 L 6.5 3.49 L 5.5 3.49 Z M 5.748 3.068 L -0.252 6.568 L 0.252 7.432 L 6.252 3.932 L 5.748 3.068 Z M 11.748 3.068 L 5.748 6.568 L 6.252 7.432 L 12.252 3.932 L 11.748 3.068 Z M 6.5 7 L 6.5 0 L 5.5 0 L 5.5 7 L 6.5 7 Z M 5.748 0.432 L 11.748 3.932 L 12.252 3.068 L 6.252 -0.432 L 5.748 0.432 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "av-fast-reverse": "<g transform=\"translate(1.5,4.5)\"><path d=\"M 6 3.49 L 5.749 3.058 L 5.5 3.202 L 5.5 3.49 L 6 3.49 Z M 12 7 L 11.748 7.432 L 12.5 7.871 L 12.5 7 L 12 7 Z M 12 0 L 12.5 0 L 12.5 -0.869 L 11.749 -0.432 L 12 0 Z M 6 3.5 L 5.5 3.5 L 5.5 3.787 L 5.748 3.932 L 6 3.5 Z M 0 3.5 L -0.252 3.068 L -0.992 3.5 L -0.252 3.932 L 0 3.5 Z M 6 7 L 5.748 7.432 L 6.5 7.871 L 6.5 7 L 6 7 Z M 6 0 L 6.5 0 L 6.5 -0.871 L 5.748 -0.432 L 6 0 Z M 12.5 7 L 12.5 0 L 11.5 0 L 11.5 7 L 12.5 7 Z M 11.749 -0.432 L 5.749 3.058 L 6.251 3.922 L 12.251 0.432 L 11.749 -0.432 Z M 5.5 3.49 L 5.5 3.5 L 6.5 3.5 L 6.5 3.49 L 5.5 3.49 Z M 5.748 3.932 L 11.748 7.432 L 12.252 6.568 L 6.252 3.068 L 5.748 3.932 Z M -0.252 3.932 L 5.748 7.432 L 6.252 6.568 L 0.252 3.068 L -0.252 3.932 Z M 6.5 7 L 6.5 0 L 5.5 0 L 5.5 7 L 6.5 7 Z M 5.748 -0.432 L -0.252 3.068 L 0.252 3.932 L 6.252 0.432 L 5.748 -0.432 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "av-film": "<g transform=\"translate(2.5,2.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 11 0 L 11.5 0 L 11.5 -0.5 L 11 -0.5 L 11 0 Z M 11 12 L 11 12.5 L 11.5 12.5 L 11.5 12 L 11 12 Z M 0 12 L -0.5 12 L -0.5 12.5 L 0 12.5 L 0 12 Z M 0 6.5 L 11 6.5 L 11 5.5 L 0 5.5 L 0 6.5 Z M 0 0.5 L 2 0.5 L 2 -0.5 L 0 -0.5 L 0 0.5 Z M 2 11.5 L 0 11.5 L 0 12.5 L 2 12.5 L 2 11.5 Z M 2 0.5 L 9 0.5 L 9 -0.5 L 2 -0.5 L 2 0.5 Z M 9 0.5 L 11 0.5 L 11 -0.5 L 9 -0.5 L 9 0.5 Z M 11 11.5 L 9 11.5 L 9 12.5 L 11 12.5 L 11 11.5 Z M 9 11.5 L 2 11.5 L 2 12.5 L 9 12.5 L 9 11.5 Z M 0.5 2 L 0.5 0 L -0.5 0 L -0.5 2 L 0.5 2 Z M 1.5 0 L 1.5 2 L 2.5 2 L 2.5 0 L 1.5 0 Z M 0 2.5 L 2 2.5 L 2 1.5 L 0 1.5 L 0 2.5 Z M 0.5 6 L 0.5 4 L -0.5 4 L -0.5 6 L 0.5 6 Z M 0.5 4 L 0.5 2 L -0.5 2 L -0.5 4 L 0.5 4 Z M 1.5 2 L 1.5 4 L 2.5 4 L 2.5 2 L 1.5 2 Z M 0 4.5 L 2 4.5 L 2 3.5 L 0 3.5 L 0 4.5 Z M 0.5 8 L 0.5 6 L -0.5 6 L -0.5 8 L 0.5 8 Z M 1.5 4 L 1.5 8 L 2.5 8 L 2.5 4 L 1.5 4 Z M 0 8.5 L 2 8.5 L 2 7.5 L 0 7.5 L 0 8.5 Z M 0.5 12 L 0.5 10 L -0.5 10 L -0.5 12 L 0.5 12 Z M 0.5 10 L 0.5 8 L -0.5 8 L -0.5 10 L 0.5 10 Z M 1.5 8 L 1.5 10 L 2.5 10 L 2.5 8 L 1.5 8 Z M 1.5 10 L 1.5 12 L 2.5 12 L 2.5 10 L 1.5 10 Z M 0 10.5 L 2 10.5 L 2 9.5 L 0 9.5 L 0 10.5 Z M 8.5 10 L 8.5 12 L 9.5 12 L 9.5 10 L 8.5 10 Z M 10.5 10 L 10.5 12 L 11.5 12 L 11.5 10 L 10.5 10 Z M 9 10.5 L 11 10.5 L 11 9.5 L 9 9.5 L 9 10.5 Z M 8.5 8 L 8.5 10 L 9.5 10 L 9.5 8 L 8.5 8 Z M 10.5 6 L 10.5 8 L 11.5 8 L 11.5 6 L 10.5 6 Z M 10.5 8 L 10.5 10 L 11.5 10 L 11.5 8 L 10.5 8 Z M 9 8.5 L 11 8.5 L 11 7.5 L 9 7.5 L 9 8.5 Z M 8.5 4 L 8.5 8 L 9.5 8 L 9.5 4 L 8.5 4 Z M 10.5 4 L 10.5 6 L 11.5 6 L 11.5 4 L 10.5 4 Z M 9 4.5 L 11 4.5 L 11 3.5 L 9 3.5 L 9 4.5 Z M 8.5 0 L 8.5 2 L 9.5 2 L 9.5 0 L 8.5 0 Z M 8.5 2 L 8.5 4 L 9.5 4 L 9.5 2 L 8.5 2 Z M 10.5 0 L 10.5 2 L 11.5 2 L 11.5 0 L 10.5 0 Z M 10.5 2 L 10.5 4 L 11.5 4 L 11.5 2 L 10.5 2 Z M 9 2.5 L 11 2.5 L 11 1.5 L 9 1.5 L 9 2.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "av-fps": "<g transform=\"translate(2.5,3.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 11 0 L 11.5 0 L 11.5 -0.5 L 11 -0.5 L 11 0 Z M 11 7 L 11 7.5 L 11.5 7.5 L 11.5 7 L 11 7 Z M 0 7 L -0.5 7 L -0.5 7.5 L 0 7.5 L 0 7 Z M 2 2 L 2 1.5 L 1.5 1.5 L 1.5 2 L 2 2 Z M 5 2 L 5 1.5 L 4.5 1.5 L 4.5 2 L 5 2 Z M 5.5 2 L 5.5 2.5 L 5.5 2.5 L 5.5 2 Z M 0 0.5 L 11 0.5 L 11 -0.5 L 0 -0.5 L 0 0.5 Z M 10.5 0 L 10.5 7 L 11.5 7 L 11.5 0 L 10.5 0 Z M 11 6.5 L 0 6.5 L 0 7.5 L 11 7.5 L 11 6.5 Z M 0.5 7 L 0.5 0 L -0.5 0 L -0.5 7 L 0.5 7 Z M 2.5 9.5 L 8.5 9.5 L 8.5 8.5 L 2.5 8.5 L 2.5 9.5 Z M 5.267 7.129 L 5.803 9.129 L 6.769 8.871 L 6.233 6.871 L 5.267 7.129 Z M 4.768 6.871 L 4.232 8.871 L 5.198 9.129 L 5.734 7.129 L 4.768 6.871 Z M 4 1.5 L 2 1.5 L 2 2.5 L 4 2.5 L 4 1.5 Z M 1.5 2 L 1.5 5.5 L 2.5 5.5 L 2.5 2 L 1.5 2 Z M 8.5 2.75 C 8.5 2.728 8.525 2.647 8.614 2.598 C 8.681 2.561 8.877 2.498 9.276 2.697 L 9.724 1.803 C 9.123 1.502 8.569 1.484 8.136 1.72 C 7.725 1.943 7.5 2.362 7.5 2.75 L 8.5 2.75 Z M 8.5 4.25 C 8.5 4.272 8.475 4.353 8.386 4.402 C 8.319 4.439 8.123 4.502 7.724 4.303 L 7.276 5.197 C 7.877 5.498 8.431 5.516 8.864 5.28 C 9.275 5.057 9.5 4.638 9.5 4.25 L 8.5 4.25 Z M 7.5 2.75 C 7.5 3.405 7.976 3.732 8.2 3.9 C 8.476 4.107 8.5 4.155 8.5 4.25 L 9.5 4.25 C 9.5 3.595 9.024 3.268 8.8 3.1 C 8.524 2.893 8.5 2.845 8.5 2.75 L 7.5 2.75 Z M 2 4 L 3.5 4 L 3.5 3 L 2 3 L 2 4 Z M 5.5 5.5 L 5.5 4 L 4.5 4 L 4.5 5.5 L 5.5 5.5 Z M 5.5 4 L 5.5 2 L 4.5 2 L 4.5 4 L 5.5 4 Z M 6 3 C 6 3.276 5.776 3.5 5.5 3.5 L 5.5 4.5 C 6.328 4.5 7 3.828 7 3 L 6 3 Z M 5.5 2.5 C 5.776 2.5 6 2.724 6 3 L 7 3 C 7 2.172 6.328 1.5 5.5 1.5 L 5.5 2.5 Z M 5.5 1.5 L 5 1.5 L 5 2.5 L 5.5 2.5 L 5.5 1.5 Z M 5.5 3.5 L 5 3.5 L 5 4.5 L 5.5 4.5 L 5.5 3.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "av-loop": "<g transform=\"translate(1.5,4.5)\"><path d=\"M -0.5 3.5 C -0.5 5.709 1.291 7.5 3.5 7.5 L 3.5 6.5 C 1.843 6.5 0.5 5.157 0.5 3.5 L -0.5 3.5 Z M 3.5 -0.5 C 1.291 -0.5 -0.5 1.291 -0.5 3.5 L 0.5 3.5 C 0.5 1.843 1.843 0.5 3.5 0.5 L 3.5 -0.5 Z M 9.5 -0.5 L 3.5 -0.5 L 3.5 0.5 L 9.5 0.5 L 9.5 -0.5 Z M 13.5 3.5 C 13.5 1.291 11.709 -0.5 9.5 -0.5 L 9.5 0.5 C 11.157 0.5 12.5 1.843 12.5 3.5 L 13.5 3.5 Z M 9.5 7.5 C 11.709 7.5 13.5 5.709 13.5 3.5 L 12.5 3.5 C 12.5 5.157 11.157 6.5 9.5 6.5 L 9.5 7.5 Z M 3.5 7.5 L 7 7.5 L 7 6.5 L 3.5 6.5 L 3.5 7.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(6.5,9.5)\"><path d=\"M 2 2 L 2.354 2.354 L 2.707 2 L 2.354 1.646 L 2 2 Z M 0.354 4.354 L 2.354 2.354 L 1.646 1.646 L -0.354 3.646 L 0.354 4.354 Z M 2.354 1.646 L 0.354 -0.354 L -0.354 0.354 L 1.646 2.354 L 2.354 1.646 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "av-microphone": "<g transform=\"translate(3.5,1.5)\"><path d=\"M 3.562 10.902 L 3.666 10.413 L 3.202 10.315 L 3.079 10.773 L 3.562 10.902 Z M 5.44 10.902 L 5.924 10.773 L 5.801 10.314 L 5.336 10.413 L 5.44 10.902 Z M 6 13 L 6 13.5 L 6.651 13.5 L 6.483 12.871 L 6 13 Z M 3 13 L 2.517 12.871 L 2.348 13.5 L 3 13.5 L 3 13 Z M 2.5 6.5 L 2.5 2.5 L 1.5 2.5 L 1.5 6.5 L 2.5 6.5 Z M 4.5 -0.5 C 2.843 -0.5 1.5 0.843 1.5 2.5 L 2.5 2.5 C 2.5 1.395 3.395 0.5 4.5 0.5 L 4.5 -0.5 Z M 7.5 2.5 C 7.5 0.843 6.157 -0.5 4.5 -0.5 L 4.5 0.5 C 5.605 0.5 6.5 1.395 6.5 2.5 L 7.5 2.5 Z M 6.5 2.5 L 6.5 6.5 L 7.5 6.5 L 7.5 2.5 L 6.5 2.5 Z M 4.5 9.5 C 6.157 9.5 7.5 8.157 7.5 6.5 L 6.5 6.5 C 6.5 7.605 5.605 8.5 4.5 8.5 L 4.5 9.5 Z M 1.5 6.5 C 1.5 8.157 2.843 9.5 4.5 9.5 L 4.5 8.5 C 3.395 8.5 2.5 7.605 2.5 6.5 L 1.5 6.5 Z M 4.5 10.5 C 2.291 10.5 0.5 8.709 0.5 6.5 L -0.5 6.5 C -0.5 9.261 1.739 11.5 4.5 11.5 L 4.5 10.5 Z M 8.5 6.5 C 8.5 8.709 6.709 10.5 4.5 10.5 L 4.5 11.5 C 7.261 11.5 9.5 9.261 9.5 6.5 L 8.5 6.5 Z M 3.458 11.391 C 3.795 11.463 4.143 11.5 4.5 11.5 L 4.5 10.5 C 4.213 10.5 3.934 10.47 3.666 10.413 L 3.458 11.391 Z M 4.5 11.5 C 4.858 11.5 5.207 11.462 5.544 11.391 L 5.336 10.413 C 5.067 10.47 4.787 10.5 4.5 10.5 L 4.5 11.5 Z M 6.483 12.871 L 5.924 10.773 L 4.957 11.03 L 5.517 13.129 L 6.483 12.871 Z M 3 13.5 L 6 13.5 L 6 12.5 L 3 12.5 L 3 13.5 Z M 3.079 10.773 L 2.517 12.871 L 3.483 13.129 L 4.045 11.032 L 3.079 10.773 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "av-microphone-off": "<g transform=\"translate(3.5,1.5)\"><path d=\"M 3.562 10.902 L 3.666 10.413 L 3.202 10.315 L 3.079 10.773 L 3.562 10.902 Z M 5.44 10.902 L 5.924 10.773 L 5.801 10.314 L 5.336 10.413 L 5.44 10.902 Z M 6 13 L 6 13.5 L 6.651 13.5 L 6.483 12.871 L 6 13 Z M 3 13 L 2.517 12.871 L 2.348 13.5 L 3 13.5 L 3 13 Z M 2.5 6.5 L 2.5 2.5 L 1.5 2.5 L 1.5 6.5 L 2.5 6.5 Z M 4.5 -0.5 C 2.843 -0.5 1.5 0.843 1.5 2.5 L 2.5 2.5 C 2.5 1.395 3.395 0.5 4.5 0.5 L 4.5 -0.5 Z M 7.5 2.5 C 7.5 0.843 6.157 -0.5 4.5 -0.5 L 4.5 0.5 C 5.605 0.5 6.5 1.395 6.5 2.5 L 7.5 2.5 Z M 6.5 2.5 L 6.5 6.5 L 7.5 6.5 L 7.5 2.5 L 6.5 2.5 Z M 4.5 9.5 C 6.157 9.5 7.5 8.157 7.5 6.5 L 6.5 6.5 C 6.5 7.605 5.605 8.5 4.5 8.5 L 4.5 9.5 Z M 1.5 6.5 C 1.5 8.157 2.843 9.5 4.5 9.5 L 4.5 8.5 C 3.395 8.5 2.5 7.605 2.5 6.5 L 1.5 6.5 Z M 4.5 10.5 C 2.291 10.5 0.5 8.709 0.5 6.5 L -0.5 6.5 C -0.5 9.261 1.739 11.5 4.5 11.5 L 4.5 10.5 Z M 8.5 6.5 C 8.5 8.709 6.709 10.5 4.5 10.5 L 4.5 11.5 C 7.261 11.5 9.5 9.261 9.5 6.5 L 8.5 6.5 Z M 3.458 11.391 C 3.795 11.463 4.143 11.5 4.5 11.5 L 4.5 10.5 C 4.213 10.5 3.934 10.47 3.666 10.413 L 3.458 11.391 Z M 4.5 11.5 C 4.858 11.5 5.207 11.462 5.544 11.391 L 5.336 10.413 C 5.067 10.47 4.787 10.5 4.5 10.5 L 4.5 11.5 Z M 6.483 12.871 L 5.924 10.773 L 4.957 11.03 L 5.517 13.129 L 6.483 12.871 Z M 3 13.5 L 6 13.5 L 6 12.5 L 3 12.5 L 3 13.5 Z M 3.079 10.773 L 2.517 12.871 L 3.483 13.129 L 4.045 11.032 L 3.079 10.773 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(0,0)\"><path d=\"M 11.646 -0.354 L -0.354 11.646 L 0.354 12.354 L 12.354 0.354 L 11.646 -0.354 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "av-next": "<g transform=\"translate(4.5,3.5)\"><path d=\"M 0 8 L -0.5 8 L -0.5 8.871 L 0.252 8.432 L 0 8 Z M 0 1 L 0.252 0.568 L -0.5 0.129 L -0.5 1 L 0 1 Z M 6 4.5 L 6.252 4.932 L 6.992 4.5 L 6.252 4.068 L 6 4.5 Z M 0.5 8 L 0.5 1 L -0.5 1 L -0.5 8 L 0.5 8 Z M 5.748 4.068 L -0.252 7.568 L 0.252 8.432 L 6.252 4.932 L 5.748 4.068 Z M -0.252 1.432 L 5.748 4.932 L 6.252 4.068 L 0.252 0.568 L -0.252 1.432 Z M 6.5 0 L 6.5 9 L 7.5 9 L 7.5 0 L 6.5 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "av-pause": "<g transform=\"translate(5.5,4)\"><path d=\"M -0.5 0 L -0.5 8 L 0.5 8 L 0.5 0 L -0.5 0 Z M 4.5 0 L 4.5 8 L 5.5 8 L 5.5 0 L 4.5 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "av-play": "<g transform=\"translate(5.5,4.5)\"><path d=\"M 6 3.5 L 6.252 3.932 L 6.992 3.5 L 6.252 3.068 L 6 3.5 Z M 0 7 L -0.5 7 L -0.5 7.871 L 0.252 7.432 L 0 7 Z M 0 0 L 0.252 -0.432 L -0.5 -0.871 L -0.5 0 L 0 0 Z M 5.748 3.068 L -0.252 6.568 L 0.252 7.432 L 6.252 3.932 L 5.748 3.068 Z M 0.5 7 L 0.5 0 L -0.5 0 L -0.5 7 L 0.5 7 Z M -0.252 0.432 L 5.748 3.932 L 6.252 3.068 L 0.252 -0.432 L -0.252 0.432 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "av-previous": "<g transform=\"translate(4.5,3.5)\"><path d=\"M 1 4.5 L 0.748 4.068 L 0.008 4.5 L 0.748 4.932 L 1 4.5 Z M 7 8 L 6.748 8.432 L 7.5 8.871 L 7.5 8 L 7 8 Z M 7 1 L 7.5 1 L 7.5 0.129 L 6.748 0.568 L 7 1 Z M 0.748 4.932 L 6.748 8.432 L 7.252 7.568 L 1.252 4.068 L 0.748 4.932 Z M 7.5 8 L 7.5 1 L 6.5 1 L 6.5 8 L 7.5 8 Z M 6.748 0.568 L 0.748 4.068 L 1.252 4.932 L 7.252 1.432 L 6.748 0.568 Z M -0.5 0 L -0.5 9 L 0.5 9 L 0.5 0 L -0.5 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "av-record": "<g transform=\"translate(3.5,3.5)\"><path d=\"M 8.5 4.5 C 8.5 6.709 6.709 8.5 4.5 8.5 L 4.5 9.5 C 7.261 9.5 9.5 7.261 9.5 4.5 L 8.5 4.5 Z M 4.5 8.5 C 2.291 8.5 0.5 6.709 0.5 4.5 L -0.5 4.5 C -0.5 7.261 1.739 9.5 4.5 9.5 L 4.5 8.5 Z M 0.5 4.5 C 0.5 2.291 2.291 0.5 4.5 0.5 L 4.5 -0.5 C 1.739 -0.5 -0.5 1.739 -0.5 4.5 L 0.5 4.5 Z M 4.5 0.5 C 6.709 0.5 8.5 2.291 8.5 4.5 L 9.5 4.5 C 9.5 1.739 7.261 -0.5 4.5 -0.5 L 4.5 0.5 Z M 5.5 4.5 C 5.5 5.052 5.052 5.5 4.5 5.5 L 4.5 6.5 C 5.605 6.5 6.5 5.605 6.5 4.5 L 5.5 4.5 Z M 4.5 5.5 C 3.948 5.5 3.5 5.052 3.5 4.5 L 2.5 4.5 C 2.5 5.605 3.395 6.5 4.5 6.5 L 4.5 5.5 Z M 3.5 4.5 C 3.5 3.948 3.948 3.5 4.5 3.5 L 4.5 2.5 C 3.395 2.5 2.5 3.395 2.5 4.5 L 3.5 4.5 Z M 4.5 3.5 C 5.052 3.5 5.5 3.948 5.5 4.5 L 6.5 4.5 C 6.5 3.395 5.605 2.5 4.5 2.5 L 4.5 3.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "av-replay": "<g transform=\"translate(3.5,2)\"><path d=\"M 5 1.5 L 5 2 L 6.207 2 L 5.354 1.147 L 5 1.5 Z M 4 10.5 L 4 10 L 2.793 10 L 3.646 10.853 L 4 10.5 Z M 4.5 5 C 5.052 5 5.5 5.448 5.5 6 L 6.5 6 C 6.5 4.895 5.605 4 4.5 4 L 4.5 5 Z M 5.5 6 C 5.5 6.552 5.052 7 4.5 7 L 4.5 8 C 5.605 8 6.5 7.105 6.5 6 L 5.5 6 Z M 4.5 7 C 3.948 7 3.5 6.552 3.5 6 L 2.5 6 C 2.5 7.105 3.395 8 4.5 8 L 4.5 7 Z M 3.5 6 C 3.5 5.448 3.948 5 4.5 5 L 4.5 4 C 3.395 4 2.5 4.895 2.5 6 L 3.5 6 Z M 2.384 9.395 C 1.251 8.688 0.5 7.432 0.5 6 L -0.5 6 C -0.5 7.791 0.441 9.361 1.854 10.243 L 2.384 9.395 Z M 0.5 6 C 0.5 3.791 2.291 2 4.5 2 L 4.5 1 C 1.739 1 -0.5 3.239 -0.5 6 L 0.5 6 Z M 4.5 2 L 5 2 L 5 1 L 4.5 1 L 4.5 2 Z M 5.354 1.147 L 3.854 -0.354 L 3.146 0.354 L 4.646 1.854 L 5.354 1.147 Z M 6.616 2.605 C 7.749 3.312 8.5 4.568 8.5 6 L 9.5 6 C 9.5 4.209 8.559 2.639 7.146 1.757 L 6.616 2.605 Z M 8.5 6 C 8.5 8.209 6.709 10 4.5 10 L 4.5 11 C 7.261 11 9.5 8.761 9.5 6 L 8.5 6 Z M 4.5 10 L 4 10 L 4 11 L 4.5 11 L 4.5 10 Z M 3.646 10.853 L 5.146 12.354 L 5.854 11.646 L 4.354 10.146 L 3.646 10.853 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "av-speaker-high": "<g transform=\"translate(2.5,3.5)\"><path d=\"M 5 0 L 5.5 0 L 5.5 -1.087 L 4.675 -0.38 L 5 0 Z M 1.5 3 L 1.5 3.5 L 1.685 3.5 L 1.825 3.38 L 1.5 3 Z M 0 3 L 0 2.5 L -0.5 2.5 L -0.5 3 L 0 3 Z M 0 6 L -0.5 6 L -0.5 6.5 L 0 6.5 L 0 6 Z M 1.5 6 L 1.825 5.62 L 1.685 5.5 L 1.5 5.5 L 1.5 6 Z M 5 9 L 4.675 9.38 L 5.5 10.087 L 5.5 9 L 5 9 Z M 4.675 -0.38 L 1.175 2.62 L 1.825 3.38 L 5.325 0.38 L 4.675 -0.38 Z M 1.5 2.5 L 0 2.5 L 0 3.5 L 1.5 3.5 L 1.5 2.5 Z M -0.5 3 L -0.5 6 L 0.5 6 L 0.5 3 L -0.5 3 Z M 0 6.5 L 1.5 6.5 L 1.5 5.5 L 0 5.5 L 0 6.5 Z M 1.175 6.38 L 4.675 9.38 L 5.325 8.62 L 1.825 5.62 L 1.175 6.38 Z M 5.5 9 L 5.5 0 L 4.5 0 L 4.5 9 L 5.5 9 Z M 6.5 2.5 L 6.5 6.5 L 7.5 6.5 L 7.5 2.5 L 6.5 2.5 Z M 8.5 1.5 L 8.5 7.5 L 9.5 7.5 L 9.5 1.5 L 8.5 1.5 Z M 10.5 0.5 L 10.5 8.5 L 11.5 8.5 L 11.5 0.5 L 10.5 0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "av-speaker-mute": "<g transform=\"translate(2.5,3.5)\"><path d=\"M 5 0 L 5.5 0 L 5.5 -1.087 L 4.675 -0.38 L 5 0 Z M 1.5 3 L 1.5 3.5 L 1.685 3.5 L 1.825 3.38 L 1.5 3 Z M 0 3 L 0 2.5 L -0.5 2.5 L -0.5 3 L 0 3 Z M 0 6 L -0.5 6 L -0.5 6.5 L 0 6.5 L 0 6 Z M 1.5 6 L 1.825 5.62 L 1.685 5.5 L 1.5 5.5 L 1.5 6 Z M 5 9 L 4.675 9.38 L 5.5 10.087 L 5.5 9 L 5 9 Z M 4.675 -0.38 L 1.175 2.62 L 1.825 3.38 L 5.325 0.38 L 4.675 -0.38 Z M 1.5 2.5 L 0 2.5 L 0 3.5 L 1.5 3.5 L 1.5 2.5 Z M -0.5 3 L -0.5 6 L 0.5 6 L 0.5 3 L -0.5 3 Z M 0 6.5 L 1.5 6.5 L 1.5 5.5 L 0 5.5 L 0 6.5 Z M 1.175 6.38 L 4.675 9.38 L 5.325 8.62 L 1.825 5.62 L 1.175 6.38 Z M 5.5 9 L 5.5 0 L 4.5 0 L 4.5 9 L 5.5 9 Z M 6.646 3.354 L 10.646 7.354 L 11.354 6.646 L 7.354 2.646 L 6.646 3.354 Z M 7.354 7.354 L 11.354 3.354 L 10.646 2.646 L 6.646 6.646 L 7.354 7.354 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "av-step-forward": "<g transform=\"translate(3.5,3.5)\"><path d=\"M 9 4.5 L 9.252 4.932 L 9.992 4.5 L 9.252 4.068 L 9 4.5 Z M 3 8 L 2.5 8 L 2.5 8.871 L 3.252 8.432 L 3 8 Z M 3 1 L 3.252 0.568 L 2.5 0.129 L 2.5 1 L 3 1 Z M 8.748 4.068 L 2.748 7.568 L 3.252 8.432 L 9.252 4.932 L 8.748 4.068 Z M 3.5 8 L 3.5 1 L 2.5 1 L 2.5 8 L 3.5 8 Z M 2.748 1.432 L 8.748 4.932 L 9.252 4.068 L 3.252 0.568 L 2.748 1.432 Z M -0.5 0 L -0.5 9 L 0.5 9 L 0.5 0 L -0.5 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "av-step-reverse": "<g transform=\"translate(3.5,3.5)\"><path d=\"M 0 4.5 L -0.252 4.068 L -0.992 4.5 L -0.252 4.932 L 0 4.5 Z M 6 8 L 5.748 8.432 L 6.5 8.871 L 6.5 8 L 6 8 Z M 6 1 L 6.5 1 L 6.5 0.129 L 5.748 0.568 L 6 1 Z M -0.252 4.932 L 5.748 8.432 L 6.252 7.568 L 0.252 4.068 L -0.252 4.932 Z M 6.5 8 L 6.5 1 L 5.5 1 L 5.5 8 L 6.5 8 Z M 5.748 0.568 L -0.252 4.068 L 0.252 4.932 L 6.252 1.432 L 5.748 0.568 Z M 8.5 0 L 8.5 9 L 9.5 9 L 9.5 0 L 8.5 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "av-videocam": "<g transform=\"translate(1.5,4.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 10 0 L 10.5 0 L 10.5 -0.5 L 10 -0.5 L 10 0 Z M 10 2 L 9.5 2 L 9.5 2.809 L 10.224 2.447 L 10 2 Z M 13 0.5 L 13.5 0.5 L 13.5 -0.309 L 12.776 0.053 L 13 0.5 Z M 13 6.5 L 12.776 6.947 L 13.5 7.309 L 13.5 6.5 L 13 6.5 Z M 10 5 L 10.224 4.553 L 9.5 4.191 L 9.5 5 L 10 5 Z M 10 7 L 10 7.5 L 10.5 7.5 L 10.5 7 L 10 7 Z M 0 7 L -0.5 7 L -0.5 7.5 L 0 7.5 L 0 7 Z M 0 0.5 L 10 0.5 L 10 -0.5 L 0 -0.5 L 0 0.5 Z M 9.5 0 L 9.5 2 L 10.5 2 L 10.5 0 L 9.5 0 Z M 10.224 2.447 L 13.224 0.947 L 12.776 0.053 L 9.776 1.553 L 10.224 2.447 Z M 12.5 0.5 L 12.5 6.5 L 13.5 6.5 L 13.5 0.5 L 12.5 0.5 Z M 13.224 6.053 L 10.224 4.553 L 9.776 5.447 L 12.776 6.947 L 13.224 6.053 Z M 9.5 5 L 9.5 7 L 10.5 7 L 10.5 5 L 9.5 5 Z M 10 6.5 L 0 6.5 L 0 7.5 L 10 7.5 L 10 6.5 Z M 0.5 7 L 0.5 0 L -0.5 0 L -0.5 7 L 0.5 7 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-add": "<g transform=\"translate(2,2)\"><path d=\"M 5.5 0 L 5.5 6 L 6.5 6 L 6.5 0 L 5.5 0 Z M 5.5 6 L 5.5 12 L 6.5 12 L 6.5 6 L 5.5 6 Z M 6 6.5 L 12 6.5 L 12 5.5 L 6 5.5 L 6 6.5 Z M 6 5.5 L 0 5.5 L 0 6.5 L 6 6.5 L 6 5.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-add-circle": "<g transform=\"translate(1.5,1.5)\"><path d=\"M 12.5 6.5 C 12.5 9.814 9.814 12.5 6.5 12.5 L 6.5 13.5 C 10.366 13.5 13.5 10.366 13.5 6.5 L 12.5 6.5 Z M 6.5 12.5 C 3.186 12.5 0.5 9.814 0.5 6.5 L -0.5 6.5 C -0.5 10.366 2.634 13.5 6.5 13.5 L 6.5 12.5 Z M 0.5 6.5 C 0.5 3.186 3.186 0.5 6.5 0.5 L 6.5 -0.5 C 2.634 -0.5 -0.5 2.634 -0.5 6.5 L 0.5 6.5 Z M 6.5 0.5 C 9.814 0.5 12.5 3.186 12.5 6.5 L 13.5 6.5 C 13.5 2.634 10.366 -0.5 6.5 -0.5 L 6.5 0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(4,4)\"><path d=\"M 4.5 8 L 4.5 0 L 3.5 0 L 3.5 8 L 4.5 8 Z M 8 3.5 L 0 3.5 L 0 4.5 L 8 4.5 L 8 3.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-alarm": "<g transform=\"translate(3,2)\"><path d=\"M 5 6 L 4.517 6.129 L 4.592 6.408 L 4.871 6.483 L 5 6 Z M 9 6 C 9 8.209 7.209 10 5 10 L 5 11 C 7.761 11 10 8.761 10 6 L 9 6 Z M 5 10 C 2.791 10 1 8.209 1 6 L 0 6 C 0 8.761 2.239 11 5 11 L 5 10 Z M 1 6 C 1 3.791 2.791 2 5 2 L 5 1 C 2.239 1 0 3.239 0 6 L 1 6 Z M 5 2 C 7.209 2 9 3.791 9 6 L 10 6 C 10 3.239 7.761 1 5 1 L 5 2 Z M 5.483 5.871 L 4.706 2.972 L 3.74 3.231 L 4.517 6.129 L 5.483 5.871 Z M 8.028 6.294 L 5.129 5.517 L 4.871 6.483 L 7.769 7.26 L 8.028 6.294 Z M 0.354 2.354 L 2.354 0.354 L 1.646 -0.354 L -0.354 1.646 L 0.354 2.354 Z M 10.354 1.646 L 8.354 -0.354 L 7.646 0.354 L 9.646 2.354 L 10.354 1.646 Z M 1.578 9.278 L 1.017 11.371 L 1.983 11.629 L 2.544 9.537 L 1.578 9.278 Z M 8.987 11.371 L 8.426 9.275 L 7.46 9.534 L 8.021 11.629 L 8.987 11.371 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-bell": "<g transform=\"translate(2.5,1.5)\"><path d=\"M 6.66 0 L 7.141 -0.134 L 7.039 -0.5 L 6.66 -0.5 L 6.66 0 Z M 7.011 1.26 L 6.529 1.394 L 6.599 1.644 L 6.843 1.731 L 7.011 1.26 Z M 10 9 L 9.5 9 L 9.5 9.118 L 9.553 9.224 L 10 9 Z M 11 11 L 11 11.5 L 11.809 11.5 L 11.447 10.776 L 11 11 Z M 0 11 L -0.447 10.776 L -0.809 11.5 L 0 11.5 L 0 11 Z M 1 9 L 1.447 9.224 L 1.5 9.118 L 1.5 9 L 1 9 Z M 3.989 1.26 L 4.157 1.731 L 4.401 1.644 L 4.471 1.394 L 3.989 1.26 Z M 4.34 0 L 4.34 -0.5 L 3.961 -0.5 L 3.859 -0.134 L 4.34 0 Z M 10.5 5.5 C 10.5 3.327 9.114 1.479 7.179 0.789 L 6.843 1.731 C 8.392 2.283 9.5 3.763 9.5 5.5 L 10.5 5.5 Z M 10.5 9 L 10.5 5.5 L 9.5 5.5 L 9.5 9 L 10.5 9 Z M 11.447 10.776 L 10.447 8.776 L 9.553 9.224 L 10.553 11.224 L 11.447 10.776 Z M 0 11.5 L 11 11.5 L 11 10.5 L 0 10.5 L 0 11.5 Z M 0.553 8.776 L -0.447 10.776 L 0.447 11.224 L 1.447 9.224 L 0.553 8.776 Z M 0.5 5.5 L 0.5 9 L 1.5 9 L 1.5 5.5 L 0.5 5.5 Z M 3.821 0.789 C 1.886 1.479 0.5 3.327 0.5 5.5 L 1.5 5.5 C 1.5 3.763 2.608 2.283 4.157 1.731 L 3.821 0.789 Z M 6.66 -0.5 L 4.34 -0.5 L 4.34 0.5 L 6.66 0.5 L 6.66 -0.5 Z M 3.859 -0.134 L 3.507 1.126 L 4.471 1.394 L 4.822 0.134 L 3.859 -0.134 Z M 7.493 1.126 L 7.141 -0.134 L 6.178 0.134 L 6.529 1.394 L 7.493 1.126 Z M 4.5 11.5 C 4.5 11.382 4.52 11.27 4.557 11.167 L 3.614 10.833 C 3.54 11.043 3.5 11.267 3.5 11.5 L 4.5 11.5 Z M 5.5 12.5 C 4.948 12.5 4.5 12.052 4.5 11.5 L 3.5 11.5 C 3.5 12.605 4.395 13.5 5.5 13.5 L 5.5 12.5 Z M 6.5 11.5 C 6.5 12.052 6.052 12.5 5.5 12.5 L 5.5 13.5 C 6.605 13.5 7.5 12.605 7.5 11.5 L 6.5 11.5 Z M 6.443 11.167 C 6.48 11.27 6.5 11.382 6.5 11.5 L 7.5 11.5 C 7.5 11.267 7.46 11.043 7.386 10.833 L 6.443 11.167 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-calendar": "<g transform=\"translate(2.5,2)\"><path d=\"M 0 1.5 L 0 1 L -0.5 1 L -0.5 1.5 L 0 1.5 Z M 11 1.5 L 11.5 1.5 L 11.5 1 L 11 1 L 11 1.5 Z M 11 11.5 L 11 12 L 11.5 12 L 11.5 11.5 L 11 11.5 Z M 0 11.5 L -0.5 11.5 L -0.5 12 L 0 12 L 0 11.5 Z M 0 2 L 11 2 L 11 1 L 0 1 L 0 2 Z M 10.5 1.5 L 10.5 11.5 L 11.5 11.5 L 11.5 1.5 L 10.5 1.5 Z M 11 11 L 0 11 L 0 12 L 11 12 L 11 11 Z M 0.5 11.5 L 0.5 1.5 L -0.5 1.5 L -0.5 11.5 L 0.5 11.5 Z M 0 5 L 11 5 L 11 4 L 0 4 L 0 5 Z M 2.5 0 L 2.5 3 L 3.5 3 L 3.5 0 L 2.5 0 Z M 7.5 0 L 7.5 3 L 8.5 3 L 8.5 0 L 7.5 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-calendar-event": "<g transform=\"translate(7,9)\"><path d=\"M 1 0 L 1.448 -0.221 L 1 -1.13 L 0.552 -0.221 L 1 0 Z M 1.309 0.626 L 0.861 0.847 L 0.977 1.083 L 1.237 1.121 L 1.309 0.626 Z M 2 0.727 L 2.349 1.085 L 3.074 0.377 L 2.072 0.232 L 2 0.727 Z M 1.5 1.214 L 1.151 0.856 L 0.963 1.039 L 1.007 1.298 L 1.5 1.214 Z M 1.618 1.902 L 1.385 2.345 L 2.282 2.816 L 2.111 1.818 L 1.618 1.902 Z M 1 1.577 L 1.233 1.135 L 1 1.012 L 0.767 1.135 L 1 1.577 Z M 0.382 1.902 L -0.111 1.818 L -0.282 2.816 L 0.615 2.345 L 0.382 1.902 Z M 0.5 1.214 L 0.993 1.298 L 1.037 1.039 L 0.849 0.856 L 0.5 1.214 Z M 0 0.727 L -0.072 0.232 L -1.074 0.377 L -0.349 1.085 L 0 0.727 Z M 0.691 0.626 L 0.763 1.121 L 1.023 1.083 L 1.139 0.847 L 0.691 0.626 Z M 0.552 0.221 L 0.861 0.847 L 1.757 0.405 L 1.448 -0.221 L 0.552 0.221 Z M 1.237 1.121 L 1.928 1.221 L 2.072 0.232 L 1.381 0.131 L 1.237 1.121 Z M 1.651 0.368 L 1.151 0.856 L 1.849 1.572 L 2.349 1.085 L 1.651 0.368 Z M 1.007 1.298 L 1.125 1.987 L 2.111 1.818 L 1.993 1.129 L 1.007 1.298 Z M 1.851 1.46 L 1.233 1.135 L 0.767 2.02 L 1.385 2.345 L 1.851 1.46 Z M 0.767 1.135 L 0.149 1.46 L 0.615 2.345 L 1.233 2.02 L 0.767 1.135 Z M 0.875 1.987 L 0.993 1.298 L 0.007 1.129 L -0.111 1.818 L 0.875 1.987 Z M 0.849 0.856 L 0.349 0.368 L -0.349 1.085 L 0.151 1.572 L 0.849 0.856 Z M 0.072 1.221 L 0.763 1.121 L 0.619 0.131 L -0.072 0.232 L 0.072 1.221 Z M 1.139 0.847 L 1.448 0.221 L 0.552 -0.221 L 0.243 0.405 L 1.139 0.847 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(2.5,2)\"><path d=\"M 0 1.5 L 0 1 L -0.5 1 L -0.5 1.5 L 0 1.5 Z M 11 1.5 L 11.5 1.5 L 11.5 1 L 11 1 L 11 1.5 Z M 11 11.5 L 11 12 L 11.5 12 L 11.5 11.5 L 11 11.5 Z M 0 11.5 L -0.5 11.5 L -0.5 12 L 0 12 L 0 11.5 Z M 0 2 L 11 2 L 11 1 L 0 1 L 0 2 Z M 10.5 1.5 L 10.5 11.5 L 11.5 11.5 L 11.5 1.5 L 10.5 1.5 Z M 11 11 L 0 11 L 0 12 L 11 12 L 11 11 Z M 0.5 11.5 L 0.5 1.5 L -0.5 1.5 L -0.5 11.5 L 0.5 11.5 Z M 0 5 L 11 5 L 11 4 L 0 4 L 0 5 Z M 2.5 0 L 2.5 3 L 3.5 3 L 3.5 0 L 2.5 0 Z M 7.5 0 L 7.5 3 L 8.5 3 L 8.5 0 L 7.5 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-cancel": "<g transform=\"translate(1.5,1.5)\"><path d=\"M 12.5 6.5 C 12.5 9.814 9.814 12.5 6.5 12.5 L 6.5 13.5 C 10.366 13.5 13.5 10.366 13.5 6.5 L 12.5 6.5 Z M 6.5 12.5 C 3.186 12.5 0.5 9.814 0.5 6.5 L -0.5 6.5 C -0.5 10.366 2.634 13.5 6.5 13.5 L 6.5 12.5 Z M 0.5 6.5 C 0.5 3.186 3.186 0.5 6.5 0.5 L 6.5 -0.5 C 2.634 -0.5 -0.5 2.634 -0.5 6.5 L 0.5 6.5 Z M 6.5 0.5 C 9.814 0.5 12.5 3.186 12.5 6.5 L 13.5 6.5 C 13.5 2.634 10.366 -0.5 6.5 -0.5 L 6.5 0.5 Z M 1.55 2.257 L 10.743 11.45 L 11.45 10.743 L 2.257 1.55 L 1.55 2.257 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-check": "<g transform=\"translate(2,4)\"><path d=\"M 4 8 L 3.646 8.354 L 4 8.707 L 4.354 8.354 L 4 8 Z M -0.354 4.354 L 3.646 8.354 L 4.354 7.646 L 0.354 3.646 L -0.354 4.354 Z M 4.354 8.354 L 12.354 0.354 L 11.646 -0.354 L 3.646 7.646 L 4.354 8.354 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-check-circle": "<g transform=\"translate(1.5,1.5)\"><path d=\"M 12.5 6.5 C 12.5 9.814 9.814 12.5 6.5 12.5 L 6.5 13.5 C 10.366 13.5 13.5 10.366 13.5 6.5 L 12.5 6.5 Z M 6.5 12.5 C 3.186 12.5 0.5 9.814 0.5 6.5 L -0.5 6.5 C -0.5 10.366 2.634 13.5 6.5 13.5 L 6.5 12.5 Z M 0.5 6.5 C 0.5 3.186 3.186 0.5 6.5 0.5 L 6.5 -0.5 C 2.634 -0.5 -0.5 2.634 -0.5 6.5 L 0.5 6.5 Z M 6.5 0.5 C 9.814 0.5 12.5 3.186 12.5 6.5 L 13.5 6.5 C 13.5 2.634 10.366 -0.5 6.5 -0.5 L 6.5 0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(4.5,5.5)\"><path d=\"M 2.5 4.5 L 2.146 4.854 L 2.5 5.207 L 2.854 4.854 L 2.5 4.5 Z M -0.354 2.354 L 2.146 4.854 L 2.854 4.146 L 0.354 1.646 L -0.354 2.354 Z M 2.854 4.854 L 7.354 0.354 L 6.646 -0.354 L 2.146 4.146 L 2.854 4.854 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-check-multi-circle": "<g transform=\"translate(1.5,1.5)\"><path d=\"M 12.5 6.5 C 12.5 9.814 9.814 12.5 6.5 12.5 L 6.5 13.5 C 10.366 13.5 13.5 10.366 13.5 6.5 L 12.5 6.5 Z M 6.5 12.5 C 3.186 12.5 0.5 9.814 0.5 6.5 L -0.5 6.5 C -0.5 10.366 2.634 13.5 6.5 13.5 L 6.5 12.5 Z M 0.5 6.5 C 0.5 3.186 3.186 0.5 6.5 0.5 L 6.5 -0.5 C 2.634 -0.5 -0.5 2.634 -0.5 6.5 L 0.5 6.5 Z M 6.5 0.5 C 9.814 0.5 12.5 3.186 12.5 6.5 L 13.5 6.5 C 13.5 2.634 10.366 -0.5 6.5 -0.5 L 6.5 0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(3,0.5)\"><path d=\"M 2.5 4.5 L 2.146 4.854 L 2.5 5.207 L 2.854 4.854 L 2.5 4.5 Z M 6.646 -0.354 L 2.146 4.146 L 2.854 4.854 L 7.354 0.354 L 6.646 -0.354 Z M 2.854 4.146 L 0.354 1.646 L -0.354 2.354 L 2.146 4.854 L 2.854 4.146 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(5,0)\"><path d=\"M 2.146 -0.354 L -0.354 2.146 L 0.354 2.854 L 2.854 0.354 L 2.146 -0.354 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(0,2.5)\"><path d=\"M 3.207 2.5 L 0.354 -0.354 L -0.354 0.354 L 2.5 3.207 L 3.207 2.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-clipboard": "<g transform=\"translate(3.5,2.5)\"><path d=\"M 6.5 0 L 6.983 0.129 L 7.152 -0.5 L 6.5 -0.5 L 6.5 0 Z M 2.5 0 L 2.5 -0.5 L 1.848 -0.5 L 2.017 0.129 L 2.5 0 Z M 5.964 2 L 5.964 2.5 L 6.348 2.5 L 6.447 2.129 L 5.964 2 Z M 3.036 2 L 2.553 2.129 L 2.652 2.5 L 3.036 2.5 L 3.036 2 Z M 0 1 L 0 0.5 L -0.5 0.5 L -0.5 1 L 0 1 Z M 0 11 L -0.5 11 L -0.5 11.5 L 0 11.5 L 0 11 Z M 9 11 L 9 11.5 L 9.5 11.5 L 9.5 11 L 9 11 Z M 9 1 L 9.5 1 L 9.5 0.5 L 9 0.5 L 9 1 Z M 6.5 -0.5 L 2.5 -0.5 L 2.5 0.5 L 6.5 0.5 L 6.5 -0.5 Z M 6.447 2.129 L 6.983 0.129 L 6.017 -0.129 L 5.481 1.871 L 6.447 2.129 Z M 3.036 2.5 L 5.964 2.5 L 5.964 1.5 L 3.036 1.5 L 3.036 2.5 Z M 2.017 0.129 L 2.553 2.129 L 3.519 1.871 L 2.983 -0.129 L 2.017 0.129 Z M 2.768 0.5 L 0 0.5 L 0 1.5 L 2.768 1.5 L 2.768 0.5 Z M -0.5 1 L -0.5 11 L 0.5 11 L 0.5 1 L -0.5 1 Z M 0 11.5 L 9 11.5 L 9 10.5 L 0 10.5 L 0 11.5 Z M 9.5 11 L 9.5 1 L 8.5 1 L 8.5 11 L 9.5 11 Z M 9 0.5 L 6.232 0.5 L 6.232 1.5 L 9 1.5 L 9 0.5 Z M 1.5 5.5 L 7.5 5.5 L 7.5 4.5 L 1.5 4.5 L 1.5 5.5 Z M 1.5 8.5 L 7.5 8.5 L 7.5 7.5 L 1.5 7.5 L 1.5 8.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-clock": "<g transform=\"translate(1.5,1.5)\"><path d=\"M 6.5 6.5 L 6 6.5 L 6 6.884 L 6.371 6.983 L 6.5 6.5 Z M 12.5 6.5 C 12.5 9.814 9.814 12.5 6.5 12.5 L 6.5 13.5 C 10.366 13.5 13.5 10.366 13.5 6.5 L 12.5 6.5 Z M 6.5 12.5 C 3.186 12.5 0.5 9.814 0.5 6.5 L -0.5 6.5 C -0.5 10.366 2.634 13.5 6.5 13.5 L 6.5 12.5 Z M 0.5 6.5 C 0.5 3.186 3.186 0.5 6.5 0.5 L 6.5 -0.5 C 2.634 -0.5 -0.5 2.634 -0.5 6.5 L 0.5 6.5 Z M 6.5 0.5 C 9.814 0.5 12.5 3.186 12.5 6.5 L 13.5 6.5 C 13.5 2.634 10.366 -0.5 6.5 -0.5 L 6.5 0.5 Z M 6 2.5 L 6 6.5 L 7 6.5 L 7 2.5 L 6 2.5 Z M 6.371 6.983 L 10.235 8.018 L 10.494 7.053 L 6.629 6.017 L 6.371 6.983 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-close": "<g transform=\"translate(3,3)\"><path d=\"M -0.354 0.354 L 9.646 10.354 L 10.354 9.646 L 0.354 -0.354 L -0.354 0.354 Z M 9.646 -0.354 L -0.354 9.646 L 0.354 10.354 L 10.354 0.354 L 9.646 -0.354 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-close-circle": "<g transform=\"translate(1.5,1.5)\"><path d=\"M 12.5 6.5 C 12.5 9.814 9.814 12.5 6.5 12.5 L 6.5 13.5 C 10.366 13.5 13.5 10.366 13.5 6.5 L 12.5 6.5 Z M 6.5 12.5 C 3.186 12.5 0.5 9.814 0.5 6.5 L -0.5 6.5 C -0.5 10.366 2.634 13.5 6.5 13.5 L 6.5 12.5 Z M 0.5 6.5 C 0.5 3.186 3.186 0.5 6.5 0.5 L 6.5 -0.5 C 2.634 -0.5 -0.5 2.634 -0.5 6.5 L 0.5 6.5 Z M 6.5 0.5 C 9.814 0.5 12.5 3.186 12.5 6.5 L 13.5 6.5 C 13.5 2.634 10.366 -0.5 6.5 -0.5 L 6.5 0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(5,5)\"><path d=\"M -0.354 0.354 L 5.646 6.354 L 6.354 5.646 L 0.354 -0.354 L -0.354 0.354 Z M 5.646 -0.354 L -0.354 5.646 L 0.354 6.354 L 6.354 0.354 L 5.646 -0.354 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-cog": "<g transform=\"translate(2.787,2.5)\"><path d=\"M 2.514 2.548 L 2.298 2.999 L 2.602 3.145 L 2.851 2.917 L 2.514 2.548 Z M 1.086 1.864 L 1.303 1.413 L 0.962 1.25 L 0.711 1.534 L 1.086 1.864 Z M 4.007 1.685 L 4.158 2.162 L 4.479 2.06 L 4.505 1.724 L 4.007 1.685 Z M 4.128 0.107 L 4.03 -0.383 L 3.659 -0.309 L 3.63 0.069 L 4.128 0.107 Z M 6.299 0.107 L 6.797 0.069 L 6.768 -0.309 L 6.397 -0.383 L 6.299 0.107 Z M 6.42 1.685 L 5.921 1.724 L 5.947 2.06 L 6.269 2.162 L 6.42 1.685 Z M 7.913 2.548 L 7.575 2.917 L 7.824 3.145 L 8.129 2.999 L 7.913 2.548 Z M 9.34 1.864 L 9.715 1.534 L 9.465 1.25 L 9.124 1.413 L 9.34 1.864 Z M 10.427 3.743 L 10.709 4.156 L 11.021 3.942 L 10.901 3.583 L 10.427 3.743 Z M 9.12 4.638 L 8.838 4.225 L 8.559 4.416 L 8.632 4.745 L 9.12 4.638 Z M 9.12 6.362 L 8.632 6.255 L 8.559 6.584 L 8.838 6.775 L 9.12 6.362 Z M 10.427 7.257 L 10.901 7.417 L 11.021 7.058 L 10.709 6.844 L 10.427 7.257 Z M 9.34 9.136 L 9.124 9.587 L 9.465 9.75 L 9.715 9.466 L 9.34 9.136 Z M 7.913 8.452 L 8.129 8.001 L 7.824 7.855 L 7.575 8.083 L 7.913 8.452 Z M 6.42 9.315 L 6.269 8.838 L 5.947 8.94 L 5.921 9.276 L 6.42 9.315 Z M 6.299 10.893 L 6.397 11.383 L 6.768 11.309 L 6.797 10.931 L 6.299 10.893 Z M 4.128 10.893 L 3.63 10.931 L 3.659 11.309 L 4.03 11.383 L 4.128 10.893 Z M 4.007 9.315 L 4.505 9.276 L 4.479 8.94 L 4.158 8.838 L 4.007 9.315 Z M 2.514 8.452 L 2.851 8.083 L 2.602 7.855 L 2.298 8.001 L 2.514 8.452 Z M 1.086 9.136 L 0.711 9.466 L 0.961 9.75 L 1.303 9.587 L 1.086 9.136 Z M 0 7.257 L -0.282 6.844 L -0.595 7.058 L -0.474 7.417 L 0 7.257 Z M 1.307 6.362 L 1.589 6.775 L 1.867 6.584 L 1.795 6.255 L 1.307 6.362 Z M 1.307 4.638 L 1.795 4.745 L 1.867 4.416 L 1.589 4.225 L 1.307 4.638 Z M 0 3.743 L -0.474 3.583 L -0.595 3.942 L -0.282 4.156 L 0 3.743 Z M 2.73 2.097 L 1.303 1.413 L 0.87 2.315 L 2.298 2.999 L 2.73 2.097 Z M 3.856 1.208 C 3.224 1.408 2.653 1.743 2.176 2.179 L 2.851 2.917 C 3.223 2.577 3.667 2.317 4.158 2.162 L 3.856 1.208 Z M 3.63 0.069 L 3.508 1.647 L 4.505 1.724 L 4.627 0.145 L 3.63 0.069 Z M 5.213 -0.5 C 4.809 -0.5 4.413 -0.46 4.03 -0.383 L 4.226 0.597 C 4.545 0.534 4.875 0.5 5.213 0.5 L 5.213 -0.5 Z M 6.397 -0.383 C 6.014 -0.46 5.618 -0.5 5.213 -0.5 L 5.213 0.5 C 5.552 0.5 5.882 0.534 6.2 0.597 L 6.397 -0.383 Z M 6.919 1.647 L 6.797 0.069 L 5.8 0.145 L 5.921 1.724 L 6.919 1.647 Z M 8.25 2.179 C 7.774 1.743 7.203 1.408 6.571 1.208 L 6.269 2.162 C 6.76 2.317 7.204 2.577 7.575 2.917 L 8.25 2.179 Z M 9.124 1.413 L 7.697 2.097 L 8.129 2.999 L 9.556 2.315 L 9.124 1.413 Z M 10.901 3.583 C 10.643 2.819 10.236 2.124 9.715 1.534 L 8.965 2.195 C 9.4 2.688 9.739 3.267 9.953 3.903 L 10.901 3.583 Z M 9.403 5.05 L 10.709 4.156 L 10.144 3.33 L 8.838 4.225 L 9.403 5.05 Z M 9.713 5.5 C 9.713 5.168 9.677 4.843 9.609 4.53 L 8.632 4.745 C 8.685 4.988 8.713 5.24 8.713 5.5 L 9.713 5.5 Z M 9.609 6.47 C 9.677 6.157 9.713 5.832 9.713 5.5 L 8.713 5.5 C 8.713 5.76 8.685 6.012 8.632 6.255 L 9.609 6.47 Z M 10.709 6.844 L 9.403 5.95 L 8.838 6.775 L 10.144 7.67 L 10.709 6.844 Z M 9.715 9.466 C 10.236 8.876 10.643 8.181 10.901 7.417 L 9.953 7.097 C 9.739 7.733 9.4 8.312 8.965 8.805 L 9.715 9.466 Z M 7.697 8.903 L 9.124 9.587 L 9.556 8.685 L 8.129 8.001 L 7.697 8.903 Z M 6.571 9.792 C 7.203 9.592 7.774 9.257 8.25 8.821 L 7.575 8.083 C 7.204 8.423 6.76 8.683 6.269 8.838 L 6.571 9.792 Z M 6.797 10.931 L 6.919 9.353 L 5.921 9.276 L 5.8 10.855 L 6.797 10.931 Z M 5.213 11.5 C 5.618 11.5 6.014 11.46 6.397 11.383 L 6.2 10.403 C 5.882 10.466 5.552 10.5 5.213 10.5 L 5.213 11.5 Z M 4.03 11.383 C 4.413 11.46 4.809 11.5 5.213 11.5 L 5.213 10.5 C 4.875 10.5 4.545 10.466 4.226 10.403 L 4.03 11.383 Z M 3.508 9.353 L 3.63 10.931 L 4.627 10.855 L 4.505 9.276 L 3.508 9.353 Z M 2.176 8.821 C 2.653 9.257 3.224 9.592 3.856 9.792 L 4.158 8.838 C 3.667 8.683 3.223 8.423 2.851 8.083 L 2.176 8.821 Z M 1.303 9.587 L 2.73 8.903 L 2.298 8.001 L 0.87 8.685 L 1.303 9.587 Z M -0.474 7.417 C -0.216 8.181 0.19 8.876 0.711 9.466 L 1.462 8.805 C 1.027 8.312 0.688 7.733 0.474 7.097 L -0.474 7.417 Z M 1.024 5.95 L -0.282 6.844 L 0.282 7.67 L 1.589 6.775 L 1.024 5.95 Z M 0.713 5.5 C 0.713 5.832 0.75 6.157 0.818 6.47 L 1.795 6.255 C 1.742 6.012 1.713 5.76 1.713 5.5 L 0.713 5.5 Z M 0.818 4.53 C 0.75 4.843 0.713 5.168 0.713 5.5 L 1.713 5.5 C 1.713 5.24 1.742 4.988 1.795 4.745 L 0.818 4.53 Z M -0.282 4.156 L 1.024 5.05 L 1.589 4.225 L 0.282 3.33 L -0.282 4.156 Z M 0.711 1.534 C 0.19 2.124 -0.216 2.819 -0.474 3.583 L 0.474 3.903 C 0.688 3.267 1.027 2.688 1.462 2.195 L 0.711 1.534 Z M 6.213 5.5 C 6.213 6.052 5.766 6.5 5.213 6.5 L 5.213 7.5 C 6.318 7.5 7.213 6.605 7.213 5.5 L 6.213 5.5 Z M 5.213 4.5 C 5.766 4.5 6.213 4.948 6.213 5.5 L 7.213 5.5 C 7.213 4.395 6.318 3.5 5.213 3.5 L 5.213 4.5 Z M 4.213 5.5 C 4.213 4.948 4.661 4.5 5.213 4.5 L 5.213 3.5 C 4.109 3.5 3.213 4.395 3.213 5.5 L 4.213 5.5 Z M 5.213 6.5 C 4.661 6.5 4.213 6.052 4.213 5.5 L 3.213 5.5 C 3.213 6.605 4.109 7.5 5.213 7.5 L 5.213 6.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-copy-doc": "<g transform=\"translate(3.5,2.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 7 2 L 7.5 2 L 7.5 1.793 L 7.354 1.646 L 7 2 Z M 7 9 L 7 9.5 L 7.5 9.5 L 7.5 9 L 7 9 Z M 0 9 L -0.5 9 L -0.5 9.5 L 0 9.5 L 0 9 Z M 5 0 L 5.354 -0.354 L 5.207 -0.5 L 5 -0.5 L 5 0 Z M 9 11 L 9 11.5 L 9.5 11.5 L 9.5 11 L 9 11 Z M 6.5 2 L 6.5 9 L 7.5 9 L 7.5 2 L 6.5 2 Z M 7 8.5 L 0 8.5 L 0 9.5 L 7 9.5 L 7 8.5 Z M 0.5 9 L 0.5 0 L -0.5 0 L -0.5 9 L 0.5 9 Z M 0 0.5 L 5 0.5 L 5 -0.5 L 0 -0.5 L 0 0.5 Z M 4.646 0.354 L 6.646 2.354 L 7.354 1.646 L 5.354 -0.354 L 4.646 0.354 Z M 2.5 11.5 L 9 11.5 L 9 10.5 L 2.5 10.5 L 2.5 11.5 Z M 9.5 11 L 9.5 2.5 L 8.5 2.5 L 8.5 11 L 9.5 11 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(8.5,2.5)\"><path d=\"M 0 2 L -0.5 2 L -0.5 2.5 L 0 2.5 L 0 2 Z M -0.5 0 L -0.5 2 L 0.5 2 L 0.5 0 L -0.5 0 Z M 0 2.5 L 2 2.5 L 2 1.5 L 0 1.5 L 0 2.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-copy-generic": "<g transform=\"translate(2.5,2.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 9 0 L 9.5 0 L 9.5 -0.5 L 9 -0.5 L 9 0 Z M 9 9 L 9 9.5 L 9.5 9.5 L 9.5 9 L 9 9 Z M 0 9 L -0.5 9 L -0.5 9.5 L 0 9.5 L 0 9 Z M 11 11 L 11 11.5 L 11.5 11.5 L 11.5 11 L 11 11 Z M 0 0.5 L 9 0.5 L 9 -0.5 L 0 -0.5 L 0 0.5 Z M 8.5 0 L 8.5 9 L 9.5 9 L 9.5 0 L 8.5 0 Z M 9 8.5 L 0 8.5 L 0 9.5 L 9 9.5 L 9 8.5 Z M 0.5 9 L 0.5 0 L -0.5 0 L -0.5 9 L 0.5 9 Z M 11 10.5 L 1.5 10.5 L 1.5 11.5 L 11 11.5 L 11 10.5 Z M 10.5 1.5 L 10.5 11 L 11.5 11 L 11.5 1.5 L 10.5 1.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-enter": "<g transform=\"translate(0,0)\"><path d=\"M 7 0 L 7.5 0 L 7.5 -0.5 L 7 -0.5 L 7 0 Z M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 0 11 L -0.5 11 L -0.5 11.5 L 0 11.5 L 0 11 Z M 7 11 L 7 11.5 L 7.5 11.5 L 7.5 11 L 7 11 Z M 7.5 1.5 L 7.5 0 L 6.5 0 L 6.5 1.5 L 7.5 1.5 Z M 7 -0.5 L 0 -0.5 L 0 0.5 L 7 0.5 L 7 -0.5 Z M -0.5 0 L -0.5 11 L 0.5 11 L 0.5 0 L -0.5 0 Z M 0 11.5 L 7 11.5 L 7 10.5 L 0 10.5 L 0 11.5 Z M 7.5 11 L 7.5 9.5 L 6.5 9.5 L 6.5 11 L 7.5 11 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(1,5.5)\"><path d=\"M 9.5 2.5 L 9.854 2.854 L 10.207 2.5 L 9.854 2.146 L 9.5 2.5 Z M 0 3 L 9.5 3 L 9.5 2 L 0 2 L 0 3 Z M 9.146 2.146 L 6.646 4.646 L 7.354 5.354 L 9.854 2.854 L 9.146 2.146 Z M 9.854 2.146 L 7.354 -0.354 L 6.646 0.354 L 9.146 2.854 L 9.854 2.146 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-exit": "<g transform=\"translate(2.5,2.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 7 0 L 7.5 0 L 7.5 -0.5 L 7 -0.5 L 7 0 Z M 7 11 L 7 11.5 L 7.5 11.5 L 7.5 11 L 7 11 Z M 0 11 L -0.5 11 L -0.5 11.5 L 0 11.5 L 0 11 Z M 0 0.5 L 7 0.5 L 7 -0.5 L 0 -0.5 L 0 0.5 Z M 7 10.5 L 0 10.5 L 0 11.5 L 7 11.5 L 7 10.5 Z M 0.5 11 L 0.5 0 L -0.5 0 L -0.5 11 L 0.5 11 Z M 6.5 0 L 6.5 1.5 L 7.5 1.5 L 7.5 0 L 6.5 0 Z M 6.5 9.5 L 6.5 11 L 7.5 11 L 7.5 9.5 L 6.5 9.5 Z M 12 5 L 2.5 5 L 2.5 6 L 12 6 L 12 5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(12,5.5)\"><path d=\"M 2.5 2.5 L 2.854 2.854 L 3.207 2.5 L 2.854 2.146 L 2.5 2.5 Z M -0.354 0.354 L 2.146 2.854 L 2.854 2.146 L 0.354 -0.354 L -0.354 0.354 Z M 2.146 2.146 L -0.354 4.646 L 0.354 5.354 L 2.854 2.854 L 2.146 2.146 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-eye": "<g transform=\"translate(1.5,4.5)\"><path d=\"M 13 3.5 L 13.435 3.746 L 13.574 3.5 L 13.435 3.254 L 13 3.5 Z M 0 3.5 L -0.435 3.254 L -0.574 3.5 L -0.435 3.746 L 0 3.5 Z M 7.5 3.5 C 7.5 4.052 7.052 4.5 6.5 4.5 L 6.5 5.5 C 7.605 5.5 8.5 4.605 8.5 3.5 L 7.5 3.5 Z M 6.5 4.5 C 5.948 4.5 5.5 4.052 5.5 3.5 L 4.5 3.5 C 4.5 4.605 5.395 5.5 6.5 5.5 L 6.5 4.5 Z M 5.5 3.5 C 5.5 2.948 5.948 2.5 6.5 2.5 L 6.5 1.5 C 5.395 1.5 4.5 2.395 4.5 3.5 L 5.5 3.5 Z M 6.5 2.5 C 7.052 2.5 7.5 2.948 7.5 3.5 L 8.5 3.5 C 8.5 2.395 7.605 1.5 6.5 1.5 L 6.5 2.5 Z M 12.565 3.254 C 12.337 3.656 11.566 4.478 10.438 5.205 C 9.322 5.924 7.932 6.5 6.5 6.5 L 6.5 7.5 C 8.19 7.5 9.767 6.826 10.979 6.045 C 12.179 5.272 13.097 4.344 13.435 3.746 L 12.565 3.254 Z M 6.5 6.5 C 5.068 6.5 3.678 5.924 2.562 5.205 C 1.434 4.478 0.663 3.656 0.435 3.254 L -0.435 3.746 C -0.097 4.344 0.821 5.272 2.021 6.045 C 3.233 6.826 4.81 7.5 6.5 7.5 L 6.5 6.5 Z M 0.435 3.746 C 0.663 3.344 1.434 2.522 2.562 1.795 C 3.678 1.076 5.068 0.5 6.5 0.5 L 6.5 -0.5 C 4.81 -0.5 3.233 0.174 2.021 0.955 C 0.821 1.728 -0.097 2.656 -0.435 3.254 L 0.435 3.746 Z M 6.5 0.5 C 7.932 0.5 9.322 1.076 10.438 1.795 C 11.566 2.522 12.337 3.344 12.565 3.746 L 13.435 3.254 C 13.097 2.656 12.179 1.728 10.979 0.955 C 9.767 0.174 8.19 -0.5 6.5 -0.5 L 6.5 0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-eye-off": "<g transform=\"translate(1.5,4.5)\"><path d=\"M 13 3.5 L 13.435 3.746 L 13.574 3.5 L 13.435 3.254 L 13 3.5 Z M 0 3.5 L -0.435 3.254 L -0.574 3.5 L -0.435 3.746 L 0 3.5 Z M 7.5 3.5 C 7.5 4.052 7.052 4.5 6.5 4.5 L 6.5 5.5 C 7.605 5.5 8.5 4.605 8.5 3.5 L 7.5 3.5 Z M 6.5 4.5 C 5.948 4.5 5.5 4.052 5.5 3.5 L 4.5 3.5 C 4.5 4.605 5.395 5.5 6.5 5.5 L 6.5 4.5 Z M 5.5 3.5 C 5.5 2.948 5.948 2.5 6.5 2.5 L 6.5 1.5 C 5.395 1.5 4.5 2.395 4.5 3.5 L 5.5 3.5 Z M 6.5 2.5 C 7.052 2.5 7.5 2.948 7.5 3.5 L 8.5 3.5 C 8.5 2.395 7.605 1.5 6.5 1.5 L 6.5 2.5 Z M 12.565 3.254 C 12.337 3.656 11.566 4.478 10.438 5.205 C 9.322 5.924 7.932 6.5 6.5 6.5 L 6.5 7.5 C 8.19 7.5 9.767 6.826 10.979 6.045 C 12.179 5.272 13.097 4.344 13.435 3.746 L 12.565 3.254 Z M 6.5 6.5 C 5.068 6.5 3.678 5.924 2.562 5.205 C 1.434 4.478 0.663 3.656 0.435 3.254 L -0.435 3.746 C -0.097 4.344 0.821 5.272 2.021 6.045 C 3.233 6.826 4.81 7.5 6.5 7.5 L 6.5 6.5 Z M 0.435 3.746 C 0.663 3.344 1.434 2.522 2.562 1.795 C 3.678 1.076 5.068 0.5 6.5 0.5 L 6.5 -0.5 C 4.81 -0.5 3.233 0.174 2.021 0.955 C 0.821 1.728 -0.097 2.656 -0.435 3.254 L 0.435 3.746 Z M 6.5 0.5 C 7.932 0.5 9.322 1.076 10.438 1.795 C 11.566 2.522 12.337 3.344 12.565 3.746 L 13.435 3.254 C 13.097 2.656 12.179 1.728 10.979 0.955 C 9.767 0.174 8.19 -0.5 6.5 -0.5 L 6.5 0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(0,0)\"><path d=\"M 11.646 -0.354 L -0.354 11.646 L 0.354 12.354 L 12.354 0.354 L 11.646 -0.354 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-filter": "<g transform=\"translate(3,5.5)\"><path d=\"M 0 0.5 L 10 0.5 L 10 -0.5 L 0 -0.5 L 0 0.5 Z M 2 3.5 L 8 3.5 L 8 2.5 L 2 2.5 L 2 3.5 Z M 4 6.5 L 6 6.5 L 6 5.5 L 4 5.5 L 4 6.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-help-circle": "<g transform=\"translate(1.5,1.5)\"><path d=\"M 12.5 6.5 C 12.5 9.814 9.814 12.5 6.5 12.5 L 6.5 13.5 C 10.366 13.5 13.5 10.366 13.5 6.5 L 12.5 6.5 Z M 6.5 12.5 C 3.186 12.5 0.5 9.814 0.5 6.5 L -0.5 6.5 C -0.5 10.366 2.634 13.5 6.5 13.5 L 6.5 12.5 Z M 0.5 6.5 C 0.5 3.186 3.186 0.5 6.5 0.5 L 6.5 -0.5 C 2.634 -0.5 -0.5 2.634 -0.5 6.5 L 0.5 6.5 Z M 6.5 0.5 C 9.814 0.5 12.5 3.186 12.5 6.5 L 13.5 6.5 C 13.5 2.634 10.366 -0.5 6.5 -0.5 L 6.5 0.5 Z M 6 9.5 L 6 10.5 L 7 10.5 L 7 9.5 L 6 9.5 Z M 7.5 4.5 C 7.5 4.784 7.431 4.973 7.341 5.119 C 7.244 5.275 7.108 5.405 6.93 5.553 C 6.773 5.684 6.531 5.867 6.355 6.073 C 6.156 6.305 6 6.604 6 7 L 7 7 C 7 6.896 7.032 6.82 7.114 6.724 C 7.219 6.602 7.352 6.503 7.57 6.322 C 7.767 6.158 8.006 5.944 8.19 5.646 C 8.381 5.339 8.5 4.966 8.5 4.5 L 7.5 4.5 Z M 5.5 4.5 C 5.5 3.948 5.948 3.5 6.5 3.5 L 6.5 2.5 C 5.395 2.5 4.5 3.395 4.5 4.5 L 5.5 4.5 Z M 6.5 3.5 C 7.052 3.5 7.5 3.948 7.5 4.5 L 8.5 4.5 C 8.5 3.395 7.605 2.5 6.5 2.5 L 6.5 3.5 Z M 6 7 L 6 8.5 L 7 8.5 L 7 7 L 6 7 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-history": "<g transform=\"translate(3.5,3.5)\"><path d=\"M 4.5 4.5 L 4 4.5 L 4 4.884 L 4.371 4.983 L 4.5 4.5 Z M 8.5 4.5 C 8.5 6.709 6.709 8.5 4.5 8.5 L 4.5 9.5 C 7.261 9.5 9.5 7.261 9.5 4.5 L 8.5 4.5 Z M 4.5 0.5 C 6.709 0.5 8.5 2.291 8.5 4.5 L 9.5 4.5 C 9.5 1.739 7.261 -0.5 4.5 -0.5 L 4.5 0.5 Z M 0.5 4.5 C 0.5 2.291 2.291 0.5 4.5 0.5 L 4.5 -0.5 C 1.739 -0.5 -0.5 1.739 -0.5 4.5 L 0.5 4.5 Z M 4.5 8.5 C 4.141 8.5 3.794 8.453 3.464 8.365 L 3.206 9.331 C 3.619 9.441 4.053 9.5 4.5 9.5 L 4.5 8.5 Z M -0.5 4.5 L -0.5 7 L 0.5 7 L 0.5 4.5 L -0.5 4.5 Z M 5 4.5 L 5 1.5 L 4 1.5 L 4 4.5 L 5 4.5 Z M 7.528 4.794 L 4.629 4.017 L 4.371 4.983 L 7.269 5.76 L 7.528 4.794 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(1.5,8.5)\"><path d=\"M 2 2 L 1.646 2.354 L 2 2.707 L 2.354 2.354 L 2 2 Z M 3.646 -0.354 L 1.646 1.646 L 2.354 2.354 L 4.354 0.354 L 3.646 -0.354 Z M 2.354 1.646 L 0.354 -0.354 L -0.354 0.354 L 1.646 2.354 L 2.354 1.646 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-home": "<g transform=\"translate(1.5,2)\"><path d=\"M 6.5 0 L 6.854 -0.354 L 6.5 -0.707 L 6.146 -0.354 L 6.5 0 Z M 11 11.5 L 11 12 L 11.5 12 L 11.5 11.5 L 11 11.5 Z M 8 11.5 L 7.5 11.5 L 7.5 12 L 8 12 L 8 11.5 Z M 8 7.5 L 8.5 7.5 L 8.5 7 L 8 7 L 8 7.5 Z M 5 7.5 L 5 7 L 4.5 7 L 4.5 7.5 L 5 7.5 Z M 5 11.5 L 5 12 L 5.5 12 L 5.5 11.5 L 5 11.5 Z M 2 11.5 L 1.5 11.5 L 1.5 12 L 2 12 L 2 11.5 Z M 0.354 6.854 L 6.854 0.354 L 6.146 -0.354 L -0.354 6.146 L 0.354 6.854 Z M 6.146 0.354 L 12.646 6.854 L 13.354 6.146 L 6.854 -0.354 L 6.146 0.354 Z M 11.5 11.5 L 11.5 4.5 L 10.5 4.5 L 10.5 11.5 L 11.5 11.5 Z M 8 12 L 11 12 L 11 11 L 8 11 L 8 12 Z M 7.5 7.5 L 7.5 11.5 L 8.5 11.5 L 8.5 7.5 L 7.5 7.5 Z M 5 8 L 8 8 L 8 7 L 5 7 L 5 8 Z M 5.5 11.5 L 5.5 7.5 L 4.5 7.5 L 4.5 11.5 L 5.5 11.5 Z M 2 12 L 5 12 L 5 11 L 2 11 L 2 12 Z M 1.5 4.5 L 1.5 11.5 L 2.5 11.5 L 2.5 4.5 L 1.5 4.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-info-circle": "<g transform=\"translate(1.5,1.5)\"><path d=\"M 12.5 6.5 C 12.5 9.814 9.814 12.5 6.5 12.5 L 6.5 13.5 C 10.366 13.5 13.5 10.366 13.5 6.5 L 12.5 6.5 Z M 6.5 12.5 C 3.186 12.5 0.5 9.814 0.5 6.5 L -0.5 6.5 C -0.5 10.366 2.634 13.5 6.5 13.5 L 6.5 12.5 Z M 0.5 6.5 C 0.5 3.186 3.186 0.5 6.5 0.5 L 6.5 -0.5 C 2.634 -0.5 -0.5 2.634 -0.5 6.5 L 0.5 6.5 Z M 6.5 0.5 C 9.814 0.5 12.5 3.186 12.5 6.5 L 13.5 6.5 C 13.5 2.634 10.366 -0.5 6.5 -0.5 L 6.5 0.5 Z M 7 3.5 L 7 2.5 L 6 2.5 L 6 3.5 L 7 3.5 Z M 7 10.5 L 7 4.5 L 6 4.5 L 6 10.5 L 7 10.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-link": "<g transform=\"translate(2.5,6.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 4 0 L 4.5 0 L 4.5 -0.5 L 4 -0.5 L 4 0 Z M 4 4 L 4 4.5 L 4.5 4.5 L 4.5 4 L 4 4 Z M 0 4 L -0.5 4 L -0.5 4.5 L 0 4.5 L 0 4 Z M 7 0 L 7 -0.5 L 6.5 -0.5 L 6.5 0 L 7 0 Z M 11 0 L 11.5 0 L 11.5 -0.5 L 11 -0.5 L 11 0 Z M 11 4 L 11 4.5 L 11.5 4.5 L 11.5 4 L 11 4 Z M 7 4 L 6.5 4 L 6.5 4.5 L 7 4.5 L 7 4 Z M 0 0.5 L 4 0.5 L 4 -0.5 L 0 -0.5 L 0 0.5 Z M 3.5 0 L 3.5 4 L 4.5 4 L 4.5 0 L 3.5 0 Z M 4 3.5 L 0 3.5 L 0 4.5 L 4 4.5 L 4 3.5 Z M 0.5 4 L 0.5 0 L -0.5 0 L -0.5 4 L 0.5 4 Z M 7 0.5 L 11 0.5 L 11 -0.5 L 7 -0.5 L 7 0.5 Z M 10.5 0 L 10.5 4 L 11.5 4 L 11.5 0 L 10.5 0 Z M 11 3.5 L 7 3.5 L 7 4.5 L 11 4.5 L 11 3.5 Z M 7.5 4 L 7.5 0 L 6.5 0 L 6.5 4 L 7.5 4 Z M 2 2.5 L 9 2.5 L 9 1.5 L 2 1.5 L 2 2.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-link-break": "<g transform=\"translate(1.051,2.5)\"><path d=\"M 0 4.586 L -0.129 4.103 L -0.612 4.232 L -0.483 4.715 L 0 4.586 Z M 3.864 3.55 L 4.347 3.421 L 4.217 2.938 L 3.734 3.067 L 3.864 3.55 Z M 4.899 7.414 L 5.028 7.897 L 5.511 7.768 L 5.382 7.285 L 4.899 7.414 Z M 1.035 8.449 L 0.552 8.579 L 0.682 9.062 L 1.165 8.932 L 1.035 8.449 Z M 10.035 3.551 L 10.165 3.068 L 9.682 2.938 L 9.552 3.421 L 10.035 3.551 Z M 13.899 4.586 L 14.382 4.715 L 14.511 4.232 L 14.028 4.103 L 13.899 4.586 Z M 12.864 8.45 L 12.734 8.932 L 13.217 9.062 L 13.347 8.579 L 12.864 8.45 Z M 9 7.414 L 8.517 7.285 L 8.388 7.768 L 8.87 7.897 L 9 7.414 Z M 0.129 5.069 L 3.993 4.033 L 3.734 3.067 L -0.129 4.103 L 0.129 5.069 Z M 3.381 3.68 L 4.416 7.544 L 5.382 7.285 L 4.347 3.421 L 3.381 3.68 Z M 4.77 6.931 L 0.906 7.966 L 1.165 8.932 L 5.028 7.897 L 4.77 6.931 Z M 1.518 8.32 L 0.483 4.456 L -0.483 4.715 L 0.552 8.579 L 1.518 8.32 Z M 9.906 4.034 L 13.769 5.069 L 14.028 4.103 L 10.165 3.068 L 9.906 4.034 Z M 13.416 4.456 L 12.381 8.32 L 13.347 8.579 L 14.382 4.715 L 13.416 4.456 Z M 12.993 7.967 L 9.129 6.931 L 8.87 7.897 L 12.734 8.932 L 12.993 7.967 Z M 9.483 7.544 L 10.518 3.68 L 9.552 3.421 L 8.517 7.285 L 9.483 7.544 Z M 2.587 6.481 L 6.087 5.481 L 5.812 4.519 L 2.312 5.519 L 2.587 6.481 Z M 11.587 5.519 L 8.087 4.519 L 7.812 5.481 L 11.312 6.481 L 11.587 5.519 Z M 7.449 2.5 L 7.449 0 L 6.449 0 L 6.449 2.5 L 7.449 2.5 Z M 6.303 2.646 L 4.803 1.146 L 4.096 1.854 L 5.596 3.354 L 6.303 2.646 Z M 8.303 3.354 L 9.803 1.854 L 9.096 1.146 L 7.596 2.646 L 8.303 3.354 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-lock-closed": "<g transform=\"translate(4.5,2.5)\"><path d=\"M 0 5 L 0 4.5 L -0.5 4.5 L -0.5 5 L 0 5 Z M 7 5 L 7.5 5 L 7.5 4.5 L 7 4.5 L 7 5 Z M 7 11 L 7 11.5 L 7.5 11.5 L 7.5 11 L 7 11 Z M 0 11 L -0.5 11 L -0.5 11.5 L 0 11.5 L 0 11 Z M 0 5.5 L 7 5.5 L 7 4.5 L 0 4.5 L 0 5.5 Z M 6.5 5 L 6.5 11 L 7.5 11 L 7.5 5 L 6.5 5 Z M 7 10.5 L 0 10.5 L 0 11.5 L 7 11.5 L 7 10.5 Z M 0.5 11 L 0.5 5 L -0.5 5 L -0.5 11 L 0.5 11 Z M 5.5 2.5 L 5.5 5 L 6.5 5 L 6.5 2.5 L 5.5 2.5 Z M 1.5 5 L 1.5 2.5 L 0.5 2.5 L 0.5 5 L 1.5 5 Z M 3.5 0.5 C 4.605 0.5 5.5 1.395 5.5 2.5 L 6.5 2.5 C 6.5 0.843 5.157 -0.5 3.5 -0.5 L 3.5 0.5 Z M 3.5 -0.5 C 1.843 -0.5 0.5 0.843 0.5 2.5 L 1.5 2.5 C 1.5 1.395 2.395 0.5 3.5 0.5 L 3.5 -0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-lock-open": "<g transform=\"translate(2.5,2.5)\"><path d=\"M 0 5 L 0 4.5 L -0.5 4.5 L -0.5 5 L 0 5 Z M 7 5 L 7.5 5 L 7.5 4.5 L 7 4.5 L 7 5 Z M 7 11 L 7 11.5 L 7.5 11.5 L 7.5 11 L 7 11 Z M 0 11 L -0.5 11 L -0.5 11.5 L 0 11.5 L 0 11 Z M 10.95 2 L 11.45 2 L 11.45 1.95 L 11.44 1.901 L 10.95 2 Z M 0 5.5 L 7 5.5 L 7 4.5 L 0 4.5 L 0 5.5 Z M 6.5 5 L 6.5 11 L 7.5 11 L 7.5 5 L 6.5 5 Z M 7 10.5 L 0 10.5 L 0 11.5 L 7 11.5 L 7 10.5 Z M 0.5 11 L 0.5 5 L -0.5 5 L -0.5 11 L 0.5 11 Z M 6.5 5 L 6.5 2.5 L 5.5 2.5 L 5.5 5 L 6.5 5 Z M 8.5 -0.5 C 6.843 -0.5 5.5 0.843 5.5 2.5 L 6.5 2.5 C 6.5 1.395 7.395 0.5 8.5 0.5 L 8.5 -0.5 Z M 8.5 0.5 C 9.467 0.5 10.275 1.187 10.46 2.099 L 11.44 1.901 C 11.162 0.531 9.952 -0.5 8.5 -0.5 L 8.5 0.5 Z M 10.45 2 L 10.45 4.5 L 11.45 4.5 L 11.45 2 L 10.45 2 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-magnifying-glass": "<g transform=\"translate(2.5,2.5)\"><path d=\"M 7.5 4 C 7.5 5.933 5.933 7.5 4 7.5 L 4 8.5 C 6.485 8.5 8.5 6.485 8.5 4 L 7.5 4 Z M 4 7.5 C 2.067 7.5 0.5 5.933 0.5 4 L -0.5 4 C -0.5 6.485 1.515 8.5 4 8.5 L 4 7.5 Z M 0.5 4 C 0.5 2.067 2.067 0.5 4 0.5 L 4 -0.5 C 1.515 -0.5 -0.5 1.515 -0.5 4 L 0.5 4 Z M 4 0.5 C 5.933 0.5 7.5 2.067 7.5 4 L 8.5 4 C 8.5 1.515 6.485 -0.5 4 -0.5 L 4 0.5 Z M 11.854 11.147 L 7.182 6.475 L 6.475 7.182 L 11.147 11.854 L 11.854 11.147 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-magnifying-glass-minus": "<g transform=\"translate(2.5,2.5)\"><path d=\"M 7.5 4 C 7.5 5.933 5.933 7.5 4 7.5 L 4 8.5 C 6.485 8.5 8.5 6.485 8.5 4 L 7.5 4 Z M 4 7.5 C 2.067 7.5 0.5 5.933 0.5 4 L -0.5 4 C -0.5 6.485 1.515 8.5 4 8.5 L 4 7.5 Z M 0.5 4 C 0.5 2.067 2.067 0.5 4 0.5 L 4 -0.5 C 1.515 -0.5 -0.5 1.515 -0.5 4 L 0.5 4 Z M 4 0.5 C 5.933 0.5 7.5 2.067 7.5 4 L 8.5 4 C 8.5 1.515 6.485 -0.5 4 -0.5 L 4 0.5 Z M 11.854 11.146 L 7.182 6.475 L 6.475 7.182 L 11.147 11.854 L 11.854 11.146 Z M 1.5 4.5 L 6.5 4.5 L 6.5 3.5 L 1.5 3.5 L 1.5 4.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-magnifying-glass-plus": "<g transform=\"translate(2.5,2.5)\"><path d=\"M 7.5 4 C 7.5 5.933 5.933 7.5 4 7.5 L 4 8.5 C 6.485 8.5 8.5 6.485 8.5 4 L 7.5 4 Z M 4 7.5 C 2.067 7.5 0.5 5.933 0.5 4 L -0.5 4 C -0.5 6.485 1.515 8.5 4 8.5 L 4 7.5 Z M 0.5 4 C 0.5 2.067 2.067 0.5 4 0.5 L 4 -0.5 C 1.515 -0.5 -0.5 1.515 -0.5 4 L 0.5 4 Z M 4 0.5 C 5.933 0.5 7.5 2.067 7.5 4 L 8.5 4 C 8.5 1.515 6.485 -0.5 4 -0.5 L 4 0.5 Z M 11.854 11.146 L 7.182 6.475 L 6.475 7.182 L 11.147 11.854 L 11.854 11.146 Z M 1.5 4.5 L 6.5 4.5 L 6.5 3.5 L 1.5 3.5 L 1.5 4.5 Z M 3.5 1.5 L 3.5 6.5 L 4.5 6.5 L 4.5 1.5 L 3.5 1.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-menu": "<g transform=\"translate(3,3.5)\"><path d=\"M 0 0.5 L 10 0.5 L 10 -0.5 L 0 -0.5 L 0 0.5 Z M 0 4.5 L 10 4.5 L 10 3.5 L 0 3.5 L 0 4.5 Z M 0 8.5 L 10 8.5 L 10 7.5 L 0 7.5 L 0 8.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-more-horiz": "<g transform=\"translate(1.5,6.5)\"><path d=\"M 2.5 1.5 C 2.5 2.052 2.052 2.5 1.5 2.5 L 1.5 3.5 C 2.605 3.5 3.5 2.605 3.5 1.5 L 2.5 1.5 Z M 1.5 2.5 C 0.948 2.5 0.5 2.052 0.5 1.5 L -0.5 1.5 C -0.5 2.605 0.395 3.5 1.5 3.5 L 1.5 2.5 Z M 0.5 1.5 C 0.5 0.948 0.948 0.5 1.5 0.5 L 1.5 -0.5 C 0.395 -0.5 -0.5 0.395 -0.5 1.5 L 0.5 1.5 Z M 1.5 0.5 C 2.052 0.5 2.5 0.948 2.5 1.5 L 3.5 1.5 C 3.5 0.395 2.605 -0.5 1.5 -0.5 L 1.5 0.5 Z M 7.5 1.5 C 7.5 2.052 7.052 2.5 6.5 2.5 L 6.5 3.5 C 7.605 3.5 8.5 2.605 8.5 1.5 L 7.5 1.5 Z M 6.5 2.5 C 5.948 2.5 5.5 2.052 5.5 1.5 L 4.5 1.5 C 4.5 2.605 5.395 3.5 6.5 3.5 L 6.5 2.5 Z M 5.5 1.5 C 5.5 0.948 5.948 0.5 6.5 0.5 L 6.5 -0.5 C 5.395 -0.5 4.5 0.395 4.5 1.5 L 5.5 1.5 Z M 6.5 0.5 C 7.052 0.5 7.5 0.948 7.5 1.5 L 8.5 1.5 C 8.5 0.395 7.605 -0.5 6.5 -0.5 L 6.5 0.5 Z M 12.5 1.5 C 12.5 2.052 12.052 2.5 11.5 2.5 L 11.5 3.5 C 12.605 3.5 13.5 2.605 13.5 1.5 L 12.5 1.5 Z M 11.5 2.5 C 10.948 2.5 10.5 2.052 10.5 1.5 L 9.5 1.5 C 9.5 2.605 10.395 3.5 11.5 3.5 L 11.5 2.5 Z M 10.5 1.5 C 10.5 0.948 10.948 0.5 11.5 0.5 L 11.5 -0.5 C 10.395 -0.5 9.5 0.395 9.5 1.5 L 10.5 1.5 Z M 11.5 0.5 C 12.052 0.5 12.5 0.948 12.5 1.5 L 13.5 1.5 C 13.5 0.395 12.605 -0.5 11.5 -0.5 L 11.5 0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-more-vert": "<g transform=\"translate(6.5,1.5)\"><path d=\"M 2.5 1.5 C 2.5 2.052 2.052 2.5 1.5 2.5 L 1.5 3.5 C 2.605 3.5 3.5 2.605 3.5 1.5 L 2.5 1.5 Z M 1.5 2.5 C 0.948 2.5 0.5 2.052 0.5 1.5 L -0.5 1.5 C -0.5 2.605 0.395 3.5 1.5 3.5 L 1.5 2.5 Z M 0.5 1.5 C 0.5 0.948 0.948 0.5 1.5 0.5 L 1.5 -0.5 C 0.395 -0.5 -0.5 0.395 -0.5 1.5 L 0.5 1.5 Z M 1.5 0.5 C 2.052 0.5 2.5 0.948 2.5 1.5 L 3.5 1.5 C 3.5 0.395 2.605 -0.5 1.5 -0.5 L 1.5 0.5 Z M 2.5 6.5 C 2.5 7.052 2.052 7.5 1.5 7.5 L 1.5 8.5 C 2.605 8.5 3.5 7.605 3.5 6.5 L 2.5 6.5 Z M 1.5 7.5 C 0.948 7.5 0.5 7.052 0.5 6.5 L -0.5 6.5 C -0.5 7.605 0.395 8.5 1.5 8.5 L 1.5 7.5 Z M 0.5 6.5 C 0.5 5.948 0.948 5.5 1.5 5.5 L 1.5 4.5 C 0.395 4.5 -0.5 5.395 -0.5 6.5 L 0.5 6.5 Z M 1.5 5.5 C 2.052 5.5 2.5 5.948 2.5 6.5 L 3.5 6.5 C 3.5 5.395 2.605 4.5 1.5 4.5 L 1.5 5.5 Z M 2.5 11.5 C 2.5 12.052 2.052 12.5 1.5 12.5 L 1.5 13.5 C 2.605 13.5 3.5 12.605 3.5 11.5 L 2.5 11.5 Z M 1.5 12.5 C 0.948 12.5 0.5 12.052 0.5 11.5 L -0.5 11.5 C -0.5 12.605 0.395 13.5 1.5 13.5 L 1.5 12.5 Z M 0.5 11.5 C 0.5 10.948 0.948 10.5 1.5 10.5 L 1.5 9.5 C 0.395 9.5 -0.5 10.395 -0.5 11.5 L 0.5 11.5 Z M 1.5 10.5 C 2.052 10.5 2.5 10.948 2.5 11.5 L 3.5 11.5 C 3.5 10.395 2.605 9.5 1.5 9.5 L 1.5 10.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-paperplane": "<g transform=\"translate(1.5,2.5)\"><path d=\"M 0 0 L 0.201 -0.458 L -0.753 -0.877 L -0.483 0.129 L 0 0 Z M 0 11 L -0.483 10.871 L -0.753 11.877 L 0.201 11.458 L 0 11 Z M 12.5 5.5 L 12.701 5.958 L 13.742 5.5 L 12.701 5.042 L 12.5 5.5 Z M 0.201 11.458 L 12.701 5.958 L 12.299 5.042 L -0.201 10.542 L 0.201 11.458 Z M 12.701 5.042 L 0.201 -0.458 L -0.201 0.458 L 12.299 5.958 L 12.701 5.042 Z M 0.991 5.371 L -0.483 10.871 L 0.483 11.129 L 1.957 5.629 L 0.991 5.371 Z M -0.483 0.129 L 0.991 5.629 L 1.957 5.371 L 0.483 -0.129 L -0.483 0.129 Z M 1.474 6 L 6.5 6 L 6.5 5 L 1.474 5 L 1.474 6 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-redo": "<g transform=\"translate(3.504,3.587)\"><path d=\"M 1.318 1.231 L 1.672 1.584 L 1.672 1.584 L 1.318 1.231 Z M 7.496 1.413 L 7.85 1.059 L 7.85 1.059 L 7.496 1.413 Z M 1.672 7.241 C 0.109 5.679 0.109 3.146 1.672 1.584 L 0.964 0.877 C -0.988 2.83 -0.988 5.995 0.964 7.948 L 1.672 7.241 Z M 1.672 1.584 C 3.207 0.049 5.553 0.176 7.143 1.766 L 7.85 1.059 C 5.925 -0.866 2.944 -1.102 0.964 0.877 L 1.672 1.584 Z M 7.143 1.766 L 9.643 4.266 L 10.35 3.559 L 7.85 1.059 L 7.143 1.766 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(10,4)\"><path d=\"M 3.498 3.498 L 3.499 3.998 L 3.998 3.998 L 3.998 3.499 L 3.498 3.498 Z M 0 4 L 3.499 3.998 L 3.498 2.998 L 0 3 L 0 4 Z M 3.998 3.499 L 4 0 L 3 0 L 2.998 3.498 L 3.998 3.499 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-refresh": "<g transform=\"translate(3.5,3.5)\"><path d=\"M -0.5 4.5 C -0.5 7.261 1.739 9.5 4.5 9.5 L 4.5 8.5 C 2.291 8.5 0.5 6.709 0.5 4.5 L -0.5 4.5 Z M 4.5 -0.5 C 1.739 -0.5 -0.5 1.739 -0.5 4.5 L 0.5 4.5 C 0.5 2.291 2.291 0.5 4.5 0.5 L 4.5 -0.5 Z M 9.5 4.5 C 9.5 1.739 7.261 -0.5 4.5 -0.5 L 4.5 0.5 C 6.709 0.5 8.5 2.291 8.5 4.5 L 9.5 4.5 Z M 4.5 9.5 C 4.947 9.5 5.381 9.441 5.794 9.331 L 5.536 8.365 C 5.206 8.453 4.859 8.5 4.5 8.5 L 4.5 9.5 Z M 8.5 4.5 L 8.5 7 L 9.5 7 L 9.5 4.5 L 8.5 4.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(10,8)\"><path d=\"M 2.5 2.5 L 2.146 2.854 L 2.5 3.207 L 2.854 2.854 L 2.5 2.5 Z M -0.354 0.354 L 2.146 2.854 L 2.854 2.146 L 0.354 -0.354 L -0.354 0.354 Z M 2.854 2.854 L 5.354 0.354 L 4.646 -0.354 L 2.146 2.146 L 2.854 2.854 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-reset": "<g transform=\"translate(3.5,3.5)\"><path d=\"M 8.5 4.5 C 8.5 6.709 6.709 8.5 4.5 8.5 L 4.5 9.5 C 7.261 9.5 9.5 7.261 9.5 4.5 L 8.5 4.5 Z M 4.5 0.5 C 6.709 0.5 8.5 2.291 8.5 4.5 L 9.5 4.5 C 9.5 1.739 7.261 -0.5 4.5 -0.5 L 4.5 0.5 Z M 0.5 4.5 C 0.5 2.291 2.291 0.5 4.5 0.5 L 4.5 -0.5 C 1.739 -0.5 -0.5 1.739 -0.5 4.5 L 0.5 4.5 Z M 4.5 8.5 C 4.141 8.5 3.794 8.453 3.464 8.365 L 3.206 9.331 C 3.619 9.441 4.053 9.5 4.5 9.5 L 4.5 8.5 Z M -0.5 4.5 L -0.5 7 L 0.5 7 L 0.5 4.5 L -0.5 4.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(1,8)\"><path d=\"M 2.5 2.5 L 2.146 2.854 L 2.5 3.207 L 2.854 2.854 L 2.5 2.5 Z M 4.646 -0.354 L 2.146 2.146 L 2.854 2.854 L 5.354 0.354 L 4.646 -0.354 Z M 2.854 2.146 L 0.354 -0.354 L -0.354 0.354 L 2.146 2.854 L 2.854 2.146 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-retry": "<g transform=\"translate(3.5,3.5)\"><path d=\"M 4.5 9.5 C 7.261 9.5 9.5 7.261 9.5 4.5 L 8.5 4.5 C 8.5 6.709 6.709 8.5 4.5 8.5 L 4.5 9.5 Z M -0.5 4.5 C -0.5 7.261 1.739 9.5 4.5 9.5 L 4.5 8.5 C 2.291 8.5 0.5 6.709 0.5 4.5 L -0.5 4.5 Z M 4.5 -0.5 C 1.739 -0.5 -0.5 1.739 -0.5 4.5 L 0.5 4.5 C 0.5 2.291 2.291 0.5 4.5 0.5 L 4.5 -0.5 Z M 4.5 0.5 L 7 0.5 L 7 -0.5 L 4.5 -0.5 L 4.5 0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(8,1)\"><path d=\"M 2.5 2.5 L 2.854 2.854 L 3.207 2.5 L 2.854 2.146 L 2.5 2.5 Z M 0.354 5.354 L 2.854 2.854 L 2.146 2.146 L -0.354 4.646 L 0.354 5.354 Z M 2.854 2.146 L 0.354 -0.354 L -0.354 0.354 L 2.146 2.854 L 2.854 2.146 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-share": "<g transform=\"translate(1.5,1.5)\"><path d=\"M 11.5 2 C 11.5 2.828 10.828 3.5 10 3.5 L 10 4.5 C 11.381 4.5 12.5 3.381 12.5 2 L 11.5 2 Z M 10 3.5 C 9.172 3.5 8.5 2.828 8.5 2 L 7.5 2 C 7.5 3.381 8.619 4.5 10 4.5 L 10 3.5 Z M 8.5 2 C 8.5 1.172 9.172 0.5 10 0.5 L 10 -0.5 C 8.619 -0.5 7.5 0.619 7.5 2 L 8.5 2 Z M 10 0.5 C 10.828 0.5 11.5 1.172 11.5 2 L 12.5 2 C 12.5 0.619 11.381 -0.5 10 -0.5 L 10 0.5 Z M 11.5 11 C 11.5 11.828 10.828 12.5 10 12.5 L 10 13.5 C 11.381 13.5 12.5 12.381 12.5 11 L 11.5 11 Z M 10 12.5 C 9.172 12.5 8.5 11.828 8.5 11 L 7.5 11 C 7.5 12.381 8.619 13.5 10 13.5 L 10 12.5 Z M 8.5 11 C 8.5 10.172 9.172 9.5 10 9.5 L 10 8.5 C 8.619 8.5 7.5 9.619 7.5 11 L 8.5 11 Z M 10 9.5 C 10.828 9.5 11.5 10.172 11.5 11 L 12.5 11 C 12.5 9.619 11.381 8.5 10 8.5 L 10 9.5 Z M 3.5 7 C 3.5 7.828 2.828 8.5 2 8.5 L 2 9.5 C 3.381 9.5 4.5 8.381 4.5 7 L 3.5 7 Z M 2 8.5 C 1.172 8.5 0.5 7.828 0.5 7 L -0.5 7 C -0.5 8.381 0.619 9.5 2 9.5 L 2 8.5 Z M 0.5 7 C 0.5 6.172 1.172 5.5 2 5.5 L 2 4.5 C 0.619 4.5 -0.5 5.619 -0.5 7 L 0.5 7 Z M 2 5.5 C 2.828 5.5 3.5 6.172 3.5 7 L 4.5 7 C 4.5 5.619 3.381 4.5 2 4.5 L 2 5.5 Z M 8.039 2.636 L 5.127 4.456 L 5.657 5.304 L 8.569 3.484 L 8.039 2.636 Z M 10.5 9 L 10.5 6 L 9.5 6 L 9.5 9 L 10.5 9 Z M 3.566 8.342 L 6.198 9.658 L 6.645 8.764 L 4.013 7.447 L 3.566 8.342 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-sort": "<g transform=\"translate(3,5.5)\"><path d=\"M 0 0.5 L 10 0.5 L 10 -0.5 L 0 -0.5 L 0 0.5 Z M 0 3.5 L 6 3.5 L 6 2.5 L 0 2.5 L 0 3.5 Z M 0 6.5 L 3 6.5 L 3 5.5 L 0 5.5 L 0 6.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-star": "<g transform=\"translate(2,2.5)\"><path d=\"M 6 0 L 6.448 -0.221 L 6 -1.13 L 5.552 -0.221 L 6 0 Z M 7.854 3.757 L 7.406 3.978 L 7.522 4.214 L 7.782 4.252 L 7.854 3.757 Z M 12 4.359 L 12.349 4.717 L 13.074 4.01 L 12.072 3.864 L 12 4.359 Z M 9 7.284 L 8.651 6.925 L 8.463 7.109 L 8.507 7.368 L 9 7.284 Z M 9.708 11.413 L 9.476 11.855 L 10.372 12.327 L 10.201 11.328 L 9.708 11.413 Z M 6 9.463 L 6.233 9.021 L 6 8.898 L 5.767 9.021 L 6 9.463 Z M 2.292 11.413 L 1.799 11.328 L 1.628 12.327 L 2.524 11.855 L 2.292 11.413 Z M 3 7.284 L 3.493 7.368 L 3.537 7.109 L 3.349 6.925 L 3 7.284 Z M 0 4.359 L -0.072 3.864 L -1.074 4.01 L -0.349 4.717 L 0 4.359 Z M 4.146 3.757 L 4.218 4.252 L 4.478 4.214 L 4.594 3.978 L 4.146 3.757 Z M 5.552 0.221 L 7.406 3.978 L 8.302 3.536 L 6.448 -0.221 L 5.552 0.221 Z M 7.782 4.252 L 11.928 4.854 L 12.072 3.864 L 7.926 3.262 L 7.782 4.252 Z M 11.651 4.001 L 8.651 6.925 L 9.349 7.642 L 12.349 4.717 L 11.651 4.001 Z M 8.507 7.368 L 9.215 11.497 L 10.201 11.328 L 9.493 7.199 L 8.507 7.368 Z M 9.941 10.97 L 6.233 9.021 L 5.767 9.906 L 9.476 11.855 L 9.941 10.97 Z M 5.767 9.021 L 2.059 10.97 L 2.524 11.855 L 6.233 9.906 L 5.767 9.021 Z M 2.785 11.497 L 3.493 7.368 L 2.507 7.199 L 1.799 11.328 L 2.785 11.497 Z M 3.349 6.925 L 0.349 4.001 L -0.349 4.717 L 2.651 7.642 L 3.349 6.925 Z M 0.072 4.854 L 4.218 4.252 L 4.074 3.262 L -0.072 3.864 L 0.072 4.854 Z M 4.594 3.978 L 6.448 0.221 L 5.552 -0.221 L 3.698 3.536 L 4.594 3.978 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-subtract": "<g transform=\"translate(3.5,8)\"><path d=\"M 9 -0.5 L 0 -0.5 L 0 0.5 L 9 0.5 L 9 -0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-subtract-circle": "<g transform=\"translate(1.5,1.5)\"><path d=\"M 12.5 6.5 C 12.5 9.814 9.814 12.5 6.5 12.5 L 6.5 13.5 C 10.366 13.5 13.5 10.366 13.5 6.5 L 12.5 6.5 Z M 6.5 12.5 C 3.186 12.5 0.5 9.814 0.5 6.5 L -0.5 6.5 C -0.5 10.366 2.634 13.5 6.5 13.5 L 6.5 12.5 Z M 0.5 6.5 C 0.5 3.186 3.186 0.5 6.5 0.5 L 6.5 -0.5 C 2.634 -0.5 -0.5 2.634 -0.5 6.5 L 0.5 6.5 Z M 6.5 0.5 C 9.814 0.5 12.5 3.186 12.5 6.5 L 13.5 6.5 C 13.5 2.634 10.366 -0.5 6.5 -0.5 L 6.5 0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(4,8)\"><path d=\"M 8 -0.5 L 0 -0.5 L 0 0.5 L 8 0.5 L 8 -0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-sync": "<g transform=\"translate(3.504,3.505)\"><path d=\"M 4.5 8.991 L 4.5 9.491 L 4.5 9.491 L 4.5 8.991 Z M -0.5 4.5 C -0.5 4.947 -0.441 5.381 -0.331 5.794 L 0.635 5.536 C 0.547 5.206 0.5 4.859 0.5 4.5 L -0.5 4.5 Z M 4.5 -0.5 C 1.739 -0.5 -0.5 1.739 -0.5 4.5 L 0.5 4.5 C 0.5 2.291 2.291 0.5 4.5 0.5 L 4.5 -0.5 Z M 7 -0.5 L 4.5 -0.5 L 4.5 0.5 L 7 0.5 L 7 -0.5 Z M 9.5 4.491 C 9.5 4.044 9.441 3.61 9.33 3.197 L 8.364 3.455 C 8.453 3.785 8.5 4.132 8.5 4.491 L 9.5 4.491 Z M 4.5 9.491 C 7.261 9.491 9.5 7.252 9.5 4.491 L 8.5 4.491 C 8.5 6.7 6.709 8.491 4.5 8.491 L 4.5 9.491 Z M 2 9.491 L 4.5 9.491 L 4.5 8.491 L 2 8.491 L 2 9.491 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(5.503,10)\"><path d=\"M 0 2.495 L -0.353 2.142 L -0.707 2.495 L -0.354 2.848 L 0 2.495 Z M 2.143 -0.354 L -0.353 2.142 L 0.353 2.849 L 2.85 0.354 L 2.143 -0.354 Z M -0.354 2.848 L 2.142 5.353 L 2.851 4.647 L 0.354 2.142 L -0.354 2.848 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(8,1)\"><path d=\"M 2.504 2.505 L 2.857 2.859 L 3.211 2.505 L 2.857 2.151 L 2.504 2.505 Z M 0.353 5.354 L 2.857 2.859 L 2.151 2.15 L -0.353 4.646 L 0.353 5.354 Z M 2.857 2.151 L 0.354 -0.353 L -0.354 0.353 L 2.15 2.858 L 2.857 2.151 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-timer": "<g transform=\"translate(3.5,2.5)\"><path d=\"M 8.5 6.5 C 8.5 8.709 6.709 10.5 4.5 10.5 L 4.5 11.5 C 7.261 11.5 9.5 9.261 9.5 6.5 L 8.5 6.5 Z M 4.5 10.5 C 2.291 10.5 0.5 8.709 0.5 6.5 L -0.5 6.5 C -0.5 9.261 1.739 11.5 4.5 11.5 L 4.5 10.5 Z M 0.5 6.5 C 0.5 4.291 2.291 2.5 4.5 2.5 L 4.5 1.5 C 1.739 1.5 -0.5 3.739 -0.5 6.5 L 0.5 6.5 Z M 4.5 2.5 C 6.709 2.5 8.5 4.291 8.5 6.5 L 9.5 6.5 C 9.5 3.739 7.261 1.5 4.5 1.5 L 4.5 2.5 Z M 2.5 0.5 L 6.5 0.5 L 6.5 -0.5 L 2.5 -0.5 L 2.5 0.5 Z M 4.5 6.5 L 4.5 7.5 C 5.052 7.5 5.5 7.052 5.5 6.5 L 4.5 6.5 Z M 4.5 6.5 L 3.5 6.5 C 3.5 7.052 3.948 7.5 4.5 7.5 L 4.5 6.5 Z M 4.5 6.5 L 4.5 5.5 C 3.948 5.5 3.5 5.948 3.5 6.5 L 4.5 6.5 Z M 4.5 6.5 L 5.5 6.5 C 5.5 5.948 5.052 5.5 4.5 5.5 L 4.5 6.5 Z M 4.983 6.371 L 4.206 3.472 L 3.24 3.731 L 4.017 6.629 L 4.983 6.371 Z M 3.5 0 L 3.5 2.027 L 4.5 2.027 L 4.5 0 L 3.5 0 Z M 5.5 2.027 L 5.5 0 L 4.5 0 L 4.5 2.027 L 5.5 2.027 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-trash": "<g transform=\"translate(3,1.5)\"><path d=\"M 8.5 12 L 8.5 12.5 L 9 12.5 L 9 12 L 8.5 12 Z M 1.5 12 L 1 12 L 1 12.5 L 1.5 12.5 L 1.5 12 Z M 3.5 0 L 3.5 -0.5 L 3 -0.5 L 3 0 L 3.5 0 Z M 6.5 0 L 7 0 L 7 -0.5 L 6.5 -0.5 L 6.5 0 Z M 8 3.5 L 8 12 L 9 12 L 9 3.5 L 8 3.5 Z M 8.5 11.5 L 1.5 11.5 L 1.5 12.5 L 8.5 12.5 L 8.5 11.5 Z M 2 12 L 2 3.5 L 1 3.5 L 1 12 L 2 12 Z M 0 2.5 L 10 2.5 L 10 1.5 L 0 1.5 L 0 2.5 Z M 4 2 L 4 0 L 3 0 L 3 2 L 4 2 Z M 3.5 0.5 L 6.5 0.5 L 6.5 -0.5 L 3.5 -0.5 L 3.5 0.5 Z M 6 0 L 6 2 L 7 2 L 7 0 L 6 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-undo": "<g transform=\"translate(2.5,3.587)\"><path d=\"M 8.678 1.231 L 8.325 1.584 L 8.325 1.584 L 8.678 1.231 Z M 2.5 1.413 L 2.147 1.059 L 2.5 1.413 Z M 9.032 7.948 C 10.984 5.995 10.984 2.83 9.032 0.877 L 8.325 1.584 C 9.887 3.146 9.887 5.679 8.325 7.241 L 9.032 7.948 Z M 9.032 0.877 C 7.052 -1.102 4.071 -0.866 2.147 1.059 L 2.854 1.766 C 4.444 0.176 6.789 0.049 8.325 1.584 L 9.032 0.877 Z M 2.147 1.059 L -0.353 3.559 L 0.354 4.266 L 2.854 1.766 L 2.147 1.059 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(2.5,4)\"><path d=\"M 0.002 3.498 L -0.498 3.499 L -0.498 3.998 L 0.001 3.998 L 0.002 3.498 Z M 3.5 3 L 0.002 2.998 L 0.001 3.998 L 3.5 4 L 3.5 3 Z M 0.502 3.498 L 0.5 0 L -0.5 0 L -0.498 3.499 L 0.502 3.498 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "common-warning": "<g transform=\"translate(1.5,2)\"><path d=\"M 6.5 0 L 6.935 -0.246 L 6.5 -1.016 L 6.065 -0.246 L 6.5 0 Z M 13 11.498 L 13 11.998 L 13.857 11.998 L 13.435 11.252 L 13 11.498 Z M 0 11.498 L -0.435 11.252 L -0.857 11.998 L 0 11.998 L 0 11.498 Z M 6.065 0.246 L 12.565 11.744 L 13.435 11.252 L 6.935 -0.246 L 6.065 0.246 Z M 13 10.998 L 0 10.998 L 0 11.998 L 13 11.998 L 13 10.998 Z M 0.435 11.744 L 6.935 0.246 L 6.065 -0.246 L -0.435 11.252 L 0.435 11.744 Z M 6 4.5 L 6 8 L 7 8 L 7 4.5 L 6 4.5 Z M 6 9 L 6 10 L 7 10 L 7 9 L 6 9 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "communication-cloud": "<g transform=\"translate(1.5,2.5)\"><path d=\"M 2.5 4.041 L 2.583 4.535 L 3.005 4.464 L 3 4.036 L 2.5 4.041 Z M 10.262 2.638 L 9.792 2.808 L 9.88 3.05 L 10.128 3.12 L 10.262 2.638 Z M 3 4.036 C 3 4.024 3 4.012 3 4 L 2 4 C 2 4.016 2 4.031 2 4.047 L 3 4.036 Z M 0.5 7 C 0.5 5.762 1.401 4.733 2.583 4.535 L 2.418 3.548 C 0.762 3.826 -0.5 5.265 -0.5 7 L 0.5 7 Z M 3 9.5 C 1.619 9.5 0.5 8.381 0.5 7 L -0.5 7 C -0.5 8.933 1.067 10.5 3 10.5 L 3 9.5 Z M 9.25 9.5 L 3 9.5 L 3 10.5 L 9.25 10.5 L 9.25 9.5 Z M 12.5 6.25 C 12.5 8.045 11.045 9.5 9.25 9.5 L 9.25 10.5 C 11.597 10.5 13.5 8.597 13.5 6.25 L 12.5 6.25 Z M 10.128 3.12 C 11.497 3.503 12.5 4.76 12.5 6.25 L 13.5 6.25 C 13.5 4.3 12.187 2.657 10.397 2.157 L 10.128 3.12 Z M 6.5 0.5 C 8.014 0.5 9.304 1.461 9.792 2.808 L 10.732 2.468 C 10.106 0.738 8.448 -0.5 6.5 -0.5 L 6.5 0.5 Z M 3 4 C 3 2.067 4.567 0.5 6.5 0.5 L 6.5 -0.5 C 4.015 -0.5 2 1.515 2 4 L 3 4 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "communication-data": "<g transform=\"translate(2.5,2)\"><path d=\"M 2 1 C 2.828 1 3.5 1.672 3.5 2.5 L 4.5 2.5 C 4.5 1.119 3.381 0 2 0 L 2 1 Z M 3.5 2.5 C 3.5 3.328 2.828 4 2 4 L 2 5 C 3.381 5 4.5 3.881 4.5 2.5 L 3.5 2.5 Z M 2 4 C 1.172 4 0.5 3.328 0.5 2.5 L -0.5 2.5 C -0.5 3.881 0.619 5 2 5 L 2 4 Z M 0.5 2.5 C 0.5 1.672 1.172 1 2 1 L 2 0 C 0.619 0 -0.5 1.119 -0.5 2.5 L 0.5 2.5 Z M 9 8 C 9.828 8 10.5 8.672 10.5 9.5 L 11.5 9.5 C 11.5 8.119 10.381 7 9 7 L 9 8 Z M 10.5 9.5 C 10.5 10.328 9.828 11 9 11 L 9 12 C 10.381 12 11.5 10.881 11.5 9.5 L 10.5 9.5 Z M 9 11 C 8.172 11 7.5 10.328 7.5 9.5 L 6.5 9.5 C 6.5 10.881 7.619 12 9 12 L 9 11 Z M 7.5 9.5 C 7.5 8.672 8.172 8 9 8 L 9 7 C 7.619 7 6.5 8.119 6.5 9.5 L 7.5 9.5 Z M 2.5 12 L 2.5 7 L 1.5 7 L 1.5 12 L 2.5 12 Z M 9.5 5 L 9.5 0 L 8.5 0 L 8.5 5 L 9.5 5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "communication-download": "<g transform=\"translate(2.33,2)\"><path d=\"M 0.67 11.5 L 0.187 11.629 L 0.286 12 L 0.67 12 L 0.67 11.5 Z M 10.67 11.5 L 10.67 12 L 11.054 12 L 11.153 11.629 L 10.67 11.5 Z M 6.17 9 L 6.17 0 L 5.17 0 L 5.17 9 L 6.17 9 Z M 1.153 11.371 L 0.483 8.871 L -0.483 9.129 L 0.187 11.629 L 1.153 11.371 Z M 10.67 11 L 0.67 11 L 0.67 12 L 10.67 12 L 10.67 11 Z M 10.857 8.871 L 10.187 11.371 L 11.153 11.629 L 11.823 9.129 L 10.857 8.871 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(4,7)\"><path d=\"M 4 4 L 3.646 4.354 L 4 4.707 L 4.354 4.354 L 4 4 Z M 7.646 -0.354 L 3.646 3.646 L 4.354 4.354 L 8.354 0.354 L 7.646 -0.354 Z M 4.354 3.646 L 0.354 -0.354 L -0.354 0.354 L 3.646 4.354 L 4.354 3.646 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "communication-envelope": "<g transform=\"translate(2.5,3.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 0 9 L -0.5 9 L -0.5 9.5 L 0 9.5 L 0 9 Z M 11 9 L 11 9.5 L 11.5 9.5 L 11.5 9 L 11 9 Z M 11 0 L 11.5 0 L 11.5 -0.5 L 11 -0.5 L 11 0 Z M 5.5 5.5 L 5.232 5.922 L 5.5 6.093 L 5.768 5.922 L 5.5 5.5 Z M -0.5 0 L -0.5 9 L 0.5 9 L 0.5 0 L -0.5 0 Z M 0 9.5 L 11 9.5 L 11 8.5 L 0 8.5 L 0 9.5 Z M 11.5 9 L 11.5 0 L 10.5 0 L 10.5 9 L 11.5 9 Z M 11 -0.5 L 0 -0.5 L 0 0.5 L 11 0.5 L 11 -0.5 Z M -0.268 2.422 L 5.232 5.922 L 5.768 5.078 L 0.268 1.578 L -0.268 2.422 Z M 5.768 5.922 L 11.268 2.422 L 10.732 1.578 L 5.232 5.078 L 5.768 5.922 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "communication-firewall": "<g transform=\"translate(2,2.5)\"><path d=\"M 4.5 12 L 4 12 L 4 12.5 L 4.5 12.5 L 4.5 12 Z M 4.5 10 L 4.5 9.5 L 4 9.5 L 4 10 L 4.5 10 Z M 7.5 12 L 7.5 12.5 L 8 12.5 L 8 12 L 7.5 12 Z M 7.5 10 L 8 10 L 8 9.5 L 7.5 9.5 L 7.5 10 Z M 11.5 7 L 11.5 7.5 L 12 7.5 L 12 7 L 11.5 7 Z M 0.5 7 L 0 7 L 0 7.5 L 0.5 7.5 L 0.5 7 Z M 11.5 0 L 12 0 L 12 -0.5 L 11.5 -0.5 L 11.5 0 Z M 9.5 0 L 9.5 -0.5 L 9 -0.5 L 9 0 L 9.5 0 Z M 9.5 1 L 9.5 1.5 L 10 1.5 L 10 1 L 9.5 1 Z M 7.5 1 L 7 1 L 7 1.5 L 7.5 1.5 L 7.5 1 Z M 7.5 0 L 8 0 L 8 -0.5 L 7.5 -0.5 L 7.5 0 Z M 4.5 0 L 4.5 -0.5 L 4 -0.5 L 4 0 L 4.5 0 Z M 4.5 1 L 4.5 1.5 L 5 1.5 L 5 1 L 4.5 1 Z M 2.5 1 L 2 1 L 2 1.5 L 2.5 1.5 L 2.5 1 Z M 2.5 0 L 3 0 L 3 -0.5 L 2.5 -0.5 L 2.5 0 Z M 0.5 0 L 0.5 -0.5 L 0 -0.5 L 0 0 L 0.5 0 Z M 12 10.5 L 7.5 10.5 L 7.5 11.5 L 12 11.5 L 12 10.5 Z M 4.5 10.5 L 0 10.5 L 0 11.5 L 4.5 11.5 L 4.5 10.5 Z M 7.5 11.5 L 4.5 11.5 L 4.5 12.5 L 7.5 12.5 L 7.5 11.5 Z M 7 10 L 7 12 L 8 12 L 8 10 L 7 10 Z M 4.5 10.5 L 7.5 10.5 L 7.5 9.5 L 4.5 9.5 L 4.5 10.5 Z M 5 12 L 5 11 L 4 11 L 4 12 L 5 12 Z M 5 11 L 5 10 L 4 10 L 4 11 L 5 11 Z M 11.5 6.5 L 0.5 6.5 L 0.5 7.5 L 11.5 7.5 L 11.5 6.5 Z M 5.5 7 L 5.5 10 L 6.5 10 L 6.5 7 L 5.5 7 Z M 12 7 L 12 0 L 11 0 L 11 7 L 12 7 Z M 11.5 -0.5 L 9.5 -0.5 L 9.5 0.5 L 11.5 0.5 L 11.5 -0.5 Z M 9 0 L 9 1 L 10 1 L 10 0 L 9 0 Z M 8 1 L 8 0 L 7 0 L 7 1 L 8 1 Z M 7.5 -0.5 L 4.5 -0.5 L 4.5 0.5 L 7.5 0.5 L 7.5 -0.5 Z M 4 0 L 4 1 L 5 1 L 5 0 L 4 0 Z M 4.5 0.5 L 2.5 0.5 L 2.5 1.5 L 4.5 1.5 L 4.5 0.5 Z M 3 1 L 3 0 L 2 0 L 2 1 L 3 1 Z M 2.5 -0.5 L 0.5 -0.5 L 0.5 0.5 L 2.5 0.5 L 2.5 -0.5 Z M 0 0 L 0 7 L 1 7 L 1 0 L 0 0 Z M 9.5 0.5 L 7.5 0.5 L 7.5 1.5 L 9.5 1.5 L 9.5 0.5 Z M 0.5 3.5 L 11.5 3.5 L 11.5 2.5 L 0.5 2.5 L 0.5 3.5 Z M 0.5 5.5 L 11.5 5.5 L 11.5 4.5 L 0.5 4.5 L 0.5 5.5 Z M 3 5 L 3 7 L 4 7 L 4 5 L 3 5 Z M 5.5 3 L 5.5 5 L 6.5 5 L 6.5 3 L 5.5 3 Z M 8 5 L 8 7 L 9 7 L 9 5 L 8 5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "communication-forward": "<g transform=\"translate(3.33,5.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.652 -0.5 L -0.483 0.129 L 0 0 Z M 0 0.5 L 9.17 0.5 L 9.17 -0.5 L 0 -0.5 L 0 0.5 Z M 2.359 6.871 L 0.483 -0.129 L -0.483 0.129 L 1.393 7.129 L 2.359 6.871 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(10,3)\"><path d=\"M 2.5 2.5 L 2.854 2.854 L 3.207 2.5 L 2.854 2.146 L 2.5 2.5 Z M 0.354 5.354 L 2.854 2.854 L 2.146 2.146 L -0.354 4.646 L 0.354 5.354 Z M 2.854 2.146 L 0.354 -0.354 L -0.354 0.354 L 2.146 2.854 L 2.854 2.146 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "communication-network-signal": "<g transform=\"translate(1.5,4.111)\"><path d=\"M 11.743 8.132 C 12.828 7.046 13.5 5.546 13.5 3.889 L 12.5 3.889 C 12.5 5.27 11.941 6.519 11.036 7.425 L 11.743 8.132 Z M 13.5 3.889 C 13.5 2.232 12.828 0.732 11.743 -0.354 L 11.036 0.354 C 11.941 1.259 12.5 2.508 12.5 3.889 L 13.5 3.889 Z M 1.257 -0.354 C 0.172 0.732 -0.5 2.232 -0.5 3.889 L 0.5 3.889 C 0.5 2.508 1.059 1.259 1.964 0.354 L 1.257 -0.354 Z M -0.5 3.889 C -0.5 5.546 0.172 7.046 1.257 8.132 L 1.964 7.425 C 1.059 6.519 0.5 5.27 0.5 3.889 L -0.5 3.889 Z M 7.5 3.889 C 7.5 4.441 7.052 4.889 6.5 4.889 L 6.5 5.889 C 7.605 5.889 8.5 4.994 8.5 3.889 L 7.5 3.889 Z M 6.5 4.889 C 5.948 4.889 5.5 4.441 5.5 3.889 L 4.5 3.889 C 4.5 4.994 5.395 5.889 6.5 5.889 L 6.5 4.889 Z M 5.5 3.889 C 5.5 3.337 5.948 2.889 6.5 2.889 L 6.5 1.889 C 5.395 1.889 4.5 2.785 4.5 3.889 L 5.5 3.889 Z M 6.5 2.889 C 7.052 2.889 7.5 3.337 7.5 3.889 L 8.5 3.889 C 8.5 2.785 7.605 1.889 6.5 1.889 L 6.5 2.889 Z M 10.328 6.717 C 11.052 5.994 11.5 4.993 11.5 3.889 L 10.5 3.889 C 10.5 4.718 10.165 5.467 9.621 6.01 L 10.328 6.717 Z M 11.5 3.889 C 11.5 2.785 11.052 1.784 10.328 1.061 L 9.621 1.768 C 10.165 2.311 10.5 3.06 10.5 3.889 L 11.5 3.889 Z M 2.672 1.061 C 1.948 1.784 1.5 2.785 1.5 3.889 L 2.5 3.889 C 2.5 3.06 2.835 2.311 3.379 1.768 L 2.672 1.061 Z M 1.5 3.889 C 1.5 4.993 1.948 5.994 2.672 6.717 L 3.379 6.01 C 2.835 5.467 2.5 4.718 2.5 3.889 L 1.5 3.889 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "communication-reply": "<g transform=\"translate(3.5,5.5)\"><path d=\"M 9.17 0 L 9.653 0.129 L 9.821 -0.5 L 9.17 -0.5 L 9.17 0 Z M 9.17 -0.5 L 0 -0.5 L 0 0.5 L 9.17 0.5 L 9.17 -0.5 Z M 7.777 7.129 L 9.653 0.129 L 8.687 -0.129 L 6.811 6.871 L 7.777 7.129 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(3.5,3)\"><path d=\"M 0 2.5 L -0.354 2.146 L -0.707 2.5 L -0.354 2.854 L 0 2.5 Z M 2.854 4.646 L 0.354 2.146 L -0.354 2.854 L 2.146 5.354 L 2.854 4.646 Z M 0.354 2.854 L 2.854 0.354 L 2.146 -0.354 L -0.354 2.146 L 0.354 2.854 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "communication-transfer-horizontal": "<g transform=\"translate(3.5,4.5)\"><path d=\"M 8.5 -0.5 L 0 -0.5 L 0 0.5 L 8.5 0.5 L 8.5 -0.5 Z M 9 6.5 L 0.5 6.5 L 0.5 7.5 L 9 7.5 L 9 6.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(9.5,8.5)\"><path d=\"M 3 3 L 3.354 3.354 L 3.707 3 L 3.354 2.646 L 3 3 Z M -0.354 0.354 L 2.646 3.354 L 3.354 2.646 L 0.354 -0.354 L -0.354 0.354 Z M 2.646 2.646 L -0.354 5.646 L 0.354 6.354 L 3.354 3.354 L 2.646 2.646 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(3.5,1.5)\"><path d=\"M 0 3 L -0.354 2.646 L -0.707 3 L -0.354 3.354 L 0 3 Z M 2.646 -0.354 L -0.354 2.646 L 0.354 3.354 L 3.354 0.354 L 2.646 -0.354 Z M -0.354 3.354 L 2.646 6.354 L 3.354 5.646 L 0.354 2.646 L -0.354 3.354 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "communication-transfer-vertical": "<g transform=\"translate(4.5,3.5)\"><path d=\"M -0.5 0.5 L -0.5 9 L 0.5 9 L 0.5 0.5 L -0.5 0.5 Z M 6.5 0 L 6.5 8.5 L 7.5 8.5 L 7.5 0 L 6.5 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(1.5,3.5)\"><path d=\"M 10 0 L 10.354 -0.354 L 10 -0.707 L 9.646 -0.354 L 10 0 Z M 3 9 L 2.646 9.354 L 3 9.707 L 3.354 9.354 L 3 9 Z M 7.354 3.354 L 10.354 0.354 L 9.646 -0.354 L 6.646 2.646 L 7.354 3.354 Z M 9.646 0.354 L 12.646 3.354 L 13.354 2.646 L 10.354 -0.354 L 9.646 0.354 Z M -0.354 6.354 L 2.646 9.354 L 3.354 8.646 L 0.354 5.646 L -0.354 6.354 Z M 3.354 9.354 L 6.354 6.354 L 5.646 5.646 L 2.646 8.646 L 3.354 9.354 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "communication-upload": "<g transform=\"translate(2.33,2.5)\"><path d=\"M 10.67 0 L 11.153 -0.129 L 11.054 -0.5 L 10.67 -0.5 L 10.67 0 Z M 0.67 0 L 0.67 -0.5 L 0.286 -0.5 L 0.187 -0.129 L 0.67 0 Z M 10.187 0.129 L 10.857 2.629 L 11.823 2.371 L 11.153 -0.129 L 10.187 0.129 Z M 0.67 0.5 L 10.67 0.5 L 10.67 -0.5 L 0.67 -0.5 L 0.67 0.5 Z M 0.483 2.629 L 1.153 0.129 L 0.187 -0.129 L -0.483 2.371 L 0.483 2.629 Z M 5.17 2.5 L 5.17 11.5 L 6.17 11.5 L 6.17 2.5 L 5.17 2.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(4,5)\"><path d=\"M 4 0 L 4.354 -0.354 L 4 -0.707 L 3.646 -0.354 L 4 0 Z M 8.354 3.646 L 4.354 -0.354 L 3.646 0.354 L 7.646 4.354 L 8.354 3.646 Z M 3.646 -0.354 L -0.354 3.646 L 0.354 4.354 L 4.354 0.354 L 3.646 -0.354 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "communication-wifi": "<g transform=\"translate(1.282,3.5)\"><path d=\"M 7.718 7.5 C 7.718 8.052 7.27 8.5 6.718 8.5 L 6.718 9.5 C 7.822 9.5 8.718 8.605 8.718 7.5 L 7.718 7.5 Z M 6.718 8.5 C 6.165 8.5 5.718 8.052 5.718 7.5 L 4.718 7.5 C 4.718 8.605 5.613 9.5 6.718 9.5 L 6.718 8.5 Z M 5.718 7.5 C 5.718 6.948 6.165 6.5 6.718 6.5 L 6.718 5.5 C 5.613 5.5 4.718 6.395 4.718 7.5 L 5.718 7.5 Z M 6.718 6.5 C 7.27 6.5 7.718 6.948 7.718 7.5 L 8.718 7.5 C 8.718 6.395 7.822 5.5 6.718 5.5 L 6.718 6.5 Z M 6.718 3.5 C 8.375 3.5 9.874 4.171 10.96 5.257 L 11.667 4.55 C 10.401 3.284 8.65 2.5 6.718 2.5 L 6.718 3.5 Z M 2.475 5.257 C 3.561 4.171 5.061 3.5 6.718 3.5 L 6.718 2.5 C 4.785 2.5 3.034 3.284 1.768 4.55 L 2.475 5.257 Z M 6.718 0.5 C 9.203 0.5 11.452 1.507 13.081 3.136 L 13.789 2.429 C 11.979 0.62 9.479 -0.5 6.718 -0.5 L 6.718 0.5 Z M 0.354 3.136 C 1.983 1.507 4.232 0.5 6.718 0.5 L 6.718 -0.5 C 3.956 -0.5 1.456 0.62 -0.354 2.429 L 0.354 3.136 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "communication-wifi-off": "<g transform=\"translate(1.282,3.5)\"><path d=\"M 7.718 7.5 C 7.718 8.052 7.27 8.5 6.718 8.5 L 6.718 9.5 C 7.822 9.5 8.718 8.605 8.718 7.5 L 7.718 7.5 Z M 6.718 8.5 C 6.165 8.5 5.718 8.052 5.718 7.5 L 4.718 7.5 C 4.718 8.605 5.613 9.5 6.718 9.5 L 6.718 8.5 Z M 5.718 7.5 C 5.718 6.948 6.165 6.5 6.718 6.5 L 6.718 5.5 C 5.613 5.5 4.718 6.395 4.718 7.5 L 5.718 7.5 Z M 6.718 6.5 C 7.27 6.5 7.718 6.948 7.718 7.5 L 8.718 7.5 C 8.718 6.395 7.822 5.5 6.718 5.5 L 6.718 6.5 Z M 6.718 3.5 C 8.375 3.5 9.874 4.171 10.96 5.257 L 11.667 4.55 C 10.401 3.284 8.65 2.5 6.718 2.5 L 6.718 3.5 Z M 2.475 5.257 C 3.561 4.171 5.061 3.5 6.718 3.5 L 6.718 2.5 C 4.785 2.5 3.034 3.284 1.768 4.55 L 2.475 5.257 Z M 6.718 0.5 C 9.203 0.5 11.452 1.507 13.081 3.136 L 13.789 2.429 C 11.979 0.62 9.479 -0.5 6.718 -0.5 L 6.718 0.5 Z M 0.354 3.136 C 1.983 1.507 4.232 0.5 6.718 0.5 L 6.718 -0.5 C 3.956 -0.5 1.456 0.62 -0.354 2.429 L 0.354 3.136 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(0,0)\"><path d=\"M 11.646 -0.354 L -0.354 11.646 L 0.354 12.354 L 12.354 0.354 L 11.646 -0.354 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "cursor-arrow": "<g transform=\"translate(2.713,2.713)\"><path d=\"M 4.848 11.172 L 4.39 11.371 L 4.787 12.287 L 5.282 11.42 L 4.848 11.172 Z M 0 0 L 0.199 -0.459 L -0.963 -0.963 L -0.459 0.199 L 0 0 Z M 6.678 7.971 L 7.031 7.617 L 6.568 7.154 L 6.244 7.723 L 6.678 7.971 Z M 9.787 11.08 L 9.434 11.434 L 9.787 11.787 L 10.141 11.434 L 9.787 11.08 Z M 11.08 9.787 L 11.434 10.141 L 11.787 9.787 L 11.434 9.434 L 11.08 9.787 Z M 7.971 6.678 L 7.723 6.244 L 7.154 6.568 L 7.617 7.031 L 7.971 6.678 Z M 11.172 4.848 L 11.42 5.282 L 12.287 4.787 L 11.371 4.39 L 11.172 4.848 Z M 5.307 10.973 L 0.459 -0.199 L -0.459 0.199 L 4.39 11.371 L 5.307 10.973 Z M 10.726 9.434 L 9.434 10.726 L 10.141 11.434 L 11.434 10.141 L 10.726 9.434 Z M -0.199 0.459 L 10.973 5.307 L 11.371 4.39 L 0.199 -0.459 L -0.199 0.459 Z M 6.244 7.723 L 4.414 10.924 L 5.282 11.42 L 7.112 8.219 L 6.244 7.723 Z M 10.141 10.726 L 7.031 7.617 L 6.324 8.324 L 9.434 11.434 L 10.141 10.726 Z M 7.617 7.031 L 10.726 10.141 L 11.434 9.434 L 8.324 6.324 L 7.617 7.031 Z M 10.924 4.414 L 7.723 6.244 L 8.219 7.112 L 11.42 5.282 L 10.924 4.414 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "cursor-crosshair": "<g transform=\"translate(1,1)\"><path d=\"M 12 7 C 12 9.761 9.761 12 7 12 L 7 13 C 10.314 13 13 10.314 13 7 L 12 7 Z M 7 12 C 4.239 12 2 9.761 2 7 L 1 7 C 1 10.314 3.686 13 7 13 L 7 12 Z M 2 7 C 2 4.239 4.239 2 7 2 L 7 1 C 3.686 1 1 3.686 1 7 L 2 7 Z M 7 2 C 9.761 2 12 4.239 12 7 L 13 7 C 13 3.686 10.314 1 7 1 L 7 2 Z M 6.5 0 L 6.5 3 L 7.5 3 L 7.5 0 L 6.5 0 Z M 6.5 11 L 6.5 14 L 7.5 14 L 7.5 11 L 6.5 11 Z M 0 7.5 L 3 7.5 L 3 6.5 L 0 6.5 L 0 7.5 Z M 11 7.5 L 14 7.5 L 14 6.5 L 11 6.5 L 11 7.5 Z M 8 7 C 8 7.552 7.552 8 7 8 L 7 9 C 8.105 9 9 8.105 9 7 L 8 7 Z M 7 8 C 6.448 8 6 7.552 6 7 L 5 7 C 5 8.105 5.895 9 7 9 L 7 8 Z M 6 7 C 6 6.448 6.448 6 7 6 L 7 5 C 5.895 5 5 5.895 5 7 L 6 7 Z M 7 6 C 7.552 6 8 6.448 8 7 L 9 7 C 9 5.895 8.105 5 7 5 L 7 6 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "cursor-hand-closed": "<g transform=\"translate(3.5,5)\"><path d=\"M 3.134 9.5 L 2.78 9.854 L 2.927 10 L 3.134 10 L 3.134 9.5 Z M 6 1.5 L 6.5 1.5 L 6.5 1.5 L 6 1.5 Z M 6 1 L 6.5 1 L 6.5 1 L 6 1 Z M 0.586 6.952 L 0.939 6.598 L 0.939 6.598 L 0.586 6.952 Z M 2 2.5 L 2.5 2.5 L 2.5 2 L 2 2 L 2 2.5 Z M 4.5 1 C 4.5 0.724 4.724 0.5 5 0.5 L 5 -0.5 C 4.172 -0.5 3.5 0.172 3.5 1 L 4.5 1 Z M 4.5 2 L 4.5 1 L 3.5 1 L 3.5 2 L 4.5 2 Z M 3.5 1.5 L 3.5 2 L 4.5 2 L 4.5 1.5 L 3.5 1.5 Z M 3 1 C 3.276 1 3.5 1.224 3.5 1.5 L 4.5 1.5 C 4.5 0.672 3.828 0 3 0 L 3 1 Z M 2.5 1.5 C 2.5 1.224 2.724 1 3 1 L 3 0 C 2.172 0 1.5 0.672 1.5 1.5 L 2.5 1.5 Z M 9.5 6.5 C 9.5 7.881 8.381 9 7 9 L 7 10 C 8.933 10 10.5 8.433 10.5 6.5 L 9.5 6.5 Z M 9.5 2 L 9.5 6.5 L 10.5 6.5 L 10.5 2 L 9.5 2 Z M 9 1.5 C 9.276 1.5 9.5 1.724 9.5 2 L 10.5 2 C 10.5 1.172 9.828 0.5 9 0.5 L 9 1.5 Z M 8.5 2 C 8.5 1.724 8.724 1.5 9 1.5 L 9 0.5 C 8.172 0.5 7.5 1.172 7.5 2 L 8.5 2 Z M 7 1 C 7.276 1 7.5 1.224 7.5 1.5 L 8.5 1.5 C 8.5 0.672 7.828 0 7 0 L 7 1 Z M 6.5 1.5 C 6.5 1.224 6.724 1 7 1 L 7 0 C 6.172 0 5.5 0.672 5.5 1.5 L 6.5 1.5 Z M 6.5 2 L 6.5 1.5 L 5.5 1.5 L 5.5 2 L 6.5 2 Z M 5.5 1 L 5.5 2 L 6.5 2 L 6.5 1 L 5.5 1 Z M 5 0.5 C 5.276 0.5 5.5 0.724 5.5 1 L 6.5 1 C 6.5 0.172 5.828 -0.5 5 -0.5 L 5 0.5 Z M 0.5 5.538 C 0.5 5.538 0.5 5.537 0.5 5.537 C 0.5 5.536 0.5 5.536 0.5 5.535 C 0.5 5.535 0.5 5.534 0.5 5.534 C 0.5 5.533 0.5 5.533 0.5 5.532 C 0.5 5.532 0.5 5.531 0.5 5.531 C 0.5 5.53 0.5 5.53 0.5 5.529 C 0.5 5.529 0.5 5.528 0.5 5.528 C 0.5 5.527 0.5 5.527 0.5 5.526 C 0.5 5.526 0.5 5.525 0.5 5.525 C 0.5 5.524 0.5 5.524 0.5 5.523 C 0.5 5.523 0.5 5.522 0.5 5.522 C 0.5 5.521 0.5 5.521 0.5 5.52 C 0.5 5.52 0.5 5.519 0.5 5.519 C 0.5 5.518 0.5 5.518 0.5 5.517 C 0.5 5.517 0.5 5.516 0.5 5.516 C 0.5 5.515 0.5 5.515 0.5 5.514 C 0.5 5.513 0.5 5.513 0.5 5.512 C 0.5 5.512 0.5 5.511 0.5 5.511 C 0.5 5.51 0.5 5.51 0.5 5.509 C 0.5 5.509 0.5 5.508 0.5 5.508 C 0.5 5.507 0.5 5.507 0.5 5.506 C 0.5 5.506 0.5 5.505 0.5 5.505 C 0.5 5.504 0.5 5.504 0.5 5.503 C 0.5 5.503 0.5 5.502 0.5 5.502 C 0.5 5.501 0.5 5.501 0.5 5.5 C 0.5 5.499 0.5 5.499 0.5 5.498 C 0.5 5.498 0.5 5.497 0.5 5.497 C 0.5 5.496 0.5 5.496 0.5 5.495 C 0.5 5.495 0.5 5.494 0.5 5.494 C 0.5 5.493 0.5 5.493 0.5 5.492 C 0.5 5.492 0.5 5.491 0.5 5.491 C 0.5 5.49 0.5 5.489 0.5 5.489 C 0.5 5.488 0.5 5.488 0.5 5.487 C 0.5 5.487 0.5 5.486 0.5 5.486 C 0.5 5.485 0.5 5.485 0.5 5.484 C 0.5 5.484 0.5 5.483 0.5 5.483 C 0.5 5.482 0.5 5.482 0.5 5.481 C 0.5 5.48 0.5 5.48 0.5 5.479 C 0.5 5.479 0.5 5.478 0.5 5.478 C 0.5 5.477 0.5 5.477 0.5 5.476 C 0.5 5.476 0.5 5.475 0.5 5.475 C 0.5 5.474 0.5 5.474 0.5 5.473 C 0.5 5.472 0.5 5.472 0.5 5.471 C 0.5 5.471 0.5 5.47 0.5 5.47 C 0.5 5.469 0.5 5.469 0.5 5.468 C 0.5 5.468 0.5 5.467 0.5 5.467 C 0.5 5.466 0.5 5.465 0.5 5.465 C 0.5 5.464 0.5 5.464 0.5 5.463 C 0.5 5.463 0.5 5.462 0.5 5.462 C 0.5 5.461 0.5 5.461 0.5 5.46 C 0.5 5.459 0.5 5.459 0.5 5.458 C 0.5 5.458 0.5 5.457 0.5 5.457 C 0.5 5.456 0.5 5.456 0.5 5.455 C 0.5 5.455 0.5 5.454 0.5 5.453 C 0.5 5.453 0.5 5.452 0.5 5.452 C 0.5 5.451 0.5 5.451 0.5 5.45 C 0.5 5.45 0.5 5.449 0.5 5.449 C 0.5 5.448 0.5 5.447 0.5 5.447 C 0.5 5.446 0.5 5.446 0.5 5.445 C 0.5 5.445 0.5 5.444 0.5 5.444 C 0.5 5.443 0.5 5.443 0.5 5.442 C 0.5 5.441 0.5 5.441 0.5 5.44 C 0.5 5.44 0.5 5.439 0.5 5.439 C 0.5 5.438 0.5 5.438 0.5 5.437 C 0.5 5.436 0.5 5.436 0.5 5.435 C 0.5 5.435 0.5 5.434 0.5 5.434 C 0.5 5.433 0.5 5.433 0.5 5.432 C 0.5 5.431 0.5 5.431 0.5 5.43 C 0.5 5.43 0.5 5.429 0.5 5.429 C 0.5 5.428 0.5 5.428 0.5 5.427 C 0.5 5.426 0.5 5.426 0.5 5.425 C 0.5 5.425 0.5 5.424 0.5 5.424 C 0.5 5.423 0.5 5.422 0.5 5.422 C 0.5 5.421 0.5 5.421 0.5 5.42 C 0.5 5.42 0.5 5.419 0.5 5.419 C 0.5 5.418 0.5 5.417 0.5 5.417 C 0.5 5.416 0.5 5.416 0.5 5.415 C 0.5 5.415 0.5 5.414 0.5 5.413 C 0.5 5.413 0.5 5.412 0.5 5.412 C 0.5 5.411 0.5 5.411 0.5 5.41 C 0.5 5.409 0.5 5.409 0.5 5.408 C 0.5 5.408 0.5 5.407 0.5 5.407 C 0.5 5.406 0.5 5.405 0.5 5.405 C 0.5 5.404 0.5 5.404 0.5 5.403 C 0.5 5.403 0.5 5.402 0.5 5.401 C 0.5 5.401 0.5 5.4 0.5 5.4 C 0.5 5.399 0.5 5.399 0.5 5.398 C 0.5 5.397 0.5 5.397 0.5 5.396 C 0.5 5.396 0.5 5.395 0.5 5.395 C 0.5 5.394 0.5 5.393 0.5 5.393 C 0.5 5.392 0.5 5.392 0.5 5.391 C 0.5 5.391 0.5 5.39 0.5 5.389 C 0.5 5.389 0.5 5.388 0.5 5.388 C 0.5 5.387 0.5 5.387 0.5 5.386 C 0.5 5.385 0.5 5.385 0.5 5.384 C 0.5 5.384 0.5 5.383 0.5 5.383 C 0.5 5.382 0.5 5.381 0.5 5.381 C 0.5 5.38 0.5 5.38 0.5 5.379 C 0.5 5.378 0.5 5.378 0.5 5.377 C 0.5 5.377 0.5 5.376 0.5 5.376 C 0.5 5.375 0.5 5.374 0.5 5.374 C 0.5 5.373 0.5 5.373 0.5 5.372 C 0.5 5.371 0.5 5.371 0.5 5.37 C 0.5 5.37 0.5 5.369 0.5 5.369 C 0.5 5.368 0.5 5.367 0.5 5.367 C 0.5 5.366 0.5 5.366 0.5 5.365 C 0.5 5.364 0.5 5.364 0.5 5.363 C 0.5 5.363 0.5 5.362 0.5 5.361 C 0.5 5.361 0.5 5.36 0.5 5.36 C 0.5 5.359 0.5 5.359 0.5 5.358 C 0.5 5.357 0.5 5.357 0.5 5.356 C 0.5 5.356 0.5 5.355 0.5 5.354 C 0.5 5.354 0.5 5.353 0.5 5.353 C 0.5 5.352 0.5 5.351 0.5 5.351 C 0.5 5.35 0.5 5.35 0.5 5.349 C 0.5 5.348 0.5 5.348 0.5 5.347 C 0.5 5.347 0.5 5.346 0.5 5.345 C 0.5 5.345 0.5 5.344 0.5 5.344 C 0.5 5.343 0.5 5.342 0.5 5.342 C 0.5 5.341 0.5 5.341 0.5 5.34 C 0.5 5.339 0.5 5.339 0.5 5.338 C 0.5 5.338 0.5 5.337 0.5 5.336 C 0.5 5.336 0.5 5.335 0.5 5.335 C 0.5 5.334 0.5 5.333 0.5 5.333 C 0.5 5.332 0.5 5.332 0.5 5.331 C 0.5 5.33 0.5 5.33 0.5 5.329 C 0.5 5.329 0.5 5.328 0.5 5.327 C 0.5 5.327 0.5 5.326 0.5 5.326 C 0.5 5.325 0.5 5.324 0.5 5.324 C 0.5 5.323 0.5 5.323 0.5 5.322 C 0.5 5.321 0.5 5.321 0.5 5.32 C 0.5 5.32 0.5 5.319 0.5 5.318 C 0.5 5.318 0.5 5.317 0.5 5.317 C 0.5 5.316 0.5 5.315 0.5 5.315 C 0.5 5.314 0.5 5.314 0.5 5.313 C 0.5 5.312 0.5 5.312 0.5 5.311 C 0.5 5.31 0.5 5.31 0.5 5.309 C 0.5 5.309 0.5 5.308 0.5 5.307 C 0.5 5.307 0.5 5.306 0.5 5.306 C 0.5 5.305 0.5 5.304 0.5 5.304 C 0.5 5.303 0.5 5.303 0.5 5.302 C 0.5 5.301 0.5 5.301 0.5 5.3 C 0.5 5.299 0.5 5.299 0.5 5.298 C 0.5 5.298 0.5 5.297 0.5 5.296 C 0.5 5.296 0.5 5.295 0.5 5.295 C 0.5 5.294 0.5 5.293 0.5 5.293 C 0.5 5.292 0.5 5.291 0.5 5.291 C 0.5 5.29 0.5 5.29 0.5 5.289 C 0.5 5.288 0.5 5.288 0.5 5.287 C 0.5 5.287 0.5 5.286 0.5 5.285 C 0.5 5.285 0.5 5.284 0.5 5.283 C 0.5 5.283 0.5 5.282 0.5 5.282 C 0.5 5.281 0.5 5.28 0.5 5.28 C 0.5 5.279 0.5 5.278 0.5 5.278 C 0.5 5.277 0.5 5.277 0.5 5.276 C 0.5 5.275 0.5 5.275 0.5 5.274 C 0.5 5.273 0.5 5.273 0.5 5.272 C 0.5 5.272 0.5 5.271 0.5 5.27 C 0.5 5.27 0.5 5.269 0.5 5.268 C 0.5 5.268 0.5 5.267 0.5 5.267 C 0.5 5.266 0.5 5.265 0.5 5.265 C 0.5 5.264 0.5 5.263 0.5 5.263 C 0.5 5.262 0.5 5.262 0.5 5.261 C 0.5 5.26 0.5 5.26 0.5 5.259 C 0.5 5.258 0.5 5.258 0.5 5.257 C 0.5 5.257 0.5 5.256 0.5 5.255 C 0.5 5.255 0.5 5.254 0.5 5.253 C 0.5 5.253 0.5 5.252 0.5 5.251 C 0.5 5.251 0.5 5.25 0.5 5.25 C 0.5 5.249 0.5 5.248 0.5 5.248 C 0.5 5.247 0.5 5.246 0.5 5.246 C 0.5 5.245 0.5 5.244 0.5 5.244 C 0.5 5.243 0.5 5.243 0.5 5.242 C 0.5 5.241 0.5 5.241 0.5 5.24 C 0.5 5.239 0.5 5.239 0.5 5.238 C 0.5 5.237 0.5 5.237 0.5 5.236 C 0.5 5.236 0.5 5.235 0.5 5.234 C 0.5 5.234 0.5 5.233 0.5 5.232 C 0.5 5.232 0.5 5.231 0.5 5.23 C 0.5 5.23 0.5 5.229 0.5 5.229 C 0.5 5.228 0.5 5.227 0.5 5.227 C 0.5 5.226 0.5 5.225 0.5 5.225 C 0.5 5.224 0.5 5.223 0.5 5.223 C 0.5 5.222 0.5 5.221 0.5 5.221 C 0.5 5.22 0.5 5.22 0.5 5.219 C 0.5 5.218 0.5 5.218 0.5 5.217 C 0.5 5.216 0.5 5.216 0.5 5.215 C 0.5 5.214 0.5 5.214 0.5 5.213 C 0.5 5.212 0.5 5.212 0.5 5.211 C 0.5 5.211 0.5 5.21 0.5 5.209 C 0.5 5.209 0.5 5.208 0.5 5.207 C 0.5 5.207 0.5 5.206 0.5 5.205 C 0.5 5.205 0.5 5.204 0.5 5.203 C 0.5 5.203 0.5 5.202 0.5 5.201 C 0.5 5.201 0.5 5.2 0.5 5.199 C 0.5 5.199 0.5 5.198 0.5 5.198 C 0.5 5.197 0.5 5.196 0.5 5.196 C 0.5 5.195 0.5 5.194 0.5 5.194 C 0.5 5.193 0.5 5.192 0.5 5.192 C 0.5 5.191 0.5 5.19 0.5 5.19 C 0.5 5.189 0.5 5.188 0.5 5.188 C 0.5 5.187 0.5 5.186 0.5 5.186 C 0.5 5.185 0.5 5.184 0.5 5.184 C 0.5 5.183 0.5 5.183 0.5 5.182 C 0.5 5.181 0.5 5.181 0.5 5.18 C 0.5 5.179 0.5 5.179 0.5 5.178 C 0.5 5.177 0.5 5.177 0.5 5.176 C 0.5 5.175 0.5 5.175 0.5 5.174 C 0.5 5.173 0.5 5.173 0.5 5.172 C 0.5 5.171 0.5 5.171 0.5 5.17 C 0.5 5.169 0.5 5.169 0.5 5.168 C 0.5 5.167 0.5 5.167 0.5 5.166 C 0.5 5.165 0.5 5.165 0.5 5.164 C 0.5 5.163 0.5 5.163 0.5 5.162 C 0.5 5.161 0.5 5.161 0.5 5.16 C 0.5 5.159 0.5 5.159 0.5 5.158 C 0.5 5.157 0.5 5.157 0.5 5.156 C 0.5 5.155 0.5 5.155 0.5 5.154 C 0.5 5.153 0.5 5.153 0.5 5.152 C 0.5 5.151 0.5 5.151 0.5 5.15 C 0.5 5.149 0.5 5.149 0.5 5.148 C 0.5 5.147 0.5 5.147 0.5 5.146 C 0.5 5.145 0.5 5.145 0.5 5.144 C 0.5 5.143 0.5 5.143 0.5 5.142 C 0.5 5.141 0.5 5.141 0.5 5.14 C 0.5 5.139 0.5 5.139 0.5 5.138 C 0.5 5.137 0.5 5.137 0.5 5.136 C 0.5 5.135 0.5 5.135 0.5 5.134 C 0.5 5.133 0.5 5.133 0.5 5.132 C 0.5 5.131 0.5 5.131 0.5 5.13 C 0.5 5.129 0.5 5.129 0.5 5.128 C 0.5 5.127 0.5 5.127 0.5 5.126 C 0.5 5.125 0.5 5.125 0.5 5.124 C 0.5 5.123 0.5 5.123 0.5 5.122 C 0.5 5.121 0.5 5.121 0.5 5.12 C 0.5 5.119 0.5 5.119 0.5 5.118 C 0.5 5.117 0.5 5.117 0.5 5.116 C 0.5 5.115 0.5 5.114 0.5 5.114 C 0.5 5.113 0.5 5.112 0.5 5.112 C 0.5 5.111 0.5 5.11 0.5 5.11 C 0.5 5.109 0.5 5.108 0.5 5.108 C 0.5 5.107 0.5 5.106 0.5 5.106 C 0.5 5.105 0.5 5.104 0.5 5.104 C 0.5 5.103 0.5 5.102 0.5 5.102 C 0.5 5.101 0.5 5.1 0.5 5.1 C 0.5 5.099 0.5 5.098 0.5 5.097 C 0.5 5.097 0.5 5.096 0.5 5.095 C 0.5 5.095 0.5 5.094 0.5 5.093 C 0.5 5.093 0.5 5.092 0.5 5.091 C 0.5 5.091 0.5 5.09 0.5 5.089 C 0.5 5.089 0.5 5.088 0.5 5.087 C 0.5 5.087 0.5 5.086 0.5 5.085 C 0.5 5.084 0.5 5.084 0.5 5.083 C 0.5 5.082 0.5 5.082 0.5 5.081 C 0.5 5.08 0.5 5.08 0.5 5.079 C 0.5 5.078 0.5 5.078 0.5 5.077 C 0.5 5.076 0.5 5.076 0.5 5.075 C 0.5 5.074 0.5 5.073 0.5 5.073 C 0.5 5.072 0.5 5.071 0.5 5.071 C 0.5 5.07 0.5 5.069 0.5 5.069 C 0.5 5.068 0.5 5.067 0.5 5.067 C 0.5 5.066 0.5 5.065 0.5 5.064 C 0.5 5.064 0.5 5.063 0.5 5.062 C 0.5 5.062 0.5 5.061 0.5 5.06 C 0.5 5.06 0.5 5.059 0.5 5.058 C 0.5 5.058 0.5 5.057 0.5 5.056 C 0.5 5.055 0.5 5.055 0.5 5.054 C 0.5 5.053 0.5 5.053 0.5 5.052 C 0.5 5.051 0.5 5.051 0.5 5.05 C 0.5 5.049 0.5 5.049 0.5 5.048 C 0.5 5.047 0.5 5.046 0.5 5.046 C 0.5 5.045 0.5 5.044 0.5 5.044 C 0.5 5.043 0.5 5.042 0.5 5.042 C 0.5 5.041 0.5 5.04 0.5 5.039 C 0.5 5.039 0.5 5.038 0.5 5.037 C 0.5 5.037 0.5 5.036 0.5 5.035 C 0.5 5.035 0.5 5.034 0.5 5.033 C 0.5 5.032 0.5 5.032 0.5 5.031 C 0.5 5.03 0.5 5.03 0.5 5.029 C 0.5 5.028 0.5 5.028 0.5 5.027 C 0.5 5.026 0.5 5.025 0.5 5.025 C 0.5 5.024 0.5 5.023 0.5 5.023 C 0.5 5.022 0.5 5.021 0.5 5.021 C 0.5 5.02 0.5 5.019 0.5 5.018 C 0.5 5.018 0.5 5.017 0.5 5.016 C 0.5 5.016 0.5 5.015 0.5 5.014 C 0.5 5.013 0.5 5.013 0.5 5.012 C 0.5 5.011 0.5 5.011 0.5 5.01 C 0.5 5.009 0.5 5.009 0.5 5.008 C 0.5 5.007 0.5 5.006 0.5 5.006 C 0.5 5.005 0.5 5.004 0.5 5.004 C 0.5 5.003 0.5 5.002 0.5 5.001 C 0.5 5.001 0.5 5 0.5 4.999 C 0.5 4.999 0.5 4.998 0.5 4.997 C 0.5 4.996 0.5 4.996 0.5 4.995 C 0.5 4.994 0.5 4.994 0.5 4.993 C 0.5 4.992 0.5 4.991 0.5 4.991 C 0.5 4.99 0.5 4.989 0.5 4.989 C 0.5 4.988 0.5 4.987 0.5 4.987 C 0.5 4.986 0.5 4.985 0.5 4.984 C 0.5 4.984 0.5 4.983 0.5 4.982 C 0.5 4.982 0.5 4.981 0.5 4.98 C 0.5 4.979 0.5 4.979 0.5 4.978 C 0.5 4.977 0.5 4.977 0.5 4.976 C 0.5 4.975 0.5 4.974 0.5 4.974 C 0.5 4.973 0.5 4.972 0.5 4.972 C 0.5 4.971 0.5 4.97 0.5 4.969 C 0.5 4.969 0.5 4.968 0.5 4.967 C 0.5 4.966 0.5 4.966 0.5 4.965 C 0.5 4.964 0.5 4.964 0.5 4.963 C 0.5 4.962 0.5 4.961 0.5 4.961 C 0.5 4.96 0.5 4.959 0.5 4.959 C 0.5 4.958 0.5 4.957 0.5 4.956 C 0.5 4.956 0.5 4.955 0.5 4.954 C 0.5 4.954 0.5 4.953 0.5 4.952 C 0.5 4.951 0.5 4.951 0.5 4.95 C 0.5 4.949 0.5 4.948 0.5 4.948 C 0.5 4.947 0.5 4.946 0.5 4.946 C 0.5 4.945 0.5 4.944 0.5 4.943 C 0.5 4.943 0.5 4.942 0.5 4.941 C 0.5 4.941 0.5 4.94 0.5 4.939 C 0.5 4.938 0.5 4.938 0.5 4.937 C 0.5 4.936 0.5 4.935 0.5 4.935 C 0.5 4.934 0.5 4.933 0.5 4.933 C 0.5 4.932 0.5 4.931 0.5 4.93 C 0.5 4.93 0.5 4.929 0.5 4.928 C 0.5 4.927 0.5 4.927 0.5 4.926 C 0.5 4.925 0.5 4.925 0.5 4.924 C 0.5 4.923 0.5 4.922 0.5 4.922 C 0.5 4.921 0.5 4.92 0.5 4.919 C 0.5 4.919 0.5 4.918 0.5 4.917 C 0.5 4.917 0.5 4.916 0.5 4.915 C 0.5 4.914 0.5 4.914 0.5 4.913 C 0.5 4.912 0.5 4.911 0.5 4.911 C 0.5 4.91 0.5 4.909 0.5 4.909 C 0.5 4.908 0.5 4.907 0.5 4.906 C 0.5 4.906 0.5 4.905 0.5 4.904 C 0.5 4.903 0.5 4.903 0.5 4.902 C 0.5 4.901 0.5 4.9 0.5 4.9 C 0.5 4.899 0.5 4.898 0.5 4.898 C 0.5 4.897 0.5 4.896 0.5 4.895 C 0.5 4.895 0.5 4.894 0.5 4.893 C 0.5 4.892 0.5 4.892 0.5 4.891 C 0.5 4.89 0.5 4.889 0.5 4.889 C 0.5 4.888 0.5 4.887 0.5 4.887 C 0.5 4.886 0.5 4.885 0.5 4.884 C 0.5 4.884 0.5 4.883 0.5 4.882 C 0.5 4.881 0.5 4.881 0.5 4.88 C 0.5 4.879 0.5 4.878 0.5 4.878 C 0.5 4.877 0.5 4.876 0.5 4.875 C 0.5 4.875 0.5 4.874 0.5 4.873 C 0.5 4.873 0.5 4.872 0.5 4.871 C 0.5 4.87 0.5 4.87 0.5 4.869 C 0.5 4.868 0.5 4.867 0.5 4.867 C 0.5 4.866 0.5 4.865 0.5 4.864 C 0.5 4.864 0.5 4.863 0.5 4.862 C 0.5 4.861 0.5 4.861 0.5 4.86 C 0.5 4.859 0.5 4.858 0.5 4.858 C 0.5 4.857 0.5 4.856 0.5 4.855 C 0.5 4.855 0.5 4.854 0.5 4.853 C 0.5 4.852 0.5 4.852 0.5 4.851 C 0.5 4.85 0.5 4.85 0.5 4.849 C 0.5 4.848 0.5 4.847 0.5 4.847 C 0.5 4.846 0.5 4.845 0.5 4.844 C 0.5 4.844 0.5 4.843 0.5 4.842 C 0.5 4.841 0.5 4.841 0.5 4.84 C 0.5 4.839 0.5 4.838 0.5 4.838 C 0.5 4.837 0.5 4.836 0.5 4.835 C 0.5 4.835 0.5 4.834 0.5 4.833 C 0.5 4.832 0.5 4.832 0.5 4.831 C 0.5 4.83 0.5 4.829 0.5 4.829 C 0.5 4.828 0.5 4.827 0.5 4.826 C 0.5 4.826 0.5 4.825 0.5 4.824 C 0.5 4.823 0.5 4.823 0.5 4.822 C 0.5 4.821 0.5 4.82 0.5 4.82 C 0.5 4.819 0.5 4.818 0.5 4.817 C 0.5 4.817 0.5 4.816 0.5 4.815 C 0.5 4.814 0.5 4.814 0.5 4.813 C 0.5 4.812 0.5 4.811 0.5 4.811 C 0.5 4.81 0.5 4.809 0.5 4.808 C 0.5 4.808 0.5 4.807 0.5 4.806 C 0.5 4.805 0.5 4.805 0.5 4.804 C 0.5 4.803 0.5 4.802 0.5 4.802 C 0.5 4.801 0.5 4.8 0.5 4.799 C 0.5 4.799 0.5 4.798 0.5 4.797 C 0.5 4.796 0.5 4.796 0.5 4.795 C 0.5 4.794 0.5 4.793 0.5 4.793 C 0.5 4.792 0.5 4.791 0.5 4.79 C 0.5 4.789 0.5 4.789 0.5 4.788 C 0.5 4.787 0.5 4.786 0.5 4.786 C 0.5 4.785 0.5 4.784 0.5 4.783 C 0.5 4.783 0.5 4.782 0.5 4.781 C 0.5 4.78 0.5 4.78 0.5 4.779 C 0.5 4.778 0.5 4.777 0.5 4.777 C 0.5 4.776 0.5 4.775 0.5 4.774 C 0.5 4.774 0.5 4.773 0.5 4.772 C 0.5 4.771 0.5 4.771 0.5 4.77 C 0.5 4.769 0.5 4.768 0.5 4.768 C 0.5 4.767 0.5 4.766 0.5 4.765 C 0.5 4.764 0.5 4.764 0.5 4.763 C 0.5 4.762 0.5 4.761 0.5 4.761 C 0.5 4.76 0.5 4.759 0.5 4.758 C 0.5 4.758 0.5 4.757 0.5 4.756 C 0.5 4.755 0.5 4.755 0.5 4.754 C 0.5 4.753 0.5 4.752 0.5 4.752 C 0.5 4.751 0.5 4.75 0.5 4.749 C 0.5 4.748 0.5 4.748 0.5 4.747 C 0.5 4.746 0.5 4.745 0.5 4.745 C 0.5 4.744 0.5 4.743 0.5 4.742 C 0.5 4.742 0.5 4.741 0.5 4.74 C 0.5 4.739 0.5 4.739 0.5 4.738 C 0.5 4.737 0.5 4.736 0.5 4.735 C 0.5 4.735 0.5 4.734 0.5 4.733 C 0.5 4.732 0.5 4.732 0.5 4.731 C 0.5 4.73 0.5 4.729 0.5 4.729 C 0.5 4.728 0.5 4.727 0.5 4.726 C 0.5 4.725 0.5 4.725 0.5 4.724 C 0.5 4.723 0.5 4.722 0.5 4.722 C 0.5 4.721 0.5 4.72 0.5 4.719 C 0.5 4.719 0.5 4.718 0.5 4.717 C 0.5 4.716 0.5 4.715 0.5 4.715 C 0.5 4.714 0.5 4.713 0.5 4.712 C 0.5 4.712 0.5 4.711 0.5 4.71 C 0.5 4.709 0.5 4.709 0.5 4.708 C 0.5 4.707 0.5 4.706 0.5 4.705 C 0.5 4.705 0.5 4.704 0.5 4.703 C 0.5 4.702 0.5 4.702 0.5 4.701 C 0.5 4.7 0.5 4.699 0.5 4.699 C 0.5 4.698 0.5 4.697 0.5 4.696 C 0.5 4.695 0.5 4.695 0.5 4.694 C 0.5 4.693 0.5 4.692 0.5 4.692 C 0.5 4.691 0.5 4.69 0.5 4.689 C 0.5 4.688 0.5 4.688 0.5 4.687 C 0.5 4.686 0.5 4.685 0.5 4.685 C 0.5 4.684 0.5 4.683 0.5 4.682 C 0.5 4.681 0.5 4.681 0.5 4.68 C 0.5 4.679 0.5 4.678 0.5 4.678 C 0.5 4.677 0.5 4.676 0.5 4.675 C 0.5 4.675 0.5 4.674 0.5 4.673 C 0.5 4.672 0.5 4.671 0.5 4.671 C 0.5 4.67 0.5 4.669 0.5 4.668 C 0.5 4.668 0.5 4.667 0.5 4.666 C 0.5 4.665 0.5 4.664 0.5 4.664 C 0.5 4.663 0.5 4.662 0.5 4.661 C 0.5 4.661 0.5 4.66 0.5 4.659 C 0.5 4.658 0.5 4.657 0.5 4.657 C 0.5 4.656 0.5 4.655 0.5 4.654 C 0.5 4.654 0.5 4.653 0.5 4.652 C 0.5 4.651 0.5 4.65 0.5 4.65 C 0.5 4.649 0.5 4.648 0.5 4.647 C 0.5 4.646 0.5 4.646 0.5 4.645 C 0.5 4.644 0.5 4.643 0.5 4.643 C 0.5 4.642 0.5 4.641 0.5 4.64 C 0.5 4.639 0.5 4.639 0.5 4.638 C 0.5 4.637 0.5 4.636 0.5 4.636 C 0.5 4.635 0.5 4.634 0.5 4.633 C 0.5 4.632 0.5 4.632 0.5 4.631 C 0.5 4.63 0.5 4.629 0.5 4.628 C 0.5 4.628 0.5 4.627 0.5 4.626 C 0.5 4.625 0.5 4.625 0.5 4.624 C 0.5 4.623 0.5 4.622 0.5 4.621 C 0.5 4.621 0.5 4.62 0.5 4.619 C 0.5 4.618 0.5 4.618 0.5 4.617 C 0.5 4.616 0.5 4.615 0.5 4.614 C 0.5 4.614 0.5 4.613 0.5 4.612 C 0.5 4.611 0.5 4.61 0.5 4.61 C 0.5 4.609 0.5 4.608 0.5 4.607 C 0.5 4.606 0.5 4.606 0.5 4.605 C 0.5 4.604 0.5 4.603 0.5 4.603 C 0.5 4.602 0.5 4.601 0.5 4.6 C 0.5 4.599 0.5 4.599 0.5 4.598 C 0.5 4.597 0.5 4.596 0.5 4.595 C 0.5 4.595 0.5 4.594 0.5 4.593 C 0.5 4.592 0.5 4.592 0.5 4.591 C 0.5 4.59 0.5 4.589 0.5 4.588 C 0.5 4.588 0.5 4.587 0.5 4.586 C 0.5 4.585 0.5 4.584 0.5 4.584 C 0.5 4.583 0.5 4.582 0.5 4.581 C 0.5 4.58 0.5 4.58 0.5 4.579 C 0.5 4.578 0.5 4.577 0.5 4.577 C 0.5 4.576 0.5 4.575 0.5 4.574 C 0.5 4.573 0.5 4.573 0.5 4.572 C 0.5 4.571 0.5 4.57 0.5 4.569 C 0.5 4.569 0.5 4.568 0.5 4.567 C 0.5 4.566 0.5 4.565 0.5 4.565 C 0.5 4.564 0.5 4.563 0.5 4.562 C 0.5 4.561 0.5 4.561 0.5 4.56 C 0.5 4.559 0.5 4.558 0.5 4.557 C 0.5 4.557 0.5 4.556 0.5 4.555 C 0.5 4.554 0.5 4.554 0.5 4.553 C 0.5 4.552 0.5 4.551 0.5 4.55 C 0.5 4.55 0.5 4.549 0.5 4.548 C 0.5 4.547 0.5 4.546 0.5 4.546 C 0.5 4.545 0.5 4.544 0.5 4.543 C 0.5 4.542 0.5 4.542 0.5 4.541 C 0.5 4.54 0.5 4.539 0.5 4.538 C 0.5 4.538 0.5 4.537 0.5 4.536 C 0.5 4.535 0.5 4.534 0.5 4.534 C 0.5 4.533 0.5 4.532 0.5 4.531 C 0.5 4.53 0.5 4.53 0.5 4.529 C 0.5 4.528 0.5 4.527 0.5 4.526 C 0.5 4.526 0.5 4.525 0.5 4.524 C 0.5 4.523 0.5 4.522 0.5 4.522 C 0.5 4.521 0.5 4.52 0.5 4.519 C 0.5 4.518 0.5 4.518 0.5 4.517 C 0.5 4.516 0.5 4.515 0.5 4.514 C 0.5 4.514 0.5 4.513 0.5 4.512 C 0.5 4.511 0.5 4.51 0.5 4.51 C 0.5 4.509 0.5 4.508 0.5 4.507 C 0.5 4.506 0.5 4.506 0.5 4.505 C 0.5 4.504 0.5 4.503 0.5 4.502 C 0.5 4.502 0.5 4.501 0.5 4.5 L -0.5 4.5 C -0.5 4.501 -0.5 4.502 -0.5 4.502 C -0.5 4.503 -0.5 4.504 -0.5 4.505 C -0.5 4.506 -0.5 4.506 -0.5 4.507 C -0.5 4.508 -0.5 4.509 -0.5 4.51 C -0.5 4.51 -0.5 4.511 -0.5 4.512 C -0.5 4.513 -0.5 4.514 -0.5 4.514 C -0.5 4.515 -0.5 4.516 -0.5 4.517 C -0.5 4.518 -0.5 4.518 -0.5 4.519 C -0.5 4.52 -0.5 4.521 -0.5 4.522 C -0.5 4.522 -0.5 4.523 -0.5 4.524 C -0.5 4.525 -0.5 4.526 -0.5 4.526 C -0.5 4.527 -0.5 4.528 -0.5 4.529 C -0.5 4.53 -0.5 4.53 -0.5 4.531 C -0.5 4.532 -0.5 4.533 -0.5 4.534 C -0.5 4.534 -0.5 4.535 -0.5 4.536 C -0.5 4.537 -0.5 4.538 -0.5 4.538 C -0.5 4.539 -0.5 4.54 -0.5 4.541 C -0.5 4.542 -0.5 4.542 -0.5 4.543 C -0.5 4.544 -0.5 4.545 -0.5 4.546 C -0.5 4.546 -0.5 4.547 -0.5 4.548 C -0.5 4.549 -0.5 4.55 -0.5 4.55 C -0.5 4.551 -0.5 4.552 -0.5 4.553 C -0.5 4.554 -0.5 4.554 -0.5 4.555 C -0.5 4.556 -0.5 4.557 -0.5 4.557 C -0.5 4.558 -0.5 4.559 -0.5 4.56 C -0.5 4.561 -0.5 4.561 -0.5 4.562 C -0.5 4.563 -0.5 4.564 -0.5 4.565 C -0.5 4.565 -0.5 4.566 -0.5 4.567 C -0.5 4.568 -0.5 4.569 -0.5 4.569 C -0.5 4.57 -0.5 4.571 -0.5 4.572 C -0.5 4.573 -0.5 4.573 -0.5 4.574 C -0.5 4.575 -0.5 4.576 -0.5 4.577 C -0.5 4.577 -0.5 4.578 -0.5 4.579 C -0.5 4.58 -0.5 4.58 -0.5 4.581 C -0.5 4.582 -0.5 4.583 -0.5 4.584 C -0.5 4.584 -0.5 4.585 -0.5 4.586 C -0.5 4.587 -0.5 4.588 -0.5 4.588 C -0.5 4.589 -0.5 4.59 -0.5 4.591 C -0.5 4.592 -0.5 4.592 -0.5 4.593 C -0.5 4.594 -0.5 4.595 -0.5 4.595 C -0.5 4.596 -0.5 4.597 -0.5 4.598 C -0.5 4.599 -0.5 4.599 -0.5 4.6 C -0.5 4.601 -0.5 4.602 -0.5 4.603 C -0.5 4.603 -0.5 4.604 -0.5 4.605 C -0.5 4.606 -0.5 4.606 -0.5 4.607 C -0.5 4.608 -0.5 4.609 -0.5 4.61 C -0.5 4.61 -0.5 4.611 -0.5 4.612 C -0.5 4.613 -0.5 4.614 -0.5 4.614 C -0.5 4.615 -0.5 4.616 -0.5 4.617 C -0.5 4.618 -0.5 4.618 -0.5 4.619 C -0.5 4.62 -0.5 4.621 -0.5 4.621 C -0.5 4.622 -0.5 4.623 -0.5 4.624 C -0.5 4.625 -0.5 4.625 -0.5 4.626 C -0.5 4.627 -0.5 4.628 -0.5 4.628 C -0.5 4.629 -0.5 4.63 -0.5 4.631 C -0.5 4.632 -0.5 4.632 -0.5 4.633 C -0.5 4.634 -0.5 4.635 -0.5 4.636 C -0.5 4.636 -0.5 4.637 -0.5 4.638 C -0.5 4.639 -0.5 4.639 -0.5 4.64 C -0.5 4.641 -0.5 4.642 -0.5 4.643 C -0.5 4.643 -0.5 4.644 -0.5 4.645 C -0.5 4.646 -0.5 4.646 -0.5 4.647 C -0.5 4.648 -0.5 4.649 -0.5 4.65 C -0.5 4.65 -0.5 4.651 -0.5 4.652 C -0.5 4.653 -0.5 4.654 -0.5 4.654 C -0.5 4.655 -0.5 4.656 -0.5 4.657 C -0.5 4.657 -0.5 4.658 -0.5 4.659 C -0.5 4.66 -0.5 4.661 -0.5 4.661 C -0.5 4.662 -0.5 4.663 -0.5 4.664 C -0.5 4.664 -0.5 4.665 -0.5 4.666 C -0.5 4.667 -0.5 4.668 -0.5 4.668 C -0.5 4.669 -0.5 4.67 -0.5 4.671 C -0.5 4.671 -0.5 4.672 -0.5 4.673 C -0.5 4.674 -0.5 4.675 -0.5 4.675 C -0.5 4.676 -0.5 4.677 -0.5 4.678 C -0.5 4.678 -0.5 4.679 -0.5 4.68 C -0.5 4.681 -0.5 4.681 -0.5 4.682 C -0.5 4.683 -0.5 4.684 -0.5 4.685 C -0.5 4.685 -0.5 4.686 -0.5 4.687 C -0.5 4.688 -0.5 4.688 -0.5 4.689 C -0.5 4.69 -0.5 4.691 -0.5 4.692 C -0.5 4.692 -0.5 4.693 -0.5 4.694 C -0.5 4.695 -0.5 4.695 -0.5 4.696 C -0.5 4.697 -0.5 4.698 -0.5 4.699 C -0.5 4.699 -0.5 4.7 -0.5 4.701 C -0.5 4.702 -0.5 4.702 -0.5 4.703 C -0.5 4.704 -0.5 4.705 -0.5 4.705 C -0.5 4.706 -0.5 4.707 -0.5 4.708 C -0.5 4.709 -0.5 4.709 -0.5 4.71 C -0.5 4.711 -0.5 4.712 -0.5 4.712 C -0.5 4.713 -0.5 4.714 -0.5 4.715 C -0.5 4.715 -0.5 4.716 -0.5 4.717 C -0.5 4.718 -0.5 4.719 -0.5 4.719 C -0.5 4.72 -0.5 4.721 -0.5 4.722 C -0.5 4.722 -0.5 4.723 -0.5 4.724 C -0.5 4.725 -0.5 4.725 -0.5 4.726 C -0.5 4.727 -0.5 4.728 -0.5 4.729 C -0.5 4.729 -0.5 4.73 -0.5 4.731 C -0.5 4.732 -0.5 4.732 -0.5 4.733 C -0.5 4.734 -0.5 4.735 -0.5 4.735 C -0.5 4.736 -0.5 4.737 -0.5 4.738 C -0.5 4.739 -0.5 4.739 -0.5 4.74 C -0.5 4.741 -0.5 4.742 -0.5 4.742 C -0.5 4.743 -0.5 4.744 -0.5 4.745 C -0.5 4.745 -0.5 4.746 -0.5 4.747 C -0.5 4.748 -0.5 4.748 -0.5 4.749 C -0.5 4.75 -0.5 4.751 -0.5 4.752 C -0.5 4.752 -0.5 4.753 -0.5 4.754 C -0.5 4.755 -0.5 4.755 -0.5 4.756 C -0.5 4.757 -0.5 4.758 -0.5 4.758 C -0.5 4.759 -0.5 4.76 -0.5 4.761 C -0.5 4.761 -0.5 4.762 -0.5 4.763 C -0.5 4.764 -0.5 4.764 -0.5 4.765 C -0.5 4.766 -0.5 4.767 -0.5 4.768 C -0.5 4.768 -0.5 4.769 -0.5 4.77 C -0.5 4.771 -0.5 4.771 -0.5 4.772 C -0.5 4.773 -0.5 4.774 -0.5 4.774 C -0.5 4.775 -0.5 4.776 -0.5 4.777 C -0.5 4.777 -0.5 4.778 -0.5 4.779 C -0.5 4.78 -0.5 4.78 -0.5 4.781 C -0.5 4.782 -0.5 4.783 -0.5 4.783 C -0.5 4.784 -0.5 4.785 -0.5 4.786 C -0.5 4.786 -0.5 4.787 -0.5 4.788 C -0.5 4.789 -0.5 4.789 -0.5 4.79 C -0.5 4.791 -0.5 4.792 -0.5 4.793 C -0.5 4.793 -0.5 4.794 -0.5 4.795 C -0.5 4.796 -0.5 4.796 -0.5 4.797 C -0.5 4.798 -0.5 4.799 -0.5 4.799 C -0.5 4.8 -0.5 4.801 -0.5 4.802 C -0.5 4.802 -0.5 4.803 -0.5 4.804 C -0.5 4.805 -0.5 4.805 -0.5 4.806 C -0.5 4.807 -0.5 4.808 -0.5 4.808 C -0.5 4.809 -0.5 4.81 -0.5 4.811 C -0.5 4.811 -0.5 4.812 -0.5 4.813 C -0.5 4.814 -0.5 4.814 -0.5 4.815 C -0.5 4.816 -0.5 4.817 -0.5 4.817 C -0.5 4.818 -0.5 4.819 -0.5 4.82 C -0.5 4.82 -0.5 4.821 -0.5 4.822 C -0.5 4.823 -0.5 4.823 -0.5 4.824 C -0.5 4.825 -0.5 4.826 -0.5 4.826 C -0.5 4.827 -0.5 4.828 -0.5 4.829 C -0.5 4.829 -0.5 4.83 -0.5 4.831 C -0.5 4.832 -0.5 4.832 -0.5 4.833 C -0.5 4.834 -0.5 4.835 -0.5 4.835 C -0.5 4.836 -0.5 4.837 -0.5 4.838 C -0.5 4.838 -0.5 4.839 -0.5 4.84 C -0.5 4.841 -0.5 4.841 -0.5 4.842 C -0.5 4.843 -0.5 4.844 -0.5 4.844 C -0.5 4.845 -0.5 4.846 -0.5 4.847 C -0.5 4.847 -0.5 4.848 -0.5 4.849 C -0.5 4.85 -0.5 4.85 -0.5 4.851 C -0.5 4.852 -0.5 4.852 -0.5 4.853 C -0.5 4.854 -0.5 4.855 -0.5 4.855 C -0.5 4.856 -0.5 4.857 -0.5 4.858 C -0.5 4.858 -0.5 4.859 -0.5 4.86 C -0.5 4.861 -0.5 4.861 -0.5 4.862 C -0.5 4.863 -0.5 4.864 -0.5 4.864 C -0.5 4.865 -0.5 4.866 -0.5 4.867 C -0.5 4.867 -0.5 4.868 -0.5 4.869 C -0.5 4.87 -0.5 4.87 -0.5 4.871 C -0.5 4.872 -0.5 4.873 -0.5 4.873 C -0.5 4.874 -0.5 4.875 -0.5 4.875 C -0.5 4.876 -0.5 4.877 -0.5 4.878 C -0.5 4.878 -0.5 4.879 -0.5 4.88 C -0.5 4.881 -0.5 4.881 -0.5 4.882 C -0.5 4.883 -0.5 4.884 -0.5 4.884 C -0.5 4.885 -0.5 4.886 -0.5 4.887 C -0.5 4.887 -0.5 4.888 -0.5 4.889 C -0.5 4.889 -0.5 4.89 -0.5 4.891 C -0.5 4.892 -0.5 4.892 -0.5 4.893 C -0.5 4.894 -0.5 4.895 -0.5 4.895 C -0.5 4.896 -0.5 4.897 -0.5 4.898 C -0.5 4.898 -0.5 4.899 -0.5 4.9 C -0.5 4.9 -0.5 4.901 -0.5 4.902 C -0.5 4.903 -0.5 4.903 -0.5 4.904 C -0.5 4.905 -0.5 4.906 -0.5 4.906 C -0.5 4.907 -0.5 4.908 -0.5 4.909 C -0.5 4.909 -0.5 4.91 -0.5 4.911 C -0.5 4.911 -0.5 4.912 -0.5 4.913 C -0.5 4.914 -0.5 4.914 -0.5 4.915 C -0.5 4.916 -0.5 4.917 -0.5 4.917 C -0.5 4.918 -0.5 4.919 -0.5 4.919 C -0.5 4.92 -0.5 4.921 -0.5 4.922 C -0.5 4.922 -0.5 4.923 -0.5 4.924 C -0.5 4.925 -0.5 4.925 -0.5 4.926 C -0.5 4.927 -0.5 4.927 -0.5 4.928 C -0.5 4.929 -0.5 4.93 -0.5 4.93 C -0.5 4.931 -0.5 4.932 -0.5 4.933 C -0.5 4.933 -0.5 4.934 -0.5 4.935 C -0.5 4.935 -0.5 4.936 -0.5 4.937 C -0.5 4.938 -0.5 4.938 -0.5 4.939 C -0.5 4.94 -0.5 4.941 -0.5 4.941 C -0.5 4.942 -0.5 4.943 -0.5 4.943 C -0.5 4.944 -0.5 4.945 -0.5 4.946 C -0.5 4.946 -0.5 4.947 -0.5 4.948 C -0.5 4.948 -0.5 4.949 -0.5 4.95 C -0.5 4.951 -0.5 4.951 -0.5 4.952 C -0.5 4.953 -0.5 4.954 -0.5 4.954 C -0.5 4.955 -0.5 4.956 -0.5 4.956 C -0.5 4.957 -0.5 4.958 -0.5 4.959 C -0.5 4.959 -0.5 4.96 -0.5 4.961 C -0.5 4.961 -0.5 4.962 -0.5 4.963 C -0.5 4.964 -0.5 4.964 -0.5 4.965 C -0.5 4.966 -0.5 4.966 -0.5 4.967 C -0.5 4.968 -0.5 4.969 -0.5 4.969 C -0.5 4.97 -0.5 4.971 -0.5 4.972 C -0.5 4.972 -0.5 4.973 -0.5 4.974 C -0.5 4.974 -0.5 4.975 -0.5 4.976 C -0.5 4.977 -0.5 4.977 -0.5 4.978 C -0.5 4.979 -0.5 4.979 -0.5 4.98 C -0.5 4.981 -0.5 4.982 -0.5 4.982 C -0.5 4.983 -0.5 4.984 -0.5 4.984 C -0.5 4.985 -0.5 4.986 -0.5 4.987 C -0.5 4.987 -0.5 4.988 -0.5 4.989 C -0.5 4.989 -0.5 4.99 -0.5 4.991 C -0.5 4.991 -0.5 4.992 -0.5 4.993 C -0.5 4.994 -0.5 4.994 -0.5 4.995 C -0.5 4.996 -0.5 4.996 -0.5 4.997 C -0.5 4.998 -0.5 4.999 -0.5 4.999 C -0.5 5 -0.5 5.001 -0.5 5.001 C -0.5 5.002 -0.5 5.003 -0.5 5.004 C -0.5 5.004 -0.5 5.005 -0.5 5.006 C -0.5 5.006 -0.5 5.007 -0.5 5.008 C -0.5 5.009 -0.5 5.009 -0.5 5.01 C -0.5 5.011 -0.5 5.011 -0.5 5.012 C -0.5 5.013 -0.5 5.013 -0.5 5.014 C -0.5 5.015 -0.5 5.016 -0.5 5.016 C -0.5 5.017 -0.5 5.018 -0.5 5.018 C -0.5 5.019 -0.5 5.02 -0.5 5.021 C -0.5 5.021 -0.5 5.022 -0.5 5.023 C -0.5 5.023 -0.5 5.024 -0.5 5.025 C -0.5 5.025 -0.5 5.026 -0.5 5.027 C -0.5 5.028 -0.5 5.028 -0.5 5.029 C -0.5 5.03 -0.5 5.03 -0.5 5.031 C -0.5 5.032 -0.5 5.032 -0.5 5.033 C -0.5 5.034 -0.5 5.035 -0.5 5.035 C -0.5 5.036 -0.5 5.037 -0.5 5.037 C -0.5 5.038 -0.5 5.039 -0.5 5.039 C -0.5 5.04 -0.5 5.041 -0.5 5.042 C -0.5 5.042 -0.5 5.043 -0.5 5.044 C -0.5 5.044 -0.5 5.045 -0.5 5.046 C -0.5 5.046 -0.5 5.047 -0.5 5.048 C -0.5 5.049 -0.5 5.049 -0.5 5.05 C -0.5 5.051 -0.5 5.051 -0.5 5.052 C -0.5 5.053 -0.5 5.053 -0.5 5.054 C -0.5 5.055 -0.5 5.055 -0.5 5.056 C -0.5 5.057 -0.5 5.058 -0.5 5.058 C -0.5 5.059 -0.5 5.06 -0.5 5.06 C -0.5 5.061 -0.5 5.062 -0.5 5.062 C -0.5 5.063 -0.5 5.064 -0.5 5.064 C -0.5 5.065 -0.5 5.066 -0.5 5.067 C -0.5 5.067 -0.5 5.068 -0.5 5.069 C -0.5 5.069 -0.5 5.07 -0.5 5.071 C -0.5 5.071 -0.5 5.072 -0.5 5.073 C -0.5 5.073 -0.5 5.074 -0.5 5.075 C -0.5 5.076 -0.5 5.076 -0.5 5.077 C -0.5 5.078 -0.5 5.078 -0.5 5.079 C -0.5 5.08 -0.5 5.08 -0.5 5.081 C -0.5 5.082 -0.5 5.082 -0.5 5.083 C -0.5 5.084 -0.5 5.084 -0.5 5.085 C -0.5 5.086 -0.5 5.087 -0.5 5.087 C -0.5 5.088 -0.5 5.089 -0.5 5.089 C -0.5 5.09 -0.5 5.091 -0.5 5.091 C -0.5 5.092 -0.5 5.093 -0.5 5.093 C -0.5 5.094 -0.5 5.095 -0.5 5.095 C -0.5 5.096 -0.5 5.097 -0.5 5.097 C -0.5 5.098 -0.5 5.099 -0.5 5.1 C -0.5 5.1 -0.5 5.101 -0.5 5.102 C -0.5 5.102 -0.5 5.103 -0.5 5.104 C -0.5 5.104 -0.5 5.105 -0.5 5.106 C -0.5 5.106 -0.5 5.107 -0.5 5.108 C -0.5 5.108 -0.5 5.109 -0.5 5.11 C -0.5 5.11 -0.5 5.111 -0.5 5.112 C -0.5 5.112 -0.5 5.113 -0.5 5.114 C -0.5 5.114 -0.5 5.115 -0.5 5.116 C -0.5 5.117 -0.5 5.117 -0.5 5.118 C -0.5 5.119 -0.5 5.119 -0.5 5.12 C -0.5 5.121 -0.5 5.121 -0.5 5.122 C -0.5 5.123 -0.5 5.123 -0.5 5.124 C -0.5 5.125 -0.5 5.125 -0.5 5.126 C -0.5 5.127 -0.5 5.127 -0.5 5.128 C -0.5 5.129 -0.5 5.129 -0.5 5.13 C -0.5 5.131 -0.5 5.131 -0.5 5.132 C -0.5 5.133 -0.5 5.133 -0.5 5.134 C -0.5 5.135 -0.5 5.135 -0.5 5.136 C -0.5 5.137 -0.5 5.137 -0.5 5.138 C -0.5 5.139 -0.5 5.139 -0.5 5.14 C -0.5 5.141 -0.5 5.141 -0.5 5.142 C -0.5 5.143 -0.5 5.143 -0.5 5.144 C -0.5 5.145 -0.5 5.145 -0.5 5.146 C -0.5 5.147 -0.5 5.147 -0.5 5.148 C -0.5 5.149 -0.5 5.149 -0.5 5.15 C -0.5 5.151 -0.5 5.151 -0.5 5.152 C -0.5 5.153 -0.5 5.153 -0.5 5.154 C -0.5 5.155 -0.5 5.155 -0.5 5.156 C -0.5 5.157 -0.5 5.157 -0.5 5.158 C -0.5 5.159 -0.5 5.159 -0.5 5.16 C -0.5 5.161 -0.5 5.161 -0.5 5.162 C -0.5 5.163 -0.5 5.163 -0.5 5.164 C -0.5 5.165 -0.5 5.165 -0.5 5.166 C -0.5 5.167 -0.5 5.167 -0.5 5.168 C -0.5 5.169 -0.5 5.169 -0.5 5.17 C -0.5 5.171 -0.5 5.171 -0.5 5.172 C -0.5 5.173 -0.5 5.173 -0.5 5.174 C -0.5 5.175 -0.5 5.175 -0.5 5.176 C -0.5 5.177 -0.5 5.177 -0.5 5.178 C -0.5 5.179 -0.5 5.179 -0.5 5.18 C -0.5 5.181 -0.5 5.181 -0.5 5.182 C -0.5 5.183 -0.5 5.183 -0.5 5.184 C -0.5 5.184 -0.5 5.185 -0.5 5.186 C -0.5 5.186 -0.5 5.187 -0.5 5.188 C -0.5 5.188 -0.5 5.189 -0.5 5.19 C -0.5 5.19 -0.5 5.191 -0.5 5.192 C -0.5 5.192 -0.5 5.193 -0.5 5.194 C -0.5 5.194 -0.5 5.195 -0.5 5.196 C -0.5 5.196 -0.5 5.197 -0.5 5.198 C -0.5 5.198 -0.5 5.199 -0.5 5.199 C -0.5 5.2 -0.5 5.201 -0.5 5.201 C -0.5 5.202 -0.5 5.203 -0.5 5.203 C -0.5 5.204 -0.5 5.205 -0.5 5.205 C -0.5 5.206 -0.5 5.207 -0.5 5.207 C -0.5 5.208 -0.5 5.209 -0.5 5.209 C -0.5 5.21 -0.5 5.211 -0.5 5.211 C -0.5 5.212 -0.5 5.212 -0.5 5.213 C -0.5 5.214 -0.5 5.214 -0.5 5.215 C -0.5 5.216 -0.5 5.216 -0.5 5.217 C -0.5 5.218 -0.5 5.218 -0.5 5.219 C -0.5 5.22 -0.5 5.22 -0.5 5.221 C -0.5 5.221 -0.5 5.222 -0.5 5.223 C -0.5 5.223 -0.5 5.224 -0.5 5.225 C -0.5 5.225 -0.5 5.226 -0.5 5.227 C -0.5 5.227 -0.5 5.228 -0.5 5.229 C -0.5 5.229 -0.5 5.23 -0.5 5.23 C -0.5 5.231 -0.5 5.232 -0.5 5.232 C -0.5 5.233 -0.5 5.234 -0.5 5.234 C -0.5 5.235 -0.5 5.236 -0.5 5.236 C -0.5 5.237 -0.5 5.237 -0.5 5.238 C -0.5 5.239 -0.5 5.239 -0.5 5.24 C -0.5 5.241 -0.5 5.241 -0.5 5.242 C -0.5 5.243 -0.5 5.243 -0.5 5.244 C -0.5 5.244 -0.5 5.245 -0.5 5.246 C -0.5 5.246 -0.5 5.247 -0.5 5.248 C -0.5 5.248 -0.5 5.249 -0.5 5.25 C -0.5 5.25 -0.5 5.251 -0.5 5.251 C -0.5 5.252 -0.5 5.253 -0.5 5.253 C -0.5 5.254 -0.5 5.255 -0.5 5.255 C -0.5 5.256 -0.5 5.257 -0.5 5.257 C -0.5 5.258 -0.5 5.258 -0.5 5.259 C -0.5 5.26 -0.5 5.26 -0.5 5.261 C -0.5 5.262 -0.5 5.262 -0.5 5.263 C -0.5 5.263 -0.5 5.264 -0.5 5.265 C -0.5 5.265 -0.5 5.266 -0.5 5.267 C -0.5 5.267 -0.5 5.268 -0.5 5.268 C -0.5 5.269 -0.5 5.27 -0.5 5.27 C -0.5 5.271 -0.5 5.272 -0.5 5.272 C -0.5 5.273 -0.5 5.273 -0.5 5.274 C -0.5 5.275 -0.5 5.275 -0.5 5.276 C -0.5 5.277 -0.5 5.277 -0.5 5.278 C -0.5 5.278 -0.5 5.279 -0.5 5.28 C -0.5 5.28 -0.5 5.281 -0.5 5.282 C -0.5 5.282 -0.5 5.283 -0.5 5.283 C -0.5 5.284 -0.5 5.285 -0.5 5.285 C -0.5 5.286 -0.5 5.287 -0.5 5.287 C -0.5 5.288 -0.5 5.288 -0.5 5.289 C -0.5 5.29 -0.5 5.29 -0.5 5.291 C -0.5 5.291 -0.5 5.292 -0.5 5.293 C -0.5 5.293 -0.5 5.294 -0.5 5.295 C -0.5 5.295 -0.5 5.296 -0.5 5.296 C -0.5 5.297 -0.5 5.298 -0.5 5.298 C -0.5 5.299 -0.5 5.299 -0.5 5.3 C -0.5 5.301 -0.5 5.301 -0.5 5.302 C -0.5 5.303 -0.5 5.303 -0.5 5.304 C -0.5 5.304 -0.5 5.305 -0.5 5.306 C -0.5 5.306 -0.5 5.307 -0.5 5.307 C -0.5 5.308 -0.5 5.309 -0.5 5.309 C -0.5 5.31 -0.5 5.31 -0.5 5.311 C -0.5 5.312 -0.5 5.312 -0.5 5.313 C -0.5 5.314 -0.5 5.314 -0.5 5.315 C -0.5 5.315 -0.5 5.316 -0.5 5.317 C -0.5 5.317 -0.5 5.318 -0.5 5.318 C -0.5 5.319 -0.5 5.32 -0.5 5.32 C -0.5 5.321 -0.5 5.321 -0.5 5.322 C -0.5 5.323 -0.5 5.323 -0.5 5.324 C -0.5 5.324 -0.5 5.325 -0.5 5.326 C -0.5 5.326 -0.5 5.327 -0.5 5.327 C -0.5 5.328 -0.5 5.329 -0.5 5.329 C -0.5 5.33 -0.5 5.33 -0.5 5.331 C -0.5 5.332 -0.5 5.332 -0.5 5.333 C -0.5 5.333 -0.5 5.334 -0.5 5.335 C -0.5 5.335 -0.5 5.336 -0.5 5.336 C -0.5 5.337 -0.5 5.338 -0.5 5.338 C -0.5 5.339 -0.5 5.339 -0.5 5.34 C -0.5 5.341 -0.5 5.341 -0.5 5.342 C -0.5 5.342 -0.5 5.343 -0.5 5.344 C -0.5 5.344 -0.5 5.345 -0.5 5.345 C -0.5 5.346 -0.5 5.347 -0.5 5.347 C -0.5 5.348 -0.5 5.348 -0.5 5.349 C -0.5 5.35 -0.5 5.35 -0.5 5.351 C -0.5 5.351 -0.5 5.352 -0.5 5.353 C -0.5 5.353 -0.5 5.354 -0.5 5.354 C -0.5 5.355 -0.5 5.356 -0.5 5.356 C -0.5 5.357 -0.5 5.357 -0.5 5.358 C -0.5 5.359 -0.5 5.359 -0.5 5.36 C -0.5 5.36 -0.5 5.361 -0.5 5.361 C -0.5 5.362 -0.5 5.363 -0.5 5.363 C -0.5 5.364 -0.5 5.364 -0.5 5.365 C -0.5 5.366 -0.5 5.366 -0.5 5.367 C -0.5 5.367 -0.5 5.368 -0.5 5.369 C -0.5 5.369 -0.5 5.37 -0.5 5.37 C -0.5 5.371 -0.5 5.371 -0.5 5.372 C -0.5 5.373 -0.5 5.373 -0.5 5.374 C -0.5 5.374 -0.5 5.375 -0.5 5.376 C -0.5 5.376 -0.5 5.377 -0.5 5.377 C -0.5 5.378 -0.5 5.378 -0.5 5.379 C -0.5 5.38 -0.5 5.38 -0.5 5.381 C -0.5 5.381 -0.5 5.382 -0.5 5.383 C -0.5 5.383 -0.5 5.384 -0.5 5.384 C -0.5 5.385 -0.5 5.385 -0.5 5.386 C -0.5 5.387 -0.5 5.387 -0.5 5.388 C -0.5 5.388 -0.5 5.389 -0.5 5.389 C -0.5 5.39 -0.5 5.391 -0.5 5.391 C -0.5 5.392 -0.5 5.392 -0.5 5.393 C -0.5 5.393 -0.5 5.394 -0.5 5.395 C -0.5 5.395 -0.5 5.396 -0.5 5.396 C -0.5 5.397 -0.5 5.397 -0.5 5.398 C -0.5 5.399 -0.5 5.399 -0.5 5.4 C -0.5 5.4 -0.5 5.401 -0.5 5.401 C -0.5 5.402 -0.5 5.403 -0.5 5.403 C -0.5 5.404 -0.5 5.404 -0.5 5.405 C -0.5 5.405 -0.5 5.406 -0.5 5.407 C -0.5 5.407 -0.5 5.408 -0.5 5.408 C -0.5 5.409 -0.5 5.409 -0.5 5.41 C -0.5 5.411 -0.5 5.411 -0.5 5.412 C -0.5 5.412 -0.5 5.413 -0.5 5.413 C -0.5 5.414 -0.5 5.415 -0.5 5.415 C -0.5 5.416 -0.5 5.416 -0.5 5.417 C -0.5 5.417 -0.5 5.418 -0.5 5.419 C -0.5 5.419 -0.5 5.42 -0.5 5.42 C -0.5 5.421 -0.5 5.421 -0.5 5.422 C -0.5 5.422 -0.5 5.423 -0.5 5.424 C -0.5 5.424 -0.5 5.425 -0.5 5.425 C -0.5 5.426 -0.5 5.426 -0.5 5.427 C -0.5 5.428 -0.5 5.428 -0.5 5.429 C -0.5 5.429 -0.5 5.43 -0.5 5.43 C -0.5 5.431 -0.5 5.431 -0.5 5.432 C -0.5 5.433 -0.5 5.433 -0.5 5.434 C -0.5 5.434 -0.5 5.435 -0.5 5.435 C -0.5 5.436 -0.5 5.436 -0.5 5.437 C -0.5 5.438 -0.5 5.438 -0.5 5.439 C -0.5 5.439 -0.5 5.44 -0.5 5.44 C -0.5 5.441 -0.5 5.441 -0.5 5.442 C -0.5 5.443 -0.5 5.443 -0.5 5.444 C -0.5 5.444 -0.5 5.445 -0.5 5.445 C -0.5 5.446 -0.5 5.446 -0.5 5.447 C -0.5 5.447 -0.5 5.448 -0.5 5.449 C -0.5 5.449 -0.5 5.45 -0.5 5.45 C -0.5 5.451 -0.5 5.451 -0.5 5.452 C -0.5 5.452 -0.5 5.453 -0.5 5.453 C -0.5 5.454 -0.5 5.455 -0.5 5.455 C -0.5 5.456 -0.5 5.456 -0.5 5.457 C -0.5 5.457 -0.5 5.458 -0.5 5.458 C -0.5 5.459 -0.5 5.459 -0.5 5.46 C -0.5 5.461 -0.5 5.461 -0.5 5.462 C -0.5 5.462 -0.5 5.463 -0.5 5.463 C -0.5 5.464 -0.5 5.464 -0.5 5.465 C -0.5 5.465 -0.5 5.466 -0.5 5.467 C -0.5 5.467 -0.5 5.468 -0.5 5.468 C -0.5 5.469 -0.5 5.469 -0.5 5.47 C -0.5 5.47 -0.5 5.471 -0.5 5.471 C -0.5 5.472 -0.5 5.472 -0.5 5.473 C -0.5 5.474 -0.5 5.474 -0.5 5.475 C -0.5 5.475 -0.5 5.476 -0.5 5.476 C -0.5 5.477 -0.5 5.477 -0.5 5.478 C -0.5 5.478 -0.5 5.479 -0.5 5.479 C -0.5 5.48 -0.5 5.48 -0.5 5.481 C -0.5 5.482 -0.5 5.482 -0.5 5.483 C -0.5 5.483 -0.5 5.484 -0.5 5.484 C -0.5 5.485 -0.5 5.485 -0.5 5.486 C -0.5 5.486 -0.5 5.487 -0.5 5.487 C -0.5 5.488 -0.5 5.488 -0.5 5.489 C -0.5 5.489 -0.5 5.49 -0.5 5.491 C -0.5 5.491 -0.5 5.492 -0.5 5.492 C -0.5 5.493 -0.5 5.493 -0.5 5.494 C -0.5 5.494 -0.5 5.495 -0.5 5.495 C -0.5 5.496 -0.5 5.496 -0.5 5.497 C -0.5 5.497 -0.5 5.498 -0.5 5.498 C -0.5 5.499 -0.5 5.499 -0.5 5.5 C -0.5 5.501 -0.5 5.501 -0.5 5.502 C -0.5 5.502 -0.5 5.503 -0.5 5.503 C -0.5 5.504 -0.5 5.504 -0.5 5.505 C -0.5 5.505 -0.5 5.506 -0.5 5.506 C -0.5 5.507 -0.5 5.507 -0.5 5.508 C -0.5 5.508 -0.5 5.509 -0.5 5.509 C -0.5 5.51 -0.5 5.51 -0.5 5.511 C -0.5 5.511 -0.5 5.512 -0.5 5.512 C -0.5 5.513 -0.5 5.513 -0.5 5.514 C -0.5 5.515 -0.5 5.515 -0.5 5.516 C -0.5 5.516 -0.5 5.517 -0.5 5.517 C -0.5 5.518 -0.5 5.518 -0.5 5.519 C -0.5 5.519 -0.5 5.52 -0.5 5.52 C -0.5 5.521 -0.5 5.521 -0.5 5.522 C -0.5 5.522 -0.5 5.523 -0.5 5.523 C -0.5 5.524 -0.5 5.524 -0.5 5.525 C -0.5 5.525 -0.5 5.526 -0.5 5.526 C -0.5 5.527 -0.5 5.527 -0.5 5.528 C -0.5 5.528 -0.5 5.529 -0.5 5.529 C -0.5 5.53 -0.5 5.53 -0.5 5.531 C -0.5 5.531 -0.5 5.532 -0.5 5.532 C -0.5 5.533 -0.5 5.533 -0.5 5.534 C -0.5 5.534 -0.5 5.535 -0.5 5.535 C -0.5 5.536 -0.5 5.536 -0.5 5.537 C -0.5 5.537 -0.5 5.538 -0.5 5.538 L 0.5 5.538 Z M 3.487 9.146 L 0.939 6.598 L 0.232 7.305 L 2.78 9.854 L 3.487 9.146 Z M -0.5 5.538 C -0.5 6.202 -0.236 6.837 0.232 7.305 L 0.939 6.598 C 0.658 6.317 0.5 5.936 0.5 5.538 L -0.5 5.538 Z M 2.5 4 L 2.5 1.5 L 1.5 1.5 L 1.5 4 L 2.5 4 Z M 1.5 2.5 L 1.5 4 L 2.5 4 L 2.5 2.5 L 1.5 2.5 Z M 7.5 1.5 L 7.5 2 L 8.5 2 L 8.5 1.5 L 7.5 1.5 Z M 7 9 L 3.134 9 L 3.134 10 L 7 10 L 7 9 Z M 7.5 2 L 7.5 2.5 L 8.5 2.5 L 8.5 2 L 7.5 2 Z M 0.5 4.5 C 0.5 3.672 1.172 3 2 3 L 2 2 C 0.619 2 -0.5 3.119 -0.5 4.5 L 0.5 4.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "cursor-hand-open": "<g transform=\"translate(2.549,1.5)\"><path d=\"M 6.951 1 L 7.451 1 L 7.451 1 L 6.951 1 Z M 6.951 2 L 7.451 2 L 7.451 2 L 6.951 2 Z M 0.292 7.293 L 0.645 7.646 L 0.645 7.646 L 0.292 7.293 Z M 1.716 7.189 L 1.505 7.643 L 1.505 7.643 L 1.716 7.189 Z M 2.951 7.818 L 2.721 8.262 L 3.451 8.64 L 3.451 7.818 L 2.951 7.818 Z M 4.085 13 L 3.711 13.332 L 3.86 13.5 L 4.085 13.5 L 4.085 13 Z M 5.951 -0.5 C 5.122 -0.5 4.451 0.172 4.451 1 L 5.451 1 C 5.451 0.724 5.675 0.5 5.951 0.5 L 5.951 -0.5 Z M 7.451 1 C 7.451 0.172 6.779 -0.5 5.951 -0.5 L 5.951 0.5 C 6.227 0.5 6.451 0.724 6.451 1 L 7.451 1 Z M 7.451 6 L 7.451 1 L 6.451 1 L 6.451 6 L 7.451 6 Z M 6.451 2 L 6.451 6 L 7.451 6 L 7.451 2 L 6.451 2 Z M 7.951 0.5 C 7.122 0.5 6.451 1.172 6.451 2 L 7.451 2 C 7.451 1.724 7.675 1.5 7.951 1.5 L 7.951 0.5 Z M 9.451 2 C 9.451 1.172 8.779 0.5 7.951 0.5 L 7.951 1.5 C 8.227 1.5 8.451 1.724 8.451 2 L 9.451 2 Z M 9.451 6 L 9.451 2 L 8.451 2 L 8.451 6 L 9.451 6 Z M 8.451 3.5 L 8.451 6 L 9.451 6 L 9.451 3.5 L 8.451 3.5 Z M 9.951 2 C 9.122 2 8.451 2.672 8.451 3.5 L 9.451 3.5 C 9.451 3.224 9.675 3 9.951 3 L 9.951 2 Z M 11.451 3.5 C 11.451 2.672 10.779 2 9.951 2 L 9.951 3 C 10.227 3 10.451 3.224 10.451 3.5 L 11.451 3.5 Z M 11.451 10 L 11.451 3.5 L 10.451 3.5 L 10.451 10 L 11.451 10 Z M 7.951 13.5 C 9.884 13.5 11.451 11.933 11.451 10 L 10.451 10 C 10.451 11.381 9.332 12.5 7.951 12.5 L 7.951 13.5 Z M -0.062 6.939 C -0.662 7.539 -0.622 8.412 -0.087 9.033 L 0.671 8.381 C 0.43 8.102 0.464 7.828 0.645 7.646 L -0.062 6.939 Z M 1.926 6.736 C 1.332 6.46 0.552 6.326 -0.062 6.939 L 0.645 7.646 C 0.813 7.479 1.056 7.434 1.505 7.643 L 1.926 6.736 Z M 3.181 7.374 C 2.561 7.054 2.075 6.805 1.926 6.736 L 1.505 7.643 C 1.629 7.7 2.089 7.936 2.721 8.262 L 3.181 7.374 Z M 2.451 2.5 L 2.451 7.818 L 3.451 7.818 L 3.451 2.5 L 2.451 2.5 Z M 3.951 1 C 3.122 1 2.451 1.672 2.451 2.5 L 3.451 2.5 C 3.451 2.224 3.675 2 3.951 2 L 3.951 1 Z M 5.451 2.5 C 5.451 1.672 4.779 1 3.951 1 L 3.951 2 C 4.227 2 4.451 2.224 4.451 2.5 L 5.451 2.5 Z M 5.451 6 L 5.451 2.5 L 4.451 2.5 L 4.451 6 L 5.451 6 Z M 4.451 1 L 4.451 6 L 5.451 6 L 5.451 1 L 4.451 1 Z M -0.087 9.033 C 0.11 9.262 1.061 10.339 1.959 11.354 C 2.409 11.862 2.847 12.357 3.172 12.724 C 3.335 12.908 3.47 13.06 3.563 13.166 C 3.61 13.219 3.647 13.26 3.672 13.288 C 3.685 13.302 3.694 13.313 3.701 13.321 C 3.704 13.324 3.706 13.327 3.708 13.329 C 3.709 13.33 3.71 13.33 3.71 13.331 C 3.71 13.331 3.71 13.331 3.71 13.331 C 3.71 13.331 3.71 13.332 3.711 13.332 C 3.711 13.332 3.711 13.332 4.085 13 C 4.459 12.668 4.459 12.668 4.459 12.668 C 4.459 12.668 4.459 12.668 4.459 12.668 C 4.459 12.668 4.459 12.668 4.458 12.668 C 4.458 12.667 4.457 12.667 4.457 12.666 C 4.455 12.664 4.452 12.661 4.449 12.657 C 4.443 12.65 4.433 12.639 4.421 12.625 C 4.396 12.597 4.359 12.555 4.312 12.502 C 4.218 12.397 4.084 12.245 3.921 12.061 C 3.596 11.694 3.158 11.199 2.708 10.691 C 1.807 9.672 0.862 8.603 0.671 8.381 L -0.087 9.033 Z M 4.085 13.5 L 7.951 13.5 L 7.951 12.5 L 4.085 12.5 L 4.085 13.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "cursor-hand-point": "<g transform=\"translate(3.5,2.5)\"><path d=\"M 4 3.5 L 3.5 3.5 L 3.5 3.5 L 4 3.5 Z M 4 1 L 3.5 1 L 3.5 1 L 4 1 Z M 3.134 12 L 2.78 12.354 L 2.927 12.5 L 3.134 12.5 L 3.134 12 Z M 6 4 L 6.5 4 L 6.5 4 L 6 4 Z M 6 3.5 L 6.5 3.5 L 6.5 3.5 L 6 3.5 Z M 0.586 9.452 L 0.939 9.098 L 0.939 9.098 L 0.586 9.452 Z M 2 5 L 2.5 5 L 2.5 4.5 L 2 4.5 L 2 5 Z M 4.5 3.5 C 4.5 3.224 4.724 3 5 3 L 5 2 C 4.172 2 3.5 2.672 3.5 3.5 L 4.5 3.5 Z M 4.5 4 L 4.5 3.5 L 3.5 3.5 L 3.5 4 L 4.5 4 Z M 3.5 1 L 3.5 4 L 4.5 4 L 4.5 1 L 3.5 1 Z M 3 0.5 C 3.276 0.5 3.5 0.724 3.5 1 L 4.5 1 C 4.5 0.172 3.828 -0.5 3 -0.5 L 3 0.5 Z M 2.5 1 C 2.5 0.724 2.724 0.5 3 0.5 L 3 -0.5 C 2.172 -0.5 1.5 0.172 1.5 1 L 2.5 1 Z M 9.5 9 C 9.5 10.381 8.381 11.5 7 11.5 L 7 12.5 C 8.933 12.5 10.5 10.933 10.5 9 L 9.5 9 Z M 9.5 4.5 L 9.5 9 L 10.5 9 L 10.5 4.5 L 9.5 4.5 Z M 9 4 C 9.276 4 9.5 4.224 9.5 4.5 L 10.5 4.5 C 10.5 3.672 9.828 3 9 3 L 9 4 Z M 8.5 4.5 C 8.5 4.224 8.724 4 9 4 L 9 3 C 8.172 3 7.5 3.672 7.5 4.5 L 8.5 4.5 Z M 7 3.5 C 7.276 3.5 7.5 3.724 7.5 4 L 8.5 4 C 8.5 3.172 7.828 2.5 7 2.5 L 7 3.5 Z M 6.5 4 C 6.5 3.724 6.724 3.5 7 3.5 L 7 2.5 C 6.172 2.5 5.5 3.172 5.5 4 L 6.5 4 Z M 6.5 4.5 L 6.5 4 L 5.5 4 L 5.5 4.5 L 6.5 4.5 Z M 5.5 3.5 L 5.5 4.5 L 6.5 4.5 L 6.5 3.5 L 5.5 3.5 Z M 5 3 C 5.276 3 5.5 3.224 5.5 3.5 L 6.5 3.5 C 6.5 2.672 5.828 2 5 2 L 5 3 Z M 0.5 8.038 C 0.5 8.038 0.5 8.037 0.5 8.037 C 0.5 8.036 0.5 8.036 0.5 8.035 C 0.5 8.035 0.5 8.034 0.5 8.034 C 0.5 8.033 0.5 8.033 0.5 8.032 C 0.5 8.032 0.5 8.031 0.5 8.031 C 0.5 8.03 0.5 8.03 0.5 8.029 C 0.5 8.029 0.5 8.028 0.5 8.028 C 0.5 8.027 0.5 8.027 0.5 8.026 C 0.5 8.026 0.5 8.025 0.5 8.025 C 0.5 8.024 0.5 8.024 0.5 8.023 C 0.5 8.023 0.5 8.022 0.5 8.022 C 0.5 8.021 0.5 8.021 0.5 8.02 C 0.5 8.02 0.5 8.019 0.5 8.019 C 0.5 8.018 0.5 8.018 0.5 8.017 C 0.5 8.017 0.5 8.016 0.5 8.016 C 0.5 8.015 0.5 8.015 0.5 8.014 C 0.5 8.013 0.5 8.013 0.5 8.012 C 0.5 8.012 0.5 8.011 0.5 8.011 C 0.5 8.01 0.5 8.01 0.5 8.009 C 0.5 8.009 0.5 8.008 0.5 8.008 C 0.5 8.007 0.5 8.007 0.5 8.006 C 0.5 8.006 0.5 8.005 0.5 8.005 C 0.5 8.004 0.5 8.004 0.5 8.003 C 0.5 8.003 0.5 8.002 0.5 8.002 C 0.5 8.001 0.5 8.001 0.5 8 C 0.5 7.999 0.5 7.999 0.5 7.998 C 0.5 7.998 0.5 7.997 0.5 7.997 C 0.5 7.996 0.5 7.996 0.5 7.995 C 0.5 7.995 0.5 7.994 0.5 7.994 C 0.5 7.993 0.5 7.993 0.5 7.992 C 0.5 7.992 0.5 7.991 0.5 7.991 C 0.5 7.99 0.5 7.989 0.5 7.989 C 0.5 7.988 0.5 7.988 0.5 7.987 C 0.5 7.987 0.5 7.986 0.5 7.986 C 0.5 7.985 0.5 7.985 0.5 7.984 C 0.5 7.984 0.5 7.983 0.5 7.983 C 0.5 7.982 0.5 7.982 0.5 7.981 C 0.5 7.98 0.5 7.98 0.5 7.979 C 0.5 7.979 0.5 7.978 0.5 7.978 C 0.5 7.977 0.5 7.977 0.5 7.976 C 0.5 7.976 0.5 7.975 0.5 7.975 C 0.5 7.974 0.5 7.974 0.5 7.973 C 0.5 7.972 0.5 7.972 0.5 7.971 C 0.5 7.971 0.5 7.97 0.5 7.97 C 0.5 7.969 0.5 7.969 0.5 7.968 C 0.5 7.968 0.5 7.967 0.5 7.967 C 0.5 7.966 0.5 7.965 0.5 7.965 C 0.5 7.964 0.5 7.964 0.5 7.963 C 0.5 7.963 0.5 7.962 0.5 7.962 C 0.5 7.961 0.5 7.961 0.5 7.96 C 0.5 7.959 0.5 7.959 0.5 7.958 C 0.5 7.958 0.5 7.957 0.5 7.957 C 0.5 7.956 0.5 7.956 0.5 7.955 C 0.5 7.955 0.5 7.954 0.5 7.953 C 0.5 7.953 0.5 7.952 0.5 7.952 C 0.5 7.951 0.5 7.951 0.5 7.95 C 0.5 7.95 0.5 7.949 0.5 7.949 C 0.5 7.948 0.5 7.947 0.5 7.947 C 0.5 7.946 0.5 7.946 0.5 7.945 C 0.5 7.945 0.5 7.944 0.5 7.944 C 0.5 7.943 0.5 7.943 0.5 7.942 C 0.5 7.941 0.5 7.941 0.5 7.94 C 0.5 7.94 0.5 7.939 0.5 7.939 C 0.5 7.938 0.5 7.938 0.5 7.937 C 0.5 7.936 0.5 7.936 0.5 7.935 C 0.5 7.935 0.5 7.934 0.5 7.934 C 0.5 7.933 0.5 7.933 0.5 7.932 C 0.5 7.931 0.5 7.931 0.5 7.93 C 0.5 7.93 0.5 7.929 0.5 7.929 C 0.5 7.928 0.5 7.928 0.5 7.927 C 0.5 7.926 0.5 7.926 0.5 7.925 C 0.5 7.925 0.5 7.924 0.5 7.924 C 0.5 7.923 0.5 7.922 0.5 7.922 C 0.5 7.921 0.5 7.921 0.5 7.92 C 0.5 7.92 0.5 7.919 0.5 7.919 C 0.5 7.918 0.5 7.917 0.5 7.917 C 0.5 7.916 0.5 7.916 0.5 7.915 C 0.5 7.915 0.5 7.914 0.5 7.913 C 0.5 7.913 0.5 7.912 0.5 7.912 C 0.5 7.911 0.5 7.911 0.5 7.91 C 0.5 7.909 0.5 7.909 0.5 7.908 C 0.5 7.908 0.5 7.907 0.5 7.907 C 0.5 7.906 0.5 7.905 0.5 7.905 C 0.5 7.904 0.5 7.904 0.5 7.903 C 0.5 7.903 0.5 7.902 0.5 7.901 C 0.5 7.901 0.5 7.9 0.5 7.9 C 0.5 7.899 0.5 7.899 0.5 7.898 C 0.5 7.897 0.5 7.897 0.5 7.896 C 0.5 7.896 0.5 7.895 0.5 7.895 C 0.5 7.894 0.5 7.893 0.5 7.893 C 0.5 7.892 0.5 7.892 0.5 7.891 C 0.5 7.891 0.5 7.89 0.5 7.889 C 0.5 7.889 0.5 7.888 0.5 7.888 C 0.5 7.887 0.5 7.887 0.5 7.886 C 0.5 7.885 0.5 7.885 0.5 7.884 C 0.5 7.884 0.5 7.883 0.5 7.883 C 0.5 7.882 0.5 7.881 0.5 7.881 C 0.5 7.88 0.5 7.88 0.5 7.879 C 0.5 7.878 0.5 7.878 0.5 7.877 C 0.5 7.877 0.5 7.876 0.5 7.876 C 0.5 7.875 0.5 7.874 0.5 7.874 C 0.5 7.873 0.5 7.873 0.5 7.872 C 0.5 7.871 0.5 7.871 0.5 7.87 C 0.5 7.87 0.5 7.869 0.5 7.869 C 0.5 7.868 0.5 7.867 0.5 7.867 C 0.5 7.866 0.5 7.866 0.5 7.865 C 0.5 7.864 0.5 7.864 0.5 7.863 C 0.5 7.863 0.5 7.862 0.5 7.861 C 0.5 7.861 0.5 7.86 0.5 7.86 C 0.5 7.859 0.5 7.859 0.5 7.858 C 0.5 7.857 0.5 7.857 0.5 7.856 C 0.5 7.856 0.5 7.855 0.5 7.854 C 0.5 7.854 0.5 7.853 0.5 7.853 C 0.5 7.852 0.5 7.851 0.5 7.851 C 0.5 7.85 0.5 7.85 0.5 7.849 C 0.5 7.848 0.5 7.848 0.5 7.847 C 0.5 7.847 0.5 7.846 0.5 7.845 C 0.5 7.845 0.5 7.844 0.5 7.844 C 0.5 7.843 0.5 7.842 0.5 7.842 C 0.5 7.841 0.5 7.841 0.5 7.84 C 0.5 7.839 0.5 7.839 0.5 7.838 C 0.5 7.838 0.5 7.837 0.5 7.836 C 0.5 7.836 0.5 7.835 0.5 7.835 C 0.5 7.834 0.5 7.833 0.5 7.833 C 0.5 7.832 0.5 7.832 0.5 7.831 C 0.5 7.83 0.5 7.83 0.5 7.829 C 0.5 7.829 0.5 7.828 0.5 7.827 C 0.5 7.827 0.5 7.826 0.5 7.826 C 0.5 7.825 0.5 7.824 0.5 7.824 C 0.5 7.823 0.5 7.823 0.5 7.822 C 0.5 7.821 0.5 7.821 0.5 7.82 C 0.5 7.82 0.5 7.819 0.5 7.818 C 0.5 7.818 0.5 7.817 0.5 7.817 C 0.5 7.816 0.5 7.815 0.5 7.815 C 0.5 7.814 0.5 7.814 0.5 7.813 C 0.5 7.812 0.5 7.812 0.5 7.811 C 0.5 7.81 0.5 7.81 0.5 7.809 C 0.5 7.809 0.5 7.808 0.5 7.807 C 0.5 7.807 0.5 7.806 0.5 7.806 C 0.5 7.805 0.5 7.804 0.5 7.804 C 0.5 7.803 0.5 7.803 0.5 7.802 C 0.5 7.801 0.5 7.801 0.5 7.8 C 0.5 7.799 0.5 7.799 0.5 7.798 C 0.5 7.798 0.5 7.797 0.5 7.796 C 0.5 7.796 0.5 7.795 0.5 7.795 C 0.5 7.794 0.5 7.793 0.5 7.793 C 0.5 7.792 0.5 7.791 0.5 7.791 C 0.5 7.79 0.5 7.79 0.5 7.789 C 0.5 7.788 0.5 7.788 0.5 7.787 C 0.5 7.787 0.5 7.786 0.5 7.785 C 0.5 7.785 0.5 7.784 0.5 7.783 C 0.5 7.783 0.5 7.782 0.5 7.782 C 0.5 7.781 0.5 7.78 0.5 7.78 C 0.5 7.779 0.5 7.778 0.5 7.778 C 0.5 7.777 0.5 7.777 0.5 7.776 C 0.5 7.775 0.5 7.775 0.5 7.774 C 0.5 7.773 0.5 7.773 0.5 7.772 C 0.5 7.772 0.5 7.771 0.5 7.77 C 0.5 7.77 0.5 7.769 0.5 7.768 C 0.5 7.768 0.5 7.767 0.5 7.767 C 0.5 7.766 0.5 7.765 0.5 7.765 C 0.5 7.764 0.5 7.763 0.5 7.763 C 0.5 7.762 0.5 7.762 0.5 7.761 C 0.5 7.76 0.5 7.76 0.5 7.759 C 0.5 7.758 0.5 7.758 0.5 7.757 C 0.5 7.757 0.5 7.756 0.5 7.755 C 0.5 7.755 0.5 7.754 0.5 7.753 C 0.5 7.753 0.5 7.752 0.5 7.751 C 0.5 7.751 0.5 7.75 0.5 7.75 C 0.5 7.749 0.5 7.748 0.5 7.748 C 0.5 7.747 0.5 7.746 0.5 7.746 C 0.5 7.745 0.5 7.744 0.5 7.744 C 0.5 7.743 0.5 7.743 0.5 7.742 C 0.5 7.741 0.5 7.741 0.5 7.74 C 0.5 7.739 0.5 7.739 0.5 7.738 C 0.5 7.737 0.5 7.737 0.5 7.736 C 0.5 7.736 0.5 7.735 0.5 7.734 C 0.5 7.734 0.5 7.733 0.5 7.732 C 0.5 7.732 0.5 7.731 0.5 7.73 C 0.5 7.73 0.5 7.729 0.5 7.729 C 0.5 7.728 0.5 7.727 0.5 7.727 C 0.5 7.726 0.5 7.725 0.5 7.725 C 0.5 7.724 0.5 7.723 0.5 7.723 C 0.5 7.722 0.5 7.721 0.5 7.721 C 0.5 7.72 0.5 7.72 0.5 7.719 C 0.5 7.718 0.5 7.718 0.5 7.717 C 0.5 7.716 0.5 7.716 0.5 7.715 C 0.5 7.714 0.5 7.714 0.5 7.713 C 0.5 7.712 0.5 7.712 0.5 7.711 C 0.5 7.711 0.5 7.71 0.5 7.709 C 0.5 7.709 0.5 7.708 0.5 7.707 C 0.5 7.707 0.5 7.706 0.5 7.705 C 0.5 7.705 0.5 7.704 0.5 7.703 C 0.5 7.703 0.5 7.702 0.5 7.701 C 0.5 7.701 0.5 7.7 0.5 7.699 C 0.5 7.699 0.5 7.698 0.5 7.698 C 0.5 7.697 0.5 7.696 0.5 7.696 C 0.5 7.695 0.5 7.694 0.5 7.694 C 0.5 7.693 0.5 7.692 0.5 7.692 C 0.5 7.691 0.5 7.69 0.5 7.69 C 0.5 7.689 0.5 7.688 0.5 7.688 C 0.5 7.687 0.5 7.686 0.5 7.686 C 0.5 7.685 0.5 7.684 0.5 7.684 C 0.5 7.683 0.5 7.683 0.5 7.682 C 0.5 7.681 0.5 7.681 0.5 7.68 C 0.5 7.679 0.5 7.679 0.5 7.678 C 0.5 7.677 0.5 7.677 0.5 7.676 C 0.5 7.675 0.5 7.675 0.5 7.674 C 0.5 7.673 0.5 7.673 0.5 7.672 C 0.5 7.671 0.5 7.671 0.5 7.67 C 0.5 7.669 0.5 7.669 0.5 7.668 C 0.5 7.667 0.5 7.667 0.5 7.666 C 0.5 7.665 0.5 7.665 0.5 7.664 C 0.5 7.663 0.5 7.663 0.5 7.662 C 0.5 7.661 0.5 7.661 0.5 7.66 C 0.5 7.659 0.5 7.659 0.5 7.658 C 0.5 7.657 0.5 7.657 0.5 7.656 C 0.5 7.655 0.5 7.655 0.5 7.654 C 0.5 7.653 0.5 7.653 0.5 7.652 C 0.5 7.651 0.5 7.651 0.5 7.65 C 0.5 7.649 0.5 7.649 0.5 7.648 C 0.5 7.647 0.5 7.647 0.5 7.646 C 0.5 7.645 0.5 7.645 0.5 7.644 C 0.5 7.643 0.5 7.643 0.5 7.642 C 0.5 7.641 0.5 7.641 0.5 7.64 C 0.5 7.639 0.5 7.639 0.5 7.638 C 0.5 7.637 0.5 7.637 0.5 7.636 C 0.5 7.635 0.5 7.635 0.5 7.634 C 0.5 7.633 0.5 7.633 0.5 7.632 C 0.5 7.631 0.5 7.631 0.5 7.63 C 0.5 7.629 0.5 7.629 0.5 7.628 C 0.5 7.627 0.5 7.627 0.5 7.626 C 0.5 7.625 0.5 7.625 0.5 7.624 C 0.5 7.623 0.5 7.623 0.5 7.622 C 0.5 7.621 0.5 7.621 0.5 7.62 C 0.5 7.619 0.5 7.619 0.5 7.618 C 0.5 7.617 0.5 7.617 0.5 7.616 C 0.5 7.615 0.5 7.614 0.5 7.614 C 0.5 7.613 0.5 7.612 0.5 7.612 C 0.5 7.611 0.5 7.61 0.5 7.61 C 0.5 7.609 0.5 7.608 0.5 7.608 C 0.5 7.607 0.5 7.606 0.5 7.606 C 0.5 7.605 0.5 7.604 0.5 7.604 C 0.5 7.603 0.5 7.602 0.5 7.602 C 0.5 7.601 0.5 7.6 0.5 7.6 C 0.5 7.599 0.5 7.598 0.5 7.597 C 0.5 7.597 0.5 7.596 0.5 7.595 C 0.5 7.595 0.5 7.594 0.5 7.593 C 0.5 7.593 0.5 7.592 0.5 7.591 C 0.5 7.591 0.5 7.59 0.5 7.589 C 0.5 7.589 0.5 7.588 0.5 7.587 C 0.5 7.587 0.5 7.586 0.5 7.585 C 0.5 7.584 0.5 7.584 0.5 7.583 C 0.5 7.582 0.5 7.582 0.5 7.581 C 0.5 7.58 0.5 7.58 0.5 7.579 C 0.5 7.578 0.5 7.578 0.5 7.577 C 0.5 7.576 0.5 7.576 0.5 7.575 C 0.5 7.574 0.5 7.573 0.5 7.573 C 0.5 7.572 0.5 7.571 0.5 7.571 C 0.5 7.57 0.5 7.569 0.5 7.569 C 0.5 7.568 0.5 7.567 0.5 7.567 C 0.5 7.566 0.5 7.565 0.5 7.564 C 0.5 7.564 0.5 7.563 0.5 7.562 C 0.5 7.562 0.5 7.561 0.5 7.56 C 0.5 7.56 0.5 7.559 0.5 7.558 C 0.5 7.558 0.5 7.557 0.5 7.556 C 0.5 7.555 0.5 7.555 0.5 7.554 C 0.5 7.553 0.5 7.553 0.5 7.552 C 0.5 7.551 0.5 7.551 0.5 7.55 C 0.5 7.549 0.5 7.549 0.5 7.548 C 0.5 7.547 0.5 7.546 0.5 7.546 C 0.5 7.545 0.5 7.544 0.5 7.544 C 0.5 7.543 0.5 7.542 0.5 7.542 C 0.5 7.541 0.5 7.54 0.5 7.539 C 0.5 7.539 0.5 7.538 0.5 7.537 C 0.5 7.537 0.5 7.536 0.5 7.535 C 0.5 7.535 0.5 7.534 0.5 7.533 C 0.5 7.532 0.5 7.532 0.5 7.531 C 0.5 7.53 0.5 7.53 0.5 7.529 C 0.5 7.528 0.5 7.528 0.5 7.527 C 0.5 7.526 0.5 7.525 0.5 7.525 C 0.5 7.524 0.5 7.523 0.5 7.523 C 0.5 7.522 0.5 7.521 0.5 7.521 C 0.5 7.52 0.5 7.519 0.5 7.518 C 0.5 7.518 0.5 7.517 0.5 7.516 C 0.5 7.516 0.5 7.515 0.5 7.514 C 0.5 7.513 0.5 7.513 0.5 7.512 C 0.5 7.511 0.5 7.511 0.5 7.51 C 0.5 7.509 0.5 7.509 0.5 7.508 C 0.5 7.507 0.5 7.506 0.5 7.506 C 0.5 7.505 0.5 7.504 0.5 7.504 C 0.5 7.503 0.5 7.502 0.5 7.501 C 0.5 7.501 0.5 7.5 0.5 7.499 C 0.5 7.499 0.5 7.498 0.5 7.497 C 0.5 7.496 0.5 7.496 0.5 7.495 C 0.5 7.494 0.5 7.494 0.5 7.493 C 0.5 7.492 0.5 7.491 0.5 7.491 C 0.5 7.49 0.5 7.489 0.5 7.489 C 0.5 7.488 0.5 7.487 0.5 7.487 C 0.5 7.486 0.5 7.485 0.5 7.484 C 0.5 7.484 0.5 7.483 0.5 7.482 C 0.5 7.482 0.5 7.481 0.5 7.48 C 0.5 7.479 0.5 7.479 0.5 7.478 C 0.5 7.477 0.5 7.477 0.5 7.476 C 0.5 7.475 0.5 7.474 0.5 7.474 C 0.5 7.473 0.5 7.472 0.5 7.472 C 0.5 7.471 0.5 7.47 0.5 7.469 C 0.5 7.469 0.5 7.468 0.5 7.467 C 0.5 7.466 0.5 7.466 0.5 7.465 C 0.5 7.464 0.5 7.464 0.5 7.463 C 0.5 7.462 0.5 7.461 0.5 7.461 C 0.5 7.46 0.5 7.459 0.5 7.459 C 0.5 7.458 0.5 7.457 0.5 7.456 C 0.5 7.456 0.5 7.455 0.5 7.454 C 0.5 7.454 0.5 7.453 0.5 7.452 C 0.5 7.451 0.5 7.451 0.5 7.45 C 0.5 7.449 0.5 7.448 0.5 7.448 C 0.5 7.447 0.5 7.446 0.5 7.446 C 0.5 7.445 0.5 7.444 0.5 7.443 C 0.5 7.443 0.5 7.442 0.5 7.441 C 0.5 7.441 0.5 7.44 0.5 7.439 C 0.5 7.438 0.5 7.438 0.5 7.437 C 0.5 7.436 0.5 7.435 0.5 7.435 C 0.5 7.434 0.5 7.433 0.5 7.433 C 0.5 7.432 0.5 7.431 0.5 7.43 C 0.5 7.43 0.5 7.429 0.5 7.428 C 0.5 7.427 0.5 7.427 0.5 7.426 C 0.5 7.425 0.5 7.425 0.5 7.424 C 0.5 7.423 0.5 7.422 0.5 7.422 C 0.5 7.421 0.5 7.42 0.5 7.419 C 0.5 7.419 0.5 7.418 0.5 7.417 C 0.5 7.417 0.5 7.416 0.5 7.415 C 0.5 7.414 0.5 7.414 0.5 7.413 C 0.5 7.412 0.5 7.411 0.5 7.411 C 0.5 7.41 0.5 7.409 0.5 7.409 C 0.5 7.408 0.5 7.407 0.5 7.406 C 0.5 7.406 0.5 7.405 0.5 7.404 C 0.5 7.403 0.5 7.403 0.5 7.402 C 0.5 7.401 0.5 7.4 0.5 7.4 C 0.5 7.399 0.5 7.398 0.5 7.398 C 0.5 7.397 0.5 7.396 0.5 7.395 C 0.5 7.395 0.5 7.394 0.5 7.393 C 0.5 7.392 0.5 7.392 0.5 7.391 C 0.5 7.39 0.5 7.389 0.5 7.389 C 0.5 7.388 0.5 7.387 0.5 7.387 C 0.5 7.386 0.5 7.385 0.5 7.384 C 0.5 7.384 0.5 7.383 0.5 7.382 C 0.5 7.381 0.5 7.381 0.5 7.38 C 0.5 7.379 0.5 7.378 0.5 7.378 C 0.5 7.377 0.5 7.376 0.5 7.375 C 0.5 7.375 0.5 7.374 0.5 7.373 C 0.5 7.373 0.5 7.372 0.5 7.371 C 0.5 7.37 0.5 7.37 0.5 7.369 C 0.5 7.368 0.5 7.367 0.5 7.367 C 0.5 7.366 0.5 7.365 0.5 7.364 C 0.5 7.364 0.5 7.363 0.5 7.362 C 0.5 7.361 0.5 7.361 0.5 7.36 C 0.5 7.359 0.5 7.358 0.5 7.358 C 0.5 7.357 0.5 7.356 0.5 7.355 C 0.5 7.355 0.5 7.354 0.5 7.353 C 0.5 7.352 0.5 7.352 0.5 7.351 C 0.5 7.35 0.5 7.35 0.5 7.349 C 0.5 7.348 0.5 7.347 0.5 7.347 C 0.5 7.346 0.5 7.345 0.5 7.344 C 0.5 7.344 0.5 7.343 0.5 7.342 C 0.5 7.341 0.5 7.341 0.5 7.34 C 0.5 7.339 0.5 7.338 0.5 7.338 C 0.5 7.337 0.5 7.336 0.5 7.335 C 0.5 7.335 0.5 7.334 0.5 7.333 C 0.5 7.332 0.5 7.332 0.5 7.331 C 0.5 7.33 0.5 7.329 0.5 7.329 C 0.5 7.328 0.5 7.327 0.5 7.326 C 0.5 7.326 0.5 7.325 0.5 7.324 C 0.5 7.323 0.5 7.323 0.5 7.322 C 0.5 7.321 0.5 7.32 0.5 7.32 C 0.5 7.319 0.5 7.318 0.5 7.317 C 0.5 7.317 0.5 7.316 0.5 7.315 C 0.5 7.314 0.5 7.314 0.5 7.313 C 0.5 7.312 0.5 7.311 0.5 7.311 C 0.5 7.31 0.5 7.309 0.5 7.308 C 0.5 7.308 0.5 7.307 0.5 7.306 C 0.5 7.305 0.5 7.305 0.5 7.304 C 0.5 7.303 0.5 7.302 0.5 7.302 C 0.5 7.301 0.5 7.3 0.5 7.299 C 0.5 7.299 0.5 7.298 0.5 7.297 C 0.5 7.296 0.5 7.296 0.5 7.295 C 0.5 7.294 0.5 7.293 0.5 7.293 C 0.5 7.292 0.5 7.291 0.5 7.29 C 0.5 7.289 0.5 7.289 0.5 7.288 C 0.5 7.287 0.5 7.286 0.5 7.286 C 0.5 7.285 0.5 7.284 0.5 7.283 C 0.5 7.283 0.5 7.282 0.5 7.281 C 0.5 7.28 0.5 7.28 0.5 7.279 C 0.5 7.278 0.5 7.277 0.5 7.277 C 0.5 7.276 0.5 7.275 0.5 7.274 C 0.5 7.274 0.5 7.273 0.5 7.272 C 0.5 7.271 0.5 7.271 0.5 7.27 C 0.5 7.269 0.5 7.268 0.5 7.268 C 0.5 7.267 0.5 7.266 0.5 7.265 C 0.5 7.264 0.5 7.264 0.5 7.263 C 0.5 7.262 0.5 7.261 0.5 7.261 C 0.5 7.26 0.5 7.259 0.5 7.258 C 0.5 7.258 0.5 7.257 0.5 7.256 C 0.5 7.255 0.5 7.255 0.5 7.254 C 0.5 7.253 0.5 7.252 0.5 7.252 C 0.5 7.251 0.5 7.25 0.5 7.249 C 0.5 7.248 0.5 7.248 0.5 7.247 C 0.5 7.246 0.5 7.245 0.5 7.245 C 0.5 7.244 0.5 7.243 0.5 7.242 C 0.5 7.242 0.5 7.241 0.5 7.24 C 0.5 7.239 0.5 7.239 0.5 7.238 C 0.5 7.237 0.5 7.236 0.5 7.235 C 0.5 7.235 0.5 7.234 0.5 7.233 C 0.5 7.232 0.5 7.232 0.5 7.231 C 0.5 7.23 0.5 7.229 0.5 7.229 C 0.5 7.228 0.5 7.227 0.5 7.226 C 0.5 7.225 0.5 7.225 0.5 7.224 C 0.5 7.223 0.5 7.222 0.5 7.222 C 0.5 7.221 0.5 7.22 0.5 7.219 C 0.5 7.219 0.5 7.218 0.5 7.217 C 0.5 7.216 0.5 7.215 0.5 7.215 C 0.5 7.214 0.5 7.213 0.5 7.212 C 0.5 7.212 0.5 7.211 0.5 7.21 C 0.5 7.209 0.5 7.209 0.5 7.208 C 0.5 7.207 0.5 7.206 0.5 7.205 C 0.5 7.205 0.5 7.204 0.5 7.203 C 0.5 7.202 0.5 7.202 0.5 7.201 C 0.5 7.2 0.5 7.199 0.5 7.199 C 0.5 7.198 0.5 7.197 0.5 7.196 C 0.5 7.195 0.5 7.195 0.5 7.194 C 0.5 7.193 0.5 7.192 0.5 7.192 C 0.5 7.191 0.5 7.19 0.5 7.189 C 0.5 7.188 0.5 7.188 0.5 7.187 C 0.5 7.186 0.5 7.185 0.5 7.185 C 0.5 7.184 0.5 7.183 0.5 7.182 C 0.5 7.181 0.5 7.181 0.5 7.18 C 0.5 7.179 0.5 7.178 0.5 7.178 C 0.5 7.177 0.5 7.176 0.5 7.175 C 0.5 7.175 0.5 7.174 0.5 7.173 C 0.5 7.172 0.5 7.171 0.5 7.171 C 0.5 7.17 0.5 7.169 0.5 7.168 C 0.5 7.168 0.5 7.167 0.5 7.166 C 0.5 7.165 0.5 7.164 0.5 7.164 C 0.5 7.163 0.5 7.162 0.5 7.161 C 0.5 7.161 0.5 7.16 0.5 7.159 C 0.5 7.158 0.5 7.157 0.5 7.157 C 0.5 7.156 0.5 7.155 0.5 7.154 C 0.5 7.154 0.5 7.153 0.5 7.152 C 0.5 7.151 0.5 7.15 0.5 7.15 C 0.5 7.149 0.5 7.148 0.5 7.147 C 0.5 7.146 0.5 7.146 0.5 7.145 C 0.5 7.144 0.5 7.143 0.5 7.143 C 0.5 7.142 0.5 7.141 0.5 7.14 C 0.5 7.139 0.5 7.139 0.5 7.138 C 0.5 7.137 0.5 7.136 0.5 7.136 C 0.5 7.135 0.5 7.134 0.5 7.133 C 0.5 7.132 0.5 7.132 0.5 7.131 C 0.5 7.13 0.5 7.129 0.5 7.128 C 0.5 7.128 0.5 7.127 0.5 7.126 C 0.5 7.125 0.5 7.125 0.5 7.124 C 0.5 7.123 0.5 7.122 0.5 7.121 C 0.5 7.121 0.5 7.12 0.5 7.119 C 0.5 7.118 0.5 7.118 0.5 7.117 C 0.5 7.116 0.5 7.115 0.5 7.114 C 0.5 7.114 0.5 7.113 0.5 7.112 C 0.5 7.111 0.5 7.11 0.5 7.11 C 0.5 7.109 0.5 7.108 0.5 7.107 C 0.5 7.106 0.5 7.106 0.5 7.105 C 0.5 7.104 0.5 7.103 0.5 7.103 C 0.5 7.102 0.5 7.101 0.5 7.1 C 0.5 7.099 0.5 7.099 0.5 7.098 C 0.5 7.097 0.5 7.096 0.5 7.095 C 0.5 7.095 0.5 7.094 0.5 7.093 C 0.5 7.092 0.5 7.092 0.5 7.091 C 0.5 7.09 0.5 7.089 0.5 7.088 C 0.5 7.088 0.5 7.087 0.5 7.086 C 0.5 7.085 0.5 7.084 0.5 7.084 C 0.5 7.083 0.5 7.082 0.5 7.081 C 0.5 7.08 0.5 7.08 0.5 7.079 C 0.5 7.078 0.5 7.077 0.5 7.077 C 0.5 7.076 0.5 7.075 0.5 7.074 C 0.5 7.073 0.5 7.073 0.5 7.072 C 0.5 7.071 0.5 7.07 0.5 7.069 C 0.5 7.069 0.5 7.068 0.5 7.067 C 0.5 7.066 0.5 7.065 0.5 7.065 C 0.5 7.064 0.5 7.063 0.5 7.062 C 0.5 7.061 0.5 7.061 0.5 7.06 C 0.5 7.059 0.5 7.058 0.5 7.057 C 0.5 7.057 0.5 7.056 0.5 7.055 C 0.5 7.054 0.5 7.054 0.5 7.053 C 0.5 7.052 0.5 7.051 0.5 7.05 C 0.5 7.05 0.5 7.049 0.5 7.048 C 0.5 7.047 0.5 7.046 0.5 7.046 C 0.5 7.045 0.5 7.044 0.5 7.043 C 0.5 7.042 0.5 7.042 0.5 7.041 C 0.5 7.04 0.5 7.039 0.5 7.038 C 0.5 7.038 0.5 7.037 0.5 7.036 C 0.5 7.035 0.5 7.034 0.5 7.034 C 0.5 7.033 0.5 7.032 0.5 7.031 C 0.5 7.03 0.5 7.03 0.5 7.029 C 0.5 7.028 0.5 7.027 0.5 7.026 C 0.5 7.026 0.5 7.025 0.5 7.024 C 0.5 7.023 0.5 7.022 0.5 7.022 C 0.5 7.021 0.5 7.02 0.5 7.019 C 0.5 7.018 0.5 7.018 0.5 7.017 C 0.5 7.016 0.5 7.015 0.5 7.014 C 0.5 7.014 0.5 7.013 0.5 7.012 C 0.5 7.011 0.5 7.01 0.5 7.01 C 0.5 7.009 0.5 7.008 0.5 7.007 C 0.5 7.006 0.5 7.006 0.5 7.005 C 0.5 7.004 0.5 7.003 0.5 7.002 C 0.5 7.002 0.5 7.001 0.5 7 L -0.5 7 C -0.5 7.001 -0.5 7.002 -0.5 7.002 C -0.5 7.003 -0.5 7.004 -0.5 7.005 C -0.5 7.006 -0.5 7.006 -0.5 7.007 C -0.5 7.008 -0.5 7.009 -0.5 7.01 C -0.5 7.01 -0.5 7.011 -0.5 7.012 C -0.5 7.013 -0.5 7.014 -0.5 7.014 C -0.5 7.015 -0.5 7.016 -0.5 7.017 C -0.5 7.018 -0.5 7.018 -0.5 7.019 C -0.5 7.02 -0.5 7.021 -0.5 7.022 C -0.5 7.022 -0.5 7.023 -0.5 7.024 C -0.5 7.025 -0.5 7.026 -0.5 7.026 C -0.5 7.027 -0.5 7.028 -0.5 7.029 C -0.5 7.03 -0.5 7.03 -0.5 7.031 C -0.5 7.032 -0.5 7.033 -0.5 7.034 C -0.5 7.034 -0.5 7.035 -0.5 7.036 C -0.5 7.037 -0.5 7.038 -0.5 7.038 C -0.5 7.039 -0.5 7.04 -0.5 7.041 C -0.5 7.042 -0.5 7.042 -0.5 7.043 C -0.5 7.044 -0.5 7.045 -0.5 7.046 C -0.5 7.046 -0.5 7.047 -0.5 7.048 C -0.5 7.049 -0.5 7.05 -0.5 7.05 C -0.5 7.051 -0.5 7.052 -0.5 7.053 C -0.5 7.054 -0.5 7.054 -0.5 7.055 C -0.5 7.056 -0.5 7.057 -0.5 7.057 C -0.5 7.058 -0.5 7.059 -0.5 7.06 C -0.5 7.061 -0.5 7.061 -0.5 7.062 C -0.5 7.063 -0.5 7.064 -0.5 7.065 C -0.5 7.065 -0.5 7.066 -0.5 7.067 C -0.5 7.068 -0.5 7.069 -0.5 7.069 C -0.5 7.07 -0.5 7.071 -0.5 7.072 C -0.5 7.073 -0.5 7.073 -0.5 7.074 C -0.5 7.075 -0.5 7.076 -0.5 7.077 C -0.5 7.077 -0.5 7.078 -0.5 7.079 C -0.5 7.08 -0.5 7.08 -0.5 7.081 C -0.5 7.082 -0.5 7.083 -0.5 7.084 C -0.5 7.084 -0.5 7.085 -0.5 7.086 C -0.5 7.087 -0.5 7.088 -0.5 7.088 C -0.5 7.089 -0.5 7.09 -0.5 7.091 C -0.5 7.092 -0.5 7.092 -0.5 7.093 C -0.5 7.094 -0.5 7.095 -0.5 7.095 C -0.5 7.096 -0.5 7.097 -0.5 7.098 C -0.5 7.099 -0.5 7.099 -0.5 7.1 C -0.5 7.101 -0.5 7.102 -0.5 7.103 C -0.5 7.103 -0.5 7.104 -0.5 7.105 C -0.5 7.106 -0.5 7.106 -0.5 7.107 C -0.5 7.108 -0.5 7.109 -0.5 7.11 C -0.5 7.11 -0.5 7.111 -0.5 7.112 C -0.5 7.113 -0.5 7.114 -0.5 7.114 C -0.5 7.115 -0.5 7.116 -0.5 7.117 C -0.5 7.118 -0.5 7.118 -0.5 7.119 C -0.5 7.12 -0.5 7.121 -0.5 7.121 C -0.5 7.122 -0.5 7.123 -0.5 7.124 C -0.5 7.125 -0.5 7.125 -0.5 7.126 C -0.5 7.127 -0.5 7.128 -0.5 7.128 C -0.5 7.129 -0.5 7.13 -0.5 7.131 C -0.5 7.132 -0.5 7.132 -0.5 7.133 C -0.5 7.134 -0.5 7.135 -0.5 7.136 C -0.5 7.136 -0.5 7.137 -0.5 7.138 C -0.5 7.139 -0.5 7.139 -0.5 7.14 C -0.5 7.141 -0.5 7.142 -0.5 7.143 C -0.5 7.143 -0.5 7.144 -0.5 7.145 C -0.5 7.146 -0.5 7.146 -0.5 7.147 C -0.5 7.148 -0.5 7.149 -0.5 7.15 C -0.5 7.15 -0.5 7.151 -0.5 7.152 C -0.5 7.153 -0.5 7.154 -0.5 7.154 C -0.5 7.155 -0.5 7.156 -0.5 7.157 C -0.5 7.157 -0.5 7.158 -0.5 7.159 C -0.5 7.16 -0.5 7.161 -0.5 7.161 C -0.5 7.162 -0.5 7.163 -0.5 7.164 C -0.5 7.164 -0.5 7.165 -0.5 7.166 C -0.5 7.167 -0.5 7.168 -0.5 7.168 C -0.5 7.169 -0.5 7.17 -0.5 7.171 C -0.5 7.171 -0.5 7.172 -0.5 7.173 C -0.5 7.174 -0.5 7.175 -0.5 7.175 C -0.5 7.176 -0.5 7.177 -0.5 7.178 C -0.5 7.178 -0.5 7.179 -0.5 7.18 C -0.5 7.181 -0.5 7.181 -0.5 7.182 C -0.5 7.183 -0.5 7.184 -0.5 7.185 C -0.5 7.185 -0.5 7.186 -0.5 7.187 C -0.5 7.188 -0.5 7.188 -0.5 7.189 C -0.5 7.19 -0.5 7.191 -0.5 7.192 C -0.5 7.192 -0.5 7.193 -0.5 7.194 C -0.5 7.195 -0.5 7.195 -0.5 7.196 C -0.5 7.197 -0.5 7.198 -0.5 7.199 C -0.5 7.199 -0.5 7.2 -0.5 7.201 C -0.5 7.202 -0.5 7.202 -0.5 7.203 C -0.5 7.204 -0.5 7.205 -0.5 7.205 C -0.5 7.206 -0.5 7.207 -0.5 7.208 C -0.5 7.209 -0.5 7.209 -0.5 7.21 C -0.5 7.211 -0.5 7.212 -0.5 7.212 C -0.5 7.213 -0.5 7.214 -0.5 7.215 C -0.5 7.215 -0.5 7.216 -0.5 7.217 C -0.5 7.218 -0.5 7.219 -0.5 7.219 C -0.5 7.22 -0.5 7.221 -0.5 7.222 C -0.5 7.222 -0.5 7.223 -0.5 7.224 C -0.5 7.225 -0.5 7.225 -0.5 7.226 C -0.5 7.227 -0.5 7.228 -0.5 7.229 C -0.5 7.229 -0.5 7.23 -0.5 7.231 C -0.5 7.232 -0.5 7.232 -0.5 7.233 C -0.5 7.234 -0.5 7.235 -0.5 7.235 C -0.5 7.236 -0.5 7.237 -0.5 7.238 C -0.5 7.239 -0.5 7.239 -0.5 7.24 C -0.5 7.241 -0.5 7.242 -0.5 7.242 C -0.5 7.243 -0.5 7.244 -0.5 7.245 C -0.5 7.245 -0.5 7.246 -0.5 7.247 C -0.5 7.248 -0.5 7.248 -0.5 7.249 C -0.5 7.25 -0.5 7.251 -0.5 7.252 C -0.5 7.252 -0.5 7.253 -0.5 7.254 C -0.5 7.255 -0.5 7.255 -0.5 7.256 C -0.5 7.257 -0.5 7.258 -0.5 7.258 C -0.5 7.259 -0.5 7.26 -0.5 7.261 C -0.5 7.261 -0.5 7.262 -0.5 7.263 C -0.5 7.264 -0.5 7.264 -0.5 7.265 C -0.5 7.266 -0.5 7.267 -0.5 7.268 C -0.5 7.268 -0.5 7.269 -0.5 7.27 C -0.5 7.271 -0.5 7.271 -0.5 7.272 C -0.5 7.273 -0.5 7.274 -0.5 7.274 C -0.5 7.275 -0.5 7.276 -0.5 7.277 C -0.5 7.277 -0.5 7.278 -0.5 7.279 C -0.5 7.28 -0.5 7.28 -0.5 7.281 C -0.5 7.282 -0.5 7.283 -0.5 7.283 C -0.5 7.284 -0.5 7.285 -0.5 7.286 C -0.5 7.286 -0.5 7.287 -0.5 7.288 C -0.5 7.289 -0.5 7.289 -0.5 7.29 C -0.5 7.291 -0.5 7.292 -0.5 7.293 C -0.5 7.293 -0.5 7.294 -0.5 7.295 C -0.5 7.296 -0.5 7.296 -0.5 7.297 C -0.5 7.298 -0.5 7.299 -0.5 7.299 C -0.5 7.3 -0.5 7.301 -0.5 7.302 C -0.5 7.302 -0.5 7.303 -0.5 7.304 C -0.5 7.305 -0.5 7.305 -0.5 7.306 C -0.5 7.307 -0.5 7.308 -0.5 7.308 C -0.5 7.309 -0.5 7.31 -0.5 7.311 C -0.5 7.311 -0.5 7.312 -0.5 7.313 C -0.5 7.314 -0.5 7.314 -0.5 7.315 C -0.5 7.316 -0.5 7.317 -0.5 7.317 C -0.5 7.318 -0.5 7.319 -0.5 7.32 C -0.5 7.32 -0.5 7.321 -0.5 7.322 C -0.5 7.323 -0.5 7.323 -0.5 7.324 C -0.5 7.325 -0.5 7.326 -0.5 7.326 C -0.5 7.327 -0.5 7.328 -0.5 7.329 C -0.5 7.329 -0.5 7.33 -0.5 7.331 C -0.5 7.332 -0.5 7.332 -0.5 7.333 C -0.5 7.334 -0.5 7.335 -0.5 7.335 C -0.5 7.336 -0.5 7.337 -0.5 7.338 C -0.5 7.338 -0.5 7.339 -0.5 7.34 C -0.5 7.341 -0.5 7.341 -0.5 7.342 C -0.5 7.343 -0.5 7.344 -0.5 7.344 C -0.5 7.345 -0.5 7.346 -0.5 7.347 C -0.5 7.347 -0.5 7.348 -0.5 7.349 C -0.5 7.35 -0.5 7.35 -0.5 7.351 C -0.5 7.352 -0.5 7.352 -0.5 7.353 C -0.5 7.354 -0.5 7.355 -0.5 7.355 C -0.5 7.356 -0.5 7.357 -0.5 7.358 C -0.5 7.358 -0.5 7.359 -0.5 7.36 C -0.5 7.361 -0.5 7.361 -0.5 7.362 C -0.5 7.363 -0.5 7.364 -0.5 7.364 C -0.5 7.365 -0.5 7.366 -0.5 7.367 C -0.5 7.367 -0.5 7.368 -0.5 7.369 C -0.5 7.37 -0.5 7.37 -0.5 7.371 C -0.5 7.372 -0.5 7.373 -0.5 7.373 C -0.5 7.374 -0.5 7.375 -0.5 7.375 C -0.5 7.376 -0.5 7.377 -0.5 7.378 C -0.5 7.378 -0.5 7.379 -0.5 7.38 C -0.5 7.381 -0.5 7.381 -0.5 7.382 C -0.5 7.383 -0.5 7.384 -0.5 7.384 C -0.5 7.385 -0.5 7.386 -0.5 7.387 C -0.5 7.387 -0.5 7.388 -0.5 7.389 C -0.5 7.389 -0.5 7.39 -0.5 7.391 C -0.5 7.392 -0.5 7.392 -0.5 7.393 C -0.5 7.394 -0.5 7.395 -0.5 7.395 C -0.5 7.396 -0.5 7.397 -0.5 7.398 C -0.5 7.398 -0.5 7.399 -0.5 7.4 C -0.5 7.4 -0.5 7.401 -0.5 7.402 C -0.5 7.403 -0.5 7.403 -0.5 7.404 C -0.5 7.405 -0.5 7.406 -0.5 7.406 C -0.5 7.407 -0.5 7.408 -0.5 7.409 C -0.5 7.409 -0.5 7.41 -0.5 7.411 C -0.5 7.411 -0.5 7.412 -0.5 7.413 C -0.5 7.414 -0.5 7.414 -0.5 7.415 C -0.5 7.416 -0.5 7.417 -0.5 7.417 C -0.5 7.418 -0.5 7.419 -0.5 7.419 C -0.5 7.42 -0.5 7.421 -0.5 7.422 C -0.5 7.422 -0.5 7.423 -0.5 7.424 C -0.5 7.425 -0.5 7.425 -0.5 7.426 C -0.5 7.427 -0.5 7.427 -0.5 7.428 C -0.5 7.429 -0.5 7.43 -0.5 7.43 C -0.5 7.431 -0.5 7.432 -0.5 7.433 C -0.5 7.433 -0.5 7.434 -0.5 7.435 C -0.5 7.435 -0.5 7.436 -0.5 7.437 C -0.5 7.438 -0.5 7.438 -0.5 7.439 C -0.5 7.44 -0.5 7.441 -0.5 7.441 C -0.5 7.442 -0.5 7.443 -0.5 7.443 C -0.5 7.444 -0.5 7.445 -0.5 7.446 C -0.5 7.446 -0.5 7.447 -0.5 7.448 C -0.5 7.448 -0.5 7.449 -0.5 7.45 C -0.5 7.451 -0.5 7.451 -0.5 7.452 C -0.5 7.453 -0.5 7.454 -0.5 7.454 C -0.5 7.455 -0.5 7.456 -0.5 7.456 C -0.5 7.457 -0.5 7.458 -0.5 7.459 C -0.5 7.459 -0.5 7.46 -0.5 7.461 C -0.5 7.461 -0.5 7.462 -0.5 7.463 C -0.5 7.464 -0.5 7.464 -0.5 7.465 C -0.5 7.466 -0.5 7.466 -0.5 7.467 C -0.5 7.468 -0.5 7.469 -0.5 7.469 C -0.5 7.47 -0.5 7.471 -0.5 7.472 C -0.5 7.472 -0.5 7.473 -0.5 7.474 C -0.5 7.474 -0.5 7.475 -0.5 7.476 C -0.5 7.477 -0.5 7.477 -0.5 7.478 C -0.5 7.479 -0.5 7.479 -0.5 7.48 C -0.5 7.481 -0.5 7.482 -0.5 7.482 C -0.5 7.483 -0.5 7.484 -0.5 7.484 C -0.5 7.485 -0.5 7.486 -0.5 7.487 C -0.5 7.487 -0.5 7.488 -0.5 7.489 C -0.5 7.489 -0.5 7.49 -0.5 7.491 C -0.5 7.491 -0.5 7.492 -0.5 7.493 C -0.5 7.494 -0.5 7.494 -0.5 7.495 C -0.5 7.496 -0.5 7.496 -0.5 7.497 C -0.5 7.498 -0.5 7.499 -0.5 7.499 C -0.5 7.5 -0.5 7.501 -0.5 7.501 C -0.5 7.502 -0.5 7.503 -0.5 7.504 C -0.5 7.504 -0.5 7.505 -0.5 7.506 C -0.5 7.506 -0.5 7.507 -0.5 7.508 C -0.5 7.509 -0.5 7.509 -0.5 7.51 C -0.5 7.511 -0.5 7.511 -0.5 7.512 C -0.5 7.513 -0.5 7.513 -0.5 7.514 C -0.5 7.515 -0.5 7.516 -0.5 7.516 C -0.5 7.517 -0.5 7.518 -0.5 7.518 C -0.5 7.519 -0.5 7.52 -0.5 7.521 C -0.5 7.521 -0.5 7.522 -0.5 7.523 C -0.5 7.523 -0.5 7.524 -0.5 7.525 C -0.5 7.525 -0.5 7.526 -0.5 7.527 C -0.5 7.528 -0.5 7.528 -0.5 7.529 C -0.5 7.53 -0.5 7.53 -0.5 7.531 C -0.5 7.532 -0.5 7.532 -0.5 7.533 C -0.5 7.534 -0.5 7.535 -0.5 7.535 C -0.5 7.536 -0.5 7.537 -0.5 7.537 C -0.5 7.538 -0.5 7.539 -0.5 7.539 C -0.5 7.54 -0.5 7.541 -0.5 7.542 C -0.5 7.542 -0.5 7.543 -0.5 7.544 C -0.5 7.544 -0.5 7.545 -0.5 7.546 C -0.5 7.546 -0.5 7.547 -0.5 7.548 C -0.5 7.549 -0.5 7.549 -0.5 7.55 C -0.5 7.551 -0.5 7.551 -0.5 7.552 C -0.5 7.553 -0.5 7.553 -0.5 7.554 C -0.5 7.555 -0.5 7.555 -0.5 7.556 C -0.5 7.557 -0.5 7.558 -0.5 7.558 C -0.5 7.559 -0.5 7.56 -0.5 7.56 C -0.5 7.561 -0.5 7.562 -0.5 7.562 C -0.5 7.563 -0.5 7.564 -0.5 7.564 C -0.5 7.565 -0.5 7.566 -0.5 7.567 C -0.5 7.567 -0.5 7.568 -0.5 7.569 C -0.5 7.569 -0.5 7.57 -0.5 7.571 C -0.5 7.571 -0.5 7.572 -0.5 7.573 C -0.5 7.573 -0.5 7.574 -0.5 7.575 C -0.5 7.576 -0.5 7.576 -0.5 7.577 C -0.5 7.578 -0.5 7.578 -0.5 7.579 C -0.5 7.58 -0.5 7.58 -0.5 7.581 C -0.5 7.582 -0.5 7.582 -0.5 7.583 C -0.5 7.584 -0.5 7.584 -0.5 7.585 C -0.5 7.586 -0.5 7.587 -0.5 7.587 C -0.5 7.588 -0.5 7.589 -0.5 7.589 C -0.5 7.59 -0.5 7.591 -0.5 7.591 C -0.5 7.592 -0.5 7.593 -0.5 7.593 C -0.5 7.594 -0.5 7.595 -0.5 7.595 C -0.5 7.596 -0.5 7.597 -0.5 7.597 C -0.5 7.598 -0.5 7.599 -0.5 7.6 C -0.5 7.6 -0.5 7.601 -0.5 7.602 C -0.5 7.602 -0.5 7.603 -0.5 7.604 C -0.5 7.604 -0.5 7.605 -0.5 7.606 C -0.5 7.606 -0.5 7.607 -0.5 7.608 C -0.5 7.608 -0.5 7.609 -0.5 7.61 C -0.5 7.61 -0.5 7.611 -0.5 7.612 C -0.5 7.612 -0.5 7.613 -0.5 7.614 C -0.5 7.614 -0.5 7.615 -0.5 7.616 C -0.5 7.617 -0.5 7.617 -0.5 7.618 C -0.5 7.619 -0.5 7.619 -0.5 7.62 C -0.5 7.621 -0.5 7.621 -0.5 7.622 C -0.5 7.623 -0.5 7.623 -0.5 7.624 C -0.5 7.625 -0.5 7.625 -0.5 7.626 C -0.5 7.627 -0.5 7.627 -0.5 7.628 C -0.5 7.629 -0.5 7.629 -0.5 7.63 C -0.5 7.631 -0.5 7.631 -0.5 7.632 C -0.5 7.633 -0.5 7.633 -0.5 7.634 C -0.5 7.635 -0.5 7.635 -0.5 7.636 C -0.5 7.637 -0.5 7.637 -0.5 7.638 C -0.5 7.639 -0.5 7.639 -0.5 7.64 C -0.5 7.641 -0.5 7.641 -0.5 7.642 C -0.5 7.643 -0.5 7.643 -0.5 7.644 C -0.5 7.645 -0.5 7.645 -0.5 7.646 C -0.5 7.647 -0.5 7.647 -0.5 7.648 C -0.5 7.649 -0.5 7.649 -0.5 7.65 C -0.5 7.651 -0.5 7.651 -0.5 7.652 C -0.5 7.653 -0.5 7.653 -0.5 7.654 C -0.5 7.655 -0.5 7.655 -0.5 7.656 C -0.5 7.657 -0.5 7.657 -0.5 7.658 C -0.5 7.659 -0.5 7.659 -0.5 7.66 C -0.5 7.661 -0.5 7.661 -0.5 7.662 C -0.5 7.663 -0.5 7.663 -0.5 7.664 C -0.5 7.665 -0.5 7.665 -0.5 7.666 C -0.5 7.667 -0.5 7.667 -0.5 7.668 C -0.5 7.669 -0.5 7.669 -0.5 7.67 C -0.5 7.671 -0.5 7.671 -0.5 7.672 C -0.5 7.673 -0.5 7.673 -0.5 7.674 C -0.5 7.675 -0.5 7.675 -0.5 7.676 C -0.5 7.677 -0.5 7.677 -0.5 7.678 C -0.5 7.679 -0.5 7.679 -0.5 7.68 C -0.5 7.681 -0.5 7.681 -0.5 7.682 C -0.5 7.683 -0.5 7.683 -0.5 7.684 C -0.5 7.684 -0.5 7.685 -0.5 7.686 C -0.5 7.686 -0.5 7.687 -0.5 7.688 C -0.5 7.688 -0.5 7.689 -0.5 7.69 C -0.5 7.69 -0.5 7.691 -0.5 7.692 C -0.5 7.692 -0.5 7.693 -0.5 7.694 C -0.5 7.694 -0.5 7.695 -0.5 7.696 C -0.5 7.696 -0.5 7.697 -0.5 7.698 C -0.5 7.698 -0.5 7.699 -0.5 7.699 C -0.5 7.7 -0.5 7.701 -0.5 7.701 C -0.5 7.702 -0.5 7.703 -0.5 7.703 C -0.5 7.704 -0.5 7.705 -0.5 7.705 C -0.5 7.706 -0.5 7.707 -0.5 7.707 C -0.5 7.708 -0.5 7.709 -0.5 7.709 C -0.5 7.71 -0.5 7.711 -0.5 7.711 C -0.5 7.712 -0.5 7.712 -0.5 7.713 C -0.5 7.714 -0.5 7.714 -0.5 7.715 C -0.5 7.716 -0.5 7.716 -0.5 7.717 C -0.5 7.718 -0.5 7.718 -0.5 7.719 C -0.5 7.72 -0.5 7.72 -0.5 7.721 C -0.5 7.721 -0.5 7.722 -0.5 7.723 C -0.5 7.723 -0.5 7.724 -0.5 7.725 C -0.5 7.725 -0.5 7.726 -0.5 7.727 C -0.5 7.727 -0.5 7.728 -0.5 7.729 C -0.5 7.729 -0.5 7.73 -0.5 7.73 C -0.5 7.731 -0.5 7.732 -0.5 7.732 C -0.5 7.733 -0.5 7.734 -0.5 7.734 C -0.5 7.735 -0.5 7.736 -0.5 7.736 C -0.5 7.737 -0.5 7.737 -0.5 7.738 C -0.5 7.739 -0.5 7.739 -0.5 7.74 C -0.5 7.741 -0.5 7.741 -0.5 7.742 C -0.5 7.743 -0.5 7.743 -0.5 7.744 C -0.5 7.744 -0.5 7.745 -0.5 7.746 C -0.5 7.746 -0.5 7.747 -0.5 7.748 C -0.5 7.748 -0.5 7.749 -0.5 7.75 C -0.5 7.75 -0.5 7.751 -0.5 7.751 C -0.5 7.752 -0.5 7.753 -0.5 7.753 C -0.5 7.754 -0.5 7.755 -0.5 7.755 C -0.5 7.756 -0.5 7.757 -0.5 7.757 C -0.5 7.758 -0.5 7.758 -0.5 7.759 C -0.5 7.76 -0.5 7.76 -0.5 7.761 C -0.5 7.762 -0.5 7.762 -0.5 7.763 C -0.5 7.763 -0.5 7.764 -0.5 7.765 C -0.5 7.765 -0.5 7.766 -0.5 7.767 C -0.5 7.767 -0.5 7.768 -0.5 7.768 C -0.5 7.769 -0.5 7.77 -0.5 7.77 C -0.5 7.771 -0.5 7.772 -0.5 7.772 C -0.5 7.773 -0.5 7.773 -0.5 7.774 C -0.5 7.775 -0.5 7.775 -0.5 7.776 C -0.5 7.777 -0.5 7.777 -0.5 7.778 C -0.5 7.778 -0.5 7.779 -0.5 7.78 C -0.5 7.78 -0.5 7.781 -0.5 7.782 C -0.5 7.782 -0.5 7.783 -0.5 7.783 C -0.5 7.784 -0.5 7.785 -0.5 7.785 C -0.5 7.786 -0.5 7.787 -0.5 7.787 C -0.5 7.788 -0.5 7.788 -0.5 7.789 C -0.5 7.79 -0.5 7.79 -0.5 7.791 C -0.5 7.791 -0.5 7.792 -0.5 7.793 C -0.5 7.793 -0.5 7.794 -0.5 7.795 C -0.5 7.795 -0.5 7.796 -0.5 7.796 C -0.5 7.797 -0.5 7.798 -0.5 7.798 C -0.5 7.799 -0.5 7.799 -0.5 7.8 C -0.5 7.801 -0.5 7.801 -0.5 7.802 C -0.5 7.803 -0.5 7.803 -0.5 7.804 C -0.5 7.804 -0.5 7.805 -0.5 7.806 C -0.5 7.806 -0.5 7.807 -0.5 7.807 C -0.5 7.808 -0.5 7.809 -0.5 7.809 C -0.5 7.81 -0.5 7.81 -0.5 7.811 C -0.5 7.812 -0.5 7.812 -0.5 7.813 C -0.5 7.814 -0.5 7.814 -0.5 7.815 C -0.5 7.815 -0.5 7.816 -0.5 7.817 C -0.5 7.817 -0.5 7.818 -0.5 7.818 C -0.5 7.819 -0.5 7.82 -0.5 7.82 C -0.5 7.821 -0.5 7.821 -0.5 7.822 C -0.5 7.823 -0.5 7.823 -0.5 7.824 C -0.5 7.824 -0.5 7.825 -0.5 7.826 C -0.5 7.826 -0.5 7.827 -0.5 7.827 C -0.5 7.828 -0.5 7.829 -0.5 7.829 C -0.5 7.83 -0.5 7.83 -0.5 7.831 C -0.5 7.832 -0.5 7.832 -0.5 7.833 C -0.5 7.833 -0.5 7.834 -0.5 7.835 C -0.5 7.835 -0.5 7.836 -0.5 7.836 C -0.5 7.837 -0.5 7.838 -0.5 7.838 C -0.5 7.839 -0.5 7.839 -0.5 7.84 C -0.5 7.841 -0.5 7.841 -0.5 7.842 C -0.5 7.842 -0.5 7.843 -0.5 7.844 C -0.5 7.844 -0.5 7.845 -0.5 7.845 C -0.5 7.846 -0.5 7.847 -0.5 7.847 C -0.5 7.848 -0.5 7.848 -0.5 7.849 C -0.5 7.85 -0.5 7.85 -0.5 7.851 C -0.5 7.851 -0.5 7.852 -0.5 7.853 C -0.5 7.853 -0.5 7.854 -0.5 7.854 C -0.5 7.855 -0.5 7.856 -0.5 7.856 C -0.5 7.857 -0.5 7.857 -0.5 7.858 C -0.5 7.859 -0.5 7.859 -0.5 7.86 C -0.5 7.86 -0.5 7.861 -0.5 7.861 C -0.5 7.862 -0.5 7.863 -0.5 7.863 C -0.5 7.864 -0.5 7.864 -0.5 7.865 C -0.5 7.866 -0.5 7.866 -0.5 7.867 C -0.5 7.867 -0.5 7.868 -0.5 7.869 C -0.5 7.869 -0.5 7.87 -0.5 7.87 C -0.5 7.871 -0.5 7.871 -0.5 7.872 C -0.5 7.873 -0.5 7.873 -0.5 7.874 C -0.5 7.874 -0.5 7.875 -0.5 7.876 C -0.5 7.876 -0.5 7.877 -0.5 7.877 C -0.5 7.878 -0.5 7.878 -0.5 7.879 C -0.5 7.88 -0.5 7.88 -0.5 7.881 C -0.5 7.881 -0.5 7.882 -0.5 7.883 C -0.5 7.883 -0.5 7.884 -0.5 7.884 C -0.5 7.885 -0.5 7.885 -0.5 7.886 C -0.5 7.887 -0.5 7.887 -0.5 7.888 C -0.5 7.888 -0.5 7.889 -0.5 7.889 C -0.5 7.89 -0.5 7.891 -0.5 7.891 C -0.5 7.892 -0.5 7.892 -0.5 7.893 C -0.5 7.893 -0.5 7.894 -0.5 7.895 C -0.5 7.895 -0.5 7.896 -0.5 7.896 C -0.5 7.897 -0.5 7.897 -0.5 7.898 C -0.5 7.899 -0.5 7.899 -0.5 7.9 C -0.5 7.9 -0.5 7.901 -0.5 7.901 C -0.5 7.902 -0.5 7.903 -0.5 7.903 C -0.5 7.904 -0.5 7.904 -0.5 7.905 C -0.5 7.905 -0.5 7.906 -0.5 7.907 C -0.5 7.907 -0.5 7.908 -0.5 7.908 C -0.5 7.909 -0.5 7.909 -0.5 7.91 C -0.5 7.911 -0.5 7.911 -0.5 7.912 C -0.5 7.912 -0.5 7.913 -0.5 7.913 C -0.5 7.914 -0.5 7.915 -0.5 7.915 C -0.5 7.916 -0.5 7.916 -0.5 7.917 C -0.5 7.917 -0.5 7.918 -0.5 7.919 C -0.5 7.919 -0.5 7.92 -0.5 7.92 C -0.5 7.921 -0.5 7.921 -0.5 7.922 C -0.5 7.922 -0.5 7.923 -0.5 7.924 C -0.5 7.924 -0.5 7.925 -0.5 7.925 C -0.5 7.926 -0.5 7.926 -0.5 7.927 C -0.5 7.928 -0.5 7.928 -0.5 7.929 C -0.5 7.929 -0.5 7.93 -0.5 7.93 C -0.5 7.931 -0.5 7.931 -0.5 7.932 C -0.5 7.933 -0.5 7.933 -0.5 7.934 C -0.5 7.934 -0.5 7.935 -0.5 7.935 C -0.5 7.936 -0.5 7.936 -0.5 7.937 C -0.5 7.938 -0.5 7.938 -0.5 7.939 C -0.5 7.939 -0.5 7.94 -0.5 7.94 C -0.5 7.941 -0.5 7.941 -0.5 7.942 C -0.5 7.943 -0.5 7.943 -0.5 7.944 C -0.5 7.944 -0.5 7.945 -0.5 7.945 C -0.5 7.946 -0.5 7.946 -0.5 7.947 C -0.5 7.947 -0.5 7.948 -0.5 7.949 C -0.5 7.949 -0.5 7.95 -0.5 7.95 C -0.5 7.951 -0.5 7.951 -0.5 7.952 C -0.5 7.952 -0.5 7.953 -0.5 7.953 C -0.5 7.954 -0.5 7.955 -0.5 7.955 C -0.5 7.956 -0.5 7.956 -0.5 7.957 C -0.5 7.957 -0.5 7.958 -0.5 7.958 C -0.5 7.959 -0.5 7.959 -0.5 7.96 C -0.5 7.961 -0.5 7.961 -0.5 7.962 C -0.5 7.962 -0.5 7.963 -0.5 7.963 C -0.5 7.964 -0.5 7.964 -0.5 7.965 C -0.5 7.965 -0.5 7.966 -0.5 7.967 C -0.5 7.967 -0.5 7.968 -0.5 7.968 C -0.5 7.969 -0.5 7.969 -0.5 7.97 C -0.5 7.97 -0.5 7.971 -0.5 7.971 C -0.5 7.972 -0.5 7.972 -0.5 7.973 C -0.5 7.974 -0.5 7.974 -0.5 7.975 C -0.5 7.975 -0.5 7.976 -0.5 7.976 C -0.5 7.977 -0.5 7.977 -0.5 7.978 C -0.5 7.978 -0.5 7.979 -0.5 7.979 C -0.5 7.98 -0.5 7.98 -0.5 7.981 C -0.5 7.982 -0.5 7.982 -0.5 7.983 C -0.5 7.983 -0.5 7.984 -0.5 7.984 C -0.5 7.985 -0.5 7.985 -0.5 7.986 C -0.5 7.986 -0.5 7.987 -0.5 7.987 C -0.5 7.988 -0.5 7.988 -0.5 7.989 C -0.5 7.989 -0.5 7.99 -0.5 7.991 C -0.5 7.991 -0.5 7.992 -0.5 7.992 C -0.5 7.993 -0.5 7.993 -0.5 7.994 C -0.5 7.994 -0.5 7.995 -0.5 7.995 C -0.5 7.996 -0.5 7.996 -0.5 7.997 C -0.5 7.997 -0.5 7.998 -0.5 7.998 C -0.5 7.999 -0.5 7.999 -0.5 8 C -0.5 8.001 -0.5 8.001 -0.5 8.002 C -0.5 8.002 -0.5 8.003 -0.5 8.003 C -0.5 8.004 -0.5 8.004 -0.5 8.005 C -0.5 8.005 -0.5 8.006 -0.5 8.006 C -0.5 8.007 -0.5 8.007 -0.5 8.008 C -0.5 8.008 -0.5 8.009 -0.5 8.009 C -0.5 8.01 -0.5 8.01 -0.5 8.011 C -0.5 8.011 -0.5 8.012 -0.5 8.012 C -0.5 8.013 -0.5 8.013 -0.5 8.014 C -0.5 8.015 -0.5 8.015 -0.5 8.016 C -0.5 8.016 -0.5 8.017 -0.5 8.017 C -0.5 8.018 -0.5 8.018 -0.5 8.019 C -0.5 8.019 -0.5 8.02 -0.5 8.02 C -0.5 8.021 -0.5 8.021 -0.5 8.022 C -0.5 8.022 -0.5 8.023 -0.5 8.023 C -0.5 8.024 -0.5 8.024 -0.5 8.025 C -0.5 8.025 -0.5 8.026 -0.5 8.026 C -0.5 8.027 -0.5 8.027 -0.5 8.028 C -0.5 8.028 -0.5 8.029 -0.5 8.029 C -0.5 8.03 -0.5 8.03 -0.5 8.031 C -0.5 8.031 -0.5 8.032 -0.5 8.032 C -0.5 8.033 -0.5 8.033 -0.5 8.034 C -0.5 8.034 -0.5 8.035 -0.5 8.035 C -0.5 8.036 -0.5 8.036 -0.5 8.037 C -0.5 8.037 -0.5 8.038 -0.5 8.038 L 0.5 8.038 Z M 3.487 11.646 L 0.939 9.098 L 0.232 9.805 L 2.78 12.354 L 3.487 11.646 Z M -0.5 8.038 C -0.5 8.702 -0.236 9.337 0.232 9.805 L 0.939 9.098 C 0.658 8.817 0.5 8.436 0.5 8.038 L -0.5 8.038 Z M 2.5 6.5 L 2.5 1 L 1.5 1 L 1.5 6.5 L 2.5 6.5 Z M 1.5 5 L 1.5 6.5 L 2.5 6.5 L 2.5 5 L 1.5 5 Z M 7.5 4 L 7.5 4.5 L 8.5 4.5 L 8.5 4 L 7.5 4 Z M 7 11.5 L 3.134 11.5 L 3.134 12.5 L 7 12.5 L 7 11.5 Z M 7.5 4.5 L 7.5 5 L 8.5 5 L 8.5 4.5 L 7.5 4.5 Z M 0.5 7 C 0.5 6.172 1.172 5.5 2 5.5 L 2 4.5 C 0.619 4.5 -0.5 5.619 -0.5 7 L 0.5 7 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "cursor-hourglass": "<g transform=\"translate(3.5,1.5)\"><path d=\"M 0 11 L -0.433 10.75 L -0.5 10.866 L -0.5 11 L 0 11 Z M 9 11 L 9.5 11 L 9.5 10.866 L 9.433 10.75 L 9 11 Z M 9 13 L 9 13.5 L 9.5 13.5 L 9.5 13 L 9 13 Z M 0 13 L -0.5 13 L -0.5 13.5 L 0 13.5 L 0 13 Z M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 9 0 L 9.5 0 L 9.5 -0.5 L 9 -0.5 L 9 0 Z M 9 2 L 9.433 2.25 L 9.5 2.134 L 9.5 2 L 9 2 Z M 0 2 L -0.5 2 L -0.5 2.134 L -0.433 2.25 L 0 2 Z M 6.402 6.5 L 5.969 6.25 L 5.825 6.5 L 5.969 6.75 L 6.402 6.5 Z M 2.598 6.5 L 3.031 6.75 L 3.175 6.5 L 3.031 6.25 L 2.598 6.5 Z M 8.5 11 L 8.5 13 L 9.5 13 L 9.5 11 L 8.5 11 Z M 9 12.5 L 0 12.5 L 0 13.5 L 9 13.5 L 9 12.5 Z M 0.5 13 L 0.5 11 L -0.5 11 L -0.5 13 L 0.5 13 Z M 0 0.5 L 9 0.5 L 9 -0.5 L 0 -0.5 L 0 0.5 Z M 8.5 0 L 8.5 2 L 9.5 2 L 9.5 0 L 8.5 0 Z M 0.5 2 L 0.5 0 L -0.5 0 L -0.5 2 L 0.5 2 Z M 8.567 1.75 L 5.969 6.25 L 6.835 6.75 L 9.433 2.25 L 8.567 1.75 Z M -0.433 2.25 L 2.165 6.75 L 3.031 6.25 L 0.433 1.75 L -0.433 2.25 Z M 2.165 6.25 L -0.433 10.75 L 0.433 11.25 L 3.031 6.75 L 2.165 6.25 Z M 5.969 6.75 L 8.567 11.25 L 9.433 10.75 L 6.835 6.25 L 5.969 6.75 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(3.5,12.5)\"><path d=\"M 0 0.5 L 9 0.5 L 9 -0.5 L 0 -0.5 L 0 0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(3.5,3.5)\"><path d=\"M 9 -0.5 L 0 -0.5 L 0 0.5 L 9 0.5 L 9 -0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-bold": "<g transform=\"translate(5.5,3.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 0 9 L -0.5 9 L -0.5 9.5 L 0 9.5 L 0 9 Z M 4.5 2 C 4.5 2.828 3.828 3.5 3 3.5 L 3 4.5 C 4.381 4.5 5.5 3.381 5.5 2 L 4.5 2 Z M 3 0.5 C 3.828 0.5 4.5 1.172 4.5 2 L 5.5 2 C 5.5 0.619 4.381 -0.5 3 -0.5 L 3 0.5 Z M 3 3.5 L 0 3.5 L 0 4.5 L 3 4.5 L 3 3.5 Z M 0 0.5 L 3 0.5 L 3 -0.5 L 0 -0.5 L 0 0.5 Z M -0.5 0 L -0.5 9 L 0.5 9 L 0.5 0 L -0.5 0 Z M 0 9.5 L 3.5 9.5 L 3.5 8.5 L 0 8.5 L 0 9.5 Z M 5.5 6.5 C 5.5 7.605 4.605 8.5 3.5 8.5 L 3.5 9.5 C 5.157 9.5 6.5 8.157 6.5 6.5 L 5.5 6.5 Z M 3.5 4.5 C 4.605 4.5 5.5 5.395 5.5 6.5 L 6.5 6.5 C 6.5 4.843 5.157 3.5 3.5 3.5 L 3.5 4.5 Z M 3.5 3.5 L 0 3.5 L 0 4.5 L 3.5 4.5 L 3.5 3.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-chart": "<g transform=\"translate(1.5,4.5)\"><path d=\"M 5 2 L 5.354 1.646 L 5 1.293 L 4.646 1.646 L 5 2 Z M 8 5 L 7.646 5.354 L 8 5.707 L 8.354 5.354 L 8 5 Z M 0.354 7.354 L 5.354 2.354 L 4.646 1.646 L -0.354 6.646 L 0.354 7.354 Z M 4.646 2.354 L 7.646 5.354 L 8.354 4.646 L 5.354 1.646 L 4.646 2.354 Z M 8.354 5.354 L 13.354 0.354 L 12.646 -0.354 L 7.646 4.646 L 8.354 5.354 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-chart-bar": "<g transform=\"translate(3.5,2.5)\"><path d=\"M 0 8 L 0 7.5 L -0.5 7.5 L -0.5 8 L 0 8 Z M 3 8 L 3.5 8 L 3.5 7.5 L 3 7.5 L 3 8 Z M 0 11 L -0.5 11 L -0.5 11.5 L 0 11.5 L 0 11 Z M 3 4 L 3 3.5 L 2.5 3.5 L 2.5 4 L 3 4 Z M 6 4 L 6.5 4 L 6.5 3.5 L 6 3.5 L 6 4 Z M 6 0 L 6 -0.5 L 5.5 -0.5 L 5.5 0 L 6 0 Z M 9 0 L 9.5 0 L 9.5 -0.5 L 9 -0.5 L 9 0 Z M 9 11 L 9 11.5 L 9.5 11.5 L 9.5 11 L 9 11 Z M 0 8.5 L 3 8.5 L 3 7.5 L 0 7.5 L 0 8.5 Z M 2.5 8 L 2.5 11 L 3.5 11 L 3.5 8 L 2.5 8 Z M 3 10.5 L 0 10.5 L 0 11.5 L 3 11.5 L 3 10.5 Z M 0.5 11 L 0.5 8 L -0.5 8 L -0.5 11 L 0.5 11 Z M 3 4.5 L 6 4.5 L 6 3.5 L 3 3.5 L 3 4.5 Z M 5.5 4 L 5.5 11 L 6.5 11 L 6.5 4 L 5.5 4 Z M 6 10.5 L 3 10.5 L 3 11.5 L 6 11.5 L 6 10.5 Z M 3.5 11 L 3.5 4 L 2.5 4 L 2.5 11 L 3.5 11 Z M 6 0.5 L 9 0.5 L 9 -0.5 L 6 -0.5 L 6 0.5 Z M 8.5 0 L 8.5 11 L 9.5 11 L 9.5 0 L 8.5 0 Z M 9 10.5 L 6 10.5 L 6 11.5 L 9 11.5 L 9 10.5 Z M 6.5 11 L 6.5 0 L 5.5 0 L 5.5 11 L 6.5 11 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-chart-pie": "<g transform=\"translate(2.5,2.5)\"><path d=\"M 3.134 0.534 L 3.617 0.404 L 3.461 -0.177 L 2.918 0.082 L 3.134 0.534 Z M 10.466 7.866 L 10.918 8.081 L 11.177 7.539 L 10.596 7.383 L 10.466 7.866 Z M 4.683 6.316 L 4.2 6.446 L 4.275 6.725 L 4.554 6.799 L 4.683 6.316 Z M 10.983 5.934 L 10.854 6.417 L 11.435 6.573 L 11.482 5.973 L 10.983 5.934 Z M 5.066 0.017 L 5.027 -0.482 L 4.427 -0.435 L 4.583 0.146 L 5.066 0.017 Z M 6.316 4.683 L 5.833 4.813 L 5.908 5.092 L 6.187 5.166 L 6.316 4.683 Z M 0.5 5.5 C 0.5 3.509 1.663 1.79 3.349 0.985 L 2.918 0.082 C 0.898 1.047 -0.5 3.11 -0.5 5.5 L 0.5 5.5 Z M 5.5 10.5 C 2.739 10.5 0.5 8.261 0.5 5.5 L -0.5 5.5 C -0.5 8.814 2.186 11.5 5.5 11.5 L 5.5 10.5 Z M 10.015 7.651 C 9.211 9.337 7.491 10.5 5.5 10.5 L 5.5 11.5 C 7.89 11.5 9.953 10.102 10.918 8.081 L 10.015 7.651 Z M 10.596 7.383 L 4.813 5.833 L 4.554 6.799 L 10.337 8.349 L 10.596 7.383 Z M 5.166 6.187 L 3.617 0.404 L 2.651 0.663 L 4.2 6.446 L 5.166 6.187 Z M 10.5 5.5 C 10.5 5.633 10.495 5.765 10.485 5.895 L 11.482 5.973 C 11.494 5.817 11.5 5.659 11.5 5.5 L 10.5 5.5 Z M 5.5 0.5 C 8.261 0.5 10.5 2.739 10.5 5.5 L 11.5 5.5 C 11.5 2.186 8.814 -0.5 5.5 -0.5 L 5.5 0.5 Z M 5.105 0.515 C 5.235 0.505 5.367 0.5 5.5 0.5 L 5.5 -0.5 C 5.341 -0.5 5.183 -0.494 5.027 -0.482 L 5.105 0.515 Z M 4.583 0.146 L 5.833 4.813 L 6.799 4.554 L 5.549 -0.113 L 4.583 0.146 Z M 6.187 5.166 L 10.854 6.417 L 11.113 5.451 L 6.446 4.2 L 6.187 5.166 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-collapse": "<g transform=\"translate(3,9.5)\"><path d=\"M 3.5 0 L 4 0 L 4 -0.5 L 3.5 -0.5 L 3.5 0 Z M 0.354 3.854 L 3.854 0.354 L 3.146 -0.354 L -0.354 3.146 L 0.354 3.854 Z M 3 0 L 3 3.5 L 4 3.5 L 4 0 L 3 0 Z M 3.5 -0.5 L 0 -0.5 L 0 0.5 L 3.5 0.5 L 3.5 -0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(0,0)\"><path d=\"M 3.5 0 L 4 0 L 4 -0.5 L 3.5 -0.5 L 3.5 0 Z M 0.354 3.854 L 3.854 0.354 L 3.146 -0.354 L -0.354 3.146 L 0.354 3.854 Z M 3 0 L 3 3.5 L 4 3.5 L 4 0 L 3 0 Z M 3.5 -0.5 L 0 -0.5 L 0 0.5 L 3.5 0.5 L 3.5 -0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-expand": "<g transform=\"translate(9,3.5)\"><path d=\"M 3.5 0 L 4 0 L 4 -0.5 L 3.5 -0.5 L 3.5 0 Z M 0.354 3.854 L 3.854 0.354 L 3.146 -0.354 L -0.354 3.146 L 0.354 3.854 Z M 3 0 L 3 3.5 L 4 3.5 L 4 0 L 3 0 Z M 3.5 -0.5 L 0 -0.5 L 0 0.5 L 3.5 0.5 L 3.5 -0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(0,0)\"><path d=\"M 3.5 0 L 4 0 L 4 -0.5 L 3.5 -0.5 L 3.5 0 Z M 0.354 3.854 L 3.854 0.354 L 3.146 -0.354 L -0.354 3.146 L 0.354 3.854 Z M 3 0 L 3 3.5 L 4 3.5 L 4 0 L 3 0 Z M 3.5 -0.5 L 0 -0.5 L 0 0.5 L 3.5 0.5 L 3.5 -0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-fit": "<g transform=\"translate(2.5,2.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 11 0 L 11.5 0 L 11.5 -0.5 L 11 -0.5 L 11 0 Z M 11 11 L 11 11.5 L 11.5 11.5 L 11.5 11 L 11 11 Z M 0 11 L -0.5 11 L -0.5 11.5 L 0 11.5 L 0 11 Z M 8.649 1.646 L 1.647 8.646 L 2.354 9.354 L 9.356 2.353 L 8.649 1.646 Z M 0 0.5 L 11 0.5 L 11 -0.5 L 0 -0.5 L 0 0.5 Z M 10.5 0 L 10.5 11 L 11.5 11 L 11.5 0 L 10.5 0 Z M 11 10.5 L 0 10.5 L 0 11.5 L 11 11.5 L 11 10.5 Z M 0.5 11 L 0.5 0 L -0.5 0 L -0.5 11 L 0.5 11 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(4.5,4.499)\"><path d=\"M 6.649 -0.354 L -0.354 6.647 L 0.354 7.354 L 7.356 0.354 L 6.649 -0.354 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(4.5,4.499)\"><path d=\"M 7.002 0 L 7.502 0 L 7.502 -0.5 L 7.002 -0.5 L 7.002 0 Z M 0.002 6.997 L -0.498 6.997 L -0.498 7.497 L 0.002 7.497 L 0.002 6.997 Z M 4.5 0.5 L 7.002 0.5 L 7.002 -0.5 L 4.5 -0.5 L 4.5 0.5 Z M 6.502 0 L 6.502 2.501 L 7.502 2.501 L 7.502 0 L 6.502 0 Z M 2.503 6.498 L 0.003 6.497 L 0.003 7.497 L 2.503 7.498 L 2.503 6.498 Z M 0.004 6.497 L 0.002 6.497 L 0.002 7.497 L 0.003 7.497 L 0.004 6.497 Z M 0.502 6.997 L 0.502 4.498 L -0.498 4.498 L -0.498 6.997 L 0.502 6.997 Z M -0.35 6.644 L -0.354 6.647 L 0.354 7.354 L 0.357 7.351 L -0.35 6.644 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-grid": "<g transform=\"translate(2,2)\"><path d=\"M 3 0 L 3 12 L 4 12 L 4 0 L 3 0 Z M 8 0 L 8 12 L 9 12 L 9 0 L 8 0 Z M 0 4 L 12 4 L 12 3 L 0 3 L 0 4 Z M 0 9 L 12 9 L 12 8 L 0 8 L 0 9 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-h-align-center": "<g transform=\"translate(8,2)\"><path d=\"M -0.5 0 L -0.5 2.5 L 0.5 2.5 L 0.5 0 L -0.5 0 Z M -0.5 4.5 L -0.5 6.5 L 0.5 6.5 L 0.5 4.5 L -0.5 4.5 Z M -0.5 8.5 L -0.5 12 L 0.5 12 L 0.5 8.5 L -0.5 8.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-image": "<g transform=\"translate(2.5,2.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 11 0 L 11.5 0 L 11.5 -0.5 L 11 -0.5 L 11 0 Z M 11 11 L 11 11.5 L 11.5 11.5 L 11.5 11 L 11 11 Z M 0 11 L -0.5 11 L -0.5 11.5 L 0 11.5 L 0 11 Z M 7.517 6.317 L 7.766 5.884 L 7.517 5.74 L 7.267 5.884 L 7.517 6.317 Z M 4.034 8.324 L 3.784 8.757 L 4.034 8.901 L 4.283 8.757 L 4.034 8.324 Z M 0 0.5 L 11 0.5 L 11 -0.5 L 0 -0.5 L 0 0.5 Z M 10.5 0 L 10.5 11 L 11.5 11 L 11.5 0 L 10.5 0 Z M 11 10.5 L 0 10.5 L 0 11.5 L 11 11.5 L 11 10.5 Z M 0.5 11 L 0.5 0 L -0.5 0 L -0.5 11 L 0.5 11 Z M 7.267 6.75 L 10.75 8.757 L 11.25 7.89 L 7.766 5.884 L 7.267 6.75 Z M 4.283 8.757 L 7.766 6.75 L 7.267 5.884 L 3.784 7.89 L 4.283 8.757 Z M -0.25 6.433 L 3.784 8.757 L 4.283 7.89 L 0.25 5.567 L -0.25 6.433 Z M 4.5 4 C 4.5 4.276 4.276 4.5 4 4.5 L 4 5.5 C 4.828 5.5 5.5 4.828 5.5 4 L 4.5 4 Z M 4 4.5 C 3.724 4.5 3.5 4.276 3.5 4 L 2.5 4 C 2.5 4.828 3.172 5.5 4 5.5 L 4 4.5 Z M 3.5 4 C 3.5 3.724 3.724 3.5 4 3.5 L 4 2.5 C 3.172 2.5 2.5 3.172 2.5 4 L 3.5 4 Z M 4 3.5 C 4.276 3.5 4.5 3.724 4.5 4 L 5.5 4 C 5.5 3.172 4.828 2.5 4 2.5 L 4 3.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-italic": "<g transform=\"translate(4.79,3.5)\"><path d=\"M 0 9.5 L 4 9.5 L 4 8.5 L 0 8.5 L 0 9.5 Z M 3.929 -0.129 L 1.517 8.871 L 2.483 9.129 L 4.895 0.129 L 3.929 -0.129 Z M 2.41 0.5 L 6.41 0.5 L 6.41 -0.5 L 2.41 -0.5 L 2.41 0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-keyframe": "<g transform=\"translate(3,3)\"><path d=\"M 5 0 L 5.354 -0.354 L 5 -0.707 L 4.646 -0.354 L 5 0 Z M 0 5 L -0.354 4.646 L -0.707 5 L -0.354 5.354 L 0 5 Z M 5 10 L 4.646 10.354 L 5 10.707 L 5.354 10.354 L 5 10 Z M 10 5 L 10.354 5.354 L 10.707 5 L 10.354 4.646 L 10 5 Z M 4.646 -0.354 L -0.354 4.646 L 0.354 5.354 L 5.354 0.354 L 4.646 -0.354 Z M -0.354 5.354 L 4.646 10.354 L 5.354 9.646 L 0.354 4.646 L -0.354 5.354 Z M 5.354 10.354 L 10.354 5.354 L 9.646 4.646 L 4.646 9.646 L 5.354 10.354 Z M 10.354 4.646 L 5.354 -0.354 L 4.646 0.354 L 9.646 5.354 L 10.354 4.646 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-layers": "<g transform=\"translate(1.5,1.5)\"><path d=\"M 6.5 0 L 6.752 -0.432 L 6.5 -0.579 L 6.248 -0.432 L 6.5 0 Z M 0.5 3.5 L 0.248 3.068 L -0.492 3.5 L 0.248 3.932 L 0.5 3.5 Z M 6.5 7 L 6.248 7.432 L 6.5 7.579 L 6.752 7.432 L 6.5 7 Z M 12.5 3.5 L 12.752 3.932 L 13.492 3.5 L 12.752 3.068 L 12.5 3.5 Z M 6.5 10 L 6.238 10.426 L 6.5 10.587 L 6.762 10.426 L 6.5 10 Z M 6.5 13 L 6.238 13.426 L 6.5 13.587 L 6.762 13.426 L 6.5 13 Z M 6.248 -0.432 L 0.248 3.068 L 0.752 3.932 L 6.752 0.432 L 6.248 -0.432 Z M 0.248 3.932 L 6.248 7.432 L 6.752 6.568 L 0.752 3.068 L 0.248 3.932 Z M 6.752 7.432 L 12.752 3.932 L 12.248 3.068 L 6.248 6.568 L 6.752 7.432 Z M 12.752 3.068 L 6.752 -0.432 L 6.248 0.432 L 12.248 3.932 L 12.752 3.068 Z M -0.262 6.426 L 6.238 10.426 L 6.762 9.574 L 0.262 5.574 L -0.262 6.426 Z M 6.762 10.426 L 13.262 6.426 L 12.738 5.574 L 6.238 9.574 L 6.762 10.426 Z M -0.262 9.426 L 6.238 13.426 L 6.762 12.574 L 0.262 8.574 L -0.262 9.426 Z M 6.762 13.426 L 13.262 9.426 L 12.738 8.574 L 6.238 12.574 L 6.762 13.426 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-list-bullet": "<g transform=\"translate(2.5,2.5)\"><path d=\"M 3.5 2 C 3.5 2.828 2.828 3.5 2 3.5 L 2 4.5 C 3.381 4.5 4.5 3.381 4.5 2 L 3.5 2 Z M 2 3.5 C 1.172 3.5 0.5 2.828 0.5 2 L -0.5 2 C -0.5 3.381 0.619 4.5 2 4.5 L 2 3.5 Z M 0.5 2 C 0.5 1.172 1.172 0.5 2 0.5 L 2 -0.5 C 0.619 -0.5 -0.5 0.619 -0.5 2 L 0.5 2 Z M 2 0.5 C 2.828 0.5 3.5 1.172 3.5 2 L 4.5 2 C 4.5 0.619 3.381 -0.5 2 -0.5 L 2 0.5 Z M 3.5 9 C 3.5 9.828 2.828 10.5 2 10.5 L 2 11.5 C 3.381 11.5 4.5 10.381 4.5 9 L 3.5 9 Z M 2 10.5 C 1.172 10.5 0.5 9.828 0.5 9 L -0.5 9 C -0.5 10.381 0.619 11.5 2 11.5 L 2 10.5 Z M 0.5 9 C 0.5 8.172 1.172 7.5 2 7.5 L 2 6.5 C 0.619 6.5 -0.5 7.619 -0.5 9 L 0.5 9 Z M 2 7.5 C 2.828 7.5 3.5 8.172 3.5 9 L 4.5 9 C 4.5 7.619 3.381 6.5 2 6.5 L 2 7.5 Z M 6.5 2.5 L 11.5 2.5 L 11.5 1.5 L 6.5 1.5 L 6.5 2.5 Z M 6.5 9.5 L 11.5 9.5 L 11.5 8.5 L 6.5 8.5 L 6.5 9.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-list-checkmark": "<g transform=\"translate(2,2)\"><path d=\"M 1.5 3.5 L 1.146 3.854 L 1.5 4.207 L 1.854 3.854 L 1.5 3.5 Z M 1.5 10.5 L 1.146 10.854 L 1.5 11.207 L 1.854 10.854 L 1.5 10.5 Z M 7 3 L 12 3 L 12 2 L 7 2 L 7 3 Z M 7 10 L 12 10 L 12 9 L 7 9 L 7 10 Z M -0.354 2.354 L 1.146 3.854 L 1.854 3.146 L 0.354 1.646 L -0.354 2.354 Z M 1.854 3.854 L 5.354 0.354 L 4.646 -0.354 L 1.146 3.146 L 1.854 3.854 Z M -0.354 9.354 L 1.146 10.854 L 1.854 10.146 L 0.354 8.646 L -0.354 9.354 Z M 1.854 10.854 L 5.354 7.354 L 4.646 6.646 L 1.146 10.146 L 1.854 10.854 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-list-number": "<g transform=\"translate(2.5,2.5)\"><path d=\"M 1 0 L 1 -0.5 L 0.5 -0.5 L 0.5 0 L 1 0 Z M 1 4 L 0.5 4 L 0.5 4.5 L 1 4.5 L 1 4 Z M 1.01 0 L 1.51 0 L 1.51 -0.5 L 1.01 -0.5 L 1.01 0 Z M 1.01 4 L 1.01 4.5 L 1.51 4.5 L 1.51 4 L 1.01 4 Z M 0 7 L 0 6.5 L -0.5 6.5 L -0.5 7 L 0 7 Z M 2 11 L 2 11.5 L 2.5 11.5 L 2.5 11 L 2 11 Z M 0 7.01 L -0.5 7.01 L -0.5 7.51 L 0 7.51 L 0 7.01 Z M 2 10.99 L 2.5 10.99 L 2.5 10.49 L 2 10.49 L 2 10.99 Z M 4.5 2.5 L 11.5 2.5 L 11.5 1.5 L 4.5 1.5 L 4.5 2.5 Z M 4.5 9.5 L 11.5 9.5 L 11.5 8.5 L 4.5 8.5 L 4.5 9.5 Z M 0.5 0 L 0.5 4 L 1.5 4 L 1.5 0 L 0.5 0 Z M 1 0.5 L 1.01 0.5 L 1.01 -0.5 L 1 -0.5 L 1 0.5 Z M 0.51 0 L 0.51 0.5 L 1.51 0.5 L 1.51 0 L 0.51 0 Z M 1 4.5 L 1.01 4.5 L 1.01 3.5 L 1 3.5 L 1 4.5 Z M 1.51 4 L 1.51 3.5 L 0.51 3.5 L 0.51 4 L 1.51 4 Z M 0 7.5 L 1.5 7.5 L 1.5 6.5 L 0 6.5 L 0 7.5 Z M 1.5 7.5 L 1.5 8.5 L 2.5 8.5 L 2.5 7.5 L 1.5 7.5 Z M 1.5 8.5 L 0.5 8.5 L 0.5 9.5 L 1.5 9.5 L 1.5 8.5 Z M -0.5 9.5 L -0.5 10.5 L 0.5 10.5 L 0.5 9.5 L -0.5 9.5 Z M 0.5 11.5 L 2 11.5 L 2 10.5 L 0.5 10.5 L 0.5 11.5 Z M -0.5 10.5 C -0.5 11.052 -0.052 11.5 0.5 11.5 L 0.5 10.5 L -0.5 10.5 Z M 0.5 8.5 C -0.052 8.5 -0.5 8.948 -0.5 9.5 L 0.5 9.5 L 0.5 8.5 Z M 1.5 8.5 L 1.5 9.5 C 2.052 9.5 2.5 9.052 2.5 8.5 L 1.5 8.5 Z M 1.5 7.5 L 2.5 7.5 C 2.5 6.948 2.052 6.5 1.5 6.5 L 1.5 7.5 Z M -0.5 7 L -0.5 7.01 L 0.5 7.01 L 0.5 7 L -0.5 7 Z M 0 7.51 L 0.5 7.51 L 0.5 6.51 L 0 6.51 L 0 7.51 Z M 2.5 11 L 2.5 10.99 L 1.5 10.99 L 1.5 11 L 2.5 11 Z M 2 10.49 L 1.5 10.49 L 1.5 11.49 L 2 11.49 L 2 10.49 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-move": "<g transform=\"translate(2,2)\"><path d=\"M 6.5 12 L 6.5 0 L 5.5 0 L 5.5 12 L 6.5 12 Z M 12 5.5 L 0 5.5 L 0 6.5 L 12 6.5 L 12 5.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(6,12)\"><path d=\"M 2 2 L 1.646 2.354 L 2 2.707 L 2.354 2.354 L 2 2 Z M 3.646 -0.354 L 1.646 1.646 L 2.354 2.354 L 4.354 0.354 L 3.646 -0.354 Z M 2.354 1.646 L 0.354 -0.354 L -0.354 0.354 L 1.646 2.354 L 2.354 1.646 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(2,6)\"><path d=\"M 0 2 L -0.354 1.646 L -0.707 2 L -0.354 2.354 L 0 2 Z M 1.646 -0.354 L -0.354 1.646 L 0.354 2.354 L 2.354 0.354 L 1.646 -0.354 Z M -0.354 2.354 L 1.646 4.354 L 2.354 3.646 L 0.354 1.646 L -0.354 2.354 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(6,2)\"><path d=\"M 2 0 L 2.354 -0.354 L 2 -0.707 L 1.646 -0.354 L 2 0 Z M 4.354 1.646 L 2.354 -0.354 L 1.646 0.354 L 3.646 2.354 L 4.354 1.646 Z M 1.646 -0.354 L -0.354 1.646 L 0.354 2.354 L 2.354 0.354 L 1.646 -0.354 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(12,6)\"><path d=\"M 2 2 L 2.354 2.354 L 2.707 2 L 2.354 1.646 L 2 2 Z M -0.354 0.354 L 1.646 2.354 L 2.354 1.646 L 0.354 -0.354 L -0.354 0.354 Z M 1.646 1.646 L -0.354 3.646 L 0.354 4.354 L 2.354 2.354 L 1.646 1.646 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-paperclip": "<g transform=\"translate(0,0)\"><path d=\"M 3 5.5 C 1.619 5.5 0.5 4.381 0.5 3 L -0.5 3 C -0.5 4.933 1.067 6.5 3 6.5 L 3 5.5 Z M 0.5 3 C 0.5 1.619 1.619 0.5 3 0.5 L 3 -0.5 C 1.067 -0.5 -0.5 1.067 -0.5 3 L 0.5 3 Z M 12.5 2 C 12.5 2.828 11.828 3.5 11 3.5 L 11 4.5 C 12.381 4.5 13.5 3.381 13.5 2 L 12.5 2 Z M 11 0.5 C 11.828 0.5 12.5 1.172 12.5 2 L 13.5 2 C 13.5 0.619 12.381 -0.5 11 -0.5 L 11 0.5 Z M 3 3.5 C 2.724 3.5 2.5 3.276 2.5 3 L 1.5 3 C 1.5 3.828 2.172 4.5 3 4.5 L 3 3.5 Z M 2.5 3 C 2.5 2.724 2.724 2.5 3 2.5 L 3 1.5 C 2.172 1.5 1.5 2.172 1.5 3 L 2.5 3 Z M 3 0.5 L 11 0.5 L 11 -0.5 L 3 -0.5 L 3 0.5 Z M 11 3.5 L 3 3.5 L 3 4.5 L 11 4.5 L 11 3.5 Z M 3 2.5 L 9.914 2.5 L 9.914 1.5 L 3 1.5 L 3 2.5 Z M 3 6.5 L 9.914 6.5 L 9.914 5.5 L 3 5.5 L 3 6.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-pencil": "<g transform=\"translate(2.5,2.5)\"><path d=\"M 0 9 L -0.354 8.646 L -0.5 8.793 L -0.5 9 L 0 9 Z M 11 2 L 11.354 2.354 L 11.707 2 L 11.354 1.646 L 11 2 Z M 2 11 L 2 11.5 L 2.207 11.5 L 2.354 11.354 L 2 11 Z M 0 11 L -0.5 11 L -0.5 11.5 L 0 11.5 L 0 11 Z M 9 0 L 9.354 -0.354 L 9 -0.707 L 8.646 -0.354 L 9 0 Z M 10.646 1.646 L 1.646 10.646 L 2.354 11.354 L 11.354 2.354 L 10.646 1.646 Z M 2 10.5 L 0 10.5 L 0 11.5 L 2 11.5 L 2 10.5 Z M 0.5 11 L 0.5 9 L -0.5 9 L -0.5 11 L 0.5 11 Z M 0.354 9.354 L 9.354 0.354 L 8.646 -0.354 L -0.354 8.646 L 0.354 9.354 Z M 8.646 0.354 L 10.646 2.354 L 11.354 1.646 L 9.354 -0.354 L 8.646 0.354 Z M 6.646 2.354 L 8.646 4.354 L 9.354 3.646 L 7.354 1.646 L 6.646 2.354 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-rotate": "<g transform=\"translate(1.5,4.5)\"><path d=\"M 6.5 5.5 C 4.761 5.5 3.21 5.174 2.113 4.667 C 0.974 4.142 0.5 3.516 0.5 3 L -0.5 3 C -0.5 4.141 0.481 5.015 1.694 5.575 C 2.95 6.155 4.65 6.5 6.5 6.5 L 6.5 5.5 Z M 0.5 3 C 0.5 2.484 0.974 1.858 2.113 1.333 C 3.21 0.826 4.761 0.5 6.5 0.5 L 6.5 -0.5 C 4.65 -0.5 2.95 -0.155 1.694 0.425 C 0.481 0.985 -0.5 1.859 -0.5 3 L 0.5 3 Z M 6.5 0.5 C 8.239 0.5 9.79 0.826 10.887 1.333 C 12.026 1.858 12.5 2.484 12.5 3 L 13.5 3 C 13.5 1.859 12.519 0.985 11.306 0.425 C 10.05 -0.155 8.35 -0.5 6.5 -0.5 L 6.5 0.5 Z M 6.5 6.5 L 8 6.5 L 8 5.5 L 6.5 5.5 L 6.5 6.5 Z M 12.5 3 C 12.5 3.298 12.35 3.628 11.981 3.97 C 11.611 4.313 11.051 4.634 10.33 4.895 L 10.67 5.835 C 11.471 5.546 12.161 5.166 12.66 4.704 C 13.16 4.241 13.5 3.663 13.5 3 L 12.5 3 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(8,9)\"><path d=\"M 1.5 1.5 L 1.854 1.854 L 2.207 1.5 L 1.854 1.146 L 1.5 1.5 Z M 1.854 1.146 L 0.354 -0.354 L -0.354 0.354 L 1.146 1.854 L 1.854 1.146 Z M 1.146 1.146 L -0.354 2.646 L 0.354 3.354 L 1.854 1.854 L 1.146 1.146 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-scissors": "<g transform=\"translate(2.5,2.5)\"><path d=\"M 3.116 7.34 L 2.837 7.755 L 2.837 7.755 L 3.116 7.34 Z M 3.5 9 C 3.5 9.828 2.828 10.5 2 10.5 L 2 11.5 C 3.381 11.5 4.5 10.381 4.5 9 L 3.5 9 Z M 2 10.5 C 1.172 10.5 0.5 9.828 0.5 9 L -0.5 9 C -0.5 10.381 0.619 11.5 2 11.5 L 2 10.5 Z M 0.5 9 C 0.5 8.172 1.172 7.5 2 7.5 L 2 6.5 C 0.619 6.5 -0.5 7.619 -0.5 9 L 0.5 9 Z M 2 3.5 C 1.172 3.5 0.5 2.828 0.5 2 L -0.5 2 C -0.5 3.381 0.619 4.5 2 4.5 L 2 3.5 Z M 0.5 2 C 0.5 1.172 1.172 0.5 2 0.5 L 2 -0.5 C 0.619 -0.5 -0.5 0.619 -0.5 2 L 0.5 2 Z M 2 0.5 C 2.828 0.5 3.5 1.172 3.5 2 L 4.5 2 C 4.5 0.619 3.381 -0.5 2 -0.5 L 2 0.5 Z M 6.554 5.933 L 11.75 2.933 L 11.25 2.067 L 6.054 5.067 L 6.554 5.933 Z M 6.054 5.933 L 11.25 8.933 L 11.75 8.067 L 6.554 5.067 L 6.054 5.933 Z M 2.866 4.093 L 6.054 5.933 L 6.554 5.067 L 3.366 3.227 L 2.866 4.093 Z M 3.5 2 C 3.5 2.518 3.238 2.975 2.837 3.245 L 3.396 4.074 C 4.061 3.626 4.5 2.864 4.5 2 L 3.5 2 Z M 2.837 3.245 C 2.598 3.406 2.311 3.5 2 3.5 L 2 4.5 C 2.516 4.5 2.997 4.343 3.396 4.074 L 2.837 3.245 Z M 3.366 7.773 L 6.554 5.933 L 6.054 5.067 L 2.866 6.907 L 3.366 7.773 Z M 2 7.5 C 2.311 7.5 2.598 7.594 2.837 7.755 L 3.396 6.926 C 2.997 6.657 2.516 6.5 2 6.5 L 2 7.5 Z M 2.837 7.755 C 3.238 8.025 3.5 8.482 3.5 9 L 4.5 9 C 4.5 8.136 4.061 7.374 3.396 6.926 L 2.837 7.755 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-sliders": "<g transform=\"translate(2,2.5)\"><path d=\"M 3.5 0 L 3.5 -0.5 L 3 -0.5 L 3 0 L 3.5 0 Z M 5.5 0 L 6 0 L 6 -0.5 L 5.5 -0.5 L 5.5 0 Z M 5.5 4 L 5.5 4.5 L 6 4.5 L 6 4 L 5.5 4 Z M 3.5 4 L 3 4 L 3 4.5 L 3.5 4.5 L 3.5 4 Z M 6.5 6 L 6.5 5.5 L 6 5.5 L 6 6 L 6.5 6 Z M 8.5 6 L 9 6 L 9 5.5 L 8.5 5.5 L 8.5 6 Z M 8.5 10 L 8.5 10.5 L 9 10.5 L 9 10 L 8.5 10 Z M 6.5 10 L 6 10 L 6 10.5 L 6.5 10.5 L 6.5 10 Z M 3.5 0.5 L 5.5 0.5 L 5.5 -0.5 L 3.5 -0.5 L 3.5 0.5 Z M 5 0 L 5 4 L 6 4 L 6 0 L 5 0 Z M 5.5 3.5 L 3.5 3.5 L 3.5 4.5 L 5.5 4.5 L 5.5 3.5 Z M 4 4 L 4 0 L 3 0 L 3 4 L 4 4 Z M 6.5 6.5 L 8.5 6.5 L 8.5 5.5 L 6.5 5.5 L 6.5 6.5 Z M 8 6 L 8 10 L 9 10 L 9 6 L 8 6 Z M 8.5 9.5 L 6.5 9.5 L 6.5 10.5 L 8.5 10.5 L 8.5 9.5 Z M 7 10 L 7 6 L 6 6 L 6 10 L 7 10 Z M 0 2.5 L 3.5 2.5 L 3.5 1.5 L 0 1.5 L 0 2.5 Z M 5.5 2.5 L 12 2.5 L 12 1.5 L 5.5 1.5 L 5.5 2.5 Z M 0 8.5 L 6.5 8.5 L 6.5 7.5 L 0 7.5 L 0 8.5 Z M 8.5 8.5 L 12 8.5 L 12 7.5 L 8.5 7.5 L 8.5 8.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-style": "<g transform=\"translate(5,2.75)\"><path d=\"M -0.354 0.354 L 1.146 1.854 L 1.854 1.146 L 0.354 -0.354 L -0.354 0.354 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(0,4.25)\"><path d=\"M 3.484 0.016 L 3.837 -0.337 L 3.719 -0.455 L 3.553 -0.479 L 3.484 0.016 Z M 1.419 1.504 L 1.852 1.755 L 1.419 1.504 Z M 0 3 L -0.158 2.526 L -0.705 2.708 L -0.447 3.224 L 0 3 Z M 3 4 L 3.07 4.495 L 3.07 4.495 L 3 4 Z M 4.984 1.516 L 5.479 1.447 L 5.455 1.281 L 5.337 1.163 L 4.984 1.516 Z M 3.553 -0.479 C 3.454 -0.493 3.353 -0.5 3.25 -0.5 L 3.25 0.5 C 3.305 0.5 3.36 0.504 3.414 0.511 L 3.553 -0.479 Z M 3.25 -0.5 C 2.618 -0.5 2.15 -0.245 1.789 0.113 C 1.451 0.449 1.2 0.887 0.987 1.252 L 1.852 1.755 C 2.08 1.362 2.267 1.048 2.494 0.822 C 2.7 0.618 2.926 0.5 3.25 0.5 L 3.25 -0.5 Z M 0.987 1.252 C 0.597 1.923 0.312 2.369 -0.158 2.526 L 0.158 3.474 C 1.05 3.177 1.508 2.345 1.852 1.755 L 0.987 1.252 Z M -0.447 3.224 C -0.112 3.895 0.509 4.25 1.143 4.417 C 1.774 4.584 2.47 4.58 3.07 4.495 L 2.93 3.505 C 2.418 3.577 1.863 3.573 1.398 3.45 C 0.935 3.328 0.612 3.105 0.447 2.776 L -0.447 3.224 Z M 3.07 4.495 C 4.396 4.308 5.5 3.294 5.5 1.75 L 4.5 1.75 C 4.5 2.731 3.83 3.378 2.93 3.505 L 3.07 4.495 Z M 5.5 1.75 C 5.5 1.647 5.493 1.546 5.479 1.447 L 4.489 1.586 C 4.496 1.64 4.5 1.695 4.5 1.75 L 5.5 1.75 Z M 3.13 0.37 L 4.63 1.87 L 5.337 1.163 L 3.837 -0.337 L 3.13 0.37 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(3.483,0)\"><path d=\"M 1.5 5.766 L 1.146 6.12 L 1.5 6.473 L 1.854 6.12 L 1.5 5.766 Z M 5.766 1.5 L 6.12 1.854 L 6.473 1.5 L 6.12 1.146 L 5.766 1.5 Z M 4.266 0 L 4.62 -0.354 L 4.266 -0.707 L 3.913 -0.354 L 4.266 0 Z M 0 4.266 L -0.354 3.913 L -0.707 4.266 L -0.354 4.62 L 0 4.266 Z M 1.854 6.12 L 6.12 1.854 L 5.413 1.146 L 1.146 5.413 L 1.854 6.12 Z M 6.12 1.146 L 4.62 -0.354 L 3.913 0.354 L 5.413 1.854 L 6.12 1.146 Z M 3.913 -0.354 L -0.354 3.913 L 0.354 4.62 L 4.62 0.354 L 3.913 -0.354 Z M 1.854 5.413 L 0.354 3.913 L -0.354 4.62 L 1.146 6.12 L 1.854 5.413 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(1.5,3.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 0 9 L -0.5 9 L -0.5 9.5 L 0 9.5 L 0 9 Z M 13 9 L 13 9.5 L 13.5 9.5 L 13.5 9 L 13 9 Z M 6.5 -0.5 L 0 -0.5 L 0 0.5 L 6.5 0.5 L 6.5 -0.5 Z M -0.5 0 L -0.5 9 L 0.5 9 L 0.5 0 L -0.5 0 Z M 0 9.5 L 13 9.5 L 13 8.5 L 0 8.5 L 0 9.5 Z M 13.5 9 L 13.5 2.5 L 12.5 2.5 L 12.5 9 L 13.5 9 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-swatches": "<g transform=\"translate(2.239,2.807)\"><path d=\"M 3.261 0.693 L 3.261 0.193 L 2.761 0.193 L 2.761 0.693 L 3.261 0.693 Z M 7.09 0 L 7.219 -0.483 L 6.736 -0.612 L 6.607 -0.129 L 7.09 0 Z M 6.209 9.082 L 6.339 8.599 L 6.339 8.599 L 6.209 9.082 Z M 0 2.32 L -0.129 1.837 L -0.612 1.967 L -0.483 2.449 L 0 2.32 Z M 1.812 9.082 L 1.329 9.211 L 1.458 9.694 L 1.941 9.565 L 1.812 9.082 Z M 6.904 0.693 L 7.387 0.823 L 7.387 0.823 L 6.904 0.693 Z M 11.922 1.295 L 12.405 1.424 L 12.534 0.941 L 12.051 0.812 L 11.922 1.295 Z M 9.592 9.988 L 9.463 10.471 L 9.946 10.6 L 10.075 10.117 L 9.592 9.988 Z M 4.761 8.193 L 3.261 8.193 L 3.261 9.193 L 4.761 9.193 L 4.761 8.193 Z M 6.339 8.599 L 4.89 8.21 L 4.631 9.176 L 6.08 9.565 L 6.339 8.599 Z M 2.295 8.952 L 0.483 2.191 L -0.483 2.449 L 1.329 9.211 L 2.295 8.952 Z M 3.131 8.21 L 1.682 8.599 L 1.941 9.565 L 3.39 9.176 L 3.131 8.21 Z M 5.244 8.823 L 7.387 0.823 L 6.421 0.564 L 4.278 8.564 L 5.244 8.823 Z M 7.387 0.823 L 7.573 0.129 L 6.607 -0.129 L 6.421 0.564 L 7.387 0.823 Z M 3.261 1.193 L 6.904 1.193 L 6.904 0.193 L 3.261 0.193 L 3.261 1.193 Z M 0.129 2.803 L 3.39 1.929 L 3.131 0.963 L -0.129 1.837 L 0.129 2.803 Z M 3.761 8.693 L 3.761 1.446 L 2.761 1.446 L 2.761 8.693 L 3.761 8.693 Z M 3.761 1.446 L 3.761 0.693 L 2.761 0.693 L 2.761 1.446 L 3.761 1.446 Z M 6.961 0.483 L 11.792 1.778 L 12.051 0.812 L 7.219 -0.483 L 6.961 0.483 Z M 10.075 10.117 L 12.405 1.424 L 11.439 1.165 L 9.109 9.859 L 10.075 10.117 Z M 9.722 9.505 L 6.339 8.599 L 6.08 9.565 L 9.463 10.471 L 9.722 9.505 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-text": "<g transform=\"translate(4,3.5)\"><path d=\"M 0 0.5 L 8 0.5 L 8 -0.5 L 0 -0.5 L 0 0.5 Z M 3.5 0 L 3.5 9.5 L 4.5 9.5 L 4.5 0 L 3.5 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-typography": "<g transform=\"translate(1.722,3.5)\"><path d=\"M 3.277 0 L 3.76 -0.129 L 3.661 -0.5 L 3.277 -0.5 L 3.277 0 Z M 2.278 0 L 2.278 -0.5 L 1.894 -0.5 L 1.795 -0.129 L 2.278 0 Z M 0.938 5 L 0.455 4.871 L 0.455 4.871 L 0.938 5 Z M 3.277 0 L 3.277 -0.5 L 2.278 -0.5 L 2.278 0 L 2.278 0.5 L 3.277 0.5 L 3.277 0 Z M 0.938 5 L 0.455 4.871 L -0.483 8.371 L 0 8.5 L 0.483 8.629 L 1.421 5.129 L 0.938 5 Z M 4.617 5 L 4.134 5.129 L 5.072 8.629 L 5.555 8.5 L 6.038 8.371 L 5.1 4.871 L 4.617 5 Z M 0.938 5 L 0.938 5.5 L 4.617 5.5 L 4.617 5 L 4.617 4.5 L 0.938 4.5 L 0.938 5 Z M 2.278 0 L 1.795 -0.129 L 0.455 4.871 L 0.938 5 L 1.421 5.129 L 2.761 0.129 L 2.278 0 Z M 3.277 0 L 2.794 0.129 L 4.134 5.129 L 4.617 5 L 5.1 4.871 L 3.76 -0.129 L 3.277 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(9.5,6.5)\"><path d=\"M 3.5 0 L 3.854 -0.354 L 3.707 -0.5 L 3.5 -0.5 L 3.5 0 Z M 3 5 L 3 5.5 L 3.207 5.5 L 3.354 5.354 L 3 5 Z M 0 4.5 L -0.5 4.5 L -0.5 4.707 L -0.354 4.854 L 0 4.5 Z M 0 3.5 L -0.354 3.146 L -0.5 3.293 L -0.5 3.5 L 0 3.5 Z M 0.5 3 L 0.371 2.517 L 0.241 2.552 L 0.146 2.646 L 0.5 3 Z M 0.5 5 L 0.146 5.354 L 0.293 5.5 L 0.5 5.5 L 0.5 5 Z M 4 0.5 L 4.5 0.5 L 4.5 0.293 L 4.354 0.146 L 4 0.5 Z M 0.5 0 L 0.5 -0.5 L 0.293 -0.5 L 0.146 -0.354 L 0.5 0 Z M 3.646 3.646 L 2.646 4.646 L 3.354 5.354 L 4.354 4.354 L 3.646 3.646 Z M 0.5 4.5 L 0.5 3.5 L -0.5 3.5 L -0.5 4.5 L 0.5 4.5 Z M 0.354 3.854 L 0.854 3.354 L 0.146 2.646 L -0.354 3.146 L 0.354 3.854 Z M 3 4.5 L 0.5 4.5 L 0.5 5.5 L 3 5.5 L 3 4.5 Z M 0.854 4.646 L 0.354 4.146 L -0.354 4.854 L 0.146 5.354 L 0.854 4.646 Z M 3.5 2.062 L 3.5 4 L 4.5 4 L 4.5 2.062 L 3.5 2.062 Z M 0.629 3.483 L 4.129 2.545 L 3.871 1.579 L 0.371 2.517 L 0.629 3.483 Z M 3.146 0.354 L 3.646 0.854 L 4.354 0.146 L 3.854 -0.354 L 3.146 0.354 Z M 3.5 0.5 L 3.5 2.062 L 4.5 2.062 L 4.5 0.5 L 3.5 0.5 Z M 0.354 0.854 L 0.854 0.354 L 0.146 -0.354 L -0.354 0.146 L 0.354 0.854 Z M 0.5 0.5 L 3.5 0.5 L 3.5 -0.5 L 0.5 -0.5 L 0.5 0.5 Z M 3.517 4.129 L 3.919 5.629 L 4.885 5.371 L 4.483 3.871 L 3.517 4.129 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-underline": "<g transform=\"translate(2,3)\"><path d=\"M 0 12 L 12 12 L 12 11 L 0 11 L 0 12 Z M 8.5 6.5 C 8.5 7.881 7.381 9 6 9 L 6 10 C 7.933 10 9.5 8.433 9.5 6.5 L 8.5 6.5 Z M 6 9 C 4.619 9 3.5 7.881 3.5 6.5 L 2.5 6.5 C 2.5 8.433 4.067 10 6 10 L 6 9 Z M 3.5 6.5 L 3.5 0 L 2.5 0 L 2.5 6.5 L 3.5 6.5 Z M 9.5 6.5 L 9.5 0 L 8.5 0 L 8.5 6.5 L 9.5 6.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-v-align-center": "<g transform=\"translate(0,0)\"><path d=\"M -0.5 0 L -0.5 2.5 L 0.5 2.5 L 0.5 0 L -0.5 0 Z M -0.5 4.5 L -0.5 6.5 L 0.5 6.5 L 0.5 4.5 L -0.5 4.5 Z M -0.5 8.5 L -0.5 12 L 0.5 12 L 0.5 8.5 L -0.5 8.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-wand": "<g transform=\"translate(2,0.686)\"><path d=\"M 0 11.814 L -0.354 11.46 L -0.707 11.814 L -0.354 12.167 L 0 11.814 Z M 9.5 5.314 L 9.854 5.667 L 10.207 5.314 L 9.854 4.96 L 9.5 5.314 Z M 1.5 13.314 L 1.146 13.667 L 1.5 14.021 L 1.854 13.667 L 1.5 13.314 Z M 8 3.814 L 8.354 3.46 L 8 3.107 L 7.646 3.46 L 8 3.814 Z M 7.25 7.564 L 7.604 7.917 L 7.604 7.917 L 7.25 7.564 Z M 7.646 4.167 L 9.146 5.667 L 9.854 4.96 L 8.354 3.46 L 7.646 4.167 Z M 1.854 12.96 L 0.354 11.46 L -0.354 12.167 L 1.146 13.667 L 1.854 12.96 Z M 0.354 12.167 L 6.104 6.417 L 5.396 5.71 L -0.354 11.46 L 0.354 12.167 Z M 6.104 6.417 L 8.354 4.167 L 7.646 3.46 L 5.396 5.71 L 6.104 6.417 Z M 5.396 6.417 L 6.896 7.917 L 7.604 7.21 L 6.104 5.71 L 5.396 6.417 Z M 9.146 4.96 L 6.896 7.21 L 7.604 7.917 L 9.854 5.667 L 9.146 4.96 Z M 6.896 7.21 L 1.146 12.96 L 1.854 13.667 L 7.604 7.917 L 6.896 7.21 Z M 2.557 4.373 L 4.489 4.891 L 4.748 3.925 L 2.815 3.407 L 2.557 4.373 Z M 6.093 0.129 L 6.611 2.062 L 7.577 1.803 L 7.059 -0.129 L 6.093 0.129 Z M 11.535 1.071 L 10.121 2.485 L 10.828 3.193 L 12.243 1.778 L 11.535 1.071 Z M 13.443 6.255 L 11.511 5.737 L 11.252 6.703 L 13.184 7.221 L 13.443 6.255 Z M 9.907 10.498 L 9.389 8.566 L 8.423 8.825 L 8.941 10.757 L 9.907 10.498 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "editor-wrench": "<g transform=\"translate(1.5,1.5)\"><path d=\"M 0 11 L -0.354 10.646 L -0.707 11 L -0.354 11.354 L 0 11 Z M 2 13 L 1.646 13.354 L 2 13.707 L 2.354 13.354 L 2 13 Z M 8.235 6.765 L 8.416 6.298 L 8.112 6.181 L 7.882 6.411 L 8.235 6.765 Z M 6.235 4.765 L 6.589 5.118 L 6.819 4.888 L 6.702 4.584 L 6.235 4.765 Z M 9 4 L 8.517 4.129 L 8.592 4.408 L 8.871 4.483 L 9 4 Z M 8.472 2.028 L 8.118 1.675 L 7.914 1.879 L 7.989 2.158 L 8.472 2.028 Z M 10.972 4.528 L 10.842 5.011 L 11.121 5.086 L 11.325 4.882 L 10.972 4.528 Z M 10 0.5 L 10.354 0.854 L 10.5 0.707 L 10.5 0.5 L 10 0.5 Z M 10 0.035 L 10.5 0.035 L 10.5 -0.398 L 10.071 -0.46 L 10 0.035 Z M 12.5 3 L 12.5 2.5 L 12.293 2.5 L 12.146 2.646 L 12.5 3 Z M 12.965 3 L 13.46 2.929 L 13.398 2.5 L 12.965 2.5 L 12.965 3 Z M 13 3.5 L 12.5 3.5 C 12.5 5.157 11.157 6.5 9.5 6.5 L 9.5 7 L 9.5 7.5 C 11.709 7.5 13.5 5.709 13.5 3.5 L 13 3.5 Z M 6 3.5 L 6.5 3.5 C 6.5 1.843 7.843 0.5 9.5 0.5 L 9.5 0 L 9.5 -0.5 C 7.291 -0.5 5.5 1.291 5.5 3.5 L 6 3.5 Z M 2 13 L 2.354 12.646 L 0.354 10.646 L 0 11 L -0.354 11.354 L 1.646 13.354 L 2 13 Z M 8.235 6.765 L 7.882 6.411 L 1.646 12.646 L 2 13 L 2.354 13.354 L 8.589 7.118 L 8.235 6.765 Z M 9.5 7 L 9.5 6.5 C 9.117 6.5 8.752 6.428 8.416 6.298 L 8.235 6.765 L 8.055 7.231 C 8.504 7.405 8.991 7.5 9.5 7.5 L 9.5 7 Z M 6.235 4.765 L 6.702 4.584 C 6.572 4.248 6.5 3.883 6.5 3.5 L 6 3.5 L 5.5 3.5 C 5.5 4.009 5.595 4.496 5.769 4.945 L 6.235 4.765 Z M 6.235 4.765 L 5.882 4.411 L -0.354 10.646 L 0 11 L 0.354 11.354 L 6.589 5.118 L 6.235 4.765 Z M 9 4 L 9.483 3.871 L 8.955 1.899 L 8.472 2.028 L 7.989 2.158 L 8.517 4.129 L 9 4 Z M 9 4 L 8.871 4.483 L 10.842 5.011 L 10.972 4.528 L 11.101 4.045 L 9.129 3.517 L 9 4 Z M 10 0.5 L 9.646 0.146 L 8.118 1.675 L 8.472 2.028 L 8.825 2.382 L 10.354 0.854 L 10 0.5 Z M 9.5 0 L 9.5 0.5 C 9.646 0.5 9.789 0.51 9.929 0.53 L 10 0.035 L 10.071 -0.46 C 9.884 -0.486 9.694 -0.5 9.5 -0.5 L 9.5 0 Z M 10 0.5 L 10.5 0.5 L 10.5 0.035 L 10 0.035 L 9.5 0.035 L 9.5 0.5 L 10 0.5 Z M 12.5 3 L 12.146 2.646 L 10.618 4.175 L 10.972 4.528 L 11.325 4.882 L 12.854 3.354 L 12.5 3 Z M 12.965 3 L 12.47 3.071 C 12.49 3.211 12.5 3.354 12.5 3.5 L 13 3.5 L 13.5 3.5 C 13.5 3.306 13.486 3.116 13.46 2.929 L 12.965 3 Z M 12.5 3 L 12.5 3.5 L 12.965 3.5 L 12.965 3 L 12.965 2.5 L 12.5 2.5 L 12.5 3 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "files-archive": "<g transform=\"translate(2.5,3.5)\"><path d=\"M 1 2 L 1 1.5 L 0.5 1.5 L 0.5 2 L 1 2 Z M 10 2 L 10.5 2 L 10.5 1.5 L 10 1.5 L 10 2 Z M 10 9 L 10 9.5 L 10.5 9.5 L 10.5 9 L 10 9 Z M 1 9 L 0.5 9 L 0.5 9.5 L 1 9.5 L 1 9 Z M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 0 2 L -0.5 2 L -0.5 2.5 L 0 2.5 L 0 2 Z M 11 0 L 11.5 0 L 11.5 -0.5 L 11 -0.5 L 11 0 Z M 11 2 L 11 2.5 L 11.5 2.5 L 11.5 2 L 11 2 Z M 1 2.5 L 10 2.5 L 10 1.5 L 1 1.5 L 1 2.5 Z M 9.5 2 L 9.5 9 L 10.5 9 L 10.5 2 L 9.5 2 Z M 10 8.5 L 1 8.5 L 1 9.5 L 10 9.5 L 10 8.5 Z M 1.5 9 L 1.5 2 L 0.5 2 L 0.5 9 L 1.5 9 Z M 3.5 4.5 L 7.5 4.5 L 7.5 3.5 L 3.5 3.5 L 3.5 4.5 Z M -0.5 0 L -0.5 2 L 0.5 2 L 0.5 0 L -0.5 0 Z M 10.5 0 L 10.5 2 L 11.5 2 L 11.5 0 L 10.5 0 Z M 0 2.5 L 11 2.5 L 11 1.5 L 0 1.5 L 0 2.5 Z M 11 -0.5 L 0 -0.5 L 0 0.5 L 11 0.5 L 11 -0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "files-db": "<g transform=\"translate(2.5,2.5)\"><path d=\"M 10.5 2.5 C 10.5 2.874 10.144 3.375 9.182 3.813 C 8.266 4.229 6.964 4.5 5.5 4.5 L 5.5 5.5 C 7.073 5.5 8.522 5.211 9.596 4.723 C 10.625 4.255 11.5 3.507 11.5 2.5 L 10.5 2.5 Z M 5.5 4.5 C 4.036 4.5 2.734 4.229 1.818 3.813 C 0.856 3.375 0.5 2.874 0.5 2.5 L -0.5 2.5 C -0.5 3.507 0.375 4.255 1.404 4.723 C 2.478 5.211 3.927 5.5 5.5 5.5 L 5.5 4.5 Z M 0.5 2.5 C 0.5 2.126 0.856 1.625 1.818 1.187 C 2.734 0.771 4.036 0.5 5.5 0.5 L 5.5 -0.5 C 3.927 -0.5 2.478 -0.211 1.404 0.277 C 0.375 0.745 -0.5 1.493 -0.5 2.5 L 0.5 2.5 Z M 5.5 0.5 C 6.964 0.5 8.266 0.771 9.182 1.187 C 10.144 1.625 10.5 2.126 10.5 2.5 L 11.5 2.5 C 11.5 1.493 10.625 0.745 9.596 0.277 C 8.522 -0.211 7.073 -0.5 5.5 -0.5 L 5.5 0.5 Z M 10.5 5.5 C 10.5 5.874 10.144 6.375 9.182 6.813 C 8.266 7.229 6.964 7.5 5.5 7.5 L 5.5 8.5 C 7.073 8.5 8.522 8.211 9.596 7.723 C 10.625 7.255 11.5 6.507 11.5 5.5 L 10.5 5.5 Z M 5.5 7.5 C 4.036 7.5 2.734 7.229 1.818 6.813 C 0.856 6.375 0.5 5.874 0.5 5.5 L -0.5 5.5 C -0.5 6.507 0.375 7.255 1.404 7.723 C 2.478 8.211 3.927 8.5 5.5 8.5 L 5.5 7.5 Z M 11.5 5.5 L 11.5 2.5 L 10.5 2.5 L 10.5 5.5 L 11.5 5.5 Z M 0.5 5.5 L 0.5 2.5 L -0.5 2.5 L -0.5 5.5 L 0.5 5.5 Z M 10.5 8.5 C 10.5 8.874 10.144 9.375 9.182 9.813 C 8.266 10.229 6.964 10.5 5.5 10.5 L 5.5 11.5 C 7.073 11.5 8.522 11.211 9.596 10.723 C 10.625 10.255 11.5 9.507 11.5 8.5 L 10.5 8.5 Z M 5.5 10.5 C 4.036 10.5 2.734 10.229 1.818 9.813 C 0.856 9.375 0.5 8.874 0.5 8.5 L -0.5 8.5 C -0.5 9.507 0.375 10.255 1.404 10.723 C 2.478 11.211 3.927 11.5 5.5 11.5 L 5.5 10.5 Z M 0.5 8.5 L 0.5 5.5 L -0.5 5.5 L -0.5 8.5 L 0.5 8.5 Z M 11.5 8.5 L 11.5 5.5 L 10.5 5.5 L 10.5 8.5 L 11.5 8.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "files-document": "<g transform=\"translate(3.5,2.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 9 3 L 9.5 3 L 9.5 2.793 L 9.354 2.646 L 9 3 Z M 9 11 L 9 11.5 L 9.5 11.5 L 9.5 11 L 9 11 Z M 0 11 L -0.5 11 L -0.5 11.5 L 0 11.5 L 0 11 Z M 6 0 L 6.354 -0.354 L 6.207 -0.5 L 6 -0.5 L 6 0 Z M 8.5 3 L 8.5 11 L 9.5 11 L 9.5 3 L 8.5 3 Z M 9 10.5 L 0 10.5 L 0 11.5 L 9 11.5 L 9 10.5 Z M 0.5 11 L 0.5 0 L -0.5 0 L -0.5 11 L 0.5 11 Z M 0 0.5 L 6 0.5 L 6 -0.5 L 0 -0.5 L 0 0.5 Z M 5.646 0.354 L 8.646 3.354 L 9.354 2.646 L 6.354 -0.354 L 5.646 0.354 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(9.5,2.5)\"><path d=\"M 0 3 L -0.5 3 L -0.5 3.5 L 0 3.5 L 0 3 Z M -0.5 0 L -0.5 3 L 0.5 3 L 0.5 0 L -0.5 0 Z M 0 3.5 L 3 3.5 L 3 2.5 L 0 2.5 L 0 3.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "files-document-checkmark": "<g transform=\"translate(3.5,2.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 9 3 L 9.5 3 L 9.5 2.793 L 9.354 2.646 L 9 3 Z M 0 11 L -0.5 11 L -0.5 11.5 L 0 11.5 L 0 11 Z M 6 0 L 6.354 -0.354 L 6.207 -0.5 L 6 -0.5 L 6 0 Z M 0.5 11 L 0.5 0 L -0.5 0 L -0.5 11 L 0.5 11 Z M 0 0.5 L 6 0.5 L 6 -0.5 L 0 -0.5 L 0 0.5 Z M 5.646 0.354 L 8.646 3.354 L 9.354 2.646 L 6.354 -0.354 L 5.646 0.354 Z M 8.5 3 L 8.5 5.5 L 9.5 5.5 L 9.5 3 L 8.5 3 Z M 3.5 10.5 L 0 10.5 L 0 11.5 L 3.5 11.5 L 3.5 10.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(9.5,2.5)\"><path d=\"M 0 3 L -0.5 3 L -0.5 3.5 L 0 3.5 L 0 3 Z M -0.5 0 L -0.5 3 L 0.5 3 L 0.5 0 L -0.5 0 Z M 0 3.5 L 3 3.5 L 3 2.5 L 0 2.5 L 0 3.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(8.5,9)\"><path d=\"M 2 4.5 L 1.646 4.854 L 2 5.207 L 2.354 4.854 L 2 4.5 Z M -0.354 2.854 L 1.646 4.854 L 2.354 4.146 L 0.354 2.146 L -0.354 2.854 Z M 2.354 4.854 L 6.854 0.354 L 6.146 -0.354 L 1.646 4.146 L 2.354 4.854 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "files-document-new": "<g transform=\"translate(3.5,2.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 9 3 L 9.5 3 L 9.5 2.793 L 9.354 2.646 L 9 3 Z M 0 11 L -0.5 11 L -0.5 11.5 L 0 11.5 L 0 11 Z M 6 0 L 6.354 -0.354 L 6.207 -0.5 L 6 -0.5 L 6 0 Z M 0.5 11 L 0.5 0 L -0.5 0 L -0.5 11 L 0.5 11 Z M 0 0.5 L 6 0.5 L 6 -0.5 L 0 -0.5 L 0 0.5 Z M 5.646 0.354 L 8.646 3.354 L 9.354 2.646 L 6.354 -0.354 L 5.646 0.354 Z M 3.5 10.5 L 0 10.5 L 0 11.5 L 3.5 11.5 L 3.5 10.5 Z M 8.5 3 L 8.5 4.5 L 9.5 4.5 L 9.5 3 L 8.5 3 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(9.5,2.5)\"><path d=\"M 0 3 L -0.5 3 L -0.5 3.5 L 0 3.5 L 0 3 Z M -0.5 0 L -0.5 3 L 0.5 3 L 0.5 0 L -0.5 0 Z M 0 3.5 L 3 3.5 L 3 2.5 L 0 2.5 L 0 3.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(11.5,8.5)\"><path d=\"M -0.5 0 L -0.5 6 L 0.5 6 L 0.5 0 L -0.5 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(8.5,11.5)\"><path d=\"M 0 0.5 L 6 0.5 L 6 -0.5 L 0 -0.5 L 0 0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "files-export": "<g transform=\"translate(3.5,2)\"><path d=\"M 0 10.5 L -0.5 10.5 L -0.5 11 L 0 11 L 0 10.5 Z M 0 3.5 L 0 3 L -0.5 3 L -0.5 3.5 L 0 3.5 Z M 9 3.5 L 9.5 3.5 L 9.5 3 L 9 3 L 9 3.5 Z M 9 10.5 L 9 11 L 9.5 11 L 9.5 10.5 L 9 10.5 Z M 0.5 10.5 L 0.5 3.5 L -0.5 3.5 L -0.5 10.5 L 0.5 10.5 Z M 8.5 3.5 L 8.5 10.5 L 9.5 10.5 L 9.5 3.5 L 8.5 3.5 Z M 9 10 L 0 10 L 0 11 L 9 11 L 9 10 Z M 0 4 L 1.5 4 L 1.5 3 L 0 3 L 0 4 Z M 7.5 4 L 9 4 L 9 3 L 7.5 3 L 7.5 4 Z M 4 0 L 4 7 L 5 7 L 5 0 L 4 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(6,2)\"><path d=\"M 2 0 L 2.354 -0.354 L 2 -0.707 L 1.646 -0.354 L 2 0 Z M 0.354 2.354 L 2.354 0.354 L 1.646 -0.354 L -0.354 1.646 L 0.354 2.354 Z M 1.646 0.354 L 3.646 2.354 L 4.354 1.646 L 2.354 -0.354 L 1.646 0.354 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "files-floppy": "<g transform=\"translate(2.5,2.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 11 2 L 11.5 2 L 11.5 1.812 L 11.376 1.671 L 11 2 Z M 11 11 L 11 11.5 L 11.5 11.5 L 11.5 11 L 11 11 Z M 0 11 L -0.5 11 L -0.5 11.5 L 0 11.5 L 0 11 Z M 9.25 0 L 9.626 -0.329 L 9.477 -0.5 L 9.25 -0.5 L 9.25 0 Z M 9 6 L 9.5 6 L 9.5 5.5 L 9 5.5 L 9 6 Z M 2 6 L 2 5.5 L 1.5 5.5 L 1.5 6 L 2 6 Z M 3 4 L 2.5 4 L 2.5 4.5 L 3 4.5 L 3 4 Z M 8 4 L 8 4.5 L 8.5 4.5 L 8.5 4 L 8 4 Z M 10.5 2 L 10.5 11 L 11.5 11 L 11.5 2 L 10.5 2 Z M 11 10.5 L 0 10.5 L 0 11.5 L 11 11.5 L 11 10.5 Z M 0.5 11 L 0.5 0 L -0.5 0 L -0.5 11 L 0.5 11 Z M 0 0.5 L 9.25 0.5 L 9.25 -0.5 L 0 -0.5 L 0 0.5 Z M 8.874 0.329 L 10.624 2.329 L 11.376 1.671 L 9.626 -0.329 L 8.874 0.329 Z M 9.5 11 L 9.5 6 L 8.5 6 L 8.5 11 L 9.5 11 Z M 9 5.5 L 2 5.5 L 2 6.5 L 9 6.5 L 9 5.5 Z M 1.5 6 L 1.5 11 L 2.5 11 L 2.5 6 L 1.5 6 Z M 2.5 0 L 2.5 4 L 3.5 4 L 3.5 0 L 2.5 0 Z M 3 4.5 L 8 4.5 L 8 3.5 L 3 3.5 L 3 4.5 Z M 8.5 4 L 8.5 0 L 7.5 0 L 7.5 4 L 8.5 4 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "files-folder-closed": "<g transform=\"translate(2.5,3.5)\"><path d=\"M 4 0 L 4.483 -0.129 L 4.384 -0.5 L 4 -0.5 L 4 0 Z M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 4.536 2 L 4.053 2.129 L 4.152 2.5 L 4.536 2.5 L 4.536 2 Z M 11 2 L 11.5 2 L 11.5 1.5 L 11 1.5 L 11 2 Z M 11 9 L 11 9.5 L 11.5 9.5 L 11.5 9 L 11 9 Z M 0 9 L -0.5 9 L -0.5 9.5 L 0 9.5 L 0 9 Z M 4 -0.5 L 0 -0.5 L 0 0.5 L 4 0.5 L 4 -0.5 Z M 5.019 1.871 L 4.483 -0.129 L 3.517 0.129 L 4.053 2.129 L 5.019 1.871 Z M 4.536 2.5 L 11 2.5 L 11 1.5 L 4.536 1.5 L 4.536 2.5 Z M 10.5 2 L 10.5 9 L 11.5 9 L 11.5 2 L 10.5 2 Z M 11 8.5 L 0 8.5 L 0 9.5 L 11 9.5 L 11 8.5 Z M 0.5 9 L 0.5 0 L -0.5 0 L -0.5 9 L 0.5 9 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "files-folder-open": "<g transform=\"translate(1.499,3.5)\"><path d=\"M 2.001 4 L 2.001 3.5 L 1.663 3.5 L 1.537 3.814 L 2.001 4 Z M 13.001 4 L 13.465 4.186 L 13.74 3.5 L 13.001 3.5 L 13.001 4 Z M 11 9 L 11 9.5 L 11.338 9.5 L 11.464 9.186 L 11 9 Z M 0.002 9 L -0.498 9 L -0.498 9.5 L 0.002 9.5 L 0.002 9 Z M 4.001 0 L 4.484 -0.129 L 4.385 -0.5 L 4.001 -0.5 L 4.001 0 Z M 0.001 0 L 0.001 -0.5 L -0.499 -0.5 L -0.499 0 L 0.001 0 Z M 4.537 2 L 4.054 2.129 L 4.153 2.5 L 4.537 2.5 L 4.537 2 Z M 11.001 2 L 11.501 2 L 11.501 1.5 L 11.001 1.5 L 11.001 2 Z M 0.464 9.186 L 0.466 9.18 L -0.462 8.809 L -0.464 8.814 L 0.464 9.186 Z M 0.466 9.18 L 2.465 4.186 L 1.537 3.814 L -0.462 8.809 L 0.466 9.18 Z M 13.001 3.5 L 11.001 3.5 L 11.001 4.5 L 13.001 4.5 L 13.001 3.5 Z M 11.001 3.5 L 2.001 3.5 L 2.001 4.5 L 11.001 4.5 L 11.001 3.5 Z M 11.464 9.186 L 13.465 4.186 L 12.537 3.814 L 10.536 8.814 L 11.464 9.186 Z M 0.002 9.5 L 11 9.5 L 11 8.5 L 0.002 8.5 L 0.002 9.5 Z M 4.001 -0.5 L 0.001 -0.5 L 0.001 0.5 L 4.001 0.5 L 4.001 -0.5 Z M 5.02 1.871 L 4.484 -0.129 L 3.518 0.129 L 4.054 2.129 L 5.02 1.871 Z M 4.537 2.5 L 11.001 2.5 L 11.001 1.5 L 4.537 1.5 L 4.537 2.5 Z M 10.501 2 L 10.501 4 L 11.501 4 L 11.501 2 L 10.501 2 Z M 0.502 9 L 0.502 8.994 L -0.498 8.995 L -0.498 9 L 0.502 9 Z M 0.502 8.994 L 0.501 0 L -0.499 0 L -0.498 8.995 L 0.502 8.994 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "files-import": "<g transform=\"translate(3.5,3)\"><path d=\"M 0 9.5 L -0.5 9.5 L -0.5 10 L 0 10 L 0 9.5 Z M 0 2.5 L 0 2 L -0.5 2 L -0.5 2.5 L 0 2.5 Z M 9 2.5 L 9.5 2.5 L 9.5 2 L 9 2 L 9 2.5 Z M 9 9.5 L 9 10 L 9.5 10 L 9.5 9.5 L 9 9.5 Z M 0.5 9.5 L 0.5 2.5 L -0.5 2.5 L -0.5 9.5 L 0.5 9.5 Z M 8.5 2.5 L 8.5 9.5 L 9.5 9.5 L 9.5 2.5 L 8.5 2.5 Z M 9 9 L 0 9 L 0 10 L 9 10 L 9 9 Z M 0 3 L 1.5 3 L 1.5 2 L 0 2 L 0 3 Z M 7.5 3 L 9 3 L 9 2 L 7.5 2 L 7.5 3 Z M 4 0 L 4 7 L 5 7 L 5 0 L 4 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(6,8)\"><path d=\"M 2 2 L 1.646 2.354 L 2 2.707 L 2.354 2.354 L 2 2 Z M 3.646 -0.354 L 1.646 1.646 L 2.354 2.354 L 4.354 0.354 L 3.646 -0.354 Z M 2.354 1.646 L 0.354 -0.354 L -0.354 0.354 L 1.646 2.354 L 2.354 1.646 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "files-split-file": "<g transform=\"translate(3.5,1.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 9 3 L 9.5 3 L 9.5 2.793 L 9.354 2.646 L 9 3 Z M 9 13 L 9 13.5 L 9.5 13.5 L 9.5 13 L 9 13 Z M 0 13 L -0.5 13 L -0.5 13.5 L 0 13.5 L 0 13 Z M 6 0 L 6.354 -0.354 L 6.207 -0.5 L 6 -0.5 L 6 0 Z M 9 13 L 9 12.5 L 0 12.5 L 0 13 L 0 13.5 L 9 13.5 L 9 13 Z M 0 0 L 0 0.5 L 6 0.5 L 6 0 L 6 -0.5 L 0 -0.5 L 0 0 Z M 6 0 L 5.646 0.354 L 8.646 3.354 L 9 3 L 9.354 2.646 L 6.354 -0.354 L 6 0 Z M 9 3 L 8.5 3 L 8.5 5.5 L 9 5.5 L 9.5 5.5 L 9.5 3 L 9 3 Z M 0 5.5 L 0.5 5.5 L 0.5 0 L 0 0 L -0.5 0 L -0.5 5.5 L 0 5.5 Z M 0 13 L 0.5 13 L 0.5 8.5 L 0 8.5 L -0.5 8.5 L -0.5 13 L 0 13 Z M 9 8.5 L 8.5 8.5 L 8.5 13 L 9 13 L 9.5 13 L 9.5 8.5 L 9 8.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(1,8.5)\"><path d=\"M 0 0 L 0 0.5 L 2 0.5 L 2 0 L 2 -0.5 L 0 -0.5 L 0 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(4,8.5)\"><path d=\"M 0 0 L 0 0.5 L 2 0.5 L 2 0 L 2 -0.5 L 0 -0.5 L 0 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(7,8.5)\"><path d=\"M 0 0 L 0 0.5 L 2 0.5 L 2 0 L 2 -0.5 L 0 -0.5 L 0 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(10,8.5)\"><path d=\"M 0 0 L 0 0.5 L 2 0.5 L 2 0 L 2 -0.5 L 0 -0.5 L 0 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(13,8.5)\"><path d=\"M 0 0 L 0 0.5 L 2 0.5 L 2 0 L 2 -0.5 L 0 -0.5 L 0 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(9.5,1.5)\"><path d=\"M 0 3 L -0.5 3 L -0.5 3.5 L 0 3.5 L 0 3 Z M 0 0 L -0.5 0 L -0.5 3 L 0 3 L 0.5 3 L 0.5 0 L 0 0 Z M 0 3 L 0 3.5 L 3 3.5 L 3 3 L 3 2.5 L 0 2.5 L 0 3 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "hardware-battery-100": "<g transform=\"translate(12.5,7.5)\"><path d=\"M 1 0 L 1.5 0 L 1.5 -0.5 L 1 -0.5 L 1 0 Z M 1 1 L 1 1.5 L 1.5 1.5 L 1.5 1 L 1 1 Z M 0 0.5 L 1 0.5 L 1 -0.5 L 0 -0.5 L 0 0.5 Z M 0.5 0 L 0.5 1 L 1.5 1 L 1.5 0 L 0.5 0 Z M 1 0.5 L 0 0.5 L 0 1.5 L 1 1.5 L 1 0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(4.5,7)\"><path d=\"M -0.5 0 L -0.5 2 L 0.5 2 L 0.5 0 L -0.5 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(6.5,7)\"><path d=\"M -0.5 0 L -0.5 2 L 0.5 2 L 0.5 0 L -0.5 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(8.5,7)\"><path d=\"M -0.5 0 L -0.5 2 L 0.5 2 L 0.5 0 L -0.5 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(10.5,7)\"><path d=\"M -0.5 0 L -0.5 2 L 0.5 2 L 0.5 0 L -0.5 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(2.5,5.5)\"><path d=\"M 10 0 L 10.5 0 L 10.5 -0.5 L 10 -0.5 L 10 0 Z M 10 5 L 10 5.5 L 10.5 5.5 L 10.5 5 L 10 5 Z M 0 5 L -0.5 5 L -0.5 5.5 L 0 5.5 L 0 5 Z M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 10 4.5 L 0 4.5 L 0 5.5 L 10 5.5 L 10 4.5 Z M 0 0.5 L 10 0.5 L 10 -0.5 L 0 -0.5 L 0 0.5 Z M 0.5 5 L 0.5 0 L -0.5 0 L -0.5 5 L 0.5 5 Z M 9.5 0 L 9.5 5 L 10.5 5 L 10.5 0 L 9.5 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "hardware-battery-charging": "<g transform=\"translate(6,4)\"><path d=\"M 2.5 0 L 3 0 L 3 -1.93 L 2.063 -0.243 L 2.5 0 Z M 0 4.5 L -0.437 4.257 L -0.85 5 L 0 5 L 0 4.5 Z M 1.5 4.5 L 2 4.5 L 2 4 L 1.5 4 L 1.5 4.5 Z M 1.5 8 L 1 8 L 1 9.93 L 1.937 8.243 L 1.5 8 Z M 4 3.5 L 4.437 3.743 L 4.85 3 L 4 3 L 4 3.5 Z M 2.5 3.5 L 2 3.5 L 2 4 L 2.5 4 L 2.5 3.5 Z M 2.063 -0.243 L -0.437 4.257 L 0.437 4.743 L 2.937 0.243 L 2.063 -0.243 Z M 0 5 L 1.5 5 L 1.5 4 L 0 4 L 0 5 Z M 1 4.5 L 1 8 L 2 8 L 2 4.5 L 1 4.5 Z M 1.937 8.243 L 4.437 3.743 L 3.563 3.257 L 1.063 7.757 L 1.937 8.243 Z M 4 3 L 2.5 3 L 2.5 4 L 4 4 L 4 3 Z M 3 3.5 L 3 0 L 2 0 L 2 3.5 L 3 3.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(12.5,7.5)\"><path d=\"M 1 0 L 1.5 0 L 1.5 -0.5 L 1 -0.5 L 1 0 Z M 1 1 L 1 1.5 L 1.5 1.5 L 1.5 1 L 1 1 Z M 0 0.5 L 1 0.5 L 1 -0.5 L 0 -0.5 L 0 0.5 Z M 0.5 0 L 0.5 1 L 1.5 1 L 1.5 0 L 0.5 0 Z M 1 0.5 L 0 0.5 L 0 1.5 L 1 1.5 L 1 0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(10,5.5)\"><path d=\"M 2.5 0 L 3 0 L 3 -0.5 L 2.5 -0.5 L 2.5 0 Z M 2.5 5 L 2.5 5.5 L 3 5.5 L 3 5 L 2.5 5 Z M 2 0 L 2 5 L 3 5 L 3 0 L 2 0 Z M 0 0.5 L 2.5 0.5 L 2.5 -0.5 L 0 -0.5 L 0 0.5 Z M 2.5 4.5 L 1 4.5 L 1 5.5 L 2.5 5.5 L 2.5 4.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(2.5,5.5)\"><path d=\"M 0 5 L -0.5 5 L -0.5 5.5 L 0 5.5 L 0 5 Z M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 3.5 4.5 L 0 4.5 L 0 5.5 L 3.5 5.5 L 3.5 4.5 Z M 0.5 5 L 0.5 0 L -0.5 0 L -0.5 5 L 0.5 5 Z M 0 0.5 L 2.5 0.5 L 2.5 -0.5 L 0 -0.5 L 0 0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "hardware-bluetooth": "<g transform=\"translate(4,2)\"><path d=\"M 6.5 3 L 6.854 3.354 L 7.207 3 L 6.854 2.646 L 6.5 3 Z M 3.5 0 L 3.854 -0.354 L 3 -1.207 L 3 0 L 3.5 0 Z M 3.5 12 L 3 12 L 3 13.207 L 3.854 12.354 L 3.5 12 Z M 6.5 9 L 6.854 9.354 L 7.207 9 L 6.854 8.646 L 6.5 9 Z M 0.354 9.854 L 6.854 3.354 L 6.146 2.646 L -0.354 9.146 L 0.354 9.854 Z M 6.854 2.646 L 3.854 -0.354 L 3.146 0.354 L 6.146 3.354 L 6.854 2.646 Z M 3 0 L 3 12 L 4 12 L 4 0 L 3 0 Z M 3.854 12.354 L 6.854 9.354 L 6.146 8.646 L 3.146 11.646 L 3.854 12.354 Z M -0.354 2.854 L 6.146 9.354 L 6.854 8.646 L 0.354 2.146 L -0.354 2.854 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "hardware-calculator": "<g transform=\"translate(4,2.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 8 0 L 8.5 0 L 8.5 -0.5 L 8 -0.5 L 8 0 Z M 8 11 L 8 11.5 L 8.5 11.5 L 8.5 11 L 8 11 Z M 0 11 L -0.5 11 L -0.5 11.5 L 0 11.5 L 0 11 Z M 0 0.5 L 8 0.5 L 8 -0.5 L 0 -0.5 L 0 0.5 Z M 7.5 0 L 7.5 11 L 8.5 11 L 8.5 0 L 7.5 0 Z M 8 10.5 L 0 10.5 L 0 11.5 L 8 11.5 L 8 10.5 Z M 0.5 11 L 0.5 0 L -0.5 0 L -0.5 11 L 0.5 11 Z M 0 3.5 L 8 3.5 L 8 2.5 L 0 2.5 L 0 3.5 Z M 1.5 9.5 L 2.5 9.5 L 2.5 8.5 L 1.5 8.5 L 1.5 9.5 Z M 1.5 7.5 L 2.5 7.5 L 2.5 6.5 L 1.5 6.5 L 1.5 7.5 Z M 1.5 5.5 L 2.5 5.5 L 2.5 4.5 L 1.5 4.5 L 1.5 5.5 Z M 3.5 9.5 L 4.5 9.5 L 4.5 8.5 L 3.5 8.5 L 3.5 9.5 Z M 3.5 7.5 L 4.5 7.5 L 4.5 6.5 L 3.5 6.5 L 3.5 7.5 Z M 3.5 5.5 L 4.5 5.5 L 4.5 4.5 L 3.5 4.5 L 3.5 5.5 Z M 5.5 9.5 L 6.5 9.5 L 6.5 8.5 L 5.5 8.5 L 5.5 9.5 Z M 5.5 7.5 L 6.5 7.5 L 6.5 6.5 L 5.5 6.5 L 5.5 7.5 Z M 5.5 5.5 L 6.5 5.5 L 6.5 4.5 L 5.5 4.5 L 5.5 5.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "hardware-cpu": "<g transform=\"translate(2,2)\"><path d=\"M 4.5 4.5 L 4.5 4 L 4 4 L 4 4.5 L 4.5 4.5 Z M 7.5 4.5 L 8 4.5 L 8 4 L 7.5 4 L 7.5 4.5 Z M 7.5 7.5 L 7.5 8 L 8 8 L 8 7.5 L 7.5 7.5 Z M 4.5 7.5 L 4 7.5 L 4 8 L 4.5 8 L 4.5 7.5 Z M 1.5 1.5 L 1.5 1 L 1 1 L 1 1.5 L 1.5 1.5 Z M 10.5 1.5 L 11 1.5 L 11 1 L 10.5 1 L 10.5 1.5 Z M 10.5 10.5 L 10.5 11 L 11 11 L 11 10.5 L 10.5 10.5 Z M 1.5 10.5 L 1 10.5 L 1 11 L 1.5 11 L 1.5 10.5 Z M 4.5 5 L 7.5 5 L 7.5 4 L 4.5 4 L 4.5 5 Z M 7 4.5 L 7 7.5 L 8 7.5 L 8 4.5 L 7 4.5 Z M 7.5 7 L 4.5 7 L 4.5 8 L 7.5 8 L 7.5 7 Z M 5 7.5 L 5 4.5 L 4 4.5 L 4 7.5 L 5 7.5 Z M 1.5 2 L 10.5 2 L 10.5 1 L 1.5 1 L 1.5 2 Z M 10 1.5 L 10 10.5 L 11 10.5 L 11 1.5 L 10 1.5 Z M 10.5 10 L 1.5 10 L 1.5 11 L 10.5 11 L 10.5 10 Z M 2 10.5 L 2 1.5 L 1 1.5 L 1 10.5 L 2 10.5 Z M 4.5 1.5 L 4.5 0 L 3.5 0 L 3.5 1.5 L 4.5 1.5 Z M 4.5 12 L 4.5 10.5 L 3.5 10.5 L 3.5 12 L 4.5 12 Z M 6.5 1.5 L 6.5 0 L 5.5 0 L 5.5 1.5 L 6.5 1.5 Z M 6.5 12 L 6.5 10.5 L 5.5 10.5 L 5.5 12 L 6.5 12 Z M 8.5 1.5 L 8.5 0 L 7.5 0 L 7.5 1.5 L 8.5 1.5 Z M 8.5 12 L 8.5 10.5 L 7.5 10.5 L 7.5 12 L 8.5 12 Z M 10.5 4.5 L 12 4.5 L 12 3.5 L 10.5 3.5 L 10.5 4.5 Z M 0 4.5 L 1.5 4.5 L 1.5 3.5 L 0 3.5 L 0 4.5 Z M 10.5 6.5 L 12 6.5 L 12 5.5 L 10.5 5.5 L 10.5 6.5 Z M 0 6.5 L 1.5 6.5 L 1.5 5.5 L 0 5.5 L 0 6.5 Z M 10.5 8.5 L 12 8.5 L 12 7.5 L 10.5 7.5 L 10.5 8.5 Z M 0 8.5 L 1.5 8.5 L 1.5 7.5 L 0 7.5 L 0 8.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "hardware-display": "<g transform=\"translate(2.5,3.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 11 0 L 11.5 0 L 11.5 -0.5 L 11 -0.5 L 11 0 Z M 11 7 L 11 7.5 L 11.5 7.5 L 11.5 7 L 11 7 Z M 0 7 L -0.5 7 L -0.5 7.5 L 0 7.5 L 0 7 Z M 0 0.5 L 11 0.5 L 11 -0.5 L 0 -0.5 L 0 0.5 Z M 10.5 0 L 10.5 7 L 11.5 7 L 11.5 0 L 10.5 0 Z M 11 6.5 L 0 6.5 L 0 7.5 L 11 7.5 L 11 6.5 Z M 0.5 7 L 0.5 0 L -0.5 0 L -0.5 7 L 0.5 7 Z M 5.198 9.129 L 5.734 7.129 L 4.768 6.871 L 4.232 8.871 L 5.198 9.129 Z M 2.5 9.5 L 8.5 9.5 L 8.5 8.5 L 2.5 8.5 L 2.5 9.5 Z M 6.769 8.871 L 6.233 6.871 L 5.267 7.129 L 5.803 9.129 L 6.769 8.871 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "hardware-display-collection": "<g transform=\"translate(1.5,2.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 11 0 L 11.5 0 L 11.5 -0.5 L 11 -0.5 L 11 0 Z M 0 7 L -0.5 7 L -0.5 7.5 L 0 7.5 L 0 7 Z M 0 0.5 L 11 0.5 L 11 -0.5 L 0 -0.5 L 0 0.5 Z M 0.5 7 L 0.5 0 L -0.5 0 L -0.5 7 L 0.5 7 Z M 0.5 6.5 L 0 6.5 L 0 7.5 L 0.5 7.5 L 0.5 6.5 Z M 10.5 0 L 10.5 0.5 L 11.5 0.5 L 11.5 0 L 10.5 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(3.5,4.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 11 0 L 11.5 0 L 11.5 -0.5 L 11 -0.5 L 11 0 Z M 11 7 L 11 7.5 L 11.5 7.5 L 11.5 7 L 11 7 Z M 0 7 L -0.5 7 L -0.5 7.5 L 0 7.5 L 0 7 Z M 0 0.5 L 11 0.5 L 11 -0.5 L 0 -0.5 L 0 0.5 Z M 10.5 0 L 10.5 7 L 11.5 7 L 11.5 0 L 10.5 0 Z M 11 6.5 L 0 6.5 L 0 7.5 L 11 7.5 L 11 6.5 Z M 0.5 7 L 0.5 0 L -0.5 0 L -0.5 7 L 0.5 7 Z M 5.198 9.129 L 5.734 7.129 L 4.768 6.871 L 4.232 8.871 L 5.198 9.129 Z M 2.5 9.5 L 8.5 9.5 L 8.5 8.5 L 2.5 8.5 L 2.5 9.5 Z M 6.769 8.871 L 6.233 6.871 L 5.267 7.129 L 5.803 9.129 L 6.769 8.871 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "hardware-drive-cloud": "<g transform=\"translate(1.5,4.5)\"><path d=\"M 7.115 9.5 C 6.223 9.5 5.5 8.828 5.5 8 C 5.5 7.257 6.183 6.446 7.009 6.629 C 6.674 5.528 7.803 4.5 9 4.5 C 10.026 4.5 11.16 5.354 10.972 6.198 C 11.78 6.051 12.5 6.939 12.5 7.75 C 12.5 8.716 11.656 9.5 10.615 9.5 C 8.898 9.5 8.727 9.5 7.115 9.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /> <path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 13 0 L 13.5 0 L 13.5 -0.5 L 13 -0.5 L 13 0 Z M 0 4 L -0.5 4 L -0.5 4.5 L 0 4.5 L 0 4 Z M 1.5 2.5 L 2.5 2.5 L 2.5 1.5 L 1.5 1.5 L 1.5 2.5 Z M 0 0.5 L 13 0.5 L 13 -0.5 L 0 -0.5 L 0 0.5 Z M 0.5 4 L 0.5 0 L -0.5 0 L -0.5 4 L 0.5 4 Z M 5.154 3.5 L 0 3.5 L 0 4.5 L 5.154 4.5 L 5.154 3.5 Z M 12.5 0 L 12.5 4.5 L 13.5 4.5 L 13.5 0 L 12.5 0 Z M 7.009 6.629 L 6.792 7.605 L 8.464 7.977 L 7.966 6.338 L 7.009 6.629 Z M 10.972 6.198 L 9.996 5.981 L 9.668 7.452 L 11.151 7.182 L 10.972 6.198 Z M 7.115 8.5 C 6.704 8.5 6.5 8.208 6.5 8 L 4.5 8 C 4.5 9.449 5.742 10.5 7.115 10.5 L 7.115 8.5 Z M 6.5 8 C 6.5 7.965 6.509 7.91 6.537 7.844 C 6.566 7.779 6.606 7.72 6.65 7.676 C 6.694 7.632 6.73 7.614 6.75 7.607 C 6.763 7.603 6.774 7.601 6.792 7.605 L 7.226 5.653 C 6.411 5.472 5.701 5.799 5.24 6.258 C 4.792 6.703 4.5 7.341 4.5 8 L 6.5 8 Z M 7.966 6.338 C 7.93 6.22 7.944 6.054 8.141 5.855 C 8.348 5.646 8.675 5.5 9 5.5 L 9 3.5 C 8.128 3.5 7.292 3.868 6.719 4.449 C 6.136 5.039 5.754 5.938 6.052 6.92 L 7.966 6.338 Z M 9 5.5 C 9.259 5.5 9.579 5.618 9.807 5.805 C 10.059 6.013 9.972 6.089 9.996 5.981 L 11.948 6.416 C 12.161 5.463 11.6 4.69 11.076 4.26 C 10.527 3.809 9.767 3.5 9 3.5 L 9 5.5 Z M 11.151 7.182 C 11.151 7.182 11.151 7.182 11.15 7.182 C 11.149 7.182 11.148 7.182 11.149 7.182 C 11.149 7.182 11.155 7.183 11.168 7.188 C 11.195 7.199 11.245 7.229 11.303 7.293 C 11.432 7.432 11.5 7.621 11.5 7.75 L 13.5 7.75 C 13.5 7.069 13.208 6.408 12.774 5.937 C 12.339 5.465 11.633 5.061 10.793 5.214 L 11.151 7.182 Z M 11.5 7.75 C 11.5 8.096 11.175 8.5 10.615 8.5 L 10.615 10.5 C 12.137 10.5 13.5 9.337 13.5 7.75 L 11.5 7.75 Z M 10.615 8.5 C 10.612 8.5 10.609 8.5 10.605 8.5 C 10.602 8.5 10.599 8.5 10.595 8.5 C 10.592 8.5 10.589 8.5 10.585 8.5 C 10.582 8.5 10.579 8.5 10.575 8.5 C 10.572 8.5 10.569 8.5 10.565 8.5 C 10.562 8.5 10.559 8.5 10.556 8.5 C 10.552 8.5 10.549 8.5 10.546 8.5 C 10.543 8.5 10.539 8.5 10.536 8.5 C 10.533 8.5 10.529 8.5 10.526 8.5 C 10.523 8.5 10.52 8.5 10.516 8.5 C 10.513 8.5 10.51 8.5 10.507 8.5 C 10.504 8.5 10.5 8.5 10.497 8.5 C 10.494 8.5 10.491 8.5 10.487 8.5 C 10.484 8.5 10.481 8.5 10.478 8.5 C 10.475 8.5 10.471 8.5 10.468 8.5 C 10.465 8.5 10.462 8.5 10.459 8.5 C 10.456 8.5 10.452 8.5 10.449 8.5 C 10.446 8.5 10.443 8.5 10.44 8.5 C 10.437 8.5 10.434 8.5 10.43 8.5 C 10.427 8.5 10.424 8.5 10.421 8.5 C 10.418 8.5 10.415 8.5 10.412 8.5 C 10.409 8.5 10.405 8.5 10.402 8.5 C 10.399 8.5 10.396 8.5 10.393 8.5 C 10.39 8.5 10.387 8.5 10.384 8.5 C 10.381 8.5 10.378 8.5 10.374 8.5 C 10.371 8.5 10.368 8.5 10.365 8.5 C 10.362 8.5 10.359 8.5 10.356 8.5 C 10.353 8.5 10.35 8.5 10.347 8.5 C 10.344 8.5 10.341 8.5 10.338 8.5 C 10.335 8.5 10.332 8.5 10.329 8.5 C 10.326 8.5 10.323 8.5 10.32 8.5 C 10.317 8.5 10.314 8.5 10.311 8.5 C 10.308 8.5 10.305 8.5 10.302 8.5 C 10.299 8.5 10.296 8.5 10.293 8.5 C 10.29 8.5 10.287 8.5 10.284 8.5 C 10.281 8.5 10.278 8.5 10.275 8.5 C 10.272 8.5 10.269 8.5 10.266 8.5 C 10.263 8.5 10.26 8.5 10.257 8.5 C 10.254 8.5 10.251 8.5 10.248 8.5 C 10.246 8.5 10.243 8.5 10.24 8.5 C 10.237 8.5 10.234 8.5 10.231 8.5 C 10.228 8.5 10.225 8.5 10.222 8.5 C 10.219 8.5 10.216 8.5 10.214 8.5 C 10.211 8.5 10.208 8.5 10.205 8.5 C 10.202 8.5 10.199 8.5 10.196 8.5 C 10.193 8.5 10.191 8.5 10.188 8.5 C 10.185 8.5 10.182 8.5 10.179 8.5 C 10.176 8.5 10.173 8.5 10.171 8.5 C 10.168 8.5 10.165 8.5 10.162 8.5 C 10.159 8.5 10.156 8.5 10.154 8.5 C 10.151 8.5 10.148 8.5 10.145 8.5 C 10.142 8.5 10.14 8.5 10.137 8.5 C 10.134 8.5 10.131 8.5 10.128 8.5 C 10.126 8.5 10.123 8.5 10.12 8.5 C 10.117 8.5 10.114 8.5 10.112 8.5 C 10.109 8.5 10.106 8.5 10.103 8.5 C 10.101 8.5 10.098 8.5 10.095 8.5 C 10.092 8.5 10.09 8.5 10.087 8.5 C 10.084 8.5 10.081 8.5 10.079 8.5 C 10.076 8.5 10.073 8.5 10.07 8.5 C 10.068 8.5 10.065 8.5 10.062 8.5 C 10.06 8.5 10.057 8.5 10.054 8.5 C 10.051 8.5 10.049 8.5 10.046 8.5 C 10.043 8.5 10.041 8.5 10.038 8.5 C 10.035 8.5 10.033 8.5 10.03 8.5 C 10.027 8.5 10.024 8.5 10.022 8.5 C 10.019 8.5 10.016 8.5 10.014 8.5 C 10.011 8.5 10.008 8.5 10.006 8.5 C 10.003 8.5 10.001 8.5 9.998 8.5 C 9.995 8.5 9.993 8.5 9.99 8.5 C 9.987 8.5 9.985 8.5 9.982 8.5 C 9.979 8.5 9.977 8.5 9.974 8.5 C 9.972 8.5 9.969 8.5 9.966 8.5 C 9.964 8.5 9.961 8.5 9.958 8.5 C 9.956 8.5 9.953 8.5 9.951 8.5 C 9.948 8.5 9.945 8.5 9.943 8.5 C 9.94 8.5 9.938 8.5 9.935 8.5 C 9.933 8.5 9.93 8.5 9.927 8.5 C 9.925 8.5 9.922 8.5 9.92 8.5 C 9.917 8.5 9.915 8.5 9.912 8.5 C 9.91 8.5 9.907 8.5 9.904 8.5 C 9.902 8.5 9.899 8.5 9.897 8.5 C 9.894 8.5 9.892 8.5 9.889 8.5 C 9.887 8.5 9.884 8.5 9.882 8.5 C 9.879 8.5 9.877 8.5 9.874 8.5 C 9.872 8.5 9.869 8.5 9.867 8.5 C 9.864 8.5 9.862 8.5 9.859 8.5 C 9.857 8.5 9.854 8.5 9.852 8.5 C 9.849 8.5 9.847 8.5 9.844 8.5 C 9.842 8.5 9.839 8.5 9.837 8.5 C 9.834 8.5 9.832 8.5 9.829 8.5 C 9.827 8.5 9.824 8.5 9.822 8.5 C 9.819 8.5 9.817 8.5 9.815 8.5 C 9.812 8.5 9.81 8.5 9.807 8.5 C 9.805 8.5 9.802 8.5 9.8 8.5 C 9.797 8.5 9.795 8.5 9.793 8.5 C 9.79 8.5 9.788 8.5 9.785 8.5 C 9.783 8.5 9.78 8.5 9.778 8.5 C 9.776 8.5 9.773 8.5 9.771 8.5 C 9.768 8.5 9.766 8.5 9.764 8.5 C 9.761 8.5 9.759 8.5 9.756 8.5 C 9.754 8.5 9.752 8.5 9.749 8.5 C 9.747 8.5 9.745 8.5 9.742 8.5 C 9.74 8.5 9.737 8.5 9.735 8.5 C 9.733 8.5 9.73 8.5 9.728 8.5 C 9.726 8.5 9.723 8.5 9.721 8.5 C 9.719 8.5 9.716 8.5 9.714 8.5 C 9.711 8.5 9.709 8.5 9.707 8.5 C 9.704 8.5 9.702 8.5 9.7 8.5 C 9.697 8.5 9.695 8.5 9.693 8.5 C 9.69 8.5 9.688 8.5 9.686 8.5 C 9.684 8.5 9.681 8.5 9.679 8.5 C 9.677 8.5 9.674 8.5 9.672 8.5 C 9.67 8.5 9.667 8.5 9.665 8.5 C 9.663 8.5 9.66 8.5 9.658 8.5 C 9.656 8.5 9.654 8.5 9.651 8.5 C 9.649 8.5 9.647 8.5 9.644 8.5 C 9.642 8.5 9.64 8.5 9.638 8.5 C 9.635 8.5 9.633 8.5 9.631 8.5 C 9.629 8.5 9.626 8.5 9.624 8.5 C 9.622 8.5 9.619 8.5 9.617 8.5 C 9.615 8.5 9.613 8.5 9.61 8.5 C 9.608 8.5 9.606 8.5 9.604 8.5 C 9.602 8.5 9.599 8.5 9.597 8.5 C 9.595 8.5 9.593 8.5 9.59 8.5 C 9.588 8.5 9.586 8.5 9.584 8.5 C 9.581 8.5 9.579 8.5 9.577 8.5 C 9.575 8.5 9.573 8.5 9.57 8.5 C 9.568 8.5 9.566 8.5 9.564 8.5 C 9.562 8.5 9.559 8.5 9.557 8.5 C 9.555 8.5 9.553 8.5 9.551 8.5 C 9.548 8.5 9.546 8.5 9.544 8.5 C 9.542 8.5 9.54 8.5 9.537 8.5 C 9.535 8.5 9.533 8.5 9.531 8.5 C 9.529 8.5 9.527 8.5 9.524 8.5 C 9.522 8.5 9.52 8.5 9.518 8.5 C 9.516 8.5 9.514 8.5 9.511 8.5 C 9.509 8.5 9.507 8.5 9.505 8.5 C 9.503 8.5 9.501 8.5 9.499 8.5 C 9.496 8.5 9.494 8.5 9.492 8.5 C 9.49 8.5 9.488 8.5 9.486 8.5 C 9.484 8.5 9.481 8.5 9.479 8.5 C 9.477 8.5 9.475 8.5 9.473 8.5 C 9.471 8.5 9.469 8.5 9.467 8.5 C 9.464 8.5 9.462 8.5 9.46 8.5 C 9.458 8.5 9.456 8.5 9.454 8.5 C 9.452 8.5 9.45 8.5 9.448 8.5 C 9.446 8.5 9.443 8.5 9.441 8.5 C 9.439 8.5 9.437 8.5 9.435 8.5 C 9.433 8.5 9.431 8.5 9.429 8.5 C 9.427 8.5 9.425 8.5 9.423 8.5 C 9.42 8.5 9.418 8.5 9.416 8.5 C 9.414 8.5 9.412 8.5 9.41 8.5 C 9.408 8.5 9.406 8.5 9.404 8.5 C 9.402 8.5 9.4 8.5 9.398 8.5 C 9.396 8.5 9.394 8.5 9.392 8.5 C 9.389 8.5 9.387 8.5 9.385 8.5 C 9.383 8.5 9.381 8.5 9.379 8.5 C 9.377 8.5 9.375 8.5 9.373 8.5 C 9.371 8.5 9.369 8.5 9.367 8.5 C 9.365 8.5 9.363 8.5 9.361 8.5 C 9.359 8.5 9.357 8.5 9.355 8.5 C 9.353 8.5 9.351 8.5 9.349 8.5 C 9.347 8.5 9.345 8.5 9.343 8.5 C 9.341 8.5 9.339 8.5 9.337 8.5 C 9.335 8.5 9.333 8.5 9.331 8.5 C 9.329 8.5 9.327 8.5 9.325 8.5 C 9.323 8.5 9.321 8.5 9.319 8.5 C 9.317 8.5 9.315 8.5 9.313 8.5 C 9.311 8.5 9.309 8.5 9.307 8.5 C 9.305 8.5 9.303 8.5 9.301 8.5 C 9.299 8.5 9.297 8.5 9.295 8.5 C 9.293 8.5 9.291 8.5 9.289 8.5 C 9.287 8.5 9.285 8.5 9.283 8.5 C 9.281 8.5 9.279 8.5 9.277 8.5 C 9.275 8.5 9.273 8.5 9.271 8.5 C 9.269 8.5 9.267 8.5 9.265 8.5 C 9.263 8.5 9.261 8.5 9.259 8.5 C 9.257 8.5 9.255 8.5 9.253 8.5 C 9.251 8.5 9.249 8.5 9.247 8.5 C 9.245 8.5 9.243 8.5 9.242 8.5 C 9.24 8.5 9.238 8.5 9.236 8.5 C 9.234 8.5 9.232 8.5 9.23 8.5 C 9.228 8.5 9.226 8.5 9.224 8.5 C 9.222 8.5 9.22 8.5 9.218 8.5 C 9.216 8.5 9.214 8.5 9.212 8.5 C 9.211 8.5 9.209 8.5 9.207 8.5 C 9.205 8.5 9.203 8.5 9.201 8.5 C 9.199 8.5 9.197 8.5 9.195 8.5 C 9.193 8.5 9.191 8.5 9.189 8.5 C 9.187 8.5 9.186 8.5 9.184 8.5 C 9.182 8.5 9.18 8.5 9.178 8.5 C 9.176 8.5 9.174 8.5 9.172 8.5 C 9.17 8.5 9.168 8.5 9.166 8.5 C 9.165 8.5 9.163 8.5 9.161 8.5 C 9.159 8.5 9.157 8.5 9.155 8.5 C 9.153 8.5 9.151 8.5 9.149 8.5 C 9.147 8.5 9.146 8.5 9.144 8.5 C 9.142 8.5 9.14 8.5 9.138 8.5 C 9.136 8.5 9.134 8.5 9.132 8.5 C 9.13 8.5 9.129 8.5 9.127 8.5 C 9.125 8.5 9.123 8.5 9.121 8.5 C 9.119 8.5 9.117 8.5 9.115 8.5 C 9.114 8.5 9.112 8.5 9.11 8.5 C 9.108 8.5 9.106 8.5 9.104 8.5 C 9.102 8.5 9.1 8.5 9.099 8.5 C 9.097 8.5 9.095 8.5 9.093 8.5 C 9.091 8.5 9.089 8.5 9.087 8.5 C 9.086 8.5 9.084 8.5 9.082 8.5 C 9.08 8.5 9.078 8.5 9.076 8.5 C 9.074 8.5 9.073 8.5 9.071 8.5 C 9.069 8.5 9.067 8.5 9.065 8.5 C 9.063 8.5 9.061 8.5 9.06 8.5 C 9.058 8.5 9.056 8.5 9.054 8.5 C 9.052 8.5 9.05 8.5 9.048 8.5 C 9.047 8.5 9.045 8.5 9.043 8.5 C 9.041 8.5 9.039 8.5 9.037 8.5 C 9.036 8.5 9.034 8.5 9.032 8.5 C 9.03 8.5 9.028 8.5 9.026 8.5 C 9.025 8.5 9.023 8.5 9.021 8.5 C 9.019 8.5 9.017 8.5 9.015 8.5 C 9.014 8.5 9.012 8.5 9.01 8.5 C 9.008 8.5 9.006 8.5 9.004 8.5 C 9.003 8.5 9.001 8.5 8.999 8.5 C 8.997 8.5 8.995 8.5 8.993 8.5 C 8.992 8.5 8.99 8.5 8.988 8.5 C 8.986 8.5 8.984 8.5 8.982 8.5 C 8.981 8.5 8.979 8.5 8.977 8.5 C 8.975 8.5 8.973 8.5 8.971 8.5 C 8.97 8.5 8.968 8.5 8.966 8.5 C 8.964 8.5 8.962 8.5 8.961 8.5 C 8.959 8.5 8.957 8.5 8.955 8.5 C 8.953 8.5 8.952 8.5 8.95 8.5 C 8.948 8.5 8.946 8.5 8.944 8.5 C 8.942 8.5 8.941 8.5 8.939 8.5 C 8.937 8.5 8.935 8.5 8.933 8.5 C 8.932 8.5 8.93 8.5 8.928 8.5 C 8.926 8.5 8.924 8.5 8.923 8.5 C 8.921 8.5 8.919 8.5 8.917 8.5 C 8.915 8.5 8.914 8.5 8.912 8.5 C 8.91 8.5 8.908 8.5 8.906 8.5 C 8.904 8.5 8.903 8.5 8.901 8.5 C 8.899 8.5 8.897 8.5 8.895 8.5 C 8.894 8.5 8.892 8.5 8.89 8.5 C 8.888 8.5 8.886 8.5 8.885 8.5 C 8.883 8.5 8.881 8.5 8.879 8.5 C 8.877 8.5 8.876 8.5 8.874 8.5 C 8.872 8.5 8.87 8.5 8.868 8.5 C 8.867 8.5 8.865 8.5 8.863 8.5 C 8.861 8.5 8.86 8.5 8.858 8.5 C 8.856 8.5 8.854 8.5 8.852 8.5 C 8.851 8.5 8.849 8.5 8.847 8.5 C 8.845 8.5 8.843 8.5 8.842 8.5 C 8.84 8.5 8.838 8.5 8.836 8.5 C 8.834 8.5 8.833 8.5 8.831 8.5 C 8.829 8.5 8.827 8.5 8.825 8.5 C 8.824 8.5 8.822 8.5 8.82 8.5 C 8.818 8.5 8.816 8.5 8.815 8.5 C 8.813 8.5 8.811 8.5 8.809 8.5 C 8.808 8.5 8.806 8.5 8.804 8.5 C 8.802 8.5 8.8 8.5 8.799 8.5 C 8.797 8.5 8.795 8.5 8.793 8.5 C 8.791 8.5 8.79 8.5 8.788 8.5 C 8.786 8.5 8.784 8.5 8.782 8.5 C 8.781 8.5 8.779 8.5 8.777 8.5 C 8.775 8.5 8.773 8.5 8.772 8.5 C 8.77 8.5 8.768 8.5 8.766 8.5 C 8.765 8.5 8.763 8.5 8.761 8.5 C 8.759 8.5 8.757 8.5 8.756 8.5 C 8.754 8.5 8.752 8.5 8.75 8.5 C 8.748 8.5 8.747 8.5 8.745 8.5 C 8.743 8.5 8.741 8.5 8.739 8.5 C 8.738 8.5 8.736 8.5 8.734 8.5 C 8.732 8.5 8.73 8.5 8.729 8.5 C 8.727 8.5 8.725 8.5 8.723 8.5 C 8.722 8.5 8.72 8.5 8.718 8.5 C 8.716 8.5 8.714 8.5 8.713 8.5 C 8.711 8.5 8.709 8.5 8.707 8.5 C 8.705 8.5 8.704 8.5 8.702 8.5 C 8.7 8.5 8.698 8.5 8.696 8.5 C 8.695 8.5 8.693 8.5 8.691 8.5 C 8.689 8.5 8.687 8.5 8.686 8.5 C 8.684 8.5 8.682 8.5 8.68 8.5 C 8.678 8.5 8.677 8.5 8.675 8.5 C 8.673 8.5 8.671 8.5 8.669 8.5 C 8.668 8.5 8.666 8.5 8.664 8.5 C 8.662 8.5 8.66 8.5 8.659 8.5 C 8.657 8.5 8.655 8.5 8.653 8.5 C 8.651 8.5 8.65 8.5 8.648 8.5 C 8.646 8.5 8.644 8.5 8.642 8.5 C 8.641 8.5 8.639 8.5 8.637 8.5 C 8.635 8.5 8.633 8.5 8.632 8.5 C 8.63 8.5 8.628 8.5 8.626 8.5 C 8.624 8.5 8.623 8.5 8.621 8.5 C 8.619 8.5 8.617 8.5 8.615 8.5 C 8.613 8.5 8.612 8.5 8.61 8.5 C 8.608 8.5 8.606 8.5 8.604 8.5 C 8.603 8.5 8.601 8.5 8.599 8.5 C 8.597 8.5 8.595 8.5 8.594 8.5 C 8.592 8.5 8.59 8.5 8.588 8.5 C 8.586 8.5 8.584 8.5 8.583 8.5 C 8.581 8.5 8.579 8.5 8.577 8.5 C 8.575 8.5 8.574 8.5 8.572 8.5 C 8.57 8.5 8.568 8.5 8.566 8.5 C 8.564 8.5 8.563 8.5 8.561 8.5 C 8.559 8.5 8.557 8.5 8.555 8.5 C 8.553 8.5 8.552 8.5 8.55 8.5 C 8.548 8.5 8.546 8.5 8.544 8.5 C 8.542 8.5 8.541 8.5 8.539 8.5 C 8.537 8.5 8.535 8.5 8.533 8.5 C 8.531 8.5 8.53 8.5 8.528 8.5 C 8.526 8.5 8.524 8.5 8.522 8.5 C 8.52 8.5 8.519 8.5 8.517 8.5 C 8.515 8.5 8.513 8.5 8.511 8.5 C 8.509 8.5 8.508 8.5 8.506 8.5 C 8.504 8.5 8.502 8.5 8.5 8.5 C 8.498 8.5 8.496 8.5 8.495 8.5 C 8.493 8.5 8.491 8.5 8.489 8.5 C 8.487 8.5 8.485 8.5 8.483 8.5 C 8.482 8.5 8.48 8.5 8.478 8.5 C 8.476 8.5 8.474 8.5 8.472 8.5 C 8.47 8.5 8.469 8.5 8.467 8.5 C 8.465 8.5 8.463 8.5 8.461 8.5 C 8.459 8.5 8.457 8.5 8.456 8.5 C 8.454 8.5 8.452 8.5 8.45 8.5 C 8.448 8.5 8.446 8.5 8.444 8.5 C 8.442 8.5 8.441 8.5 8.439 8.5 C 8.437 8.5 8.435 8.5 8.433 8.5 C 8.431 8.5 8.429 8.5 8.427 8.5 C 8.426 8.5 8.424 8.5 8.422 8.5 C 8.42 8.5 8.418 8.5 8.416 8.5 C 8.414 8.5 8.412 8.5 8.41 8.5 C 8.409 8.5 8.407 8.5 8.405 8.5 C 8.403 8.5 8.401 8.5 8.399 8.5 C 8.397 8.5 8.395 8.5 8.393 8.5 C 8.392 8.5 8.39 8.5 8.388 8.5 C 8.386 8.5 8.384 8.5 8.382 8.5 C 8.38 8.5 8.378 8.5 8.376 8.5 C 8.374 8.5 8.372 8.5 8.371 8.5 C 8.369 8.5 8.367 8.5 8.365 8.5 C 8.363 8.5 8.361 8.5 8.359 8.5 C 8.357 8.5 8.355 8.5 8.353 8.5 C 8.351 8.5 8.349 8.5 8.348 8.5 C 8.346 8.5 8.344 8.5 8.342 8.5 C 8.34 8.5 8.338 8.5 8.336 8.5 C 8.334 8.5 8.332 8.5 8.33 8.5 C 8.328 8.5 8.326 8.5 8.324 8.5 C 8.322 8.5 8.32 8.5 8.319 8.5 C 8.317 8.5 8.315 8.5 8.313 8.5 C 8.311 8.5 8.309 8.5 8.307 8.5 C 8.305 8.5 8.303 8.5 8.301 8.5 C 8.299 8.5 8.297 8.5 8.295 8.5 C 8.293 8.5 8.291 8.5 8.289 8.5 C 8.287 8.5 8.285 8.5 8.283 8.5 C 8.281 8.5 8.279 8.5 8.277 8.5 C 8.275 8.5 8.273 8.5 8.272 8.5 C 8.27 8.5 8.268 8.5 8.266 8.5 C 8.264 8.5 8.262 8.5 8.26 8.5 C 8.258 8.5 8.256 8.5 8.254 8.5 C 8.252 8.5 8.25 8.5 8.248 8.5 C 8.246 8.5 8.244 8.5 8.242 8.5 C 8.24 8.5 8.238 8.5 8.236 8.5 C 8.234 8.5 8.232 8.5 8.23 8.5 C 8.228 8.5 8.226 8.5 8.224 8.5 C 8.222 8.5 8.22 8.5 8.218 8.5 C 8.216 8.5 8.214 8.5 8.212 8.5 C 8.21 8.5 8.208 8.5 8.206 8.5 C 8.204 8.5 8.202 8.5 8.2 8.5 C 8.198 8.5 8.195 8.5 8.193 8.5 C 8.191 8.5 8.189 8.5 8.187 8.5 C 8.185 8.5 8.183 8.5 8.181 8.5 C 8.179 8.5 8.177 8.5 8.175 8.5 C 8.173 8.5 8.171 8.5 8.169 8.5 C 8.167 8.5 8.165 8.5 8.163 8.5 C 8.161 8.5 8.159 8.5 8.157 8.5 C 8.155 8.5 8.153 8.5 8.15 8.5 C 8.148 8.5 8.146 8.5 8.144 8.5 C 8.142 8.5 8.14 8.5 8.138 8.5 C 8.136 8.5 8.134 8.5 8.132 8.5 C 8.13 8.5 8.128 8.5 8.126 8.5 C 8.123 8.5 8.121 8.5 8.119 8.5 C 8.117 8.5 8.115 8.5 8.113 8.5 C 8.111 8.5 8.109 8.5 8.107 8.5 C 8.105 8.5 8.103 8.5 8.1 8.5 C 8.098 8.5 8.096 8.5 8.094 8.5 C 8.092 8.5 8.09 8.5 8.088 8.5 C 8.086 8.5 8.084 8.5 8.081 8.5 C 8.079 8.5 8.077 8.5 8.075 8.5 C 8.073 8.5 8.071 8.5 8.069 8.5 C 8.067 8.5 8.064 8.5 8.062 8.5 C 8.06 8.5 8.058 8.5 8.056 8.5 C 8.054 8.5 8.052 8.5 8.049 8.5 C 8.047 8.5 8.045 8.5 8.043 8.5 C 8.041 8.5 8.039 8.5 8.036 8.5 C 8.034 8.5 8.032 8.5 8.03 8.5 C 8.028 8.5 8.026 8.5 8.023 8.5 C 8.021 8.5 8.019 8.5 8.017 8.5 C 8.015 8.5 8.013 8.5 8.01 8.5 C 8.008 8.5 8.006 8.5 8.004 8.5 C 8.002 8.5 7.999 8.5 7.997 8.5 C 7.995 8.5 7.993 8.5 7.991 8.5 C 7.988 8.5 7.986 8.5 7.984 8.5 C 7.982 8.5 7.98 8.5 7.977 8.5 C 7.975 8.5 7.973 8.5 7.971 8.5 C 7.969 8.5 7.966 8.5 7.964 8.5 C 7.962 8.5 7.96 8.5 7.957 8.5 C 7.955 8.5 7.953 8.5 7.951 8.5 C 7.948 8.5 7.946 8.5 7.944 8.5 C 7.942 8.5 7.939 8.5 7.937 8.5 C 7.935 8.5 7.933 8.5 7.93 8.5 C 7.928 8.5 7.926 8.5 7.924 8.5 C 7.921 8.5 7.919 8.5 7.917 8.5 C 7.915 8.5 7.912 8.5 7.91 8.5 C 7.908 8.5 7.906 8.5 7.903 8.5 C 7.901 8.5 7.899 8.5 7.896 8.5 C 7.894 8.5 7.892 8.5 7.889 8.5 C 7.887 8.5 7.885 8.5 7.883 8.5 C 7.88 8.5 7.878 8.5 7.876 8.5 C 7.873 8.5 7.871 8.5 7.869 8.5 C 7.866 8.5 7.864 8.5 7.862 8.5 C 7.859 8.5 7.857 8.5 7.855 8.5 C 7.852 8.5 7.85 8.5 7.848 8.5 C 7.845 8.5 7.843 8.5 7.841 8.5 C 7.838 8.5 7.836 8.5 7.834 8.5 C 7.831 8.5 7.829 8.5 7.827 8.5 C 7.824 8.5 7.822 8.5 7.82 8.5 C 7.817 8.5 7.815 8.5 7.812 8.5 C 7.81 8.5 7.808 8.5 7.805 8.5 C 7.803 8.5 7.801 8.5 7.798 8.5 C 7.796 8.5 7.793 8.5 7.791 8.5 C 7.789 8.5 7.786 8.5 7.784 8.5 C 7.781 8.5 7.779 8.5 7.777 8.5 C 7.774 8.5 7.772 8.5 7.769 8.5 C 7.767 8.5 7.764 8.5 7.762 8.5 C 7.76 8.5 7.757 8.5 7.755 8.5 C 7.752 8.5 7.75 8.5 7.747 8.5 C 7.745 8.5 7.743 8.5 7.74 8.5 C 7.738 8.5 7.735 8.5 7.733 8.5 C 7.73 8.5 7.728 8.5 7.725 8.5 C 7.723 8.5 7.72 8.5 7.718 8.5 C 7.716 8.5 7.713 8.5 7.711 8.5 C 7.708 8.5 7.706 8.5 7.703 8.5 C 7.701 8.5 7.698 8.5 7.696 8.5 C 7.693 8.5 7.691 8.5 7.688 8.5 C 7.686 8.5 7.683 8.5 7.681 8.5 C 7.678 8.5 7.676 8.5 7.673 8.5 C 7.671 8.5 7.668 8.5 7.666 8.5 C 7.663 8.5 7.66 8.5 7.658 8.5 C 7.655 8.5 7.653 8.5 7.65 8.5 C 7.648 8.5 7.645 8.5 7.643 8.5 C 7.64 8.5 7.638 8.5 7.635 8.5 C 7.632 8.5 7.63 8.5 7.627 8.5 C 7.625 8.5 7.622 8.5 7.62 8.5 C 7.617 8.5 7.614 8.5 7.612 8.5 C 7.609 8.5 7.607 8.5 7.604 8.5 C 7.601 8.5 7.599 8.5 7.596 8.5 C 7.594 8.5 7.591 8.5 7.588 8.5 C 7.586 8.5 7.583 8.5 7.581 8.5 C 7.578 8.5 7.575 8.5 7.573 8.5 C 7.57 8.5 7.567 8.5 7.565 8.5 C 7.562 8.5 7.56 8.5 7.557 8.5 C 7.554 8.5 7.552 8.5 7.549 8.5 C 7.546 8.5 7.544 8.5 7.541 8.5 C 7.538 8.5 7.536 8.5 7.533 8.5 C 7.53 8.5 7.528 8.5 7.525 8.5 C 7.522 8.5 7.52 8.5 7.517 8.5 C 7.514 8.5 7.512 8.5 7.509 8.5 C 7.506 8.5 7.503 8.5 7.501 8.5 C 7.498 8.5 7.495 8.5 7.493 8.5 C 7.49 8.5 7.487 8.5 7.484 8.5 C 7.482 8.5 7.479 8.5 7.476 8.5 C 7.474 8.5 7.471 8.5 7.468 8.5 C 7.465 8.5 7.463 8.5 7.46 8.5 C 7.457 8.5 7.454 8.5 7.452 8.5 C 7.449 8.5 7.446 8.5 7.443 8.5 C 7.44 8.5 7.438 8.5 7.435 8.5 C 7.432 8.5 7.429 8.5 7.427 8.5 C 7.424 8.5 7.421 8.5 7.418 8.5 C 7.415 8.5 7.413 8.5 7.41 8.5 C 7.407 8.5 7.404 8.5 7.401 8.5 C 7.399 8.5 7.396 8.5 7.393 8.5 C 7.39 8.5 7.387 8.5 7.384 8.5 C 7.382 8.5 7.379 8.5 7.376 8.5 C 7.373 8.5 7.37 8.5 7.367 8.5 C 7.364 8.5 7.362 8.5 7.359 8.5 C 7.356 8.5 7.353 8.5 7.35 8.5 C 7.347 8.5 7.344 8.5 7.341 8.5 C 7.339 8.5 7.336 8.5 7.333 8.5 C 7.33 8.5 7.327 8.5 7.324 8.5 C 7.321 8.5 7.318 8.5 7.315 8.5 C 7.312 8.5 7.31 8.5 7.307 8.5 C 7.304 8.5 7.301 8.5 7.298 8.5 C 7.295 8.5 7.292 8.5 7.289 8.5 C 7.286 8.5 7.283 8.5 7.28 8.5 C 7.277 8.5 7.274 8.5 7.271 8.5 C 7.268 8.5 7.265 8.5 7.262 8.5 C 7.259 8.5 7.256 8.5 7.253 8.5 C 7.25 8.5 7.247 8.5 7.244 8.5 C 7.241 8.5 7.238 8.5 7.235 8.5 C 7.232 8.5 7.229 8.5 7.226 8.5 C 7.223 8.5 7.22 8.5 7.217 8.5 C 7.214 8.5 7.211 8.5 7.208 8.5 C 7.205 8.5 7.202 8.5 7.199 8.5 C 7.196 8.5 7.193 8.5 7.19 8.5 C 7.187 8.5 7.184 8.5 7.181 8.5 C 7.178 8.5 7.175 8.5 7.171 8.5 C 7.168 8.5 7.165 8.5 7.162 8.5 C 7.159 8.5 7.156 8.5 7.153 8.5 C 7.15 8.5 7.147 8.5 7.144 8.5 C 7.14 8.5 7.137 8.5 7.134 8.5 C 7.131 8.5 7.128 8.5 7.125 8.5 C 7.122 8.5 7.119 8.5 7.115 8.5 L 7.115 10.5 C 7.119 10.5 7.122 10.5 7.125 10.5 C 7.128 10.5 7.131 10.5 7.134 10.5 C 7.137 10.5 7.14 10.5 7.144 10.5 C 7.147 10.5 7.15 10.5 7.153 10.5 C 7.156 10.5 7.159 10.5 7.162 10.5 C 7.165 10.5 7.168 10.5 7.171 10.5 C 7.175 10.5 7.178 10.5 7.181 10.5 C 7.184 10.5 7.187 10.5 7.19 10.5 C 7.193 10.5 7.196 10.5 7.199 10.5 C 7.202 10.5 7.205 10.5 7.208 10.5 C 7.211 10.5 7.214 10.5 7.217 10.5 C 7.22 10.5 7.223 10.5 7.226 10.5 C 7.229 10.5 7.232 10.5 7.235 10.5 C 7.238 10.5 7.241 10.5 7.244 10.5 C 7.247 10.5 7.25 10.5 7.253 10.5 C 7.256 10.5 7.259 10.5 7.262 10.5 C 7.265 10.5 7.268 10.5 7.271 10.5 C 7.274 10.5 7.277 10.5 7.28 10.5 C 7.283 10.5 7.286 10.5 7.289 10.5 C 7.292 10.5 7.295 10.5 7.298 10.5 C 7.301 10.5 7.304 10.5 7.307 10.5 C 7.31 10.5 7.312 10.5 7.315 10.5 C 7.318 10.5 7.321 10.5 7.324 10.5 C 7.327 10.5 7.33 10.5 7.333 10.5 C 7.336 10.5 7.339 10.5 7.341 10.5 C 7.344 10.5 7.347 10.5 7.35 10.5 C 7.353 10.5 7.356 10.5 7.359 10.5 C 7.362 10.5 7.364 10.5 7.367 10.5 C 7.37 10.5 7.373 10.5 7.376 10.5 C 7.379 10.5 7.382 10.5 7.384 10.5 C 7.387 10.5 7.39 10.5 7.393 10.5 C 7.396 10.5 7.399 10.5 7.401 10.5 C 7.404 10.5 7.407 10.5 7.41 10.5 C 7.413 10.5 7.415 10.5 7.418 10.5 C 7.421 10.5 7.424 10.5 7.427 10.5 C 7.429 10.5 7.432 10.5 7.435 10.5 C 7.438 10.5 7.44 10.5 7.443 10.5 C 7.446 10.5 7.449 10.5 7.452 10.5 C 7.454 10.5 7.457 10.5 7.46 10.5 C 7.463 10.5 7.465 10.5 7.468 10.5 C 7.471 10.5 7.474 10.5 7.476 10.5 C 7.479 10.5 7.482 10.5 7.484 10.5 C 7.487 10.5 7.49 10.5 7.493 10.5 C 7.495 10.5 7.498 10.5 7.501 10.5 C 7.503 10.5 7.506 10.5 7.509 10.5 C 7.512 10.5 7.514 10.5 7.517 10.5 C 7.52 10.5 7.522 10.5 7.525 10.5 C 7.528 10.5 7.53 10.5 7.533 10.5 C 7.536 10.5 7.538 10.5 7.541 10.5 C 7.544 10.5 7.546 10.5 7.549 10.5 C 7.552 10.5 7.554 10.5 7.557 10.5 C 7.56 10.5 7.562 10.5 7.565 10.5 C 7.567 10.5 7.57 10.5 7.573 10.5 C 7.575 10.5 7.578 10.5 7.581 10.5 C 7.583 10.5 7.586 10.5 7.588 10.5 C 7.591 10.5 7.594 10.5 7.596 10.5 C 7.599 10.5 7.601 10.5 7.604 10.5 C 7.607 10.5 7.609 10.5 7.612 10.5 C 7.614 10.5 7.617 10.5 7.62 10.5 C 7.622 10.5 7.625 10.5 7.627 10.5 C 7.63 10.5 7.632 10.5 7.635 10.5 C 7.638 10.5 7.64 10.5 7.643 10.5 C 7.645 10.5 7.648 10.5 7.65 10.5 C 7.653 10.5 7.655 10.5 7.658 10.5 C 7.66 10.5 7.663 10.5 7.666 10.5 C 7.668 10.5 7.671 10.5 7.673 10.5 C 7.676 10.5 7.678 10.5 7.681 10.5 C 7.683 10.5 7.686 10.5 7.688 10.5 C 7.691 10.5 7.693 10.5 7.696 10.5 C 7.698 10.5 7.701 10.5 7.703 10.5 C 7.706 10.5 7.708 10.5 7.711 10.5 C 7.713 10.5 7.716 10.5 7.718 10.5 C 7.72 10.5 7.723 10.5 7.725 10.5 C 7.728 10.5 7.73 10.5 7.733 10.5 C 7.735 10.5 7.738 10.5 7.74 10.5 C 7.743 10.5 7.745 10.5 7.747 10.5 C 7.75 10.5 7.752 10.5 7.755 10.5 C 7.757 10.5 7.76 10.5 7.762 10.5 C 7.764 10.5 7.767 10.5 7.769 10.5 C 7.772 10.5 7.774 10.5 7.777 10.5 C 7.779 10.5 7.781 10.5 7.784 10.5 C 7.786 10.5 7.789 10.5 7.791 10.5 C 7.793 10.5 7.796 10.5 7.798 10.5 C 7.801 10.5 7.803 10.5 7.805 10.5 C 7.808 10.5 7.81 10.5 7.812 10.5 C 7.815 10.5 7.817 10.5 7.82 10.5 C 7.822 10.5 7.824 10.5 7.827 10.5 C 7.829 10.5 7.831 10.5 7.834 10.5 C 7.836 10.5 7.838 10.5 7.841 10.5 C 7.843 10.5 7.845 10.5 7.848 10.5 C 7.85 10.5 7.852 10.5 7.855 10.5 C 7.857 10.5 7.859 10.5 7.862 10.5 C 7.864 10.5 7.866 10.5 7.869 10.5 C 7.871 10.5 7.873 10.5 7.876 10.5 C 7.878 10.5 7.88 10.5 7.883 10.5 C 7.885 10.5 7.887 10.5 7.889 10.5 C 7.892 10.5 7.894 10.5 7.896 10.5 C 7.899 10.5 7.901 10.5 7.903 10.5 C 7.906 10.5 7.908 10.5 7.91 10.5 C 7.912 10.5 7.915 10.5 7.917 10.5 C 7.919 10.5 7.921 10.5 7.924 10.5 C 7.926 10.5 7.928 10.5 7.93 10.5 C 7.933 10.5 7.935 10.5 7.937 10.5 C 7.939 10.5 7.942 10.5 7.944 10.5 C 7.946 10.5 7.948 10.5 7.951 10.5 C 7.953 10.5 7.955 10.5 7.957 10.5 C 7.96 10.5 7.962 10.5 7.964 10.5 C 7.966 10.5 7.969 10.5 7.971 10.5 C 7.973 10.5 7.975 10.5 7.977 10.5 C 7.98 10.5 7.982 10.5 7.984 10.5 C 7.986 10.5 7.988 10.5 7.991 10.5 C 7.993 10.5 7.995 10.5 7.997 10.5 C 7.999 10.5 8.002 10.5 8.004 10.5 C 8.006 10.5 8.008 10.5 8.01 10.5 C 8.013 10.5 8.015 10.5 8.017 10.5 C 8.019 10.5 8.021 10.5 8.023 10.5 C 8.026 10.5 8.028 10.5 8.03 10.5 C 8.032 10.5 8.034 10.5 8.036 10.5 C 8.039 10.5 8.041 10.5 8.043 10.5 C 8.045 10.5 8.047 10.5 8.049 10.5 C 8.052 10.5 8.054 10.5 8.056 10.5 C 8.058 10.5 8.06 10.5 8.062 10.5 C 8.064 10.5 8.067 10.5 8.069 10.5 C 8.071 10.5 8.073 10.5 8.075 10.5 C 8.077 10.5 8.079 10.5 8.081 10.5 C 8.084 10.5 8.086 10.5 8.088 10.5 C 8.09 10.5 8.092 10.5 8.094 10.5 C 8.096 10.5 8.098 10.5 8.1 10.5 C 8.103 10.5 8.105 10.5 8.107 10.5 C 8.109 10.5 8.111 10.5 8.113 10.5 C 8.115 10.5 8.117 10.5 8.119 10.5 C 8.121 10.5 8.123 10.5 8.126 10.5 C 8.128 10.5 8.13 10.5 8.132 10.5 C 8.134 10.5 8.136 10.5 8.138 10.5 C 8.14 10.5 8.142 10.5 8.144 10.5 C 8.146 10.5 8.148 10.5 8.15 10.5 C 8.153 10.5 8.155 10.5 8.157 10.5 C 8.159 10.5 8.161 10.5 8.163 10.5 C 8.165 10.5 8.167 10.5 8.169 10.5 C 8.171 10.5 8.173 10.5 8.175 10.5 C 8.177 10.5 8.179 10.5 8.181 10.5 C 8.183 10.5 8.185 10.5 8.187 10.5 C 8.189 10.5 8.191 10.5 8.193 10.5 C 8.195 10.5 8.198 10.5 8.2 10.5 C 8.202 10.5 8.204 10.5 8.206 10.5 C 8.208 10.5 8.21 10.5 8.212 10.5 C 8.214 10.5 8.216 10.5 8.218 10.5 C 8.22 10.5 8.222 10.5 8.224 10.5 C 8.226 10.5 8.228 10.5 8.23 10.5 C 8.232 10.5 8.234 10.5 8.236 10.5 C 8.238 10.5 8.24 10.5 8.242 10.5 C 8.244 10.5 8.246 10.5 8.248 10.5 C 8.25 10.5 8.252 10.5 8.254 10.5 C 8.256 10.5 8.258 10.5 8.26 10.5 C 8.262 10.5 8.264 10.5 8.266 10.5 C 8.268 10.5 8.27 10.5 8.272 10.5 C 8.273 10.5 8.275 10.5 8.277 10.5 C 8.279 10.5 8.281 10.5 8.283 10.5 C 8.285 10.5 8.287 10.5 8.289 10.5 C 8.291 10.5 8.293 10.5 8.295 10.5 C 8.297 10.5 8.299 10.5 8.301 10.5 C 8.303 10.5 8.305 10.5 8.307 10.5 C 8.309 10.5 8.311 10.5 8.313 10.5 C 8.315 10.5 8.317 10.5 8.319 10.5 C 8.32 10.5 8.322 10.5 8.324 10.5 C 8.326 10.5 8.328 10.5 8.33 10.5 C 8.332 10.5 8.334 10.5 8.336 10.5 C 8.338 10.5 8.34 10.5 8.342 10.5 C 8.344 10.5 8.346 10.5 8.348 10.5 C 8.349 10.5 8.351 10.5 8.353 10.5 C 8.355 10.5 8.357 10.5 8.359 10.5 C 8.361 10.5 8.363 10.5 8.365 10.5 C 8.367 10.5 8.369 10.5 8.371 10.5 C 8.372 10.5 8.374 10.5 8.376 10.5 C 8.378 10.5 8.38 10.5 8.382 10.5 C 8.384 10.5 8.386 10.5 8.388 10.5 C 8.39 10.5 8.392 10.5 8.393 10.5 C 8.395 10.5 8.397 10.5 8.399 10.5 C 8.401 10.5 8.403 10.5 8.405 10.5 C 8.407 10.5 8.409 10.5 8.41 10.5 C 8.412 10.5 8.414 10.5 8.416 10.5 C 8.418 10.5 8.42 10.5 8.422 10.5 C 8.424 10.5 8.426 10.5 8.427 10.5 C 8.429 10.5 8.431 10.5 8.433 10.5 C 8.435 10.5 8.437 10.5 8.439 10.5 C 8.441 10.5 8.442 10.5 8.444 10.5 C 8.446 10.5 8.448 10.5 8.45 10.5 C 8.452 10.5 8.454 10.5 8.456 10.5 C 8.457 10.5 8.459 10.5 8.461 10.5 C 8.463 10.5 8.465 10.5 8.467 10.5 C 8.469 10.5 8.47 10.5 8.472 10.5 C 8.474 10.5 8.476 10.5 8.478 10.5 C 8.48 10.5 8.482 10.5 8.483 10.5 C 8.485 10.5 8.487 10.5 8.489 10.5 C 8.491 10.5 8.493 10.5 8.495 10.5 C 8.496 10.5 8.498 10.5 8.5 10.5 C 8.502 10.5 8.504 10.5 8.506 10.5 C 8.508 10.5 8.509 10.5 8.511 10.5 C 8.513 10.5 8.515 10.5 8.517 10.5 C 8.519 10.5 8.52 10.5 8.522 10.5 C 8.524 10.5 8.526 10.5 8.528 10.5 C 8.53 10.5 8.531 10.5 8.533 10.5 C 8.535 10.5 8.537 10.5 8.539 10.5 C 8.541 10.5 8.542 10.5 8.544 10.5 C 8.546 10.5 8.548 10.5 8.55 10.5 C 8.552 10.5 8.553 10.5 8.555 10.5 C 8.557 10.5 8.559 10.5 8.561 10.5 C 8.563 10.5 8.564 10.5 8.566 10.5 C 8.568 10.5 8.57 10.5 8.572 10.5 C 8.574 10.5 8.575 10.5 8.577 10.5 C 8.579 10.5 8.581 10.5 8.583 10.5 C 8.584 10.5 8.586 10.5 8.588 10.5 C 8.59 10.5 8.592 10.5 8.594 10.5 C 8.595 10.5 8.597 10.5 8.599 10.5 C 8.601 10.5 8.603 10.5 8.604 10.5 C 8.606 10.5 8.608 10.5 8.61 10.5 C 8.612 10.5 8.613 10.5 8.615 10.5 C 8.617 10.5 8.619 10.5 8.621 10.5 C 8.623 10.5 8.624 10.5 8.626 10.5 C 8.628 10.5 8.63 10.5 8.632 10.5 C 8.633 10.5 8.635 10.5 8.637 10.5 C 8.639 10.5 8.641 10.5 8.642 10.5 C 8.644 10.5 8.646 10.5 8.648 10.5 C 8.65 10.5 8.651 10.5 8.653 10.5 C 8.655 10.5 8.657 10.5 8.659 10.5 C 8.66 10.5 8.662 10.5 8.664 10.5 C 8.666 10.5 8.668 10.5 8.669 10.5 C 8.671 10.5 8.673 10.5 8.675 10.5 C 8.677 10.5 8.678 10.5 8.68 10.5 C 8.682 10.5 8.684 10.5 8.686 10.5 C 8.687 10.5 8.689 10.5 8.691 10.5 C 8.693 10.5 8.695 10.5 8.696 10.5 C 8.698 10.5 8.7 10.5 8.702 10.5 C 8.704 10.5 8.705 10.5 8.707 10.5 C 8.709 10.5 8.711 10.5 8.713 10.5 C 8.714 10.5 8.716 10.5 8.718 10.5 C 8.72 10.5 8.722 10.5 8.723 10.5 C 8.725 10.5 8.727 10.5 8.729 10.5 C 8.73 10.5 8.732 10.5 8.734 10.5 C 8.736 10.5 8.738 10.5 8.739 10.5 C 8.741 10.5 8.743 10.5 8.745 10.5 C 8.747 10.5 8.748 10.5 8.75 10.5 C 8.752 10.5 8.754 10.5 8.756 10.5 C 8.757 10.5 8.759 10.5 8.761 10.5 C 8.763 10.5 8.765 10.5 8.766 10.5 C 8.768 10.5 8.77 10.5 8.772 10.5 C 8.773 10.5 8.775 10.5 8.777 10.5 C 8.779 10.5 8.781 10.5 8.782 10.5 C 8.784 10.5 8.786 10.5 8.788 10.5 C 8.79 10.5 8.791 10.5 8.793 10.5 C 8.795 10.5 8.797 10.5 8.799 10.5 C 8.8 10.5 8.802 10.5 8.804 10.5 C 8.806 10.5 8.808 10.5 8.809 10.5 C 8.811 10.5 8.813 10.5 8.815 10.5 C 8.816 10.5 8.818 10.5 8.82 10.5 C 8.822 10.5 8.824 10.5 8.825 10.5 C 8.827 10.5 8.829 10.5 8.831 10.5 C 8.833 10.5 8.834 10.5 8.836 10.5 C 8.838 10.5 8.84 10.5 8.842 10.5 C 8.843 10.5 8.845 10.5 8.847 10.5 C 8.849 10.5 8.851 10.5 8.852 10.5 C 8.854 10.5 8.856 10.5 8.858 10.5 C 8.86 10.5 8.861 10.5 8.863 10.5 C 8.865 10.5 8.867 10.5 8.868 10.5 C 8.87 10.5 8.872 10.5 8.874 10.5 C 8.876 10.5 8.877 10.5 8.879 10.5 C 8.881 10.5 8.883 10.5 8.885 10.5 C 8.886 10.5 8.888 10.5 8.89 10.5 C 8.892 10.5 8.894 10.5 8.895 10.5 C 8.897 10.5 8.899 10.5 8.901 10.5 C 8.903 10.5 8.904 10.5 8.906 10.5 C 8.908 10.5 8.91 10.5 8.912 10.5 C 8.914 10.5 8.915 10.5 8.917 10.5 C 8.919 10.5 8.921 10.5 8.923 10.5 C 8.924 10.5 8.926 10.5 8.928 10.5 C 8.93 10.5 8.932 10.5 8.933 10.5 C 8.935 10.5 8.937 10.5 8.939 10.5 C 8.941 10.5 8.942 10.5 8.944 10.5 C 8.946 10.5 8.948 10.5 8.95 10.5 C 8.952 10.5 8.953 10.5 8.955 10.5 C 8.957 10.5 8.959 10.5 8.961 10.5 C 8.962 10.5 8.964 10.5 8.966 10.5 C 8.968 10.5 8.97 10.5 8.971 10.5 C 8.973 10.5 8.975 10.5 8.977 10.5 C 8.979 10.5 8.981 10.5 8.982 10.5 C 8.984 10.5 8.986 10.5 8.988 10.5 C 8.99 10.5 8.992 10.5 8.993 10.5 C 8.995 10.5 8.997 10.5 8.999 10.5 C 9.001 10.5 9.003 10.5 9.004 10.5 C 9.006 10.5 9.008 10.5 9.01 10.5 C 9.012 10.5 9.014 10.5 9.015 10.5 C 9.017 10.5 9.019 10.5 9.021 10.5 C 9.023 10.5 9.025 10.5 9.026 10.5 C 9.028 10.5 9.03 10.5 9.032 10.5 C 9.034 10.5 9.036 10.5 9.037 10.5 C 9.039 10.5 9.041 10.5 9.043 10.5 C 9.045 10.5 9.047 10.5 9.048 10.5 C 9.05 10.5 9.052 10.5 9.054 10.5 C 9.056 10.5 9.058 10.5 9.06 10.5 C 9.061 10.5 9.063 10.5 9.065 10.5 C 9.067 10.5 9.069 10.5 9.071 10.5 C 9.073 10.5 9.074 10.5 9.076 10.5 C 9.078 10.5 9.08 10.5 9.082 10.5 C 9.084 10.5 9.086 10.5 9.087 10.5 C 9.089 10.5 9.091 10.5 9.093 10.5 C 9.095 10.5 9.097 10.5 9.099 10.5 C 9.1 10.5 9.102 10.5 9.104 10.5 C 9.106 10.5 9.108 10.5 9.11 10.5 C 9.112 10.5 9.114 10.5 9.115 10.5 C 9.117 10.5 9.119 10.5 9.121 10.5 C 9.123 10.5 9.125 10.5 9.127 10.5 C 9.129 10.5 9.13 10.5 9.132 10.5 C 9.134 10.5 9.136 10.5 9.138 10.5 C 9.14 10.5 9.142 10.5 9.144 10.5 C 9.146 10.5 9.147 10.5 9.149 10.5 C 9.151 10.5 9.153 10.5 9.155 10.5 C 9.157 10.5 9.159 10.5 9.161 10.5 C 9.163 10.5 9.165 10.5 9.166 10.5 C 9.168 10.5 9.17 10.5 9.172 10.5 C 9.174 10.5 9.176 10.5 9.178 10.5 C 9.18 10.5 9.182 10.5 9.184 10.5 C 9.186 10.5 9.187 10.5 9.189 10.5 C 9.191 10.5 9.193 10.5 9.195 10.5 C 9.197 10.5 9.199 10.5 9.201 10.5 C 9.203 10.5 9.205 10.5 9.207 10.5 C 9.209 10.5 9.211 10.5 9.212 10.5 C 9.214 10.5 9.216 10.5 9.218 10.5 C 9.22 10.5 9.222 10.5 9.224 10.5 C 9.226 10.5 9.228 10.5 9.23 10.5 C 9.232 10.5 9.234 10.5 9.236 10.5 C 9.238 10.5 9.24 10.5 9.242 10.5 C 9.243 10.5 9.245 10.5 9.247 10.5 C 9.249 10.5 9.251 10.5 9.253 10.5 C 9.255 10.5 9.257 10.5 9.259 10.5 C 9.261 10.5 9.263 10.5 9.265 10.5 C 9.267 10.5 9.269 10.5 9.271 10.5 C 9.273 10.5 9.275 10.5 9.277 10.5 C 9.279 10.5 9.281 10.5 9.283 10.5 C 9.285 10.5 9.287 10.5 9.289 10.5 C 9.291 10.5 9.293 10.5 9.295 10.5 C 9.297 10.5 9.299 10.5 9.301 10.5 C 9.303 10.5 9.305 10.5 9.307 10.5 C 9.309 10.5 9.311 10.5 9.313 10.5 C 9.315 10.5 9.317 10.5 9.319 10.5 C 9.321 10.5 9.323 10.5 9.325 10.5 C 9.327 10.5 9.329 10.5 9.331 10.5 C 9.333 10.5 9.335 10.5 9.337 10.5 C 9.339 10.5 9.341 10.5 9.343 10.5 C 9.345 10.5 9.347 10.5 9.349 10.5 C 9.351 10.5 9.353 10.5 9.355 10.5 C 9.357 10.5 9.359 10.5 9.361 10.5 C 9.363 10.5 9.365 10.5 9.367 10.5 C 9.369 10.5 9.371 10.5 9.373 10.5 C 9.375 10.5 9.377 10.5 9.379 10.5 C 9.381 10.5 9.383 10.5 9.385 10.5 C 9.387 10.5 9.389 10.5 9.392 10.5 C 9.394 10.5 9.396 10.5 9.398 10.5 C 9.4 10.5 9.402 10.5 9.404 10.5 C 9.406 10.5 9.408 10.5 9.41 10.5 C 9.412 10.5 9.414 10.5 9.416 10.5 C 9.418 10.5 9.42 10.5 9.423 10.5 C 9.425 10.5 9.427 10.5 9.429 10.5 C 9.431 10.5 9.433 10.5 9.435 10.5 C 9.437 10.5 9.439 10.5 9.441 10.5 C 9.443 10.5 9.446 10.5 9.448 10.5 C 9.45 10.5 9.452 10.5 9.454 10.5 C 9.456 10.5 9.458 10.5 9.46 10.5 C 9.462 10.5 9.464 10.5 9.467 10.5 C 9.469 10.5 9.471 10.5 9.473 10.5 C 9.475 10.5 9.477 10.5 9.479 10.5 C 9.481 10.5 9.484 10.5 9.486 10.5 C 9.488 10.5 9.49 10.5 9.492 10.5 C 9.494 10.5 9.496 10.5 9.499 10.5 C 9.501 10.5 9.503 10.5 9.505 10.5 C 9.507 10.5 9.509 10.5 9.511 10.5 C 9.514 10.5 9.516 10.5 9.518 10.5 C 9.52 10.5 9.522 10.5 9.524 10.5 C 9.527 10.5 9.529 10.5 9.531 10.5 C 9.533 10.5 9.535 10.5 9.537 10.5 C 9.54 10.5 9.542 10.5 9.544 10.5 C 9.546 10.5 9.548 10.5 9.551 10.5 C 9.553 10.5 9.555 10.5 9.557 10.5 C 9.559 10.5 9.562 10.5 9.564 10.5 C 9.566 10.5 9.568 10.5 9.57 10.5 C 9.573 10.5 9.575 10.5 9.577 10.5 C 9.579 10.5 9.581 10.5 9.584 10.5 C 9.586 10.5 9.588 10.5 9.59 10.5 C 9.593 10.5 9.595 10.5 9.597 10.5 C 9.599 10.5 9.602 10.5 9.604 10.5 C 9.606 10.5 9.608 10.5 9.61 10.5 C 9.613 10.5 9.615 10.5 9.617 10.5 C 9.619 10.5 9.622 10.5 9.624 10.5 C 9.626 10.5 9.629 10.5 9.631 10.5 C 9.633 10.5 9.635 10.5 9.638 10.5 C 9.64 10.5 9.642 10.5 9.644 10.5 C 9.647 10.5 9.649 10.5 9.651 10.5 C 9.654 10.5 9.656 10.5 9.658 10.5 C 9.66 10.5 9.663 10.5 9.665 10.5 C 9.667 10.5 9.67 10.5 9.672 10.5 C 9.674 10.5 9.677 10.5 9.679 10.5 C 9.681 10.5 9.684 10.5 9.686 10.5 C 9.688 10.5 9.69 10.5 9.693 10.5 C 9.695 10.5 9.697 10.5 9.7 10.5 C 9.702 10.5 9.704 10.5 9.707 10.5 C 9.709 10.5 9.711 10.5 9.714 10.5 C 9.716 10.5 9.719 10.5 9.721 10.5 C 9.723 10.5 9.726 10.5 9.728 10.5 C 9.73 10.5 9.733 10.5 9.735 10.5 C 9.737 10.5 9.74 10.5 9.742 10.5 C 9.745 10.5 9.747 10.5 9.749 10.5 C 9.752 10.5 9.754 10.5 9.756 10.5 C 9.759 10.5 9.761 10.5 9.764 10.5 C 9.766 10.5 9.768 10.5 9.771 10.5 C 9.773 10.5 9.776 10.5 9.778 10.5 C 9.78 10.5 9.783 10.5 9.785 10.5 C 9.788 10.5 9.79 10.5 9.793 10.5 C 9.795 10.5 9.797 10.5 9.8 10.5 C 9.802 10.5 9.805 10.5 9.807 10.5 C 9.81 10.5 9.812 10.5 9.815 10.5 C 9.817 10.5 9.819 10.5 9.822 10.5 C 9.824 10.5 9.827 10.5 9.829 10.5 C 9.832 10.5 9.834 10.5 9.837 10.5 C 9.839 10.5 9.842 10.5 9.844 10.5 C 9.847 10.5 9.849 10.5 9.852 10.5 C 9.854 10.5 9.857 10.5 9.859 10.5 C 9.862 10.5 9.864 10.5 9.867 10.5 C 9.869 10.5 9.872 10.5 9.874 10.5 C 9.877 10.5 9.879 10.5 9.882 10.5 C 9.884 10.5 9.887 10.5 9.889 10.5 C 9.892 10.5 9.894 10.5 9.897 10.5 C 9.899 10.5 9.902 10.5 9.904 10.5 C 9.907 10.5 9.91 10.5 9.912 10.5 C 9.915 10.5 9.917 10.5 9.92 10.5 C 9.922 10.5 9.925 10.5 9.927 10.5 C 9.93 10.5 9.933 10.5 9.935 10.5 C 9.938 10.5 9.94 10.5 9.943 10.5 C 9.945 10.5 9.948 10.5 9.951 10.5 C 9.953 10.5 9.956 10.5 9.958 10.5 C 9.961 10.5 9.964 10.5 9.966 10.5 C 9.969 10.5 9.972 10.5 9.974 10.5 C 9.977 10.5 9.979 10.5 9.982 10.5 C 9.985 10.5 9.987 10.5 9.99 10.5 C 9.993 10.5 9.995 10.5 9.998 10.5 C 10.001 10.5 10.003 10.5 10.006 10.5 C 10.008 10.5 10.011 10.5 10.014 10.5 C 10.016 10.5 10.019 10.5 10.022 10.5 C 10.024 10.5 10.027 10.5 10.03 10.5 C 10.033 10.5 10.035 10.5 10.038 10.5 C 10.041 10.5 10.043 10.5 10.046 10.5 C 10.049 10.5 10.051 10.5 10.054 10.5 C 10.057 10.5 10.06 10.5 10.062 10.5 C 10.065 10.5 10.068 10.5 10.07 10.5 C 10.073 10.5 10.076 10.5 10.079 10.5 C 10.081 10.5 10.084 10.5 10.087 10.5 C 10.09 10.5 10.092 10.5 10.095 10.5 C 10.098 10.5 10.101 10.5 10.103 10.5 C 10.106 10.5 10.109 10.5 10.112 10.5 C 10.114 10.5 10.117 10.5 10.12 10.5 C 10.123 10.5 10.126 10.5 10.128 10.5 C 10.131 10.5 10.134 10.5 10.137 10.5 C 10.14 10.5 10.142 10.5 10.145 10.5 C 10.148 10.5 10.151 10.5 10.154 10.5 C 10.156 10.5 10.159 10.5 10.162 10.5 C 10.165 10.5 10.168 10.5 10.171 10.5 C 10.173 10.5 10.176 10.5 10.179 10.5 C 10.182 10.5 10.185 10.5 10.188 10.5 C 10.191 10.5 10.193 10.5 10.196 10.5 C 10.199 10.5 10.202 10.5 10.205 10.5 C 10.208 10.5 10.211 10.5 10.214 10.5 C 10.216 10.5 10.219 10.5 10.222 10.5 C 10.225 10.5 10.228 10.5 10.231 10.5 C 10.234 10.5 10.237 10.5 10.24 10.5 C 10.243 10.5 10.246 10.5 10.248 10.5 C 10.251 10.5 10.254 10.5 10.257 10.5 C 10.26 10.5 10.263 10.5 10.266 10.5 C 10.269 10.5 10.272 10.5 10.275 10.5 C 10.278 10.5 10.281 10.5 10.284 10.5 C 10.287 10.5 10.29 10.5 10.293 10.5 C 10.296 10.5 10.299 10.5 10.302 10.5 C 10.305 10.5 10.308 10.5 10.311 10.5 C 10.314 10.5 10.317 10.5 10.32 10.5 C 10.323 10.5 10.326 10.5 10.329 10.5 C 10.332 10.5 10.335 10.5 10.338 10.5 C 10.341 10.5 10.344 10.5 10.347 10.5 C 10.35 10.5 10.353 10.5 10.356 10.5 C 10.359 10.5 10.362 10.5 10.365 10.5 C 10.368 10.5 10.371 10.5 10.374 10.5 C 10.378 10.5 10.381 10.5 10.384 10.5 C 10.387 10.5 10.39 10.5 10.393 10.5 C 10.396 10.5 10.399 10.5 10.402 10.5 C 10.405 10.5 10.409 10.5 10.412 10.5 C 10.415 10.5 10.418 10.5 10.421 10.5 C 10.424 10.5 10.427 10.5 10.43 10.5 C 10.434 10.5 10.437 10.5 10.44 10.5 C 10.443 10.5 10.446 10.5 10.449 10.5 C 10.452 10.5 10.456 10.5 10.459 10.5 C 10.462 10.5 10.465 10.5 10.468 10.5 C 10.471 10.5 10.475 10.5 10.478 10.5 C 10.481 10.5 10.484 10.5 10.487 10.5 C 10.491 10.5 10.494 10.5 10.497 10.5 C 10.5 10.5 10.504 10.5 10.507 10.5 C 10.51 10.5 10.513 10.5 10.516 10.5 C 10.52 10.5 10.523 10.5 10.526 10.5 C 10.529 10.5 10.533 10.5 10.536 10.5 C 10.539 10.5 10.543 10.5 10.546 10.5 C 10.549 10.5 10.552 10.5 10.556 10.5 C 10.559 10.5 10.562 10.5 10.565 10.5 C 10.569 10.5 10.572 10.5 10.575 10.5 C 10.579 10.5 10.582 10.5 10.585 10.5 C 10.589 10.5 10.592 10.5 10.595 10.5 C 10.599 10.5 10.602 10.5 10.605 10.5 C 10.609 10.5 10.612 10.5 10.615 10.5 L 10.615 8.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "hardware-drive-network": "<g transform=\"translate(1.5,4.5)\"><path d=\"M 5 10 L 4.5 10 L 4.5 10.5 L 5 10.5 L 5 10 Z M 5 8 L 5 7.5 L 4.5 7.5 L 4.5 8 L 5 8 Z M 8 10 L 8 10.5 L 8.5 10.5 L 8.5 10 L 8 10 Z M 8 8 L 8.5 8 L 8.5 7.5 L 8 7.5 L 8 8 Z M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 13 0 L 13.5 0 L 13.5 -0.5 L 13 -0.5 L 13 0 Z M 13 4 L 13 4.5 L 13.5 4.5 L 13.5 4 L 13 4 Z M 0 4 L -0.5 4 L -0.5 4.5 L 0 4.5 L 0 4 Z M 12.5 8.5 L 8 8.5 L 8 9.5 L 12.5 9.5 L 12.5 8.5 Z M 5 8.5 L 0.5 8.5 L 0.5 9.5 L 5 9.5 L 5 8.5 Z M 8 9.5 L 5 9.5 L 5 10.5 L 8 10.5 L 8 9.5 Z M 7.5 8 L 7.5 10 L 8.5 10 L 8.5 8 L 7.5 8 Z M 5 8.5 L 8 8.5 L 8 7.5 L 5 7.5 L 5 8.5 Z M 5.5 10 L 5.5 9 L 4.5 9 L 4.5 10 L 5.5 10 Z M 5.5 9 L 5.5 8 L 4.5 8 L 4.5 9 L 5.5 9 Z M 0 0.5 L 13 0.5 L 13 -0.5 L 0 -0.5 L 0 0.5 Z M 12.5 0 L 12.5 4 L 13.5 4 L 13.5 0 L 12.5 0 Z M 13 3.5 L 0 3.5 L 0 4.5 L 13 4.5 L 13 3.5 Z M 0.5 4 L 0.5 0 L -0.5 0 L -0.5 4 L 0.5 4 Z M 6 4 L 6 8 L 7 8 L 7 4 L 6 4 Z M 1.5 2.5 L 2.5 2.5 L 2.5 1.5 L 1.5 1.5 L 1.5 2.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "hardware-drive-usb": "<g transform=\"translate(5.5,5.5)\"><path d=\"M 5 0 L 5.5 0 L 5.5 -0.5 L 5 -0.5 L 5 0 Z M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 5 7.33 L 5.129 7.813 L 5.5 7.714 L 5.5 7.33 L 5 7.33 Z M 0 7.33 L -0.5 7.33 L -0.5 7.714 L -0.129 7.813 L 0 7.33 Z M 2.5 8 L 2.371 8.483 L 2.5 8.518 L 2.629 8.483 L 2.5 8 Z M 5 -0.5 L 0 -0.5 L 0 0.5 L 5 0.5 L 5 -0.5 Z M 5.5 7.33 L 5.5 0 L 4.5 0 L 4.5 7.33 L 5.5 7.33 Z M -0.5 0 L -0.5 7.33 L 0.5 7.33 L 0.5 0 L -0.5 0 Z M 4.871 6.847 L 2.371 7.517 L 2.629 8.483 L 5.129 7.813 L 4.871 6.847 Z M 2.629 7.517 L 0.129 6.847 L -0.129 7.813 L 2.371 8.483 L 2.629 7.517 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(6.5,2.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 3 0 L 3.5 0 L 3.5 -0.5 L 3 -0.5 L 3 0 Z M 0.5 3 L 0.5 0 L -0.5 0 L -0.5 3 L 0.5 3 Z M 0 0.5 L 3 0.5 L 3 -0.5 L 0 -0.5 L 0 0.5 Z M 2.5 0 L 2.5 3 L 3.5 3 L 3.5 0 L 2.5 0 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "hardware-drone": "<g transform=\"translate(1.5,1.5)\"><path d=\"M 5 10.5 C 5 11.881 3.881 13 2.5 13 C 1.119 13 0 11.881 0 10.5 C 0 9.119 1.119 8 2.5 8 C 2.878 8 3.236 8.084 3.557 8.234 C 4.088 8.482 4.518 8.912 4.766 9.443 C 4.916 9.764 5 10.122 5 10.5 Z\" fill=\"rgb(255,255,255)\" fill-rule=\"nonzero\" /> <path d=\"M 5 2.5 C 5 2.878 4.916 3.236 4.766 3.557 C 4.518 4.088 4.088 4.518 3.557 4.766 C 3.236 4.916 2.878 5 2.5 5 C 1.119 5 0 3.881 0 2.5 C 0 1.119 1.119 0 2.5 0 C 3.881 0 5 1.119 5 2.5 Z\" fill=\"rgb(255,255,255)\" fill-rule=\"nonzero\" /> <path d=\"M 13 10.5 C 13 11.881 11.881 13 10.5 13 C 9.119 13 8 11.881 8 10.5 C 8 10.122 8.084 9.764 8.234 9.443 C 8.482 8.912 8.912 8.482 9.443 8.234 C 9.764 8.084 10.122 8 10.5 8 C 11.881 8 13 9.119 13 10.5 Z\" fill=\"rgb(255,255,255)\" fill-rule=\"nonzero\" /> <path d=\"M 13 2.5 C 13 3.881 11.881 5 10.5 5 C 10.122 5 9.764 4.916 9.443 4.766 C 8.912 4.518 8.482 4.088 8.234 3.557 C 8.084 3.236 8 2.878 8 2.5 C 8 1.119 9.119 0 10.5 0 C 11.881 0 13 1.119 13 2.5 Z\" fill=\"rgb(255,255,255)\" fill-rule=\"nonzero\" /> <path d=\"M 4.766 3.557 L 4.313 3.345 L 4.766 3.557 Z M 9.443 4.766 L 9.655 4.313 L 9.443 4.766 Z M 3.557 4.766 L 3.345 4.313 L 3.557 4.766 Z M 4.5 10.5 C 4.5 11.605 3.605 12.5 2.5 12.5 L 2.5 13.5 C 4.157 13.5 5.5 12.157 5.5 10.5 L 4.5 10.5 Z M 2.5 12.5 C 1.395 12.5 0.5 11.605 0.5 10.5 L -0.5 10.5 C -0.5 12.157 0.843 13.5 2.5 13.5 L 2.5 12.5 Z M 0.5 10.5 C 0.5 9.395 1.395 8.5 2.5 8.5 L 2.5 7.5 C 0.843 7.5 -0.5 8.843 -0.5 10.5 L 0.5 10.5 Z M 2.5 4.5 C 1.395 4.5 0.5 3.605 0.5 2.5 L -0.5 2.5 C -0.5 4.157 0.843 5.5 2.5 5.5 L 2.5 4.5 Z M 0.5 2.5 C 0.5 1.395 1.395 0.5 2.5 0.5 L 2.5 -0.5 C 0.843 -0.5 -0.5 0.843 -0.5 2.5 L 0.5 2.5 Z M 2.5 0.5 C 3.605 0.5 4.5 1.395 4.5 2.5 L 5.5 2.5 C 5.5 0.843 4.157 -0.5 2.5 -0.5 L 2.5 0.5 Z M 12.5 10.5 C 12.5 11.605 11.605 12.5 10.5 12.5 L 10.5 13.5 C 12.157 13.5 13.5 12.157 13.5 10.5 L 12.5 10.5 Z M 10.5 12.5 C 9.395 12.5 8.5 11.605 8.5 10.5 L 7.5 10.5 C 7.5 12.157 8.843 13.5 10.5 13.5 L 10.5 12.5 Z M 10.5 8.5 C 11.605 8.5 12.5 9.395 12.5 10.5 L 13.5 10.5 C 13.5 8.843 12.157 7.5 10.5 7.5 L 10.5 8.5 Z M 12.5 2.5 C 12.5 3.605 11.605 4.5 10.5 4.5 L 10.5 5.5 C 12.157 5.5 13.5 4.157 13.5 2.5 L 12.5 2.5 Z M 8.5 2.5 C 8.5 1.395 9.395 0.5 10.5 0.5 L 10.5 -0.5 C 8.843 -0.5 7.5 0.843 7.5 2.5 L 8.5 2.5 Z M 10.5 0.5 C 11.605 0.5 12.5 1.395 12.5 2.5 L 13.5 2.5 C 13.5 0.843 12.157 -0.5 10.5 -0.5 L 10.5 0.5 Z M 4.5 2.5 C 4.5 2.803 4.433 3.089 4.313 3.345 L 5.219 3.768 C 5.4 3.382 5.5 2.952 5.5 2.5 L 4.5 2.5 Z M 8.687 3.345 C 8.567 3.089 8.5 2.803 8.5 2.5 L 7.5 2.5 C 7.5 2.952 7.6 3.382 7.781 3.768 L 8.687 3.345 Z M 10.5 4.5 C 10.197 4.5 9.911 4.433 9.655 4.313 L 9.232 5.219 C 9.618 5.4 10.048 5.5 10.5 5.5 L 10.5 4.5 Z M 9.655 4.313 C 9.23 4.115 8.885 3.77 8.687 3.345 L 7.781 3.768 C 8.078 4.406 8.594 4.922 9.232 5.219 L 9.655 4.313 Z M 9.004 4.528 C 8.682 5.122 8.5 5.792 8.5 6.5 L 9.5 6.5 C 9.5 5.966 9.637 5.459 9.883 5.005 L 9.004 4.528 Z M 8.5 6.5 C 8.5 7.208 8.682 7.878 9.004 8.472 L 9.883 7.995 C 9.637 7.541 9.5 7.034 9.5 6.5 L 8.5 6.5 Z M 9.655 8.687 C 9.911 8.567 10.197 8.5 10.5 8.5 L 10.5 7.5 C 10.048 7.5 9.618 7.6 9.232 7.781 L 9.655 8.687 Z M 8.5 10.5 C 8.5 10.197 8.567 9.911 8.687 9.655 L 7.781 9.232 C 7.6 9.618 7.5 10.048 7.5 10.5 L 8.5 10.5 Z M 8.687 9.655 C 8.885 9.23 9.23 8.885 9.655 8.687 L 9.232 7.781 C 8.594 8.078 8.078 8.594 7.781 9.232 L 8.687 9.655 Z M 6.5 9.5 C 7.034 9.5 7.541 9.637 7.995 9.883 L 8.472 9.004 C 7.878 8.682 7.208 8.5 6.5 8.5 L 6.5 9.5 Z M 5.005 9.883 C 5.459 9.637 5.966 9.5 6.5 9.5 L 6.5 8.5 C 5.792 8.5 5.122 8.682 4.528 9.004 L 5.005 9.883 Z M 4.313 9.655 C 4.433 9.911 4.5 10.197 4.5 10.5 L 5.5 10.5 C 5.5 10.048 5.4 9.618 5.219 9.232 L 4.313 9.655 Z M 2.5 8.5 C 2.803 8.5 3.089 8.567 3.345 8.687 L 3.768 7.781 C 3.382 7.6 2.952 7.5 2.5 7.5 L 2.5 8.5 Z M 3.345 8.687 C 3.77 8.885 4.115 9.23 4.313 9.655 L 5.219 9.232 C 4.922 8.594 4.406 8.078 3.768 7.781 L 3.345 8.687 Z M 3.5 6.5 C 3.5 7.034 3.363 7.541 3.117 7.995 L 3.996 8.472 C 4.318 7.878 4.5 7.208 4.5 6.5 L 3.5 6.5 Z M 4.313 3.345 C 4.115 3.77 3.77 4.115 3.345 4.313 L 3.768 5.219 C 4.406 4.922 4.922 4.406 5.219 3.768 L 4.313 3.345 Z M 3.345 4.313 C 3.089 4.433 2.803 4.5 2.5 4.5 L 2.5 5.5 C 2.952 5.5 3.382 5.4 3.768 5.219 L 3.345 4.313 Z M 3.117 5.005 C 3.363 5.459 3.5 5.966 3.5 6.5 L 4.5 6.5 C 4.5 5.792 4.318 5.122 3.996 4.528 L 3.117 5.005 Z M 6 4.5 L 7 4.5 L 7 3.5 L 6 3.5 L 6 4.5 Z M 6.5 4 L 6.5 5 L 7.5 5 L 7.5 4 L 6.5 4 Z M 6.5 5 L 6.5 4 L 5.5 4 L 5.5 5 L 6.5 5 Z M 6.169 3.529 L 4.936 3.086 L 4.597 4.027 L 5.831 4.471 L 6.169 3.529 Z M 8.064 3.086 L 6.831 3.529 L 7.169 4.471 L 8.403 4.027 L 8.064 3.086 Z M 6.5 5 L 6.5 6 C 7.052 6 7.5 5.552 7.5 5 L 6.5 5 Z M 6.5 5 L 5.5 5 C 5.5 5.552 5.948 6 6.5 6 L 6.5 5 Z\" fill=\"rgb(255,0,0)\" fill-rule=\"nonzero\" /></g><g transform=\"translate(0,0)\"><path d=\"M 1.646 -0.354 L -0.354 1.646 L 0.354 2.354 L 2.354 0.354 L 1.646 -0.354 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(0,0)\"><path d=\"M 1.646 -0.354 L -0.354 1.646 L 0.354 2.354 L 2.354 0.354 L 1.646 -0.354 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(0,0)\"><path d=\"M 1.646 -0.354 L -0.354 1.646 L 0.354 2.354 L 2.354 0.354 L 1.646 -0.354 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(0,0)\"><path d=\"M 1.646 -0.354 L -0.354 1.646 L 0.354 2.354 L 2.354 0.354 L 1.646 -0.354 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "hardware-fan": "<g transform=\"translate(2.5,2.5)\"><path d=\"M 11 5 L 10.5 5 L 11 5 Z M 11 3 L 11.5 3 L 11.5 2.5 L 11 2.5 L 11 3 Z M 2 4 L 2 3.5 L 2 4 Z M 0 6 L -0.5 6 L 0 6 Z M 0 8 L -0.5 8 L -0.5 8.5 L 0 8.5 L 0 8 Z M 7 2 L 6.5 2 L 7 2 Z M 3 0 L 3 -0.5 L 2.5 -0.5 L 2.5 0 L 3 0 Z M 8 11 L 8 11.5 L 8.5 11.5 L 8.5 11 L 8 11 Z M 9 6.5 C 8.384 6.5 7.605 6.181 6.927 5.811 C 6.599 5.632 6.315 5.453 6.113 5.318 C 6.013 5.251 5.933 5.196 5.879 5.157 C 5.852 5.138 5.831 5.123 5.818 5.113 C 5.811 5.108 5.807 5.105 5.803 5.103 C 5.802 5.101 5.801 5.101 5.8 5.1 C 5.8 5.1 5.8 5.1 5.8 5.1 C 5.8 5.1 5.8 5.1 5.8 5.1 C 5.8 5.1 5.8 5.1 5.8 5.1 C 5.8 5.1 5.8 5.1 5.5 5.5 C 5.2 5.9 5.2 5.9 5.2 5.9 C 5.2 5.9 5.2 5.9 5.2 5.9 C 5.201 5.9 5.201 5.901 5.201 5.901 C 5.201 5.901 5.202 5.901 5.203 5.902 C 5.204 5.903 5.206 5.904 5.208 5.906 C 5.213 5.909 5.219 5.914 5.228 5.92 C 5.244 5.933 5.268 5.95 5.299 5.972 C 5.36 6.015 5.448 6.077 5.559 6.15 C 5.778 6.297 6.088 6.493 6.448 6.689 C 7.145 7.069 8.116 7.5 9 7.5 L 9 6.5 Z M 5.5 5.5 C 5.89 5.812 5.89 5.812 5.89 5.812 C 5.89 5.812 5.89 5.813 5.89 5.813 C 5.89 5.813 5.89 5.813 5.89 5.812 C 5.89 5.812 5.891 5.812 5.891 5.812 C 5.892 5.811 5.893 5.809 5.894 5.807 C 5.898 5.804 5.903 5.797 5.909 5.789 C 5.923 5.772 5.944 5.747 5.971 5.715 C 6.026 5.65 6.106 5.557 6.208 5.444 C 6.411 5.218 6.696 4.918 7.024 4.62 C 7.354 4.32 7.716 4.032 8.074 3.822 C 8.44 3.606 8.754 3.5 9 3.5 L 9 2.5 C 8.496 2.5 7.998 2.706 7.567 2.96 C 7.128 3.218 6.709 3.555 6.351 3.88 C 5.992 4.207 5.683 4.532 5.464 4.775 C 5.355 4.897 5.267 4.998 5.207 5.07 C 5.176 5.106 5.153 5.135 5.136 5.155 C 5.128 5.165 5.122 5.173 5.117 5.178 C 5.115 5.181 5.113 5.183 5.112 5.185 C 5.111 5.186 5.111 5.186 5.11 5.187 C 5.11 5.187 5.11 5.187 5.11 5.187 C 5.11 5.187 5.11 5.187 5.11 5.187 C 5.11 5.188 5.11 5.188 5.5 5.5 Z M 9 3.5 C 9.001 3.5 9.002 3.5 9.004 3.5 C 9.005 3.5 9.006 3.5 9.007 3.5 C 9.008 3.5 9.01 3.5 9.011 3.5 C 9.012 3.5 9.013 3.5 9.014 3.5 C 9.016 3.5 9.017 3.5 9.018 3.5 C 9.019 3.5 9.021 3.5 9.022 3.5 C 9.023 3.5 9.024 3.5 9.025 3.5 C 9.027 3.5 9.028 3.5 9.029 3.5 C 9.03 3.5 9.032 3.5 9.033 3.5 C 9.034 3.5 9.035 3.5 9.037 3.5 C 9.038 3.5 9.039 3.5 9.04 3.5 C 9.042 3.5 9.043 3.5 9.044 3.5 C 9.045 3.5 9.047 3.5 9.048 3.5 C 9.049 3.5 9.051 3.5 9.052 3.5 C 9.053 3.5 9.054 3.5 9.056 3.5 C 9.057 3.5 9.058 3.5 9.059 3.5 C 9.061 3.5 9.062 3.5 9.063 3.5 C 9.065 3.5 9.066 3.5 9.067 3.5 C 9.069 3.5 9.07 3.5 9.071 3.5 C 9.072 3.5 9.074 3.5 9.075 3.5 C 9.076 3.5 9.078 3.5 9.079 3.5 C 9.08 3.5 9.082 3.5 9.083 3.5 C 9.084 3.5 9.086 3.5 9.087 3.5 C 9.088 3.5 9.089 3.5 9.091 3.5 C 9.092 3.5 9.093 3.5 9.095 3.5 C 9.096 3.5 9.097 3.5 9.099 3.5 C 9.1 3.5 9.101 3.5 9.103 3.5 C 9.104 3.5 9.105 3.5 9.107 3.5 C 9.108 3.5 9.11 3.5 9.111 3.5 C 9.112 3.5 9.114 3.5 9.115 3.5 C 9.116 3.5 9.118 3.5 9.119 3.5 C 9.12 3.5 9.122 3.5 9.123 3.5 C 9.124 3.5 9.126 3.5 9.127 3.5 C 9.129 3.5 9.13 3.5 9.131 3.5 C 9.133 3.5 9.134 3.5 9.135 3.5 C 9.137 3.5 9.138 3.5 9.14 3.5 C 9.141 3.5 9.142 3.5 9.144 3.5 C 9.145 3.5 9.147 3.5 9.148 3.5 C 9.149 3.5 9.151 3.5 9.152 3.5 C 9.154 3.5 9.155 3.5 9.156 3.5 C 9.158 3.5 9.159 3.5 9.161 3.5 C 9.162 3.5 9.163 3.5 9.165 3.5 C 9.166 3.5 9.168 3.5 9.169 3.5 C 9.171 3.5 9.172 3.5 9.173 3.5 C 9.175 3.5 9.176 3.5 9.178 3.5 C 9.179 3.5 9.18 3.5 9.182 3.5 C 9.183 3.5 9.185 3.5 9.186 3.5 C 9.188 3.5 9.189 3.5 9.191 3.5 C 9.192 3.5 9.193 3.5 9.195 3.5 C 9.196 3.5 9.198 3.5 9.199 3.5 C 9.201 3.5 9.202 3.5 9.204 3.5 C 9.205 3.5 9.206 3.5 9.208 3.5 C 9.209 3.5 9.211 3.5 9.212 3.5 C 9.214 3.5 9.215 3.5 9.217 3.5 C 9.218 3.5 9.22 3.5 9.221 3.5 C 9.223 3.5 9.224 3.5 9.226 3.5 C 9.227 3.5 9.228 3.5 9.23 3.5 C 9.231 3.5 9.233 3.5 9.234 3.5 C 9.236 3.5 9.237 3.5 9.239 3.5 C 9.24 3.5 9.242 3.5 9.243 3.5 C 9.245 3.5 9.246 3.5 9.248 3.5 C 9.249 3.5 9.251 3.5 9.252 3.5 C 9.254 3.5 9.255 3.5 9.257 3.5 C 9.258 3.5 9.26 3.5 9.261 3.5 C 9.263 3.5 9.264 3.5 9.266 3.5 C 9.267 3.5 9.269 3.5 9.27 3.5 C 9.272 3.5 9.273 3.5 9.275 3.5 C 9.276 3.5 9.278 3.5 9.279 3.5 C 9.281 3.5 9.283 3.5 9.284 3.5 C 9.286 3.5 9.287 3.5 9.289 3.5 C 9.29 3.5 9.292 3.5 9.293 3.5 C 9.295 3.5 9.296 3.5 9.298 3.5 C 9.299 3.5 9.301 3.5 9.302 3.5 C 9.304 3.5 9.306 3.5 9.307 3.5 C 9.309 3.5 9.31 3.5 9.312 3.5 C 9.313 3.5 9.315 3.5 9.316 3.5 C 9.318 3.5 9.319 3.5 9.321 3.5 C 9.323 3.5 9.324 3.5 9.326 3.5 C 9.327 3.5 9.329 3.5 9.33 3.5 C 9.332 3.5 9.334 3.5 9.335 3.5 C 9.337 3.5 9.338 3.5 9.34 3.5 C 9.341 3.5 9.343 3.5 9.344 3.5 C 9.346 3.5 9.348 3.5 9.349 3.5 C 9.351 3.5 9.352 3.5 9.354 3.5 C 9.356 3.5 9.357 3.5 9.359 3.5 C 9.36 3.5 9.362 3.5 9.363 3.5 C 9.365 3.5 9.367 3.5 9.368 3.5 C 9.37 3.5 9.371 3.5 9.373 3.5 C 9.375 3.5 9.376 3.5 9.378 3.5 C 9.379 3.5 9.381 3.5 9.382 3.5 C 9.384 3.5 9.386 3.5 9.387 3.5 C 9.389 3.5 9.39 3.5 9.392 3.5 C 9.394 3.5 9.395 3.5 9.397 3.5 C 9.399 3.5 9.4 3.5 9.402 3.5 C 9.403 3.5 9.405 3.5 9.407 3.5 C 9.408 3.5 9.41 3.5 9.411 3.5 C 9.413 3.5 9.415 3.5 9.416 3.5 C 9.418 3.5 9.419 3.5 9.421 3.5 C 9.423 3.5 9.424 3.5 9.426 3.5 C 9.428 3.5 9.429 3.5 9.431 3.5 C 9.432 3.5 9.434 3.5 9.436 3.5 C 9.437 3.5 9.439 3.5 9.441 3.5 C 9.442 3.5 9.444 3.5 9.446 3.5 C 9.447 3.5 9.449 3.5 9.45 3.5 C 9.452 3.5 9.454 3.5 9.455 3.5 C 9.457 3.5 9.459 3.5 9.46 3.5 C 9.462 3.5 9.464 3.5 9.465 3.5 C 9.467 3.5 9.468 3.5 9.47 3.5 C 9.472 3.5 9.473 3.5 9.475 3.5 C 9.477 3.5 9.478 3.5 9.48 3.5 C 9.482 3.5 9.483 3.5 9.485 3.5 C 9.487 3.5 9.488 3.5 9.49 3.5 C 9.492 3.5 9.493 3.5 9.495 3.5 C 9.497 3.5 9.498 3.5 9.5 3.5 C 9.502 3.5 9.503 3.5 9.505 3.5 C 9.507 3.5 9.508 3.5 9.51 3.5 C 9.511 3.5 9.513 3.5 9.515 3.5 C 9.516 3.5 9.518 3.5 9.52 3.5 C 9.521 3.5 9.523 3.5 9.525 3.5 C 9.527 3.5 9.528 3.5 9.53 3.5 C 9.532 3.5 9.533 3.5 9.535 3.5 C 9.537 3.5 9.538 3.5 9.54 3.5 C 9.542 3.5 9.543 3.5 9.545 3.5 C 9.547 3.5 9.548 3.5 9.55 3.5 C 9.552 3.5 9.553 3.5 9.555 3.5 C 9.557 3.5 9.558 3.5 9.56 3.5 C 9.562 3.5 9.563 3.5 9.565 3.5 C 9.567 3.5 9.569 3.5 9.57 3.5 C 9.572 3.5 9.574 3.5 9.575 3.5 C 9.577 3.5 9.579 3.5 9.58 3.5 C 9.582 3.5 9.584 3.5 9.585 3.5 C 9.587 3.5 9.589 3.5 9.591 3.5 C 9.592 3.5 9.594 3.5 9.596 3.5 C 9.597 3.5 9.599 3.5 9.601 3.5 C 9.602 3.5 9.604 3.5 9.606 3.5 C 9.607 3.5 9.609 3.5 9.611 3.5 C 9.613 3.5 9.614 3.5 9.616 3.5 C 9.618 3.5 9.619 3.5 9.621 3.5 C 9.623 3.5 9.625 3.5 9.626 3.5 C 9.628 3.5 9.63 3.5 9.631 3.5 C 9.633 3.5 9.635 3.5 9.636 3.5 C 9.638 3.5 9.64 3.5 9.642 3.5 C 9.643 3.5 9.645 3.5 9.647 3.5 C 9.648 3.5 9.65 3.5 9.652 3.5 C 9.654 3.5 9.655 3.5 9.657 3.5 C 9.659 3.5 9.66 3.5 9.662 3.5 C 9.664 3.5 9.666 3.5 9.667 3.5 C 9.669 3.5 9.671 3.5 9.672 3.5 C 9.674 3.5 9.676 3.5 9.678 3.5 C 9.679 3.5 9.681 3.5 9.683 3.5 C 9.685 3.5 9.686 3.5 9.688 3.5 C 9.69 3.5 9.691 3.5 9.693 3.5 C 9.695 3.5 9.697 3.5 9.698 3.5 C 9.7 3.5 9.702 3.5 9.703 3.5 C 9.705 3.5 9.707 3.5 9.709 3.5 C 9.71 3.5 9.712 3.5 9.714 3.5 C 9.716 3.5 9.717 3.5 9.719 3.5 C 9.721 3.5 9.722 3.5 9.724 3.5 C 9.726 3.5 9.728 3.5 9.729 3.5 C 9.731 3.5 9.733 3.5 9.735 3.5 C 9.736 3.5 9.738 3.5 9.74 3.5 C 9.741 3.5 9.743 3.5 9.745 3.5 C 9.747 3.5 9.748 3.5 9.75 3.5 C 9.752 3.5 9.754 3.5 9.755 3.5 C 9.757 3.5 9.759 3.5 9.761 3.5 C 9.762 3.5 9.764 3.5 9.766 3.5 C 9.767 3.5 9.769 3.5 9.771 3.5 C 9.773 3.5 9.774 3.5 9.776 3.5 C 9.778 3.5 9.78 3.5 9.781 3.5 C 9.783 3.5 9.785 3.5 9.787 3.5 C 9.788 3.5 9.79 3.5 9.792 3.5 C 9.793 3.5 9.795 3.5 9.797 3.5 C 9.799 3.5 9.8 3.5 9.802 3.5 C 9.804 3.5 9.806 3.5 9.807 3.5 C 9.809 3.5 9.811 3.5 9.813 3.5 C 9.814 3.5 9.816 3.5 9.818 3.5 C 9.82 3.5 9.821 3.5 9.823 3.5 C 9.825 3.5 9.826 3.5 9.828 3.5 C 9.83 3.5 9.832 3.5 9.833 3.5 C 9.835 3.5 9.837 3.5 9.839 3.5 C 9.84 3.5 9.842 3.5 9.844 3.5 C 9.846 3.5 9.847 3.5 9.849 3.5 C 9.851 3.5 9.853 3.5 9.854 3.5 C 9.856 3.5 9.858 3.5 9.86 3.5 C 9.861 3.5 9.863 3.5 9.865 3.5 C 9.866 3.5 9.868 3.5 9.87 3.5 C 9.872 3.5 9.873 3.5 9.875 3.5 C 9.877 3.5 9.879 3.5 9.88 3.5 C 9.882 3.5 9.884 3.5 9.886 3.5 C 9.887 3.5 9.889 3.5 9.891 3.5 C 9.893 3.5 9.894 3.5 9.896 3.5 C 9.898 3.5 9.9 3.5 9.901 3.5 C 9.903 3.5 9.905 3.5 9.906 3.5 C 9.908 3.5 9.91 3.5 9.912 3.5 C 9.913 3.5 9.915 3.5 9.917 3.5 C 9.919 3.5 9.92 3.5 9.922 3.5 C 9.924 3.5 9.926 3.5 9.927 3.5 C 9.929 3.5 9.931 3.5 9.933 3.5 C 9.934 3.5 9.936 3.5 9.938 3.5 C 9.939 3.5 9.941 3.5 9.943 3.5 C 9.945 3.5 9.946 3.5 9.948 3.5 C 9.95 3.5 9.952 3.5 9.953 3.5 C 9.955 3.5 9.957 3.5 9.959 3.5 C 9.96 3.5 9.962 3.5 9.964 3.5 C 9.965 3.5 9.967 3.5 9.969 3.5 C 9.971 3.5 9.972 3.5 9.974 3.5 C 9.976 3.5 9.978 3.5 9.979 3.5 C 9.981 3.5 9.983 3.5 9.984 3.5 C 9.986 3.5 9.988 3.5 9.99 3.5 C 9.991 3.5 9.993 3.5 9.995 3.5 C 9.997 3.5 9.998 3.5 10 3.5 C 10.002 3.5 10.004 3.5 10.005 3.5 C 10.007 3.5 10.009 3.5 10.01 3.5 C 10.012 3.5 10.014 3.5 10.016 3.5 C 10.017 3.5 10.019 3.5 10.021 3.5 C 10.022 3.5 10.024 3.5 10.026 3.5 C 10.028 3.5 10.029 3.5 10.031 3.5 C 10.033 3.5 10.035 3.5 10.036 3.5 C 10.038 3.5 10.04 3.5 10.041 3.5 C 10.043 3.5 10.045 3.5 10.047 3.5 C 10.048 3.5 10.05 3.5 10.052 3.5 C 10.053 3.5 10.055 3.5 10.057 3.5 C 10.059 3.5 10.06 3.5 10.062 3.5 C 10.064 3.5 10.065 3.5 10.067 3.5 C 10.069 3.5 10.071 3.5 10.072 3.5 C 10.074 3.5 10.076 3.5 10.077 3.5 C 10.079 3.5 10.081 3.5 10.083 3.5 C 10.084 3.5 10.086 3.5 10.088 3.5 C 10.089 3.5 10.091 3.5 10.093 3.5 C 10.094 3.5 10.096 3.5 10.098 3.5 C 10.1 3.5 10.101 3.5 10.103 3.5 C 10.105 3.5 10.106 3.5 10.108 3.5 C 10.11 3.5 10.112 3.5 10.113 3.5 C 10.115 3.5 10.117 3.5 10.118 3.5 C 10.12 3.5 10.122 3.5 10.123 3.5 C 10.125 3.5 10.127 3.5 10.129 3.5 C 10.13 3.5 10.132 3.5 10.134 3.5 C 10.135 3.5 10.137 3.5 10.139 3.5 C 10.14 3.5 10.142 3.5 10.144 3.5 C 10.145 3.5 10.147 3.5 10.149 3.5 C 10.15 3.5 10.152 3.5 10.154 3.5 C 10.156 3.5 10.157 3.5 10.159 3.5 C 10.161 3.5 10.162 3.5 10.164 3.5 C 10.166 3.5 10.167 3.5 10.169 3.5 C 10.171 3.5 10.172 3.5 10.174 3.5 C 10.176 3.5 10.177 3.5 10.179 3.5 C 10.181 3.5 10.182 3.5 10.184 3.5 C 10.186 3.5 10.187 3.5 10.189 3.5 C 10.191 3.5 10.192 3.5 10.194 3.5 C 10.196 3.5 10.197 3.5 10.199 3.5 C 10.201 3.5 10.202 3.5 10.204 3.5 C 10.206 3.5 10.207 3.5 10.209 3.5 C 10.211 3.5 10.212 3.5 10.214 3.5 C 10.216 3.5 10.217 3.5 10.219 3.5 C 10.221 3.5 10.222 3.5 10.224 3.5 C 10.226 3.5 10.227 3.5 10.229 3.5 C 10.231 3.5 10.232 3.5 10.234 3.5 C 10.236 3.5 10.237 3.5 10.239 3.5 C 10.241 3.5 10.242 3.5 10.244 3.5 C 10.246 3.5 10.247 3.5 10.249 3.5 C 10.251 3.5 10.252 3.5 10.254 3.5 C 10.255 3.5 10.257 3.5 10.259 3.5 C 10.26 3.5 10.262 3.5 10.264 3.5 C 10.265 3.5 10.267 3.5 10.269 3.5 C 10.27 3.5 10.272 3.5 10.273 3.5 C 10.275 3.5 10.277 3.5 10.278 3.5 C 10.28 3.5 10.282 3.5 10.283 3.5 C 10.285 3.5 10.286 3.5 10.288 3.5 C 10.29 3.5 10.291 3.5 10.293 3.5 C 10.295 3.5 10.296 3.5 10.298 3.5 C 10.299 3.5 10.301 3.5 10.303 3.5 C 10.304 3.5 10.306 3.5 10.308 3.5 C 10.309 3.5 10.311 3.5 10.312 3.5 C 10.314 3.5 10.316 3.5 10.317 3.5 C 10.319 3.5 10.32 3.5 10.322 3.5 C 10.324 3.5 10.325 3.5 10.327 3.5 C 10.328 3.5 10.33 3.5 10.332 3.5 C 10.333 3.5 10.335 3.5 10.336 3.5 C 10.338 3.5 10.34 3.5 10.341 3.5 C 10.343 3.5 10.344 3.5 10.346 3.5 C 10.348 3.5 10.349 3.5 10.351 3.5 C 10.352 3.5 10.354 3.5 10.355 3.5 C 10.357 3.5 10.359 3.5 10.36 3.5 C 10.362 3.5 10.363 3.5 10.365 3.5 C 10.367 3.5 10.368 3.5 10.37 3.5 C 10.371 3.5 10.373 3.5 10.374 3.5 C 10.376 3.5 10.378 3.5 10.379 3.5 C 10.381 3.5 10.382 3.5 10.384 3.5 C 10.385 3.5 10.387 3.5 10.389 3.5 C 10.39 3.5 10.392 3.5 10.393 3.5 C 10.395 3.5 10.396 3.5 10.398 3.5 C 10.399 3.5 10.401 3.5 10.403 3.5 C 10.404 3.5 10.406 3.5 10.407 3.5 C 10.409 3.5 10.41 3.5 10.412 3.5 C 10.413 3.5 10.415 3.5 10.416 3.5 C 10.418 3.5 10.419 3.5 10.421 3.5 C 10.423 3.5 10.424 3.5 10.426 3.5 C 10.427 3.5 10.429 3.5 10.43 3.5 C 10.432 3.5 10.433 3.5 10.435 3.5 C 10.436 3.5 10.438 3.5 10.439 3.5 C 10.441 3.5 10.442 3.5 10.444 3.5 C 10.445 3.5 10.447 3.5 10.448 3.5 C 10.45 3.5 10.451 3.5 10.453 3.5 C 10.455 3.5 10.456 3.5 10.458 3.5 C 10.459 3.5 10.461 3.5 10.462 3.5 C 10.464 3.5 10.465 3.5 10.467 3.5 C 10.468 3.5 10.47 3.5 10.471 3.5 C 10.473 3.5 10.474 3.5 10.476 3.5 C 10.477 3.5 10.478 3.5 10.48 3.5 C 10.481 3.5 10.483 3.5 10.484 3.5 C 10.486 3.5 10.487 3.5 10.489 3.5 C 10.49 3.5 10.492 3.5 10.493 3.5 C 10.495 3.5 10.496 3.5 10.498 3.5 C 10.499 3.5 10.501 3.5 10.502 3.5 C 10.504 3.5 10.505 3.5 10.506 3.5 C 10.508 3.5 10.509 3.5 10.511 3.5 C 10.512 3.5 10.514 3.5 10.515 3.5 C 10.517 3.5 10.518 3.5 10.52 3.5 C 10.521 3.5 10.522 3.5 10.524 3.5 C 10.525 3.5 10.527 3.5 10.528 3.5 C 10.53 3.5 10.531 3.5 10.533 3.5 C 10.534 3.5 10.535 3.5 10.537 3.5 C 10.538 3.5 10.54 3.5 10.541 3.5 C 10.543 3.5 10.544 3.5 10.545 3.5 C 10.547 3.5 10.548 3.5 10.55 3.5 C 10.551 3.5 10.553 3.5 10.554 3.5 C 10.555 3.5 10.557 3.5 10.558 3.5 C 10.56 3.5 10.561 3.5 10.562 3.5 C 10.564 3.5 10.565 3.5 10.567 3.5 C 10.568 3.5 10.569 3.5 10.571 3.5 C 10.572 3.5 10.574 3.5 10.575 3.5 C 10.576 3.5 10.578 3.5 10.579 3.5 C 10.58 3.5 10.582 3.5 10.583 3.5 C 10.585 3.5 10.586 3.5 10.587 3.5 C 10.589 3.5 10.59 3.5 10.591 3.5 C 10.593 3.5 10.594 3.5 10.596 3.5 C 10.597 3.5 10.598 3.5 10.6 3.5 C 10.601 3.5 10.602 3.5 10.604 3.5 C 10.605 3.5 10.606 3.5 10.608 3.5 C 10.609 3.5 10.611 3.5 10.612 3.5 C 10.613 3.5 10.615 3.5 10.616 3.5 C 10.617 3.5 10.619 3.5 10.62 3.5 C 10.621 3.5 10.623 3.5 10.624 3.5 C 10.625 3.5 10.627 3.5 10.628 3.5 C 10.629 3.5 10.631 3.5 10.632 3.5 C 10.633 3.5 10.634 3.5 10.636 3.5 C 10.637 3.5 10.638 3.5 10.64 3.5 C 10.641 3.5 10.642 3.5 10.644 3.5 C 10.645 3.5 10.646 3.5 10.648 3.5 C 10.649 3.5 10.65 3.5 10.651 3.5 C 10.653 3.5 10.654 3.5 10.655 3.5 C 10.657 3.5 10.658 3.5 10.659 3.5 C 10.66 3.5 10.662 3.5 10.663 3.5 C 10.664 3.5 10.666 3.5 10.667 3.5 C 10.668 3.5 10.669 3.5 10.671 3.5 C 10.672 3.5 10.673 3.5 10.674 3.5 C 10.676 3.5 10.677 3.5 10.678 3.5 C 10.679 3.5 10.681 3.5 10.682 3.5 C 10.683 3.5 10.684 3.5 10.686 3.5 C 10.687 3.5 10.688 3.5 10.689 3.5 C 10.691 3.5 10.692 3.5 10.693 3.5 C 10.694 3.5 10.696 3.5 10.697 3.5 C 10.698 3.5 10.699 3.5 10.7 3.5 C 10.702 3.5 10.703 3.5 10.704 3.5 C 10.705 3.5 10.707 3.5 10.708 3.5 C 10.709 3.5 10.71 3.5 10.711 3.5 C 10.713 3.5 10.714 3.5 10.715 3.5 C 10.716 3.5 10.717 3.5 10.719 3.5 C 10.72 3.5 10.721 3.5 10.722 3.5 C 10.723 3.5 10.724 3.5 10.726 3.5 C 10.727 3.5 10.728 3.5 10.729 3.5 C 10.73 3.5 10.732 3.5 10.733 3.5 C 10.734 3.5 10.735 3.5 10.736 3.5 C 10.737 3.5 10.739 3.5 10.74 3.5 C 10.741 3.5 10.742 3.5 10.743 3.5 C 10.744 3.5 10.745 3.5 10.747 3.5 C 10.748 3.5 10.749 3.5 10.75 3.5 C 10.751 3.5 10.752 3.5 10.753 3.5 C 10.755 3.5 10.756 3.5 10.757 3.5 C 10.758 3.5 10.759 3.5 10.76 3.5 C 10.761 3.5 10.762 3.5 10.763 3.5 C 10.765 3.5 10.766 3.5 10.767 3.5 C 10.768 3.5 10.769 3.5 10.77 3.5 C 10.771 3.5 10.772 3.5 10.773 3.5 C 10.774 3.5 10.776 3.5 10.777 3.5 C 10.778 3.5 10.779 3.5 10.78 3.5 C 10.781 3.5 10.782 3.5 10.783 3.5 C 10.784 3.5 10.785 3.5 10.786 3.5 C 10.787 3.5 10.788 3.5 10.79 3.5 C 10.791 3.5 10.792 3.5 10.793 3.5 C 10.794 3.5 10.795 3.5 10.796 3.5 C 10.797 3.5 10.798 3.5 10.799 3.5 C 10.8 3.5 10.801 3.5 10.802 3.5 C 10.803 3.5 10.804 3.5 10.805 3.5 C 10.806 3.5 10.807 3.5 10.808 3.5 C 10.809 3.5 10.81 3.5 10.811 3.5 C 10.812 3.5 10.813 3.5 10.814 3.5 C 10.815 3.5 10.816 3.5 10.817 3.5 C 10.818 3.5 10.819 3.5 10.82 3.5 C 10.821 3.5 10.822 3.5 10.823 3.5 C 10.824 3.5 10.825 3.5 10.826 3.5 C 10.827 3.5 10.828 3.5 10.829 3.5 C 10.83 3.5 10.831 3.5 10.832 3.5 C 10.833 3.5 10.834 3.5 10.835 3.5 C 10.836 3.5 10.837 3.5 10.838 3.5 C 10.839 3.5 10.84 3.5 10.841 3.5 C 10.842 3.5 10.842 3.5 10.843 3.5 C 10.844 3.5 10.845 3.5 10.846 3.5 C 10.847 3.5 10.848 3.5 10.849 3.5 C 10.85 3.5 10.851 3.5 10.852 3.5 C 10.853 3.5 10.854 3.5 10.854 3.5 C 10.855 3.5 10.856 3.5 10.857 3.5 C 10.858 3.5 10.859 3.5 10.86 3.5 C 10.861 3.5 10.862 3.5 10.863 3.5 C 10.863 3.5 10.864 3.5 10.865 3.5 C 10.866 3.5 10.867 3.5 10.868 3.5 C 10.869 3.5 10.87 3.5 10.87 3.5 C 10.871 3.5 10.872 3.5 10.873 3.5 C 10.874 3.5 10.875 3.5 10.875 3.5 C 10.876 3.5 10.877 3.5 10.878 3.5 C 10.879 3.5 10.88 3.5 10.881 3.5 C 10.881 3.5 10.882 3.5 10.883 3.5 C 10.884 3.5 10.885 3.5 10.885 3.5 C 10.886 3.5 10.887 3.5 10.888 3.5 C 10.889 3.5 10.89 3.5 10.89 3.5 C 10.891 3.5 10.892 3.5 10.893 3.5 C 10.893 3.5 10.894 3.5 10.895 3.5 C 10.896 3.5 10.897 3.5 10.897 3.5 C 10.898 3.5 10.899 3.5 10.9 3.5 C 10.9 3.5 10.901 3.5 10.902 3.5 C 10.903 3.5 10.904 3.5 10.904 3.5 C 10.905 3.5 10.906 3.5 10.907 3.5 C 10.907 3.5 10.908 3.5 10.909 3.5 C 10.909 3.5 10.91 3.5 10.911 3.5 C 10.912 3.5 10.912 3.5 10.913 3.5 C 10.914 3.5 10.915 3.5 10.915 3.5 C 10.916 3.5 10.917 3.5 10.917 3.5 C 10.918 3.5 10.919 3.5 10.919 3.5 C 10.92 3.5 10.921 3.5 10.922 3.5 C 10.922 3.5 10.923 3.5 10.924 3.5 C 10.924 3.5 10.925 3.5 10.926 3.5 C 10.926 3.5 10.927 3.5 10.928 3.5 C 10.928 3.5 10.929 3.5 10.93 3.5 C 10.93 3.5 10.931 3.5 10.932 3.5 C 10.932 3.5 10.933 3.5 10.933 3.5 C 10.934 3.5 10.935 3.5 10.935 3.5 C 10.936 3.5 10.937 3.5 10.937 3.5 C 10.938 3.5 10.939 3.5 10.939 3.5 C 10.94 3.5 10.94 3.5 10.941 3.5 C 10.942 3.5 10.942 3.5 10.943 3.5 C 10.943 3.5 10.944 3.5 10.945 3.5 C 10.945 3.5 10.946 3.5 10.946 3.5 C 10.947 3.5 10.947 3.5 10.948 3.5 C 10.949 3.5 10.949 3.5 10.95 3.5 C 10.95 3.5 10.951 3.5 10.951 3.5 C 10.952 3.5 10.952 3.5 10.953 3.5 C 10.954 3.5 10.954 3.5 10.955 3.5 C 10.955 3.5 10.956 3.5 10.956 3.5 C 10.957 3.5 10.957 3.5 10.958 3.5 C 10.958 3.5 10.959 3.5 10.959 3.5 C 10.96 3.5 10.96 3.5 10.961 3.5 C 10.961 3.5 10.962 3.5 10.962 3.5 C 10.963 3.5 10.963 3.5 10.964 3.5 C 10.964 3.5 10.965 3.5 10.965 3.5 C 10.966 3.5 10.966 3.5 10.967 3.5 C 10.967 3.5 10.967 3.5 10.968 3.5 C 10.968 3.5 10.969 3.5 10.969 3.5 C 10.97 3.5 10.97 3.5 10.971 3.5 C 10.971 3.5 10.971 3.5 10.972 3.5 C 10.972 3.5 10.973 3.5 10.973 3.5 C 10.974 3.5 10.974 3.5 10.974 3.5 C 10.975 3.5 10.975 3.5 10.976 3.5 C 10.976 3.5 10.976 3.5 10.977 3.5 C 10.977 3.5 10.978 3.5 10.978 3.5 C 10.978 3.5 10.979 3.5 10.979 3.5 C 10.979 3.5 10.98 3.5 10.98 3.5 C 10.981 3.5 10.981 3.5 10.981 3.5 C 10.982 3.5 10.982 3.5 10.982 3.5 C 10.983 3.5 10.983 3.5 10.983 3.5 C 10.984 3.5 10.984 3.5 10.984 3.5 C 10.985 3.5 10.985 3.5 10.985 3.5 C 10.986 3.5 10.986 3.5 10.986 3.5 C 10.986 3.5 10.987 3.5 10.987 3.5 C 10.987 3.5 10.988 3.5 10.988 3.5 C 10.988 3.5 10.988 3.5 10.989 3.5 C 10.989 3.5 10.989 3.5 10.99 3.5 C 10.99 3.5 10.99 3.5 10.99 3.5 C 10.991 3.5 10.991 3.5 10.991 3.5 C 10.991 3.5 10.992 3.5 10.992 3.5 C 10.992 3.5 10.992 3.5 10.992 3.5 C 10.993 3.5 10.993 3.5 10.993 3.5 C 10.993 3.5 10.994 3.5 10.994 3.5 C 10.994 3.5 10.994 3.5 10.994 3.5 C 10.995 3.5 10.995 3.5 10.995 3.5 C 10.995 3.5 10.995 3.5 10.995 3.5 C 10.996 3.5 10.996 3.5 10.996 3.5 C 10.996 3.5 10.996 3.5 10.996 3.5 C 10.997 3.5 10.997 3.5 10.997 3.5 C 10.997 3.5 10.997 3.5 10.997 3.5 C 10.997 3.5 10.998 3.5 10.998 3.5 C 10.998 3.5 10.998 3.5 10.998 3.5 C 10.998 3.5 10.998 3.5 10.998 3.5 C 10.999 3.5 10.999 3.5 10.999 3.5 C 10.999 3.5 10.999 3.5 10.999 3.5 C 10.999 3.5 10.999 3.5 10.999 3.5 C 10.999 3.5 10.999 3.5 10.999 3.5 C 10.999 3.5 11 3.5 11 3.5 C 11 3.5 11 3.5 11 3.5 C 11 3.5 11 3.5 11 3.5 C 11 3.5 11 3.5 11 3.5 C 11 3.5 11 3.5 11 3.5 C 11 3.5 11 3.5 11 3 C 11 2.5 11 2.5 11 2.5 C 11 2.5 11 2.5 11 2.5 C 11 2.5 11 2.5 11 2.5 C 11 2.5 11 2.5 11 2.5 C 11 2.5 11 2.5 11 2.5 C 11 2.5 10.999 2.5 10.999 2.5 C 10.999 2.5 10.999 2.5 10.999 2.5 C 10.999 2.5 10.999 2.5 10.999 2.5 C 10.999 2.5 10.999 2.5 10.999 2.5 C 10.999 2.5 10.999 2.5 10.998 2.5 C 10.998 2.5 10.998 2.5 10.998 2.5 C 10.998 2.5 10.998 2.5 10.998 2.5 C 10.998 2.5 10.997 2.5 10.997 2.5 C 10.997 2.5 10.997 2.5 10.997 2.5 C 10.997 2.5 10.997 2.5 10.996 2.5 C 10.996 2.5 10.996 2.5 10.996 2.5 C 10.996 2.5 10.996 2.5 10.995 2.5 C 10.995 2.5 10.995 2.5 10.995 2.5 C 10.995 2.5 10.995 2.5 10.994 2.5 C 10.994 2.5 10.994 2.5 10.994 2.5 C 10.994 2.5 10.993 2.5 10.993 2.5 C 10.993 2.5 10.993 2.5 10.992 2.5 C 10.992 2.5 10.992 2.5 10.992 2.5 C 10.992 2.5 10.991 2.5 10.991 2.5 C 10.991 2.5 10.991 2.5 10.99 2.5 C 10.99 2.5 10.99 2.5 10.99 2.5 C 10.989 2.5 10.989 2.5 10.989 2.5 C 10.988 2.5 10.988 2.5 10.988 2.5 C 10.988 2.5 10.987 2.5 10.987 2.5 C 10.987 2.5 10.986 2.5 10.986 2.5 C 10.986 2.5 10.986 2.5 10.985 2.5 C 10.985 2.5 10.985 2.5 10.984 2.5 C 10.984 2.5 10.984 2.5 10.983 2.5 C 10.983 2.5 10.983 2.5 10.982 2.5 C 10.982 2.5 10.982 2.5 10.981 2.5 C 10.981 2.5 10.981 2.5 10.98 2.5 C 10.98 2.5 10.979 2.5 10.979 2.5 C 10.979 2.5 10.978 2.5 10.978 2.5 C 10.978 2.5 10.977 2.5 10.977 2.5 C 10.976 2.5 10.976 2.5 10.976 2.5 C 10.975 2.5 10.975 2.5 10.974 2.5 C 10.974 2.5 10.974 2.5 10.973 2.5 C 10.973 2.5 10.972 2.5 10.972 2.5 C 10.971 2.5 10.971 2.5 10.971 2.5 C 10.97 2.5 10.97 2.5 10.969 2.5 C 10.969 2.5 10.968 2.5 10.968 2.5 C 10.967 2.5 10.967 2.5 10.967 2.5 C 10.966 2.5 10.966 2.5 10.965 2.5 C 10.965 2.5 10.964 2.5 10.964 2.5 C 10.963 2.5 10.963 2.5 10.962 2.5 C 10.962 2.5 10.961 2.5 10.961 2.5 C 10.96 2.5 10.96 2.5 10.959 2.5 C 10.959 2.5 10.958 2.5 10.958 2.5 C 10.957 2.5 10.957 2.5 10.956 2.5 C 10.956 2.5 10.955 2.5 10.955 2.5 C 10.954 2.5 10.954 2.5 10.953 2.5 C 10.952 2.5 10.952 2.5 10.951 2.5 C 10.951 2.5 10.95 2.5 10.95 2.5 C 10.949 2.5 10.949 2.5 10.948 2.5 C 10.947 2.5 10.947 2.5 10.946 2.5 C 10.946 2.5 10.945 2.5 10.945 2.5 C 10.944 2.5 10.943 2.5 10.943 2.5 C 10.942 2.5 10.942 2.5 10.941 2.5 C 10.94 2.5 10.94 2.5 10.939 2.5 C 10.939 2.5 10.938 2.5 10.937 2.5 C 10.937 2.5 10.936 2.5 10.935 2.5 C 10.935 2.5 10.934 2.5 10.933 2.5 C 10.933 2.5 10.932 2.5 10.932 2.5 C 10.931 2.5 10.93 2.5 10.93 2.5 C 10.929 2.5 10.928 2.5 10.928 2.5 C 10.927 2.5 10.926 2.5 10.926 2.5 C 10.925 2.5 10.924 2.5 10.924 2.5 C 10.923 2.5 10.922 2.5 10.922 2.5 C 10.921 2.5 10.92 2.5 10.919 2.5 C 10.919 2.5 10.918 2.5 10.917 2.5 C 10.917 2.5 10.916 2.5 10.915 2.5 C 10.915 2.5 10.914 2.5 10.913 2.5 C 10.912 2.5 10.912 2.5 10.911 2.5 C 10.91 2.5 10.909 2.5 10.909 2.5 C 10.908 2.5 10.907 2.5 10.907 2.5 C 10.906 2.5 10.905 2.5 10.904 2.5 C 10.904 2.5 10.903 2.5 10.902 2.5 C 10.901 2.5 10.9 2.5 10.9 2.5 C 10.899 2.5 10.898 2.5 10.897 2.5 C 10.897 2.5 10.896 2.5 10.895 2.5 C 10.894 2.5 10.893 2.5 10.893 2.5 C 10.892 2.5 10.891 2.5 10.89 2.5 C 10.89 2.5 10.889 2.5 10.888 2.5 C 10.887 2.5 10.886 2.5 10.885 2.5 C 10.885 2.5 10.884 2.5 10.883 2.5 C 10.882 2.5 10.881 2.5 10.881 2.5 C 10.88 2.5 10.879 2.5 10.878 2.5 C 10.877 2.5 10.876 2.5 10.875 2.5 C 10.875 2.5 10.874 2.5 10.873 2.5 C 10.872 2.5 10.871 2.5 10.87 2.5 C 10.87 2.5 10.869 2.5 10.868 2.5 C 10.867 2.5 10.866 2.5 10.865 2.5 C 10.864 2.5 10.863 2.5 10.863 2.5 C 10.862 2.5 10.861 2.5 10.86 2.5 C 10.859 2.5 10.858 2.5 10.857 2.5 C 10.856 2.5 10.855 2.5 10.854 2.5 C 10.854 2.5 10.853 2.5 10.852 2.5 C 10.851 2.5 10.85 2.5 10.849 2.5 C 10.848 2.5 10.847 2.5 10.846 2.5 C 10.845 2.5 10.844 2.5 10.843 2.5 C 10.842 2.5 10.842 2.5 10.841 2.5 C 10.84 2.5 10.839 2.5 10.838 2.5 C 10.837 2.5 10.836 2.5 10.835 2.5 C 10.834 2.5 10.833 2.5 10.832 2.5 C 10.831 2.5 10.83 2.5 10.829 2.5 C 10.828 2.5 10.827 2.5 10.826 2.5 C 10.825 2.5 10.824 2.5 10.823 2.5 C 10.822 2.5 10.821 2.5 10.82 2.5 C 10.819 2.5 10.818 2.5 10.817 2.5 C 10.816 2.5 10.815 2.5 10.814 2.5 C 10.813 2.5 10.812 2.5 10.811 2.5 C 10.81 2.5 10.809 2.5 10.808 2.5 C 10.807 2.5 10.806 2.5 10.805 2.5 C 10.804 2.5 10.803 2.5 10.802 2.5 C 10.801 2.5 10.8 2.5 10.799 2.5 C 10.798 2.5 10.797 2.5 10.796 2.5 C 10.795 2.5 10.794 2.5 10.793 2.5 C 10.792 2.5 10.791 2.5 10.79 2.5 C 10.788 2.5 10.787 2.5 10.786 2.5 C 10.785 2.5 10.784 2.5 10.783 2.5 C 10.782 2.5 10.781 2.5 10.78 2.5 C 10.779 2.5 10.778 2.5 10.777 2.5 C 10.776 2.5 10.774 2.5 10.773 2.5 C 10.772 2.5 10.771 2.5 10.77 2.5 C 10.769 2.5 10.768 2.5 10.767 2.5 C 10.766 2.5 10.765 2.5 10.763 2.5 C 10.762 2.5 10.761 2.5 10.76 2.5 C 10.759 2.5 10.758 2.5 10.757 2.5 C 10.756 2.5 10.755 2.5 10.753 2.5 C 10.752 2.5 10.751 2.5 10.75 2.5 C 10.749 2.5 10.748 2.5 10.747 2.5 C 10.745 2.5 10.744 2.5 10.743 2.5 C 10.742 2.5 10.741 2.5 10.74 2.5 C 10.739 2.5 10.737 2.5 10.736 2.5 C 10.735 2.5 10.734 2.5 10.733 2.5 C 10.732 2.5 10.73 2.5 10.729 2.5 C 10.728 2.5 10.727 2.5 10.726 2.5 C 10.724 2.5 10.723 2.5 10.722 2.5 C 10.721 2.5 10.72 2.5 10.719 2.5 C 10.717 2.5 10.716 2.5 10.715 2.5 C 10.714 2.5 10.713 2.5 10.711 2.5 C 10.71 2.5 10.709 2.5 10.708 2.5 C 10.707 2.5 10.705 2.5 10.704 2.5 C 10.703 2.5 10.702 2.5 10.7 2.5 C 10.699 2.5 10.698 2.5 10.697 2.5 C 10.696 2.5 10.694 2.5 10.693 2.5 C 10.692 2.5 10.691 2.5 10.689 2.5 C 10.688 2.5 10.687 2.5 10.686 2.5 C 10.684 2.5 10.683 2.5 10.682 2.5 C 10.681 2.5 10.679 2.5 10.678 2.5 C 10.677 2.5 10.676 2.5 10.674 2.5 C 10.673 2.5 10.672 2.5 10.671 2.5 C 10.669 2.5 10.668 2.5 10.667 2.5 C 10.666 2.5 10.664 2.5 10.663 2.5 C 10.662 2.5 10.66 2.5 10.659 2.5 C 10.658 2.5 10.657 2.5 10.655 2.5 C 10.654 2.5 10.653 2.5 10.651 2.5 C 10.65 2.5 10.649 2.5 10.648 2.5 C 10.646 2.5 10.645 2.5 10.644 2.5 C 10.642 2.5 10.641 2.5 10.64 2.5 C 10.638 2.5 10.637 2.5 10.636 2.5 C 10.634 2.5 10.633 2.5 10.632 2.5 C 10.631 2.5 10.629 2.5 10.628 2.5 C 10.627 2.5 10.625 2.5 10.624 2.5 C 10.623 2.5 10.621 2.5 10.62 2.5 C 10.619 2.5 10.617 2.5 10.616 2.5 C 10.615 2.5 10.613 2.5 10.612 2.5 C 10.611 2.5 10.609 2.5 10.608 2.5 C 10.606 2.5 10.605 2.5 10.604 2.5 C 10.602 2.5 10.601 2.5 10.6 2.5 C 10.598 2.5 10.597 2.5 10.596 2.5 C 10.594 2.5 10.593 2.5 10.591 2.5 C 10.59 2.5 10.589 2.5 10.587 2.5 C 10.586 2.5 10.585 2.5 10.583 2.5 C 10.582 2.5 10.58 2.5 10.579 2.5 C 10.578 2.5 10.576 2.5 10.575 2.5 C 10.574 2.5 10.572 2.5 10.571 2.5 C 10.569 2.5 10.568 2.5 10.567 2.5 C 10.565 2.5 10.564 2.5 10.562 2.5 C 10.561 2.5 10.56 2.5 10.558 2.5 C 10.557 2.5 10.555 2.5 10.554 2.5 C 10.553 2.5 10.551 2.5 10.55 2.5 C 10.548 2.5 10.547 2.5 10.545 2.5 C 10.544 2.5 10.543 2.5 10.541 2.5 C 10.54 2.5 10.538 2.5 10.537 2.5 C 10.535 2.5 10.534 2.5 10.533 2.5 C 10.531 2.5 10.53 2.5 10.528 2.5 C 10.527 2.5 10.525 2.5 10.524 2.5 C 10.522 2.5 10.521 2.5 10.52 2.5 C 10.518 2.5 10.517 2.5 10.515 2.5 C 10.514 2.5 10.512 2.5 10.511 2.5 C 10.509 2.5 10.508 2.5 10.506 2.5 C 10.505 2.5 10.504 2.5 10.502 2.5 C 10.501 2.5 10.499 2.5 10.498 2.5 C 10.496 2.5 10.495 2.5 10.493 2.5 C 10.492 2.5 10.49 2.5 10.489 2.5 C 10.487 2.5 10.486 2.5 10.484 2.5 C 10.483 2.5 10.481 2.5 10.48 2.5 C 10.478 2.5 10.477 2.5 10.476 2.5 C 10.474 2.5 10.473 2.5 10.471 2.5 C 10.47 2.5 10.468 2.5 10.467 2.5 C 10.465 2.5 10.464 2.5 10.462 2.5 C 10.461 2.5 10.459 2.5 10.458 2.5 C 10.456 2.5 10.455 2.5 10.453 2.5 C 10.451 2.5 10.45 2.5 10.448 2.5 C 10.447 2.5 10.445 2.5 10.444 2.5 C 10.442 2.5 10.441 2.5 10.439 2.5 C 10.438 2.5 10.436 2.5 10.435 2.5 C 10.433 2.5 10.432 2.5 10.43 2.5 C 10.429 2.5 10.427 2.5 10.426 2.5 C 10.424 2.5 10.423 2.5 10.421 2.5 C 10.419 2.5 10.418 2.5 10.416 2.5 C 10.415 2.5 10.413 2.5 10.412 2.5 C 10.41 2.5 10.409 2.5 10.407 2.5 C 10.406 2.5 10.404 2.5 10.403 2.5 C 10.401 2.5 10.399 2.5 10.398 2.5 C 10.396 2.5 10.395 2.5 10.393 2.5 C 10.392 2.5 10.39 2.5 10.389 2.5 C 10.387 2.5 10.385 2.5 10.384 2.5 C 10.382 2.5 10.381 2.5 10.379 2.5 C 10.378 2.5 10.376 2.5 10.374 2.5 C 10.373 2.5 10.371 2.5 10.37 2.5 C 10.368 2.5 10.367 2.5 10.365 2.5 C 10.363 2.5 10.362 2.5 10.36 2.5 C 10.359 2.5 10.357 2.5 10.355 2.5 C 10.354 2.5 10.352 2.5 10.351 2.5 C 10.349 2.5 10.348 2.5 10.346 2.5 C 10.344 2.5 10.343 2.5 10.341 2.5 C 10.34 2.5 10.338 2.5 10.336 2.5 C 10.335 2.5 10.333 2.5 10.332 2.5 C 10.33 2.5 10.328 2.5 10.327 2.5 C 10.325 2.5 10.324 2.5 10.322 2.5 C 10.32 2.5 10.319 2.5 10.317 2.5 C 10.316 2.5 10.314 2.5 10.312 2.5 C 10.311 2.5 10.309 2.5 10.308 2.5 C 10.306 2.5 10.304 2.5 10.303 2.5 C 10.301 2.5 10.299 2.5 10.298 2.5 C 10.296 2.5 10.295 2.5 10.293 2.5 C 10.291 2.5 10.29 2.5 10.288 2.5 C 10.286 2.5 10.285 2.5 10.283 2.5 C 10.282 2.5 10.28 2.5 10.278 2.5 C 10.277 2.5 10.275 2.5 10.273 2.5 C 10.272 2.5 10.27 2.5 10.269 2.5 C 10.267 2.5 10.265 2.5 10.264 2.5 C 10.262 2.5 10.26 2.5 10.259 2.5 C 10.257 2.5 10.255 2.5 10.254 2.5 C 10.252 2.5 10.251 2.5 10.249 2.5 C 10.247 2.5 10.246 2.5 10.244 2.5 C 10.242 2.5 10.241 2.5 10.239 2.5 C 10.237 2.5 10.236 2.5 10.234 2.5 C 10.232 2.5 10.231 2.5 10.229 2.5 C 10.227 2.5 10.226 2.5 10.224 2.5 C 10.222 2.5 10.221 2.5 10.219 2.5 C 10.217 2.5 10.216 2.5 10.214 2.5 C 10.212 2.5 10.211 2.5 10.209 2.5 C 10.207 2.5 10.206 2.5 10.204 2.5 C 10.202 2.5 10.201 2.5 10.199 2.5 C 10.197 2.5 10.196 2.5 10.194 2.5 C 10.192 2.5 10.191 2.5 10.189 2.5 C 10.187 2.5 10.186 2.5 10.184 2.5 C 10.182 2.5 10.181 2.5 10.179 2.5 C 10.177 2.5 10.176 2.5 10.174 2.5 C 10.172 2.5 10.171 2.5 10.169 2.5 C 10.167 2.5 10.166 2.5 10.164 2.5 C 10.162 2.5 10.161 2.5 10.159 2.5 C 10.157 2.5 10.156 2.5 10.154 2.5 C 10.152 2.5 10.15 2.5 10.149 2.5 C 10.147 2.5 10.145 2.5 10.144 2.5 C 10.142 2.5 10.14 2.5 10.139 2.5 C 10.137 2.5 10.135 2.5 10.134 2.5 C 10.132 2.5 10.13 2.5 10.129 2.5 C 10.127 2.5 10.125 2.5 10.123 2.5 C 10.122 2.5 10.12 2.5 10.118 2.5 C 10.117 2.5 10.115 2.5 10.113 2.5 C 10.112 2.5 10.11 2.5 10.108 2.5 C 10.106 2.5 10.105 2.5 10.103 2.5 C 10.101 2.5 10.1 2.5 10.098 2.5 C 10.096 2.5 10.094 2.5 10.093 2.5 C 10.091 2.5 10.089 2.5 10.088 2.5 C 10.086 2.5 10.084 2.5 10.083 2.5 C 10.081 2.5 10.079 2.5 10.077 2.5 C 10.076 2.5 10.074 2.5 10.072 2.5 C 10.071 2.5 10.069 2.5 10.067 2.5 C 10.065 2.5 10.064 2.5 10.062 2.5 C 10.06 2.5 10.059 2.5 10.057 2.5 C 10.055 2.5 10.053 2.5 10.052 2.5 C 10.05 2.5 10.048 2.5 10.047 2.5 C 10.045 2.5 10.043 2.5 10.041 2.5 C 10.04 2.5 10.038 2.5 10.036 2.5 C 10.035 2.5 10.033 2.5 10.031 2.5 C 10.029 2.5 10.028 2.5 10.026 2.5 C 10.024 2.5 10.022 2.5 10.021 2.5 C 10.019 2.5 10.017 2.5 10.016 2.5 C 10.014 2.5 10.012 2.5 10.01 2.5 C 10.009 2.5 10.007 2.5 10.005 2.5 C 10.004 2.5 10.002 2.5 10 2.5 C 9.998 2.5 9.997 2.5 9.995 2.5 C 9.993 2.5 9.991 2.5 9.99 2.5 C 9.988 2.5 9.986 2.5 9.984 2.5 C 9.983 2.5 9.981 2.5 9.979 2.5 C 9.978 2.5 9.976 2.5 9.974 2.5 C 9.972 2.5 9.971 2.5 9.969 2.5 C 9.967 2.5 9.965 2.5 9.964 2.5 C 9.962 2.5 9.96 2.5 9.959 2.5 C 9.957 2.5 9.955 2.5 9.953 2.5 C 9.952 2.5 9.95 2.5 9.948 2.5 C 9.946 2.5 9.945 2.5 9.943 2.5 C 9.941 2.5 9.939 2.5 9.938 2.5 C 9.936 2.5 9.934 2.5 9.933 2.5 C 9.931 2.5 9.929 2.5 9.927 2.5 C 9.926 2.5 9.924 2.5 9.922 2.5 C 9.92 2.5 9.919 2.5 9.917 2.5 C 9.915 2.5 9.913 2.5 9.912 2.5 C 9.91 2.5 9.908 2.5 9.906 2.5 C 9.905 2.5 9.903 2.5 9.901 2.5 C 9.9 2.5 9.898 2.5 9.896 2.5 C 9.894 2.5 9.893 2.5 9.891 2.5 C 9.889 2.5 9.887 2.5 9.886 2.5 C 9.884 2.5 9.882 2.5 9.88 2.5 C 9.879 2.5 9.877 2.5 9.875 2.5 C 9.873 2.5 9.872 2.5 9.87 2.5 C 9.868 2.5 9.866 2.5 9.865 2.5 C 9.863 2.5 9.861 2.5 9.86 2.5 C 9.858 2.5 9.856 2.5 9.854 2.5 C 9.853 2.5 9.851 2.5 9.849 2.5 C 9.847 2.5 9.846 2.5 9.844 2.5 C 9.842 2.5 9.84 2.5 9.839 2.5 C 9.837 2.5 9.835 2.5 9.833 2.5 C 9.832 2.5 9.83 2.5 9.828 2.5 C 9.826 2.5 9.825 2.5 9.823 2.5 C 9.821 2.5 9.82 2.5 9.818 2.5 C 9.816 2.5 9.814 2.5 9.813 2.5 C 9.811 2.5 9.809 2.5 9.807 2.5 C 9.806 2.5 9.804 2.5 9.802 2.5 C 9.8 2.5 9.799 2.5 9.797 2.5 C 9.795 2.5 9.793 2.5 9.792 2.5 C 9.79 2.5 9.788 2.5 9.787 2.5 C 9.785 2.5 9.783 2.5 9.781 2.5 C 9.78 2.5 9.778 2.5 9.776 2.5 C 9.774 2.5 9.773 2.5 9.771 2.5 C 9.769 2.5 9.767 2.5 9.766 2.5 C 9.764 2.5 9.762 2.5 9.761 2.5 C 9.759 2.5 9.757 2.5 9.755 2.5 C 9.754 2.5 9.752 2.5 9.75 2.5 C 9.748 2.5 9.747 2.5 9.745 2.5 C 9.743 2.5 9.741 2.5 9.74 2.5 C 9.738 2.5 9.736 2.5 9.735 2.5 C 9.733 2.5 9.731 2.5 9.729 2.5 C 9.728 2.5 9.726 2.5 9.724 2.5 C 9.722 2.5 9.721 2.5 9.719 2.5 C 9.717 2.5 9.716 2.5 9.714 2.5 C 9.712 2.5 9.71 2.5 9.709 2.5 C 9.707 2.5 9.705 2.5 9.703 2.5 C 9.702 2.5 9.7 2.5 9.698 2.5 C 9.697 2.5 9.695 2.5 9.693 2.5 C 9.691 2.5 9.69 2.5 9.688 2.5 C 9.686 2.5 9.685 2.5 9.683 2.5 C 9.681 2.5 9.679 2.5 9.678 2.5 C 9.676 2.5 9.674 2.5 9.672 2.5 C 9.671 2.5 9.669 2.5 9.667 2.5 C 9.666 2.5 9.664 2.5 9.662 2.5 C 9.66 2.5 9.659 2.5 9.657 2.5 C 9.655 2.5 9.654 2.5 9.652 2.5 C 9.65 2.5 9.648 2.5 9.647 2.5 C 9.645 2.5 9.643 2.5 9.642 2.5 C 9.64 2.5 9.638 2.5 9.636 2.5 C 9.635 2.5 9.633 2.5 9.631 2.5 C 9.63 2.5 9.628 2.5 9.626 2.5 C 9.625 2.5 9.623 2.5 9.621 2.5 C 9.619 2.5 9.618 2.5 9.616 2.5 C 9.614 2.5 9.613 2.5 9.611 2.5 C 9.609 2.5 9.607 2.5 9.606 2.5 C 9.604 2.5 9.602 2.5 9.601 2.5 C 9.599 2.5 9.597 2.5 9.596 2.5 C 9.594 2.5 9.592 2.5 9.591 2.5 C 9.589 2.5 9.587 2.5 9.585 2.5 C 9.584 2.5 9.582 2.5 9.58 2.5 C 9.579 2.5 9.577 2.5 9.575 2.5 C 9.574 2.5 9.572 2.5 9.57 2.5 C 9.569 2.5 9.567 2.5 9.565 2.5 C 9.563 2.5 9.562 2.5 9.56 2.5 C 9.558 2.5 9.557 2.5 9.555 2.5 C 9.553 2.5 9.552 2.5 9.55 2.5 C 9.548 2.5 9.547 2.5 9.545 2.5 C 9.543 2.5 9.542 2.5 9.54 2.5 C 9.538 2.5 9.537 2.5 9.535 2.5 C 9.533 2.5 9.532 2.5 9.53 2.5 C 9.528 2.5 9.527 2.5 9.525 2.5 C 9.523 2.5 9.521 2.5 9.52 2.5 C 9.518 2.5 9.516 2.5 9.515 2.5 C 9.513 2.5 9.511 2.5 9.51 2.5 C 9.508 2.5 9.507 2.5 9.505 2.5 C 9.503 2.5 9.502 2.5 9.5 2.5 C 9.498 2.5 9.497 2.5 9.495 2.5 C 9.493 2.5 9.492 2.5 9.49 2.5 C 9.488 2.5 9.487 2.5 9.485 2.5 C 9.483 2.5 9.482 2.5 9.48 2.5 C 9.478 2.5 9.477 2.5 9.475 2.5 C 9.473 2.5 9.472 2.5 9.47 2.5 C 9.468 2.5 9.467 2.5 9.465 2.5 C 9.464 2.5 9.462 2.5 9.46 2.5 C 9.459 2.5 9.457 2.5 9.455 2.5 C 9.454 2.5 9.452 2.5 9.45 2.5 C 9.449 2.5 9.447 2.5 9.446 2.5 C 9.444 2.5 9.442 2.5 9.441 2.5 C 9.439 2.5 9.437 2.5 9.436 2.5 C 9.434 2.5 9.432 2.5 9.431 2.5 C 9.429 2.5 9.428 2.5 9.426 2.5 C 9.424 2.5 9.423 2.5 9.421 2.5 C 9.419 2.5 9.418 2.5 9.416 2.5 C 9.415 2.5 9.413 2.5 9.411 2.5 C 9.41 2.5 9.408 2.5 9.407 2.5 C 9.405 2.5 9.403 2.5 9.402 2.5 C 9.4 2.5 9.399 2.5 9.397 2.5 C 9.395 2.5 9.394 2.5 9.392 2.5 C 9.39 2.5 9.389 2.5 9.387 2.5 C 9.386 2.5 9.384 2.5 9.382 2.5 C 9.381 2.5 9.379 2.5 9.378 2.5 C 9.376 2.5 9.375 2.5 9.373 2.5 C 9.371 2.5 9.37 2.5 9.368 2.5 C 9.367 2.5 9.365 2.5 9.363 2.5 C 9.362 2.5 9.36 2.5 9.359 2.5 C 9.357 2.5 9.356 2.5 9.354 2.5 C 9.352 2.5 9.351 2.5 9.349 2.5 C 9.348 2.5 9.346 2.5 9.344 2.5 C 9.343 2.5 9.341 2.5 9.34 2.5 C 9.338 2.5 9.337 2.5 9.335 2.5 C 9.334 2.5 9.332 2.5 9.33 2.5 C 9.329 2.5 9.327 2.5 9.326 2.5 C 9.324 2.5 9.323 2.5 9.321 2.5 C 9.319 2.5 9.318 2.5 9.316 2.5 C 9.315 2.5 9.313 2.5 9.312 2.5 C 9.31 2.5 9.309 2.5 9.307 2.5 C 9.306 2.5 9.304 2.5 9.302 2.5 C 9.301 2.5 9.299 2.5 9.298 2.5 C 9.296 2.5 9.295 2.5 9.293 2.5 C 9.292 2.5 9.29 2.5 9.289 2.5 C 9.287 2.5 9.286 2.5 9.284 2.5 C 9.283 2.5 9.281 2.5 9.279 2.5 C 9.278 2.5 9.276 2.5 9.275 2.5 C 9.273 2.5 9.272 2.5 9.27 2.5 C 9.269 2.5 9.267 2.5 9.266 2.5 C 9.264 2.5 9.263 2.5 9.261 2.5 C 9.26 2.5 9.258 2.5 9.257 2.5 C 9.255 2.5 9.254 2.5 9.252 2.5 C 9.251 2.5 9.249 2.5 9.248 2.5 C 9.246 2.5 9.245 2.5 9.243 2.5 C 9.242 2.5 9.24 2.5 9.239 2.5 C 9.237 2.5 9.236 2.5 9.234 2.5 C 9.233 2.5 9.231 2.5 9.23 2.5 C 9.228 2.5 9.227 2.5 9.226 2.5 C 9.224 2.5 9.223 2.5 9.221 2.5 C 9.22 2.5 9.218 2.5 9.217 2.5 C 9.215 2.5 9.214 2.5 9.212 2.5 C 9.211 2.5 9.209 2.5 9.208 2.5 C 9.206 2.5 9.205 2.5 9.204 2.5 C 9.202 2.5 9.201 2.5 9.199 2.5 C 9.198 2.5 9.196 2.5 9.195 2.5 C 9.193 2.5 9.192 2.5 9.191 2.5 C 9.189 2.5 9.188 2.5 9.186 2.5 C 9.185 2.5 9.183 2.5 9.182 2.5 C 9.18 2.5 9.179 2.5 9.178 2.5 C 9.176 2.5 9.175 2.5 9.173 2.5 C 9.172 2.5 9.171 2.5 9.169 2.5 C 9.168 2.5 9.166 2.5 9.165 2.5 C 9.163 2.5 9.162 2.5 9.161 2.5 C 9.159 2.5 9.158 2.5 9.156 2.5 C 9.155 2.5 9.154 2.5 9.152 2.5 C 9.151 2.5 9.149 2.5 9.148 2.5 C 9.147 2.5 9.145 2.5 9.144 2.5 C 9.142 2.5 9.141 2.5 9.14 2.5 C 9.138 2.5 9.137 2.5 9.135 2.5 C 9.134 2.5 9.133 2.5 9.131 2.5 C 9.13 2.5 9.129 2.5 9.127 2.5 C 9.126 2.5 9.124 2.5 9.123 2.5 C 9.122 2.5 9.12 2.5 9.119 2.5 C 9.118 2.5 9.116 2.5 9.115 2.5 C 9.114 2.5 9.112 2.5 9.111 2.5 C 9.11 2.5 9.108 2.5 9.107 2.5 C 9.105 2.5 9.104 2.5 9.103 2.5 C 9.101 2.5 9.1 2.5 9.099 2.5 C 9.097 2.5 9.096 2.5 9.095 2.5 C 9.093 2.5 9.092 2.5 9.091 2.5 C 9.089 2.5 9.088 2.5 9.087 2.5 C 9.086 2.5 9.084 2.5 9.083 2.5 C 9.082 2.5 9.08 2.5 9.079 2.5 C 9.078 2.5 9.076 2.5 9.075 2.5 C 9.074 2.5 9.072 2.5 9.071 2.5 C 9.07 2.5 9.069 2.5 9.067 2.5 C 9.066 2.5 9.065 2.5 9.063 2.5 C 9.062 2.5 9.061 2.5 9.059 2.5 C 9.058 2.5 9.057 2.5 9.056 2.5 C 9.054 2.5 9.053 2.5 9.052 2.5 C 9.051 2.5 9.049 2.5 9.048 2.5 C 9.047 2.5 9.045 2.5 9.044 2.5 C 9.043 2.5 9.042 2.5 9.04 2.5 C 9.039 2.5 9.038 2.5 9.037 2.5 C 9.035 2.5 9.034 2.5 9.033 2.5 C 9.032 2.5 9.03 2.5 9.029 2.5 C 9.028 2.5 9.027 2.5 9.025 2.5 C 9.024 2.5 9.023 2.5 9.022 2.5 C 9.021 2.5 9.019 2.5 9.018 2.5 C 9.017 2.5 9.016 2.5 9.014 2.5 C 9.013 2.5 9.012 2.5 9.011 2.5 C 9.01 2.5 9.008 2.5 9.007 2.5 C 9.006 2.5 9.005 2.5 9.004 2.5 C 9.002 2.5 9.001 2.5 9 2.5 L 9 3.5 Z M 10.5 3 L 10.5 5 L 11.5 5 L 11.5 3 L 10.5 3 Z M 10.5 5 C 10.5 5.828 9.828 6.5 9 6.5 L 9 7.5 C 10.381 7.5 11.5 6.381 11.5 5 L 10.5 5 Z M 2 4.5 C 2.616 4.5 3.395 4.819 4.073 5.189 C 4.401 5.368 4.685 5.547 4.887 5.682 C 4.987 5.749 5.067 5.804 5.121 5.843 C 5.148 5.862 5.169 5.877 5.182 5.887 C 5.189 5.892 5.193 5.895 5.197 5.897 C 5.198 5.899 5.199 5.899 5.2 5.9 C 5.2 5.9 5.2 5.9 5.2 5.9 C 5.2 5.9 5.2 5.9 5.2 5.9 C 5.2 5.9 5.2 5.9 5.2 5.9 C 5.2 5.9 5.2 5.9 5.5 5.5 C 5.8 5.1 5.8 5.1 5.8 5.1 C 5.8 5.1 5.8 5.1 5.8 5.1 C 5.799 5.1 5.799 5.099 5.799 5.099 C 5.799 5.099 5.798 5.099 5.797 5.098 C 5.796 5.097 5.794 5.096 5.792 5.094 C 5.787 5.091 5.781 5.086 5.772 5.08 C 5.756 5.067 5.732 5.05 5.701 5.028 C 5.64 4.985 5.552 4.923 5.441 4.85 C 5.222 4.703 4.912 4.507 4.552 4.311 C 3.855 3.931 2.884 3.5 2 3.5 L 2 4.5 Z M 5.5 5.5 C 5.11 5.188 5.11 5.188 5.11 5.188 C 5.11 5.188 5.11 5.187 5.11 5.187 C 5.11 5.187 5.11 5.187 5.11 5.188 C 5.11 5.188 5.109 5.188 5.109 5.188 C 5.108 5.189 5.107 5.191 5.106 5.193 C 5.102 5.196 5.097 5.203 5.091 5.211 C 5.077 5.228 5.056 5.253 5.029 5.285 C 4.974 5.35 4.894 5.443 4.792 5.556 C 4.589 5.782 4.304 6.082 3.976 6.38 C 3.646 6.68 3.284 6.968 2.926 7.178 C 2.56 7.394 2.246 7.5 2 7.5 L 2 8.5 C 2.504 8.5 3.002 8.294 3.433 8.04 C 3.872 7.782 4.291 7.445 4.649 7.12 C 5.008 6.793 5.317 6.468 5.536 6.225 C 5.645 6.103 5.733 6.002 5.793 5.93 C 5.824 5.894 5.847 5.865 5.864 5.845 C 5.872 5.835 5.878 5.827 5.883 5.822 C 5.885 5.819 5.887 5.817 5.888 5.815 C 5.889 5.814 5.889 5.814 5.89 5.813 C 5.89 5.813 5.89 5.813 5.89 5.813 C 5.89 5.813 5.89 5.813 5.89 5.813 C 5.89 5.812 5.89 5.812 5.5 5.5 Z M 2 7.5 C 1.999 7.5 1.998 7.5 1.996 7.5 C 1.995 7.5 1.994 7.5 1.993 7.5 C 1.992 7.5 1.99 7.5 1.989 7.5 C 1.988 7.5 1.987 7.5 1.986 7.5 C 1.984 7.5 1.983 7.5 1.982 7.5 C 1.981 7.5 1.979 7.5 1.978 7.5 C 1.977 7.5 1.976 7.5 1.975 7.5 C 1.973 7.5 1.972 7.5 1.971 7.5 C 1.97 7.5 1.968 7.5 1.967 7.5 C 1.966 7.5 1.965 7.5 1.963 7.5 C 1.962 7.5 1.961 7.5 1.96 7.5 C 1.958 7.5 1.957 7.5 1.956 7.5 C 1.955 7.5 1.953 7.5 1.952 7.5 C 1.951 7.5 1.949 7.5 1.948 7.5 C 1.947 7.5 1.946 7.5 1.944 7.5 C 1.943 7.5 1.942 7.5 1.941 7.5 C 1.939 7.5 1.938 7.5 1.937 7.5 C 1.935 7.5 1.934 7.5 1.933 7.5 C 1.931 7.5 1.93 7.5 1.929 7.5 C 1.928 7.5 1.926 7.5 1.925 7.5 C 1.924 7.5 1.922 7.5 1.921 7.5 C 1.92 7.5 1.918 7.5 1.917 7.5 C 1.916 7.5 1.914 7.5 1.913 7.5 C 1.912 7.5 1.911 7.5 1.909 7.5 C 1.908 7.5 1.907 7.5 1.905 7.5 C 1.904 7.5 1.903 7.5 1.901 7.5 C 1.9 7.5 1.899 7.5 1.897 7.5 C 1.896 7.5 1.895 7.5 1.893 7.5 C 1.892 7.5 1.89 7.5 1.889 7.5 C 1.888 7.5 1.886 7.5 1.885 7.5 C 1.884 7.5 1.882 7.5 1.881 7.5 C 1.88 7.5 1.878 7.5 1.877 7.5 C 1.876 7.5 1.874 7.5 1.873 7.5 C 1.871 7.5 1.87 7.5 1.869 7.5 C 1.867 7.5 1.866 7.5 1.865 7.5 C 1.863 7.5 1.862 7.5 1.86 7.5 C 1.859 7.5 1.858 7.5 1.856 7.5 C 1.855 7.5 1.853 7.5 1.852 7.5 C 1.851 7.5 1.849 7.5 1.848 7.5 C 1.846 7.5 1.845 7.5 1.844 7.5 C 1.842 7.5 1.841 7.5 1.839 7.5 C 1.838 7.5 1.837 7.5 1.835 7.5 C 1.834 7.5 1.832 7.5 1.831 7.5 C 1.829 7.5 1.828 7.5 1.827 7.5 C 1.825 7.5 1.824 7.5 1.822 7.5 C 1.821 7.5 1.82 7.5 1.818 7.5 C 1.817 7.5 1.815 7.5 1.814 7.5 C 1.812 7.5 1.811 7.5 1.809 7.5 C 1.808 7.5 1.807 7.5 1.805 7.5 C 1.804 7.5 1.802 7.5 1.801 7.5 C 1.799 7.5 1.798 7.5 1.796 7.5 C 1.795 7.5 1.794 7.5 1.792 7.5 C 1.791 7.5 1.789 7.5 1.788 7.5 C 1.786 7.5 1.785 7.5 1.783 7.5 C 1.782 7.5 1.78 7.5 1.779 7.5 C 1.777 7.5 1.776 7.5 1.774 7.5 C 1.773 7.5 1.772 7.5 1.77 7.5 C 1.769 7.5 1.767 7.5 1.766 7.5 C 1.764 7.5 1.763 7.5 1.761 7.5 C 1.76 7.5 1.758 7.5 1.757 7.5 C 1.755 7.5 1.754 7.5 1.752 7.5 C 1.751 7.5 1.749 7.5 1.748 7.5 C 1.746 7.5 1.745 7.5 1.743 7.5 C 1.742 7.5 1.74 7.5 1.739 7.5 C 1.737 7.5 1.736 7.5 1.734 7.5 C 1.733 7.5 1.731 7.5 1.73 7.5 C 1.728 7.5 1.727 7.5 1.725 7.5 C 1.724 7.5 1.722 7.5 1.721 7.5 C 1.719 7.5 1.717 7.5 1.716 7.5 C 1.714 7.5 1.713 7.5 1.711 7.5 C 1.71 7.5 1.708 7.5 1.707 7.5 C 1.705 7.5 1.704 7.5 1.702 7.5 C 1.701 7.5 1.699 7.5 1.698 7.5 C 1.696 7.5 1.694 7.5 1.693 7.5 C 1.691 7.5 1.69 7.5 1.688 7.5 C 1.687 7.5 1.685 7.5 1.684 7.5 C 1.682 7.5 1.681 7.5 1.679 7.5 C 1.677 7.5 1.676 7.5 1.674 7.5 C 1.673 7.5 1.671 7.5 1.67 7.5 C 1.668 7.5 1.666 7.5 1.665 7.5 C 1.663 7.5 1.662 7.5 1.66 7.5 C 1.659 7.5 1.657 7.5 1.656 7.5 C 1.654 7.5 1.652 7.5 1.651 7.5 C 1.649 7.5 1.648 7.5 1.646 7.5 C 1.644 7.5 1.643 7.5 1.641 7.5 C 1.64 7.5 1.638 7.5 1.637 7.5 C 1.635 7.5 1.633 7.5 1.632 7.5 C 1.63 7.5 1.629 7.5 1.627 7.5 C 1.625 7.5 1.624 7.5 1.622 7.5 C 1.621 7.5 1.619 7.5 1.618 7.5 C 1.616 7.5 1.614 7.5 1.613 7.5 C 1.611 7.5 1.61 7.5 1.608 7.5 C 1.606 7.5 1.605 7.5 1.603 7.5 C 1.601 7.5 1.6 7.5 1.598 7.5 C 1.597 7.5 1.595 7.5 1.593 7.5 C 1.592 7.5 1.59 7.5 1.589 7.5 C 1.587 7.5 1.585 7.5 1.584 7.5 C 1.582 7.5 1.581 7.5 1.579 7.5 C 1.577 7.5 1.576 7.5 1.574 7.5 C 1.572 7.5 1.571 7.5 1.569 7.5 C 1.568 7.5 1.566 7.5 1.564 7.5 C 1.563 7.5 1.561 7.5 1.559 7.5 C 1.558 7.5 1.556 7.5 1.554 7.5 C 1.553 7.5 1.551 7.5 1.55 7.5 C 1.548 7.5 1.546 7.5 1.545 7.5 C 1.543 7.5 1.541 7.5 1.54 7.5 C 1.538 7.5 1.536 7.5 1.535 7.5 C 1.533 7.5 1.532 7.5 1.53 7.5 C 1.528 7.5 1.527 7.5 1.525 7.5 C 1.523 7.5 1.522 7.5 1.52 7.5 C 1.518 7.5 1.517 7.5 1.515 7.5 C 1.513 7.5 1.512 7.5 1.51 7.5 C 1.508 7.5 1.507 7.5 1.505 7.5 C 1.503 7.5 1.502 7.5 1.5 7.5 C 1.498 7.5 1.497 7.5 1.495 7.5 C 1.493 7.5 1.492 7.5 1.49 7.5 C 1.489 7.5 1.487 7.5 1.485 7.5 C 1.484 7.5 1.482 7.5 1.48 7.5 C 1.479 7.5 1.477 7.5 1.475 7.5 C 1.473 7.5 1.472 7.5 1.47 7.5 C 1.468 7.5 1.467 7.5 1.465 7.5 C 1.463 7.5 1.462 7.5 1.46 7.5 C 1.458 7.5 1.457 7.5 1.455 7.5 C 1.453 7.5 1.452 7.5 1.45 7.5 C 1.448 7.5 1.447 7.5 1.445 7.5 C 1.443 7.5 1.442 7.5 1.44 7.5 C 1.438 7.5 1.437 7.5 1.435 7.5 C 1.433 7.5 1.431 7.5 1.43 7.5 C 1.428 7.5 1.426 7.5 1.425 7.5 C 1.423 7.5 1.421 7.5 1.42 7.5 C 1.418 7.5 1.416 7.5 1.415 7.5 C 1.413 7.5 1.411 7.5 1.409 7.5 C 1.408 7.5 1.406 7.5 1.404 7.5 C 1.403 7.5 1.401 7.5 1.399 7.5 C 1.398 7.5 1.396 7.5 1.394 7.5 C 1.393 7.5 1.391 7.5 1.389 7.5 C 1.387 7.5 1.386 7.5 1.384 7.5 C 1.382 7.5 1.381 7.5 1.379 7.5 C 1.377 7.5 1.375 7.5 1.374 7.5 C 1.372 7.5 1.37 7.5 1.369 7.5 C 1.367 7.5 1.365 7.5 1.364 7.5 C 1.362 7.5 1.36 7.5 1.358 7.5 C 1.357 7.5 1.355 7.5 1.353 7.5 C 1.352 7.5 1.35 7.5 1.348 7.5 C 1.346 7.5 1.345 7.5 1.343 7.5 C 1.341 7.5 1.34 7.5 1.338 7.5 C 1.336 7.5 1.334 7.5 1.333 7.5 C 1.331 7.5 1.329 7.5 1.328 7.5 C 1.326 7.5 1.324 7.5 1.322 7.5 C 1.321 7.5 1.319 7.5 1.317 7.5 C 1.315 7.5 1.314 7.5 1.312 7.5 C 1.31 7.5 1.309 7.5 1.307 7.5 C 1.305 7.5 1.303 7.5 1.302 7.5 C 1.3 7.5 1.298 7.5 1.297 7.5 C 1.295 7.5 1.293 7.5 1.291 7.5 C 1.29 7.5 1.288 7.5 1.286 7.5 C 1.284 7.5 1.283 7.5 1.281 7.5 C 1.279 7.5 1.278 7.5 1.276 7.5 C 1.274 7.5 1.272 7.5 1.271 7.5 C 1.269 7.5 1.267 7.5 1.265 7.5 C 1.264 7.5 1.262 7.5 1.26 7.5 C 1.259 7.5 1.257 7.5 1.255 7.5 C 1.253 7.5 1.252 7.5 1.25 7.5 C 1.248 7.5 1.246 7.5 1.245 7.5 C 1.243 7.5 1.241 7.5 1.239 7.5 C 1.238 7.5 1.236 7.5 1.234 7.5 C 1.233 7.5 1.231 7.5 1.229 7.5 C 1.227 7.5 1.226 7.5 1.224 7.5 C 1.222 7.5 1.22 7.5 1.219 7.5 C 1.217 7.5 1.215 7.5 1.213 7.5 C 1.212 7.5 1.21 7.5 1.208 7.5 C 1.207 7.5 1.205 7.5 1.203 7.5 C 1.201 7.5 1.2 7.5 1.198 7.5 C 1.196 7.5 1.194 7.5 1.193 7.5 C 1.191 7.5 1.189 7.5 1.187 7.5 C 1.186 7.5 1.184 7.5 1.182 7.5 C 1.18 7.5 1.179 7.5 1.177 7.5 C 1.175 7.5 1.174 7.5 1.172 7.5 C 1.17 7.5 1.168 7.5 1.167 7.5 C 1.165 7.5 1.163 7.5 1.161 7.5 C 1.16 7.5 1.158 7.5 1.156 7.5 C 1.154 7.5 1.153 7.5 1.151 7.5 C 1.149 7.5 1.147 7.5 1.146 7.5 C 1.144 7.5 1.142 7.5 1.14 7.5 C 1.139 7.5 1.137 7.5 1.135 7.5 C 1.134 7.5 1.132 7.5 1.13 7.5 C 1.128 7.5 1.127 7.5 1.125 7.5 C 1.123 7.5 1.121 7.5 1.12 7.5 C 1.118 7.5 1.116 7.5 1.114 7.5 C 1.113 7.5 1.111 7.5 1.109 7.5 C 1.107 7.5 1.106 7.5 1.104 7.5 C 1.102 7.5 1.1 7.5 1.099 7.5 C 1.097 7.5 1.095 7.5 1.094 7.5 C 1.092 7.5 1.09 7.5 1.088 7.5 C 1.087 7.5 1.085 7.5 1.083 7.5 C 1.081 7.5 1.08 7.5 1.078 7.5 C 1.076 7.5 1.074 7.5 1.073 7.5 C 1.071 7.5 1.069 7.5 1.067 7.5 C 1.066 7.5 1.064 7.5 1.062 7.5 C 1.061 7.5 1.059 7.5 1.057 7.5 C 1.055 7.5 1.054 7.5 1.052 7.5 C 1.05 7.5 1.048 7.5 1.047 7.5 C 1.045 7.5 1.043 7.5 1.041 7.5 C 1.04 7.5 1.038 7.5 1.036 7.5 C 1.035 7.5 1.033 7.5 1.031 7.5 C 1.029 7.5 1.028 7.5 1.026 7.5 C 1.024 7.5 1.022 7.5 1.021 7.5 C 1.019 7.5 1.017 7.5 1.016 7.5 C 1.014 7.5 1.012 7.5 1.01 7.5 C 1.009 7.5 1.007 7.5 1.005 7.5 C 1.003 7.5 1.002 7.5 1 7.5 C 0.998 7.5 0.996 7.5 0.995 7.5 C 0.993 7.5 0.991 7.5 0.99 7.5 C 0.988 7.5 0.986 7.5 0.984 7.5 C 0.983 7.5 0.981 7.5 0.979 7.5 C 0.978 7.5 0.976 7.5 0.974 7.5 C 0.972 7.5 0.971 7.5 0.969 7.5 C 0.967 7.5 0.965 7.5 0.964 7.5 C 0.962 7.5 0.96 7.5 0.959 7.5 C 0.957 7.5 0.955 7.5 0.953 7.5 C 0.952 7.5 0.95 7.5 0.948 7.5 C 0.947 7.5 0.945 7.5 0.943 7.5 C 0.941 7.5 0.94 7.5 0.938 7.5 C 0.936 7.5 0.935 7.5 0.933 7.5 C 0.931 7.5 0.929 7.5 0.928 7.5 C 0.926 7.5 0.924 7.5 0.923 7.5 C 0.921 7.5 0.919 7.5 0.917 7.5 C 0.916 7.5 0.914 7.5 0.912 7.5 C 0.911 7.5 0.909 7.5 0.907 7.5 C 0.906 7.5 0.904 7.5 0.902 7.5 C 0.9 7.5 0.899 7.5 0.897 7.5 C 0.895 7.5 0.894 7.5 0.892 7.5 C 0.89 7.5 0.888 7.5 0.887 7.5 C 0.885 7.5 0.883 7.5 0.882 7.5 C 0.88 7.5 0.878 7.5 0.877 7.5 C 0.875 7.5 0.873 7.5 0.871 7.5 C 0.87 7.5 0.868 7.5 0.866 7.5 C 0.865 7.5 0.863 7.5 0.861 7.5 C 0.86 7.5 0.858 7.5 0.856 7.5 C 0.855 7.5 0.853 7.5 0.851 7.5 C 0.85 7.5 0.848 7.5 0.846 7.5 C 0.844 7.5 0.843 7.5 0.841 7.5 C 0.839 7.5 0.838 7.5 0.836 7.5 C 0.834 7.5 0.833 7.5 0.831 7.5 C 0.829 7.5 0.828 7.5 0.826 7.5 C 0.824 7.5 0.823 7.5 0.821 7.5 C 0.819 7.5 0.818 7.5 0.816 7.5 C 0.814 7.5 0.813 7.5 0.811 7.5 C 0.809 7.5 0.808 7.5 0.806 7.5 C 0.804 7.5 0.803 7.5 0.801 7.5 C 0.799 7.5 0.798 7.5 0.796 7.5 C 0.794 7.5 0.793 7.5 0.791 7.5 C 0.789 7.5 0.788 7.5 0.786 7.5 C 0.784 7.5 0.783 7.5 0.781 7.5 C 0.779 7.5 0.778 7.5 0.776 7.5 C 0.774 7.5 0.773 7.5 0.771 7.5 C 0.769 7.5 0.768 7.5 0.766 7.5 C 0.764 7.5 0.763 7.5 0.761 7.5 C 0.759 7.5 0.758 7.5 0.756 7.5 C 0.754 7.5 0.753 7.5 0.751 7.5 C 0.749 7.5 0.748 7.5 0.746 7.5 C 0.745 7.5 0.743 7.5 0.741 7.5 C 0.74 7.5 0.738 7.5 0.736 7.5 C 0.735 7.5 0.733 7.5 0.731 7.5 C 0.73 7.5 0.728 7.5 0.727 7.5 C 0.725 7.5 0.723 7.5 0.722 7.5 C 0.72 7.5 0.718 7.5 0.717 7.5 C 0.715 7.5 0.714 7.5 0.712 7.5 C 0.71 7.5 0.709 7.5 0.707 7.5 C 0.705 7.5 0.704 7.5 0.702 7.5 C 0.701 7.5 0.699 7.5 0.697 7.5 C 0.696 7.5 0.694 7.5 0.692 7.5 C 0.691 7.5 0.689 7.5 0.688 7.5 C 0.686 7.5 0.684 7.5 0.683 7.5 C 0.681 7.5 0.68 7.5 0.678 7.5 C 0.676 7.5 0.675 7.5 0.673 7.5 C 0.672 7.5 0.67 7.5 0.668 7.5 C 0.667 7.5 0.665 7.5 0.664 7.5 C 0.662 7.5 0.66 7.5 0.659 7.5 C 0.657 7.5 0.656 7.5 0.654 7.5 C 0.652 7.5 0.651 7.5 0.649 7.5 C 0.648 7.5 0.646 7.5 0.645 7.5 C 0.643 7.5 0.641 7.5 0.64 7.5 C 0.638 7.5 0.637 7.5 0.635 7.5 C 0.633 7.5 0.632 7.5 0.63 7.5 C 0.629 7.5 0.627 7.5 0.626 7.5 C 0.624 7.5 0.622 7.5 0.621 7.5 C 0.619 7.5 0.618 7.5 0.616 7.5 C 0.615 7.5 0.613 7.5 0.611 7.5 C 0.61 7.5 0.608 7.5 0.607 7.5 C 0.605 7.5 0.604 7.5 0.602 7.5 C 0.601 7.5 0.599 7.5 0.597 7.5 C 0.596 7.5 0.594 7.5 0.593 7.5 C 0.591 7.5 0.59 7.5 0.588 7.5 C 0.587 7.5 0.585 7.5 0.584 7.5 C 0.582 7.5 0.581 7.5 0.579 7.5 C 0.577 7.5 0.576 7.5 0.574 7.5 C 0.573 7.5 0.571 7.5 0.57 7.5 C 0.568 7.5 0.567 7.5 0.565 7.5 C 0.564 7.5 0.562 7.5 0.561 7.5 C 0.559 7.5 0.558 7.5 0.556 7.5 C 0.555 7.5 0.553 7.5 0.552 7.5 C 0.55 7.5 0.549 7.5 0.547 7.5 C 0.545 7.5 0.544 7.5 0.542 7.5 C 0.541 7.5 0.539 7.5 0.538 7.5 C 0.536 7.5 0.535 7.5 0.533 7.5 C 0.532 7.5 0.53 7.5 0.529 7.5 C 0.527 7.5 0.526 7.5 0.524 7.5 C 0.523 7.5 0.522 7.5 0.52 7.5 C 0.519 7.5 0.517 7.5 0.516 7.5 C 0.514 7.5 0.513 7.5 0.511 7.5 C 0.51 7.5 0.508 7.5 0.507 7.5 C 0.505 7.5 0.504 7.5 0.502 7.5 C 0.501 7.5 0.499 7.5 0.498 7.5 C 0.496 7.5 0.495 7.5 0.494 7.5 C 0.492 7.5 0.491 7.5 0.489 7.5 C 0.488 7.5 0.486 7.5 0.485 7.5 C 0.483 7.5 0.482 7.5 0.48 7.5 C 0.479 7.5 0.478 7.5 0.476 7.5 C 0.475 7.5 0.473 7.5 0.472 7.5 C 0.47 7.5 0.469 7.5 0.467 7.5 C 0.466 7.5 0.465 7.5 0.463 7.5 C 0.462 7.5 0.46 7.5 0.459 7.5 C 0.457 7.5 0.456 7.5 0.455 7.5 C 0.453 7.5 0.452 7.5 0.45 7.5 C 0.449 7.5 0.447 7.5 0.446 7.5 C 0.445 7.5 0.443 7.5 0.442 7.5 C 0.44 7.5 0.439 7.5 0.438 7.5 C 0.436 7.5 0.435 7.5 0.433 7.5 C 0.432 7.5 0.431 7.5 0.429 7.5 C 0.428 7.5 0.426 7.5 0.425 7.5 C 0.424 7.5 0.422 7.5 0.421 7.5 C 0.42 7.5 0.418 7.5 0.417 7.5 C 0.415 7.5 0.414 7.5 0.413 7.5 C 0.411 7.5 0.41 7.5 0.409 7.5 C 0.407 7.5 0.406 7.5 0.404 7.5 C 0.403 7.5 0.402 7.5 0.4 7.5 C 0.399 7.5 0.398 7.5 0.396 7.5 C 0.395 7.5 0.394 7.5 0.392 7.5 C 0.391 7.5 0.389 7.5 0.388 7.5 C 0.387 7.5 0.385 7.5 0.384 7.5 C 0.383 7.5 0.381 7.5 0.38 7.5 C 0.379 7.5 0.377 7.5 0.376 7.5 C 0.375 7.5 0.373 7.5 0.372 7.5 C 0.371 7.5 0.369 7.5 0.368 7.5 C 0.367 7.5 0.366 7.5 0.364 7.5 C 0.363 7.5 0.362 7.5 0.36 7.5 C 0.359 7.5 0.358 7.5 0.356 7.5 C 0.355 7.5 0.354 7.5 0.352 7.5 C 0.351 7.5 0.35 7.5 0.349 7.5 C 0.347 7.5 0.346 7.5 0.345 7.5 C 0.343 7.5 0.342 7.5 0.341 7.5 C 0.34 7.5 0.338 7.5 0.337 7.5 C 0.336 7.5 0.334 7.5 0.333 7.5 C 0.332 7.5 0.331 7.5 0.329 7.5 C 0.328 7.5 0.327 7.5 0.326 7.5 C 0.324 7.5 0.323 7.5 0.322 7.5 C 0.321 7.5 0.319 7.5 0.318 7.5 C 0.317 7.5 0.316 7.5 0.314 7.5 C 0.313 7.5 0.312 7.5 0.311 7.5 C 0.309 7.5 0.308 7.5 0.307 7.5 C 0.306 7.5 0.304 7.5 0.303 7.5 C 0.302 7.5 0.301 7.5 0.3 7.5 C 0.298 7.5 0.297 7.5 0.296 7.5 C 0.295 7.5 0.293 7.5 0.292 7.5 C 0.291 7.5 0.29 7.5 0.289 7.5 C 0.287 7.5 0.286 7.5 0.285 7.5 C 0.284 7.5 0.283 7.5 0.281 7.5 C 0.28 7.5 0.279 7.5 0.278 7.5 C 0.277 7.5 0.276 7.5 0.274 7.5 C 0.273 7.5 0.272 7.5 0.271 7.5 C 0.27 7.5 0.268 7.5 0.267 7.5 C 0.266 7.5 0.265 7.5 0.264 7.5 C 0.263 7.5 0.261 7.5 0.26 7.5 C 0.259 7.5 0.258 7.5 0.257 7.5 C 0.256 7.5 0.255 7.5 0.253 7.5 C 0.252 7.5 0.251 7.5 0.25 7.5 C 0.249 7.5 0.248 7.5 0.247 7.5 C 0.245 7.5 0.244 7.5 0.243 7.5 C 0.242 7.5 0.241 7.5 0.24 7.5 C 0.239 7.5 0.238 7.5 0.237 7.5 C 0.235 7.5 0.234 7.5 0.233 7.5 C 0.232 7.5 0.231 7.5 0.23 7.5 C 0.229 7.5 0.228 7.5 0.227 7.5 C 0.226 7.5 0.224 7.5 0.223 7.5 C 0.222 7.5 0.221 7.5 0.22 7.5 C 0.219 7.5 0.218 7.5 0.217 7.5 C 0.216 7.5 0.215 7.5 0.214 7.5 C 0.213 7.5 0.212 7.5 0.21 7.5 C 0.209 7.5 0.208 7.5 0.207 7.5 C 0.206 7.5 0.205 7.5 0.204 7.5 C 0.203 7.5 0.202 7.5 0.201 7.5 C 0.2 7.5 0.199 7.5 0.198 7.5 C 0.197 7.5 0.196 7.5 0.195 7.5 C 0.194 7.5 0.193 7.5 0.192 7.5 C 0.191 7.5 0.19 7.5 0.189 7.5 C 0.188 7.5 0.187 7.5 0.186 7.5 C 0.185 7.5 0.184 7.5 0.183 7.5 C 0.182 7.5 0.181 7.5 0.18 7.5 C 0.179 7.5 0.178 7.5 0.177 7.5 C 0.176 7.5 0.175 7.5 0.174 7.5 C 0.173 7.5 0.172 7.5 0.171 7.5 C 0.17 7.5 0.169 7.5 0.168 7.5 C 0.167 7.5 0.166 7.5 0.165 7.5 C 0.164 7.5 0.163 7.5 0.162 7.5 C 0.161 7.5 0.16 7.5 0.159 7.5 C 0.158 7.5 0.158 7.5 0.157 7.5 C 0.156 7.5 0.155 7.5 0.154 7.5 C 0.153 7.5 0.152 7.5 0.151 7.5 C 0.15 7.5 0.149 7.5 0.148 7.5 C 0.147 7.5 0.146 7.5 0.146 7.5 C 0.145 7.5 0.144 7.5 0.143 7.5 C 0.142 7.5 0.141 7.5 0.14 7.5 C 0.139 7.5 0.138 7.5 0.137 7.5 C 0.137 7.5 0.136 7.5 0.135 7.5 C 0.134 7.5 0.133 7.5 0.132 7.5 C 0.131 7.5 0.13 7.5 0.13 7.5 C 0.129 7.5 0.128 7.5 0.127 7.5 C 0.126 7.5 0.125 7.5 0.125 7.5 C 0.124 7.5 0.123 7.5 0.122 7.5 C 0.121 7.5 0.12 7.5 0.119 7.5 C 0.119 7.5 0.118 7.5 0.117 7.5 C 0.116 7.5 0.115 7.5 0.115 7.5 C 0.114 7.5 0.113 7.5 0.112 7.5 C 0.111 7.5 0.11 7.5 0.11 7.5 C 0.109 7.5 0.108 7.5 0.107 7.5 C 0.107 7.5 0.106 7.5 0.105 7.5 C 0.104 7.5 0.103 7.5 0.103 7.5 C 0.102 7.5 0.101 7.5 0.1 7.5 C 0.1 7.5 0.099 7.5 0.098 7.5 C 0.097 7.5 0.096 7.5 0.096 7.5 C 0.095 7.5 0.094 7.5 0.093 7.5 C 0.093 7.5 0.092 7.5 0.091 7.5 C 0.091 7.5 0.09 7.5 0.089 7.5 C 0.088 7.5 0.088 7.5 0.087 7.5 C 0.086 7.5 0.085 7.5 0.085 7.5 C 0.084 7.5 0.083 7.5 0.083 7.5 C 0.082 7.5 0.081 7.5 0.081 7.5 C 0.08 7.5 0.079 7.5 0.078 7.5 C 0.078 7.5 0.077 7.5 0.076 7.5 C 0.076 7.5 0.075 7.5 0.074 7.5 C 0.074 7.5 0.073 7.5 0.072 7.5 C 0.072 7.5 0.071 7.5 0.07 7.5 C 0.07 7.5 0.069 7.5 0.068 7.5 C 0.068 7.5 0.067 7.5 0.067 7.5 C 0.066 7.5 0.065 7.5 0.065 7.5 C 0.064 7.5 0.063 7.5 0.063 7.5 C 0.062 7.5 0.061 7.5 0.061 7.5 C 0.06 7.5 0.06 7.5 0.059 7.5 C 0.058 7.5 0.058 7.5 0.057 7.5 C 0.057 7.5 0.056 7.5 0.055 7.5 C 0.055 7.5 0.054 7.5 0.054 7.5 C 0.053 7.5 0.053 7.5 0.052 7.5 C 0.051 7.5 0.051 7.5 0.05 7.5 C 0.05 7.5 0.049 7.5 0.049 7.5 C 0.048 7.5 0.048 7.5 0.047 7.5 C 0.046 7.5 0.046 7.5 0.045 7.5 C 0.045 7.5 0.044 7.5 0.044 7.5 C 0.043 7.5 0.043 7.5 0.042 7.5 C 0.042 7.5 0.041 7.5 0.041 7.5 C 0.04 7.5 0.04 7.5 0.039 7.5 C 0.039 7.5 0.038 7.5 0.038 7.5 C 0.037 7.5 0.037 7.5 0.036 7.5 C 0.036 7.5 0.035 7.5 0.035 7.5 C 0.034 7.5 0.034 7.5 0.033 7.5 C 0.033 7.5 0.033 7.5 0.032 7.5 C 0.032 7.5 0.031 7.5 0.031 7.5 C 0.03 7.5 0.03 7.5 0.029 7.5 C 0.029 7.5 0.029 7.5 0.028 7.5 C 0.028 7.5 0.027 7.5 0.027 7.5 C 0.026 7.5 0.026 7.5 0.026 7.5 C 0.025 7.5 0.025 7.5 0.024 7.5 C 0.024 7.5 0.024 7.5 0.023 7.5 C 0.023 7.5 0.022 7.5 0.022 7.5 C 0.022 7.5 0.021 7.5 0.021 7.5 C 0.021 7.5 0.02 7.5 0.02 7.5 C 0.019 7.5 0.019 7.5 0.019 7.5 C 0.018 7.5 0.018 7.5 0.018 7.5 C 0.017 7.5 0.017 7.5 0.017 7.5 C 0.016 7.5 0.016 7.5 0.016 7.5 C 0.015 7.5 0.015 7.5 0.015 7.5 C 0.014 7.5 0.014 7.5 0.014 7.5 C 0.014 7.5 0.013 7.5 0.013 7.5 C 0.013 7.5 0.012 7.5 0.012 7.5 C 0.012 7.5 0.012 7.5 0.011 7.5 C 0.011 7.5 0.011 7.5 0.01 7.5 C 0.01 7.5 0.01 7.5 0.01 7.5 C 0.009 7.5 0.009 7.5 0.009 7.5 C 0.009 7.5 0.008 7.5 0.008 7.5 C 0.008 7.5 0.008 7.5 0.008 7.5 C 0.007 7.5 0.007 7.5 0.007 7.5 C 0.007 7.5 0.006 7.5 0.006 7.5 C 0.006 7.5 0.006 7.5 0.006 7.5 C 0.005 7.5 0.005 7.5 0.005 7.5 C 0.005 7.5 0.005 7.5 0.005 7.5 C 0.004 7.5 0.004 7.5 0.004 7.5 C 0.004 7.5 0.004 7.5 0.004 7.5 C 0.003 7.5 0.003 7.5 0.003 7.5 C 0.003 7.5 0.003 7.5 0.003 7.5 C 0.003 7.5 0.002 7.5 0.002 7.5 C 0.002 7.5 0.002 7.5 0.002 7.5 C 0.002 7.5 0.002 7.5 0.002 7.5 C 0.001 7.5 0.001 7.5 0.001 7.5 C 0.001 7.5 0.001 7.5 0.001 7.5 C 0.001 7.5 0.001 7.5 0.001 7.5 C 0.001 7.5 0.001 7.5 0.001 7.5 C 0.001 7.5 0 7.5 0 7.5 C 0 7.5 0 7.5 0 7.5 C 0 7.5 0 7.5 0 7.5 C 0 7.5 0 7.5 0 7.5 C 0 7.5 0 7.5 0 7.5 C 0 7.5 0 7.5 0 8 C 0 8.5 0 8.5 0 8.5 C 0 8.5 0 8.5 0 8.5 C 0 8.5 0 8.5 0 8.5 C 0 8.5 0 8.5 0 8.5 C 0 8.5 0 8.5 0 8.5 C 0 8.5 0.001 8.5 0.001 8.5 C 0.001 8.5 0.001 8.5 0.001 8.5 C 0.001 8.5 0.001 8.5 0.001 8.5 C 0.001 8.5 0.001 8.5 0.001 8.5 C 0.001 8.5 0.001 8.5 0.002 8.5 C 0.002 8.5 0.002 8.5 0.002 8.5 C 0.002 8.5 0.002 8.5 0.002 8.5 C 0.002 8.5 0.003 8.5 0.003 8.5 C 0.003 8.5 0.003 8.5 0.003 8.5 C 0.003 8.5 0.003 8.5 0.004 8.5 C 0.004 8.5 0.004 8.5 0.004 8.5 C 0.004 8.5 0.004 8.5 0.005 8.5 C 0.005 8.5 0.005 8.5 0.005 8.5 C 0.005 8.5 0.005 8.5 0.006 8.5 C 0.006 8.5 0.006 8.5 0.006 8.5 C 0.006 8.5 0.007 8.5 0.007 8.5 C 0.007 8.5 0.007 8.5 0.008 8.5 C 0.008 8.5 0.008 8.5 0.008 8.5 C 0.008 8.5 0.009 8.5 0.009 8.5 C 0.009 8.5 0.009 8.5 0.01 8.5 C 0.01 8.5 0.01 8.5 0.01 8.5 C 0.011 8.5 0.011 8.5 0.011 8.5 C 0.012 8.5 0.012 8.5 0.012 8.5 C 0.012 8.5 0.013 8.5 0.013 8.5 C 0.013 8.5 0.014 8.5 0.014 8.5 C 0.014 8.5 0.014 8.5 0.015 8.5 C 0.015 8.5 0.015 8.5 0.016 8.5 C 0.016 8.5 0.016 8.5 0.017 8.5 C 0.017 8.5 0.017 8.5 0.018 8.5 C 0.018 8.5 0.018 8.5 0.019 8.5 C 0.019 8.5 0.019 8.5 0.02 8.5 C 0.02 8.5 0.021 8.5 0.021 8.5 C 0.021 8.5 0.022 8.5 0.022 8.5 C 0.022 8.5 0.023 8.5 0.023 8.5 C 0.024 8.5 0.024 8.5 0.024 8.5 C 0.025 8.5 0.025 8.5 0.026 8.5 C 0.026 8.5 0.026 8.5 0.027 8.5 C 0.027 8.5 0.028 8.5 0.028 8.5 C 0.029 8.5 0.029 8.5 0.029 8.5 C 0.03 8.5 0.03 8.5 0.031 8.5 C 0.031 8.5 0.032 8.5 0.032 8.5 C 0.033 8.5 0.033 8.5 0.033 8.5 C 0.034 8.5 0.034 8.5 0.035 8.5 C 0.035 8.5 0.036 8.5 0.036 8.5 C 0.037 8.5 0.037 8.5 0.038 8.5 C 0.038 8.5 0.039 8.5 0.039 8.5 C 0.04 8.5 0.04 8.5 0.041 8.5 C 0.041 8.5 0.042 8.5 0.042 8.5 C 0.043 8.5 0.043 8.5 0.044 8.5 C 0.044 8.5 0.045 8.5 0.045 8.5 C 0.046 8.5 0.046 8.5 0.047 8.5 C 0.048 8.5 0.048 8.5 0.049 8.5 C 0.049 8.5 0.05 8.5 0.05 8.5 C 0.051 8.5 0.051 8.5 0.052 8.5 C 0.053 8.5 0.053 8.5 0.054 8.5 C 0.054 8.5 0.055 8.5 0.055 8.5 C 0.056 8.5 0.057 8.5 0.057 8.5 C 0.058 8.5 0.058 8.5 0.059 8.5 C 0.06 8.5 0.06 8.5 0.061 8.5 C 0.061 8.5 0.062 8.5 0.063 8.5 C 0.063 8.5 0.064 8.5 0.065 8.5 C 0.065 8.5 0.066 8.5 0.067 8.5 C 0.067 8.5 0.068 8.5 0.068 8.5 C 0.069 8.5 0.07 8.5 0.07 8.5 C 0.071 8.5 0.072 8.5 0.072 8.5 C 0.073 8.5 0.074 8.5 0.074 8.5 C 0.075 8.5 0.076 8.5 0.076 8.5 C 0.077 8.5 0.078 8.5 0.078 8.5 C 0.079 8.5 0.08 8.5 0.081 8.5 C 0.081 8.5 0.082 8.5 0.083 8.5 C 0.083 8.5 0.084 8.5 0.085 8.5 C 0.085 8.5 0.086 8.5 0.087 8.5 C 0.088 8.5 0.088 8.5 0.089 8.5 C 0.09 8.5 0.091 8.5 0.091 8.5 C 0.092 8.5 0.093 8.5 0.093 8.5 C 0.094 8.5 0.095 8.5 0.096 8.5 C 0.096 8.5 0.097 8.5 0.098 8.5 C 0.099 8.5 0.1 8.5 0.1 8.5 C 0.101 8.5 0.102 8.5 0.103 8.5 C 0.103 8.5 0.104 8.5 0.105 8.5 C 0.106 8.5 0.107 8.5 0.107 8.5 C 0.108 8.5 0.109 8.5 0.11 8.5 C 0.11 8.5 0.111 8.5 0.112 8.5 C 0.113 8.5 0.114 8.5 0.115 8.5 C 0.115 8.5 0.116 8.5 0.117 8.5 C 0.118 8.5 0.119 8.5 0.119 8.5 C 0.12 8.5 0.121 8.5 0.122 8.5 C 0.123 8.5 0.124 8.5 0.125 8.5 C 0.125 8.5 0.126 8.5 0.127 8.5 C 0.128 8.5 0.129 8.5 0.13 8.5 C 0.13 8.5 0.131 8.5 0.132 8.5 C 0.133 8.5 0.134 8.5 0.135 8.5 C 0.136 8.5 0.137 8.5 0.137 8.5 C 0.138 8.5 0.139 8.5 0.14 8.5 C 0.141 8.5 0.142 8.5 0.143 8.5 C 0.144 8.5 0.145 8.5 0.146 8.5 C 0.146 8.5 0.147 8.5 0.148 8.5 C 0.149 8.5 0.15 8.5 0.151 8.5 C 0.152 8.5 0.153 8.5 0.154 8.5 C 0.155 8.5 0.156 8.5 0.157 8.5 C 0.158 8.5 0.158 8.5 0.159 8.5 C 0.16 8.5 0.161 8.5 0.162 8.5 C 0.163 8.5 0.164 8.5 0.165 8.5 C 0.166 8.5 0.167 8.5 0.168 8.5 C 0.169 8.5 0.17 8.5 0.171 8.5 C 0.172 8.5 0.173 8.5 0.174 8.5 C 0.175 8.5 0.176 8.5 0.177 8.5 C 0.178 8.5 0.179 8.5 0.18 8.5 C 0.181 8.5 0.182 8.5 0.183 8.5 C 0.184 8.5 0.185 8.5 0.186 8.5 C 0.187 8.5 0.188 8.5 0.189 8.5 C 0.19 8.5 0.191 8.5 0.192 8.5 C 0.193 8.5 0.194 8.5 0.195 8.5 C 0.196 8.5 0.197 8.5 0.198 8.5 C 0.199 8.5 0.2 8.5 0.201 8.5 C 0.202 8.5 0.203 8.5 0.204 8.5 C 0.205 8.5 0.206 8.5 0.207 8.5 C 0.208 8.5 0.209 8.5 0.21 8.5 C 0.212 8.5 0.213 8.5 0.214 8.5 C 0.215 8.5 0.216 8.5 0.217 8.5 C 0.218 8.5 0.219 8.5 0.22 8.5 C 0.221 8.5 0.222 8.5 0.223 8.5 C 0.224 8.5 0.226 8.5 0.227 8.5 C 0.228 8.5 0.229 8.5 0.23 8.5 C 0.231 8.5 0.232 8.5 0.233 8.5 C 0.234 8.5 0.235 8.5 0.237 8.5 C 0.238 8.5 0.239 8.5 0.24 8.5 C 0.241 8.5 0.242 8.5 0.243 8.5 C 0.244 8.5 0.245 8.5 0.247 8.5 C 0.248 8.5 0.249 8.5 0.25 8.5 C 0.251 8.5 0.252 8.5 0.253 8.5 C 0.255 8.5 0.256 8.5 0.257 8.5 C 0.258 8.5 0.259 8.5 0.26 8.5 C 0.261 8.5 0.263 8.5 0.264 8.5 C 0.265 8.5 0.266 8.5 0.267 8.5 C 0.268 8.5 0.27 8.5 0.271 8.5 C 0.272 8.5 0.273 8.5 0.274 8.5 C 0.276 8.5 0.277 8.5 0.278 8.5 C 0.279 8.5 0.28 8.5 0.281 8.5 C 0.283 8.5 0.284 8.5 0.285 8.5 C 0.286 8.5 0.287 8.5 0.289 8.5 C 0.29 8.5 0.291 8.5 0.292 8.5 C 0.293 8.5 0.295 8.5 0.296 8.5 C 0.297 8.5 0.298 8.5 0.3 8.5 C 0.301 8.5 0.302 8.5 0.303 8.5 C 0.304 8.5 0.306 8.5 0.307 8.5 C 0.308 8.5 0.309 8.5 0.311 8.5 C 0.312 8.5 0.313 8.5 0.314 8.5 C 0.316 8.5 0.317 8.5 0.318 8.5 C 0.319 8.5 0.321 8.5 0.322 8.5 C 0.323 8.5 0.324 8.5 0.326 8.5 C 0.327 8.5 0.328 8.5 0.329 8.5 C 0.331 8.5 0.332 8.5 0.333 8.5 C 0.334 8.5 0.336 8.5 0.337 8.5 C 0.338 8.5 0.34 8.5 0.341 8.5 C 0.342 8.5 0.343 8.5 0.345 8.5 C 0.346 8.5 0.347 8.5 0.349 8.5 C 0.35 8.5 0.351 8.5 0.352 8.5 C 0.354 8.5 0.355 8.5 0.356 8.5 C 0.358 8.5 0.359 8.5 0.36 8.5 C 0.362 8.5 0.363 8.5 0.364 8.5 C 0.366 8.5 0.367 8.5 0.368 8.5 C 0.369 8.5 0.371 8.5 0.372 8.5 C 0.373 8.5 0.375 8.5 0.376 8.5 C 0.377 8.5 0.379 8.5 0.38 8.5 C 0.381 8.5 0.383 8.5 0.384 8.5 C 0.385 8.5 0.387 8.5 0.388 8.5 C 0.389 8.5 0.391 8.5 0.392 8.5 C 0.394 8.5 0.395 8.5 0.396 8.5 C 0.398 8.5 0.399 8.5 0.4 8.5 C 0.402 8.5 0.403 8.5 0.404 8.5 C 0.406 8.5 0.407 8.5 0.409 8.5 C 0.41 8.5 0.411 8.5 0.413 8.5 C 0.414 8.5 0.415 8.5 0.417 8.5 C 0.418 8.5 0.42 8.5 0.421 8.5 C 0.422 8.5 0.424 8.5 0.425 8.5 C 0.426 8.5 0.428 8.5 0.429 8.5 C 0.431 8.5 0.432 8.5 0.433 8.5 C 0.435 8.5 0.436 8.5 0.438 8.5 C 0.439 8.5 0.44 8.5 0.442 8.5 C 0.443 8.5 0.445 8.5 0.446 8.5 C 0.447 8.5 0.449 8.5 0.45 8.5 C 0.452 8.5 0.453 8.5 0.455 8.5 C 0.456 8.5 0.457 8.5 0.459 8.5 C 0.46 8.5 0.462 8.5 0.463 8.5 C 0.465 8.5 0.466 8.5 0.467 8.5 C 0.469 8.5 0.47 8.5 0.472 8.5 C 0.473 8.5 0.475 8.5 0.476 8.5 C 0.478 8.5 0.479 8.5 0.48 8.5 C 0.482 8.5 0.483 8.5 0.485 8.5 C 0.486 8.5 0.488 8.5 0.489 8.5 C 0.491 8.5 0.492 8.5 0.494 8.5 C 0.495 8.5 0.496 8.5 0.498 8.5 C 0.499 8.5 0.501 8.5 0.502 8.5 C 0.504 8.5 0.505 8.5 0.507 8.5 C 0.508 8.5 0.51 8.5 0.511 8.5 C 0.513 8.5 0.514 8.5 0.516 8.5 C 0.517 8.5 0.519 8.5 0.52 8.5 C 0.522 8.5 0.523 8.5 0.524 8.5 C 0.526 8.5 0.527 8.5 0.529 8.5 C 0.53 8.5 0.532 8.5 0.533 8.5 C 0.535 8.5 0.536 8.5 0.538 8.5 C 0.539 8.5 0.541 8.5 0.542 8.5 C 0.544 8.5 0.545 8.5 0.547 8.5 C 0.549 8.5 0.55 8.5 0.552 8.5 C 0.553 8.5 0.555 8.5 0.556 8.5 C 0.558 8.5 0.559 8.5 0.561 8.5 C 0.562 8.5 0.564 8.5 0.565 8.5 C 0.567 8.5 0.568 8.5 0.57 8.5 C 0.571 8.5 0.573 8.5 0.574 8.5 C 0.576 8.5 0.577 8.5 0.579 8.5 C 0.581 8.5 0.582 8.5 0.584 8.5 C 0.585 8.5 0.587 8.5 0.588 8.5 C 0.59 8.5 0.591 8.5 0.593 8.5 C 0.594 8.5 0.596 8.5 0.597 8.5 C 0.599 8.5 0.601 8.5 0.602 8.5 C 0.604 8.5 0.605 8.5 0.607 8.5 C 0.608 8.5 0.61 8.5 0.611 8.5 C 0.613 8.5 0.615 8.5 0.616 8.5 C 0.618 8.5 0.619 8.5 0.621 8.5 C 0.622 8.5 0.624 8.5 0.626 8.5 C 0.627 8.5 0.629 8.5 0.63 8.5 C 0.632 8.5 0.633 8.5 0.635 8.5 C 0.637 8.5 0.638 8.5 0.64 8.5 C 0.641 8.5 0.643 8.5 0.645 8.5 C 0.646 8.5 0.648 8.5 0.649 8.5 C 0.651 8.5 0.652 8.5 0.654 8.5 C 0.656 8.5 0.657 8.5 0.659 8.5 C 0.66 8.5 0.662 8.5 0.664 8.5 C 0.665 8.5 0.667 8.5 0.668 8.5 C 0.67 8.5 0.672 8.5 0.673 8.5 C 0.675 8.5 0.676 8.5 0.678 8.5 C 0.68 8.5 0.681 8.5 0.683 8.5 C 0.684 8.5 0.686 8.5 0.688 8.5 C 0.689 8.5 0.691 8.5 0.692 8.5 C 0.694 8.5 0.696 8.5 0.697 8.5 C 0.699 8.5 0.701 8.5 0.702 8.5 C 0.704 8.5 0.705 8.5 0.707 8.5 C 0.709 8.5 0.71 8.5 0.712 8.5 C 0.714 8.5 0.715 8.5 0.717 8.5 C 0.718 8.5 0.72 8.5 0.722 8.5 C 0.723 8.5 0.725 8.5 0.727 8.5 C 0.728 8.5 0.73 8.5 0.731 8.5 C 0.733 8.5 0.735 8.5 0.736 8.5 C 0.738 8.5 0.74 8.5 0.741 8.5 C 0.743 8.5 0.745 8.5 0.746 8.5 C 0.748 8.5 0.749 8.5 0.751 8.5 C 0.753 8.5 0.754 8.5 0.756 8.5 C 0.758 8.5 0.759 8.5 0.761 8.5 C 0.763 8.5 0.764 8.5 0.766 8.5 C 0.768 8.5 0.769 8.5 0.771 8.5 C 0.773 8.5 0.774 8.5 0.776 8.5 C 0.778 8.5 0.779 8.5 0.781 8.5 C 0.783 8.5 0.784 8.5 0.786 8.5 C 0.788 8.5 0.789 8.5 0.791 8.5 C 0.793 8.5 0.794 8.5 0.796 8.5 C 0.798 8.5 0.799 8.5 0.801 8.5 C 0.803 8.5 0.804 8.5 0.806 8.5 C 0.808 8.5 0.809 8.5 0.811 8.5 C 0.813 8.5 0.814 8.5 0.816 8.5 C 0.818 8.5 0.819 8.5 0.821 8.5 C 0.823 8.5 0.824 8.5 0.826 8.5 C 0.828 8.5 0.829 8.5 0.831 8.5 C 0.833 8.5 0.834 8.5 0.836 8.5 C 0.838 8.5 0.839 8.5 0.841 8.5 C 0.843 8.5 0.844 8.5 0.846 8.5 C 0.848 8.5 0.85 8.5 0.851 8.5 C 0.853 8.5 0.855 8.5 0.856 8.5 C 0.858 8.5 0.86 8.5 0.861 8.5 C 0.863 8.5 0.865 8.5 0.866 8.5 C 0.868 8.5 0.87 8.5 0.871 8.5 C 0.873 8.5 0.875 8.5 0.877 8.5 C 0.878 8.5 0.88 8.5 0.882 8.5 C 0.883 8.5 0.885 8.5 0.887 8.5 C 0.888 8.5 0.89 8.5 0.892 8.5 C 0.894 8.5 0.895 8.5 0.897 8.5 C 0.899 8.5 0.9 8.5 0.902 8.5 C 0.904 8.5 0.906 8.5 0.907 8.5 C 0.909 8.5 0.911 8.5 0.912 8.5 C 0.914 8.5 0.916 8.5 0.917 8.5 C 0.919 8.5 0.921 8.5 0.923 8.5 C 0.924 8.5 0.926 8.5 0.928 8.5 C 0.929 8.5 0.931 8.5 0.933 8.5 C 0.935 8.5 0.936 8.5 0.938 8.5 C 0.94 8.5 0.941 8.5 0.943 8.5 C 0.945 8.5 0.947 8.5 0.948 8.5 C 0.95 8.5 0.952 8.5 0.953 8.5 C 0.955 8.5 0.957 8.5 0.959 8.5 C 0.96 8.5 0.962 8.5 0.964 8.5 C 0.965 8.5 0.967 8.5 0.969 8.5 C 0.971 8.5 0.972 8.5 0.974 8.5 C 0.976 8.5 0.978 8.5 0.979 8.5 C 0.981 8.5 0.983 8.5 0.984 8.5 C 0.986 8.5 0.988 8.5 0.99 8.5 C 0.991 8.5 0.993 8.5 0.995 8.5 C 0.996 8.5 0.998 8.5 1 8.5 C 1.002 8.5 1.003 8.5 1.005 8.5 C 1.007 8.5 1.009 8.5 1.01 8.5 C 1.012 8.5 1.014 8.5 1.016 8.5 C 1.017 8.5 1.019 8.5 1.021 8.5 C 1.022 8.5 1.024 8.5 1.026 8.5 C 1.028 8.5 1.029 8.5 1.031 8.5 C 1.033 8.5 1.035 8.5 1.036 8.5 C 1.038 8.5 1.04 8.5 1.041 8.5 C 1.043 8.5 1.045 8.5 1.047 8.5 C 1.048 8.5 1.05 8.5 1.052 8.5 C 1.054 8.5 1.055 8.5 1.057 8.5 C 1.059 8.5 1.061 8.5 1.062 8.5 C 1.064 8.5 1.066 8.5 1.067 8.5 C 1.069 8.5 1.071 8.5 1.073 8.5 C 1.074 8.5 1.076 8.5 1.078 8.5 C 1.08 8.5 1.081 8.5 1.083 8.5 C 1.085 8.5 1.087 8.5 1.088 8.5 C 1.09 8.5 1.092 8.5 1.094 8.5 C 1.095 8.5 1.097 8.5 1.099 8.5 C 1.1 8.5 1.102 8.5 1.104 8.5 C 1.106 8.5 1.107 8.5 1.109 8.5 C 1.111 8.5 1.113 8.5 1.114 8.5 C 1.116 8.5 1.118 8.5 1.12 8.5 C 1.121 8.5 1.123 8.5 1.125 8.5 C 1.127 8.5 1.128 8.5 1.13 8.5 C 1.132 8.5 1.134 8.5 1.135 8.5 C 1.137 8.5 1.139 8.5 1.14 8.5 C 1.142 8.5 1.144 8.5 1.146 8.5 C 1.147 8.5 1.149 8.5 1.151 8.5 C 1.153 8.5 1.154 8.5 1.156 8.5 C 1.158 8.5 1.16 8.5 1.161 8.5 C 1.163 8.5 1.165 8.5 1.167 8.5 C 1.168 8.5 1.17 8.5 1.172 8.5 C 1.174 8.5 1.175 8.5 1.177 8.5 C 1.179 8.5 1.18 8.5 1.182 8.5 C 1.184 8.5 1.186 8.5 1.187 8.5 C 1.189 8.5 1.191 8.5 1.193 8.5 C 1.194 8.5 1.196 8.5 1.198 8.5 C 1.2 8.5 1.201 8.5 1.203 8.5 C 1.205 8.5 1.207 8.5 1.208 8.5 C 1.21 8.5 1.212 8.5 1.213 8.5 C 1.215 8.5 1.217 8.5 1.219 8.5 C 1.22 8.5 1.222 8.5 1.224 8.5 C 1.226 8.5 1.227 8.5 1.229 8.5 C 1.231 8.5 1.233 8.5 1.234 8.5 C 1.236 8.5 1.238 8.5 1.239 8.5 C 1.241 8.5 1.243 8.5 1.245 8.5 C 1.246 8.5 1.248 8.5 1.25 8.5 C 1.252 8.5 1.253 8.5 1.255 8.5 C 1.257 8.5 1.259 8.5 1.26 8.5 C 1.262 8.5 1.264 8.5 1.265 8.5 C 1.267 8.5 1.269 8.5 1.271 8.5 C 1.272 8.5 1.274 8.5 1.276 8.5 C 1.278 8.5 1.279 8.5 1.281 8.5 C 1.283 8.5 1.284 8.5 1.286 8.5 C 1.288 8.5 1.29 8.5 1.291 8.5 C 1.293 8.5 1.295 8.5 1.297 8.5 C 1.298 8.5 1.3 8.5 1.302 8.5 C 1.303 8.5 1.305 8.5 1.307 8.5 C 1.309 8.5 1.31 8.5 1.312 8.5 C 1.314 8.5 1.315 8.5 1.317 8.5 C 1.319 8.5 1.321 8.5 1.322 8.5 C 1.324 8.5 1.326 8.5 1.328 8.5 C 1.329 8.5 1.331 8.5 1.333 8.5 C 1.334 8.5 1.336 8.5 1.338 8.5 C 1.34 8.5 1.341 8.5 1.343 8.5 C 1.345 8.5 1.346 8.5 1.348 8.5 C 1.35 8.5 1.352 8.5 1.353 8.5 C 1.355 8.5 1.357 8.5 1.358 8.5 C 1.36 8.5 1.362 8.5 1.364 8.5 C 1.365 8.5 1.367 8.5 1.369 8.5 C 1.37 8.5 1.372 8.5 1.374 8.5 C 1.375 8.5 1.377 8.5 1.379 8.5 C 1.381 8.5 1.382 8.5 1.384 8.5 C 1.386 8.5 1.387 8.5 1.389 8.5 C 1.391 8.5 1.393 8.5 1.394 8.5 C 1.396 8.5 1.398 8.5 1.399 8.5 C 1.401 8.5 1.403 8.5 1.404 8.5 C 1.406 8.5 1.408 8.5 1.409 8.5 C 1.411 8.5 1.413 8.5 1.415 8.5 C 1.416 8.5 1.418 8.5 1.42 8.5 C 1.421 8.5 1.423 8.5 1.425 8.5 C 1.426 8.5 1.428 8.5 1.43 8.5 C 1.431 8.5 1.433 8.5 1.435 8.5 C 1.437 8.5 1.438 8.5 1.44 8.5 C 1.442 8.5 1.443 8.5 1.445 8.5 C 1.447 8.5 1.448 8.5 1.45 8.5 C 1.452 8.5 1.453 8.5 1.455 8.5 C 1.457 8.5 1.458 8.5 1.46 8.5 C 1.462 8.5 1.463 8.5 1.465 8.5 C 1.467 8.5 1.468 8.5 1.47 8.5 C 1.472 8.5 1.473 8.5 1.475 8.5 C 1.477 8.5 1.479 8.5 1.48 8.5 C 1.482 8.5 1.484 8.5 1.485 8.5 C 1.487 8.5 1.489 8.5 1.49 8.5 C 1.492 8.5 1.493 8.5 1.495 8.5 C 1.497 8.5 1.498 8.5 1.5 8.5 C 1.502 8.5 1.503 8.5 1.505 8.5 C 1.507 8.5 1.508 8.5 1.51 8.5 C 1.512 8.5 1.513 8.5 1.515 8.5 C 1.517 8.5 1.518 8.5 1.52 8.5 C 1.522 8.5 1.523 8.5 1.525 8.5 C 1.527 8.5 1.528 8.5 1.53 8.5 C 1.532 8.5 1.533 8.5 1.535 8.5 C 1.536 8.5 1.538 8.5 1.54 8.5 C 1.541 8.5 1.543 8.5 1.545 8.5 C 1.546 8.5 1.548 8.5 1.55 8.5 C 1.551 8.5 1.553 8.5 1.554 8.5 C 1.556 8.5 1.558 8.5 1.559 8.5 C 1.561 8.5 1.563 8.5 1.564 8.5 C 1.566 8.5 1.568 8.5 1.569 8.5 C 1.571 8.5 1.572 8.5 1.574 8.5 C 1.576 8.5 1.577 8.5 1.579 8.5 C 1.581 8.5 1.582 8.5 1.584 8.5 C 1.585 8.5 1.587 8.5 1.589 8.5 C 1.59 8.5 1.592 8.5 1.593 8.5 C 1.595 8.5 1.597 8.5 1.598 8.5 C 1.6 8.5 1.601 8.5 1.603 8.5 C 1.605 8.5 1.606 8.5 1.608 8.5 C 1.61 8.5 1.611 8.5 1.613 8.5 C 1.614 8.5 1.616 8.5 1.618 8.5 C 1.619 8.5 1.621 8.5 1.622 8.5 C 1.624 8.5 1.625 8.5 1.627 8.5 C 1.629 8.5 1.63 8.5 1.632 8.5 C 1.633 8.5 1.635 8.5 1.637 8.5 C 1.638 8.5 1.64 8.5 1.641 8.5 C 1.643 8.5 1.644 8.5 1.646 8.5 C 1.648 8.5 1.649 8.5 1.651 8.5 C 1.652 8.5 1.654 8.5 1.656 8.5 C 1.657 8.5 1.659 8.5 1.66 8.5 C 1.662 8.5 1.663 8.5 1.665 8.5 C 1.666 8.5 1.668 8.5 1.67 8.5 C 1.671 8.5 1.673 8.5 1.674 8.5 C 1.676 8.5 1.677 8.5 1.679 8.5 C 1.681 8.5 1.682 8.5 1.684 8.5 C 1.685 8.5 1.687 8.5 1.688 8.5 C 1.69 8.5 1.691 8.5 1.693 8.5 C 1.694 8.5 1.696 8.5 1.698 8.5 C 1.699 8.5 1.701 8.5 1.702 8.5 C 1.704 8.5 1.705 8.5 1.707 8.5 C 1.708 8.5 1.71 8.5 1.711 8.5 C 1.713 8.5 1.714 8.5 1.716 8.5 C 1.717 8.5 1.719 8.5 1.721 8.5 C 1.722 8.5 1.724 8.5 1.725 8.5 C 1.727 8.5 1.728 8.5 1.73 8.5 C 1.731 8.5 1.733 8.5 1.734 8.5 C 1.736 8.5 1.737 8.5 1.739 8.5 C 1.74 8.5 1.742 8.5 1.743 8.5 C 1.745 8.5 1.746 8.5 1.748 8.5 C 1.749 8.5 1.751 8.5 1.752 8.5 C 1.754 8.5 1.755 8.5 1.757 8.5 C 1.758 8.5 1.76 8.5 1.761 8.5 C 1.763 8.5 1.764 8.5 1.766 8.5 C 1.767 8.5 1.769 8.5 1.77 8.5 C 1.772 8.5 1.773 8.5 1.774 8.5 C 1.776 8.5 1.777 8.5 1.779 8.5 C 1.78 8.5 1.782 8.5 1.783 8.5 C 1.785 8.5 1.786 8.5 1.788 8.5 C 1.789 8.5 1.791 8.5 1.792 8.5 C 1.794 8.5 1.795 8.5 1.796 8.5 C 1.798 8.5 1.799 8.5 1.801 8.5 C 1.802 8.5 1.804 8.5 1.805 8.5 C 1.807 8.5 1.808 8.5 1.809 8.5 C 1.811 8.5 1.812 8.5 1.814 8.5 C 1.815 8.5 1.817 8.5 1.818 8.5 C 1.82 8.5 1.821 8.5 1.822 8.5 C 1.824 8.5 1.825 8.5 1.827 8.5 C 1.828 8.5 1.829 8.5 1.831 8.5 C 1.832 8.5 1.834 8.5 1.835 8.5 C 1.837 8.5 1.838 8.5 1.839 8.5 C 1.841 8.5 1.842 8.5 1.844 8.5 C 1.845 8.5 1.846 8.5 1.848 8.5 C 1.849 8.5 1.851 8.5 1.852 8.5 C 1.853 8.5 1.855 8.5 1.856 8.5 C 1.858 8.5 1.859 8.5 1.86 8.5 C 1.862 8.5 1.863 8.5 1.865 8.5 C 1.866 8.5 1.867 8.5 1.869 8.5 C 1.87 8.5 1.871 8.5 1.873 8.5 C 1.874 8.5 1.876 8.5 1.877 8.5 C 1.878 8.5 1.88 8.5 1.881 8.5 C 1.882 8.5 1.884 8.5 1.885 8.5 C 1.886 8.5 1.888 8.5 1.889 8.5 C 1.89 8.5 1.892 8.5 1.893 8.5 C 1.895 8.5 1.896 8.5 1.897 8.5 C 1.899 8.5 1.9 8.5 1.901 8.5 C 1.903 8.5 1.904 8.5 1.905 8.5 C 1.907 8.5 1.908 8.5 1.909 8.5 C 1.911 8.5 1.912 8.5 1.913 8.5 C 1.914 8.5 1.916 8.5 1.917 8.5 C 1.918 8.5 1.92 8.5 1.921 8.5 C 1.922 8.5 1.924 8.5 1.925 8.5 C 1.926 8.5 1.928 8.5 1.929 8.5 C 1.93 8.5 1.931 8.5 1.933 8.5 C 1.934 8.5 1.935 8.5 1.937 8.5 C 1.938 8.5 1.939 8.5 1.941 8.5 C 1.942 8.5 1.943 8.5 1.944 8.5 C 1.946 8.5 1.947 8.5 1.948 8.5 C 1.949 8.5 1.951 8.5 1.952 8.5 C 1.953 8.5 1.955 8.5 1.956 8.5 C 1.957 8.5 1.958 8.5 1.96 8.5 C 1.961 8.5 1.962 8.5 1.963 8.5 C 1.965 8.5 1.966 8.5 1.967 8.5 C 1.968 8.5 1.97 8.5 1.971 8.5 C 1.972 8.5 1.973 8.5 1.975 8.5 C 1.976 8.5 1.977 8.5 1.978 8.5 C 1.979 8.5 1.981 8.5 1.982 8.5 C 1.983 8.5 1.984 8.5 1.986 8.5 C 1.987 8.5 1.988 8.5 1.989 8.5 C 1.99 8.5 1.992 8.5 1.993 8.5 C 1.994 8.5 1.995 8.5 1.996 8.5 C 1.998 8.5 1.999 8.5 2 8.5 L 2 7.5 Z M 0.5 8 L 0.5 6 L -0.5 6 L -0.5 8 L 0.5 8 Z M 0.5 6 C 0.5 5.172 1.172 4.5 2 4.5 L 2 3.5 C 0.619 3.5 -0.5 4.619 -0.5 6 L 0.5 6 Z M 6.5 2 C 6.5 2.616 6.181 3.395 5.811 4.073 C 5.632 4.401 5.453 4.685 5.318 4.887 C 5.251 4.987 5.196 5.067 5.157 5.121 C 5.138 5.148 5.123 5.169 5.113 5.182 C 5.108 5.189 5.105 5.193 5.103 5.197 C 5.101 5.198 5.101 5.199 5.1 5.2 C 5.1 5.2 5.1 5.2 5.1 5.2 C 5.1 5.2 5.1 5.2 5.1 5.2 C 5.1 5.2 5.1 5.2 5.1 5.2 C 5.1 5.2 5.1 5.2 5.5 5.5 C 5.9 5.8 5.9 5.8 5.9 5.8 C 5.9 5.8 5.9 5.8 5.9 5.8 C 5.9 5.799 5.901 5.799 5.901 5.799 C 5.901 5.799 5.901 5.798 5.902 5.797 C 5.903 5.796 5.904 5.794 5.906 5.792 C 5.909 5.787 5.914 5.781 5.92 5.772 C 5.933 5.756 5.95 5.732 5.972 5.701 C 6.015 5.64 6.077 5.552 6.15 5.441 C 6.297 5.222 6.493 4.912 6.689 4.552 C 7.069 3.855 7.5 2.884 7.5 2 L 6.5 2 Z M 5.5 5.5 C 5.812 5.11 5.812 5.11 5.812 5.11 C 5.812 5.11 5.813 5.11 5.813 5.11 C 5.813 5.11 5.813 5.11 5.812 5.11 C 5.812 5.11 5.812 5.109 5.812 5.109 C 5.811 5.108 5.809 5.107 5.807 5.106 C 5.804 5.102 5.797 5.097 5.789 5.091 C 5.772 5.077 5.747 5.056 5.715 5.029 C 5.65 4.974 5.557 4.894 5.444 4.792 C 5.218 4.589 4.918 4.304 4.62 3.976 C 4.32 3.646 4.032 3.284 3.822 2.926 C 3.606 2.56 3.5 2.246 3.5 2 L 2.5 2 C 2.5 2.504 2.706 3.002 2.96 3.433 C 3.218 3.872 3.555 4.291 3.88 4.649 C 4.207 5.008 4.532 5.317 4.775 5.536 C 4.897 5.645 4.998 5.733 5.07 5.793 C 5.106 5.824 5.135 5.847 5.155 5.864 C 5.165 5.872 5.173 5.878 5.178 5.883 C 5.181 5.885 5.183 5.887 5.185 5.888 C 5.186 5.889 5.186 5.889 5.187 5.89 C 5.187 5.89 5.187 5.89 5.187 5.89 C 5.187 5.89 5.187 5.89 5.187 5.89 C 5.188 5.89 5.188 5.89 5.5 5.5 Z M 3.5 2 C 3.5 1.999 3.5 1.998 3.5 1.996 C 3.5 1.995 3.5 1.994 3.5 1.993 C 3.5 1.992 3.5 1.99 3.5 1.989 C 3.5 1.988 3.5 1.987 3.5 1.986 C 3.5 1.984 3.5 1.983 3.5 1.982 C 3.5 1.981 3.5 1.979 3.5 1.978 C 3.5 1.977 3.5 1.976 3.5 1.975 C 3.5 1.973 3.5 1.972 3.5 1.971 C 3.5 1.97 3.5 1.968 3.5 1.967 C 3.5 1.966 3.5 1.965 3.5 1.963 C 3.5 1.962 3.5 1.961 3.5 1.96 C 3.5 1.958 3.5 1.957 3.5 1.956 C 3.5 1.955 3.5 1.953 3.5 1.952 C 3.5 1.951 3.5 1.949 3.5 1.948 C 3.5 1.947 3.5 1.946 3.5 1.944 C 3.5 1.943 3.5 1.942 3.5 1.941 C 3.5 1.939 3.5 1.938 3.5 1.937 C 3.5 1.935 3.5 1.934 3.5 1.933 C 3.5 1.931 3.5 1.93 3.5 1.929 C 3.5 1.928 3.5 1.926 3.5 1.925 C 3.5 1.924 3.5 1.922 3.5 1.921 C 3.5 1.92 3.5 1.918 3.5 1.917 C 3.5 1.916 3.5 1.914 3.5 1.913 C 3.5 1.912 3.5 1.911 3.5 1.909 C 3.5 1.908 3.5 1.907 3.5 1.905 C 3.5 1.904 3.5 1.903 3.5 1.901 C 3.5 1.9 3.5 1.899 3.5 1.897 C 3.5 1.896 3.5 1.895 3.5 1.893 C 3.5 1.892 3.5 1.89 3.5 1.889 C 3.5 1.888 3.5 1.886 3.5 1.885 C 3.5 1.884 3.5 1.882 3.5 1.881 C 3.5 1.88 3.5 1.878 3.5 1.877 C 3.5 1.876 3.5 1.874 3.5 1.873 C 3.5 1.871 3.5 1.87 3.5 1.869 C 3.5 1.867 3.5 1.866 3.5 1.865 C 3.5 1.863 3.5 1.862 3.5 1.86 C 3.5 1.859 3.5 1.858 3.5 1.856 C 3.5 1.855 3.5 1.853 3.5 1.852 C 3.5 1.851 3.5 1.849 3.5 1.848 C 3.5 1.846 3.5 1.845 3.5 1.844 C 3.5 1.842 3.5 1.841 3.5 1.839 C 3.5 1.838 3.5 1.837 3.5 1.835 C 3.5 1.834 3.5 1.832 3.5 1.831 C 3.5 1.829 3.5 1.828 3.5 1.827 C 3.5 1.825 3.5 1.824 3.5 1.822 C 3.5 1.821 3.5 1.82 3.5 1.818 C 3.5 1.817 3.5 1.815 3.5 1.814 C 3.5 1.812 3.5 1.811 3.5 1.809 C 3.5 1.808 3.5 1.807 3.5 1.805 C 3.5 1.804 3.5 1.802 3.5 1.801 C 3.5 1.799 3.5 1.798 3.5 1.796 C 3.5 1.795 3.5 1.794 3.5 1.792 C 3.5 1.791 3.5 1.789 3.5 1.788 C 3.5 1.786 3.5 1.785 3.5 1.783 C 3.5 1.782 3.5 1.78 3.5 1.779 C 3.5 1.777 3.5 1.776 3.5 1.774 C 3.5 1.773 3.5 1.772 3.5 1.77 C 3.5 1.769 3.5 1.767 3.5 1.766 C 3.5 1.764 3.5 1.763 3.5 1.761 C 3.5 1.76 3.5 1.758 3.5 1.757 C 3.5 1.755 3.5 1.754 3.5 1.752 C 3.5 1.751 3.5 1.749 3.5 1.748 C 3.5 1.746 3.5 1.745 3.5 1.743 C 3.5 1.742 3.5 1.74 3.5 1.739 C 3.5 1.737 3.5 1.736 3.5 1.734 C 3.5 1.733 3.5 1.731 3.5 1.73 C 3.5 1.728 3.5 1.727 3.5 1.725 C 3.5 1.724 3.5 1.722 3.5 1.721 C 3.5 1.719 3.5 1.717 3.5 1.716 C 3.5 1.714 3.5 1.713 3.5 1.711 C 3.5 1.71 3.5 1.708 3.5 1.707 C 3.5 1.705 3.5 1.704 3.5 1.702 C 3.5 1.701 3.5 1.699 3.5 1.698 C 3.5 1.696 3.5 1.694 3.5 1.693 C 3.5 1.691 3.5 1.69 3.5 1.688 C 3.5 1.687 3.5 1.685 3.5 1.684 C 3.5 1.682 3.5 1.681 3.5 1.679 C 3.5 1.677 3.5 1.676 3.5 1.674 C 3.5 1.673 3.5 1.671 3.5 1.67 C 3.5 1.668 3.5 1.666 3.5 1.665 C 3.5 1.663 3.5 1.662 3.5 1.66 C 3.5 1.659 3.5 1.657 3.5 1.656 C 3.5 1.654 3.5 1.652 3.5 1.651 C 3.5 1.649 3.5 1.648 3.5 1.646 C 3.5 1.644 3.5 1.643 3.5 1.641 C 3.5 1.64 3.5 1.638 3.5 1.637 C 3.5 1.635 3.5 1.633 3.5 1.632 C 3.5 1.63 3.5 1.629 3.5 1.627 C 3.5 1.625 3.5 1.624 3.5 1.622 C 3.5 1.621 3.5 1.619 3.5 1.618 C 3.5 1.616 3.5 1.614 3.5 1.613 C 3.5 1.611 3.5 1.61 3.5 1.608 C 3.5 1.606 3.5 1.605 3.5 1.603 C 3.5 1.601 3.5 1.6 3.5 1.598 C 3.5 1.597 3.5 1.595 3.5 1.593 C 3.5 1.592 3.5 1.59 3.5 1.589 C 3.5 1.587 3.5 1.585 3.5 1.584 C 3.5 1.582 3.5 1.581 3.5 1.579 C 3.5 1.577 3.5 1.576 3.5 1.574 C 3.5 1.572 3.5 1.571 3.5 1.569 C 3.5 1.568 3.5 1.566 3.5 1.564 C 3.5 1.563 3.5 1.561 3.5 1.559 C 3.5 1.558 3.5 1.556 3.5 1.554 C 3.5 1.553 3.5 1.551 3.5 1.55 C 3.5 1.548 3.5 1.546 3.5 1.545 C 3.5 1.543 3.5 1.541 3.5 1.54 C 3.5 1.538 3.5 1.536 3.5 1.535 C 3.5 1.533 3.5 1.532 3.5 1.53 C 3.5 1.528 3.5 1.527 3.5 1.525 C 3.5 1.523 3.5 1.522 3.5 1.52 C 3.5 1.518 3.5 1.517 3.5 1.515 C 3.5 1.513 3.5 1.512 3.5 1.51 C 3.5 1.508 3.5 1.507 3.5 1.505 C 3.5 1.503 3.5 1.502 3.5 1.5 C 3.5 1.498 3.5 1.497 3.5 1.495 C 3.5 1.493 3.5 1.492 3.5 1.49 C 3.5 1.489 3.5 1.487 3.5 1.485 C 3.5 1.484 3.5 1.482 3.5 1.48 C 3.5 1.479 3.5 1.477 3.5 1.475 C 3.5 1.473 3.5 1.472 3.5 1.47 C 3.5 1.468 3.5 1.467 3.5 1.465 C 3.5 1.463 3.5 1.462 3.5 1.46 C 3.5 1.458 3.5 1.457 3.5 1.455 C 3.5 1.453 3.5 1.452 3.5 1.45 C 3.5 1.448 3.5 1.447 3.5 1.445 C 3.5 1.443 3.5 1.442 3.5 1.44 C 3.5 1.438 3.5 1.437 3.5 1.435 C 3.5 1.433 3.5 1.431 3.5 1.43 C 3.5 1.428 3.5 1.426 3.5 1.425 C 3.5 1.423 3.5 1.421 3.5 1.42 C 3.5 1.418 3.5 1.416 3.5 1.415 C 3.5 1.413 3.5 1.411 3.5 1.409 C 3.5 1.408 3.5 1.406 3.5 1.404 C 3.5 1.403 3.5 1.401 3.5 1.399 C 3.5 1.398 3.5 1.396 3.5 1.394 C 3.5 1.393 3.5 1.391 3.5 1.389 C 3.5 1.387 3.5 1.386 3.5 1.384 C 3.5 1.382 3.5 1.381 3.5 1.379 C 3.5 1.377 3.5 1.375 3.5 1.374 C 3.5 1.372 3.5 1.37 3.5 1.369 C 3.5 1.367 3.5 1.365 3.5 1.364 C 3.5 1.362 3.5 1.36 3.5 1.358 C 3.5 1.357 3.5 1.355 3.5 1.353 C 3.5 1.352 3.5 1.35 3.5 1.348 C 3.5 1.346 3.5 1.345 3.5 1.343 C 3.5 1.341 3.5 1.34 3.5 1.338 C 3.5 1.336 3.5 1.334 3.5 1.333 C 3.5 1.331 3.5 1.329 3.5 1.328 C 3.5 1.326 3.5 1.324 3.5 1.322 C 3.5 1.321 3.5 1.319 3.5 1.317 C 3.5 1.315 3.5 1.314 3.5 1.312 C 3.5 1.31 3.5 1.309 3.5 1.307 C 3.5 1.305 3.5 1.303 3.5 1.302 C 3.5 1.3 3.5 1.298 3.5 1.297 C 3.5 1.295 3.5 1.293 3.5 1.291 C 3.5 1.29 3.5 1.288 3.5 1.286 C 3.5 1.284 3.5 1.283 3.5 1.281 C 3.5 1.279 3.5 1.278 3.5 1.276 C 3.5 1.274 3.5 1.272 3.5 1.271 C 3.5 1.269 3.5 1.267 3.5 1.265 C 3.5 1.264 3.5 1.262 3.5 1.26 C 3.5 1.259 3.5 1.257 3.5 1.255 C 3.5 1.253 3.5 1.252 3.5 1.25 C 3.5 1.248 3.5 1.246 3.5 1.245 C 3.5 1.243 3.5 1.241 3.5 1.239 C 3.5 1.238 3.5 1.236 3.5 1.234 C 3.5 1.233 3.5 1.231 3.5 1.229 C 3.5 1.227 3.5 1.226 3.5 1.224 C 3.5 1.222 3.5 1.22 3.5 1.219 C 3.5 1.217 3.5 1.215 3.5 1.213 C 3.5 1.212 3.5 1.21 3.5 1.208 C 3.5 1.207 3.5 1.205 3.5 1.203 C 3.5 1.201 3.5 1.2 3.5 1.198 C 3.5 1.196 3.5 1.194 3.5 1.193 C 3.5 1.191 3.5 1.189 3.5 1.187 C 3.5 1.186 3.5 1.184 3.5 1.182 C 3.5 1.18 3.5 1.179 3.5 1.177 C 3.5 1.175 3.5 1.174 3.5 1.172 C 3.5 1.17 3.5 1.168 3.5 1.167 C 3.5 1.165 3.5 1.163 3.5 1.161 C 3.5 1.16 3.5 1.158 3.5 1.156 C 3.5 1.154 3.5 1.153 3.5 1.151 C 3.5 1.149 3.5 1.147 3.5 1.146 C 3.5 1.144 3.5 1.142 3.5 1.14 C 3.5 1.139 3.5 1.137 3.5 1.135 C 3.5 1.134 3.5 1.132 3.5 1.13 C 3.5 1.128 3.5 1.127 3.5 1.125 C 3.5 1.123 3.5 1.121 3.5 1.12 C 3.5 1.118 3.5 1.116 3.5 1.114 C 3.5 1.113 3.5 1.111 3.5 1.109 C 3.5 1.107 3.5 1.106 3.5 1.104 C 3.5 1.102 3.5 1.1 3.5 1.099 C 3.5 1.097 3.5 1.095 3.5 1.094 C 3.5 1.092 3.5 1.09 3.5 1.088 C 3.5 1.087 3.5 1.085 3.5 1.083 C 3.5 1.081 3.5 1.08 3.5 1.078 C 3.5 1.076 3.5 1.074 3.5 1.073 C 3.5 1.071 3.5 1.069 3.5 1.067 C 3.5 1.066 3.5 1.064 3.5 1.062 C 3.5 1.061 3.5 1.059 3.5 1.057 C 3.5 1.055 3.5 1.054 3.5 1.052 C 3.5 1.05 3.5 1.048 3.5 1.047 C 3.5 1.045 3.5 1.043 3.5 1.041 C 3.5 1.04 3.5 1.038 3.5 1.036 C 3.5 1.035 3.5 1.033 3.5 1.031 C 3.5 1.029 3.5 1.028 3.5 1.026 C 3.5 1.024 3.5 1.022 3.5 1.021 C 3.5 1.019 3.5 1.017 3.5 1.016 C 3.5 1.014 3.5 1.012 3.5 1.01 C 3.5 1.009 3.5 1.007 3.5 1.005 C 3.5 1.003 3.5 1.002 3.5 1 C 3.5 0.998 3.5 0.996 3.5 0.995 C 3.5 0.993 3.5 0.991 3.5 0.99 C 3.5 0.988 3.5 0.986 3.5 0.984 C 3.5 0.983 3.5 0.981 3.5 0.979 C 3.5 0.978 3.5 0.976 3.5 0.974 C 3.5 0.972 3.5 0.971 3.5 0.969 C 3.5 0.967 3.5 0.965 3.5 0.964 C 3.5 0.962 3.5 0.96 3.5 0.959 C 3.5 0.957 3.5 0.955 3.5 0.953 C 3.5 0.952 3.5 0.95 3.5 0.948 C 3.5 0.947 3.5 0.945 3.5 0.943 C 3.5 0.941 3.5 0.94 3.5 0.938 C 3.5 0.936 3.5 0.935 3.5 0.933 C 3.5 0.931 3.5 0.929 3.5 0.928 C 3.5 0.926 3.5 0.924 3.5 0.923 C 3.5 0.921 3.5 0.919 3.5 0.917 C 3.5 0.916 3.5 0.914 3.5 0.912 C 3.5 0.911 3.5 0.909 3.5 0.907 C 3.5 0.906 3.5 0.904 3.5 0.902 C 3.5 0.9 3.5 0.899 3.5 0.897 C 3.5 0.895 3.5 0.894 3.5 0.892 C 3.5 0.89 3.5 0.888 3.5 0.887 C 3.5 0.885 3.5 0.883 3.5 0.882 C 3.5 0.88 3.5 0.878 3.5 0.877 C 3.5 0.875 3.5 0.873 3.5 0.871 C 3.5 0.87 3.5 0.868 3.5 0.866 C 3.5 0.865 3.5 0.863 3.5 0.861 C 3.5 0.86 3.5 0.858 3.5 0.856 C 3.5 0.855 3.5 0.853 3.5 0.851 C 3.5 0.85 3.5 0.848 3.5 0.846 C 3.5 0.844 3.5 0.843 3.5 0.841 C 3.5 0.839 3.5 0.838 3.5 0.836 C 3.5 0.834 3.5 0.833 3.5 0.831 C 3.5 0.829 3.5 0.828 3.5 0.826 C 3.5 0.824 3.5 0.823 3.5 0.821 C 3.5 0.819 3.5 0.818 3.5 0.816 C 3.5 0.814 3.5 0.813 3.5 0.811 C 3.5 0.809 3.5 0.808 3.5 0.806 C 3.5 0.804 3.5 0.803 3.5 0.801 C 3.5 0.799 3.5 0.798 3.5 0.796 C 3.5 0.794 3.5 0.793 3.5 0.791 C 3.5 0.789 3.5 0.788 3.5 0.786 C 3.5 0.784 3.5 0.783 3.5 0.781 C 3.5 0.779 3.5 0.778 3.5 0.776 C 3.5 0.774 3.5 0.773 3.5 0.771 C 3.5 0.769 3.5 0.768 3.5 0.766 C 3.5 0.764 3.5 0.763 3.5 0.761 C 3.5 0.759 3.5 0.758 3.5 0.756 C 3.5 0.754 3.5 0.753 3.5 0.751 C 3.5 0.749 3.5 0.748 3.5 0.746 C 3.5 0.745 3.5 0.743 3.5 0.741 C 3.5 0.74 3.5 0.738 3.5 0.736 C 3.5 0.735 3.5 0.733 3.5 0.731 C 3.5 0.73 3.5 0.728 3.5 0.727 C 3.5 0.725 3.5 0.723 3.5 0.722 C 3.5 0.72 3.5 0.718 3.5 0.717 C 3.5 0.715 3.5 0.714 3.5 0.712 C 3.5 0.71 3.5 0.709 3.5 0.707 C 3.5 0.705 3.5 0.704 3.5 0.702 C 3.5 0.701 3.5 0.699 3.5 0.697 C 3.5 0.696 3.5 0.694 3.5 0.692 C 3.5 0.691 3.5 0.689 3.5 0.688 C 3.5 0.686 3.5 0.684 3.5 0.683 C 3.5 0.681 3.5 0.68 3.5 0.678 C 3.5 0.676 3.5 0.675 3.5 0.673 C 3.5 0.672 3.5 0.67 3.5 0.668 C 3.5 0.667 3.5 0.665 3.5 0.664 C 3.5 0.662 3.5 0.66 3.5 0.659 C 3.5 0.657 3.5 0.656 3.5 0.654 C 3.5 0.652 3.5 0.651 3.5 0.649 C 3.5 0.648 3.5 0.646 3.5 0.645 C 3.5 0.643 3.5 0.641 3.5 0.64 C 3.5 0.638 3.5 0.637 3.5 0.635 C 3.5 0.633 3.5 0.632 3.5 0.63 C 3.5 0.629 3.5 0.627 3.5 0.626 C 3.5 0.624 3.5 0.622 3.5 0.621 C 3.5 0.619 3.5 0.618 3.5 0.616 C 3.5 0.615 3.5 0.613 3.5 0.611 C 3.5 0.61 3.5 0.608 3.5 0.607 C 3.5 0.605 3.5 0.604 3.5 0.602 C 3.5 0.601 3.5 0.599 3.5 0.597 C 3.5 0.596 3.5 0.594 3.5 0.593 C 3.5 0.591 3.5 0.59 3.5 0.588 C 3.5 0.587 3.5 0.585 3.5 0.584 C 3.5 0.582 3.5 0.581 3.5 0.579 C 3.5 0.577 3.5 0.576 3.5 0.574 C 3.5 0.573 3.5 0.571 3.5 0.57 C 3.5 0.568 3.5 0.567 3.5 0.565 C 3.5 0.564 3.5 0.562 3.5 0.561 C 3.5 0.559 3.5 0.558 3.5 0.556 C 3.5 0.555 3.5 0.553 3.5 0.552 C 3.5 0.55 3.5 0.549 3.5 0.547 C 3.5 0.545 3.5 0.544 3.5 0.542 C 3.5 0.541 3.5 0.539 3.5 0.538 C 3.5 0.536 3.5 0.535 3.5 0.533 C 3.5 0.532 3.5 0.53 3.5 0.529 C 3.5 0.527 3.5 0.526 3.5 0.524 C 3.5 0.523 3.5 0.522 3.5 0.52 C 3.5 0.519 3.5 0.517 3.5 0.516 C 3.5 0.514 3.5 0.513 3.5 0.511 C 3.5 0.51 3.5 0.508 3.5 0.507 C 3.5 0.505 3.5 0.504 3.5 0.502 C 3.5 0.501 3.5 0.499 3.5 0.498 C 3.5 0.496 3.5 0.495 3.5 0.494 C 3.5 0.492 3.5 0.491 3.5 0.489 C 3.5 0.488 3.5 0.486 3.5 0.485 C 3.5 0.483 3.5 0.482 3.5 0.48 C 3.5 0.479 3.5 0.478 3.5 0.476 C 3.5 0.475 3.5 0.473 3.5 0.472 C 3.5 0.47 3.5 0.469 3.5 0.467 C 3.5 0.466 3.5 0.465 3.5 0.463 C 3.5 0.462 3.5 0.46 3.5 0.459 C 3.5 0.457 3.5 0.456 3.5 0.455 C 3.5 0.453 3.5 0.452 3.5 0.45 C 3.5 0.449 3.5 0.447 3.5 0.446 C 3.5 0.445 3.5 0.443 3.5 0.442 C 3.5 0.44 3.5 0.439 3.5 0.438 C 3.5 0.436 3.5 0.435 3.5 0.433 C 3.5 0.432 3.5 0.431 3.5 0.429 C 3.5 0.428 3.5 0.426 3.5 0.425 C 3.5 0.424 3.5 0.422 3.5 0.421 C 3.5 0.42 3.5 0.418 3.5 0.417 C 3.5 0.415 3.5 0.414 3.5 0.413 C 3.5 0.411 3.5 0.41 3.5 0.409 C 3.5 0.407 3.5 0.406 3.5 0.404 C 3.5 0.403 3.5 0.402 3.5 0.4 C 3.5 0.399 3.5 0.398 3.5 0.396 C 3.5 0.395 3.5 0.394 3.5 0.392 C 3.5 0.391 3.5 0.389 3.5 0.388 C 3.5 0.387 3.5 0.385 3.5 0.384 C 3.5 0.383 3.5 0.381 3.5 0.38 C 3.5 0.379 3.5 0.377 3.5 0.376 C 3.5 0.375 3.5 0.373 3.5 0.372 C 3.5 0.371 3.5 0.369 3.5 0.368 C 3.5 0.367 3.5 0.366 3.5 0.364 C 3.5 0.363 3.5 0.362 3.5 0.36 C 3.5 0.359 3.5 0.358 3.5 0.356 C 3.5 0.355 3.5 0.354 3.5 0.352 C 3.5 0.351 3.5 0.35 3.5 0.349 C 3.5 0.347 3.5 0.346 3.5 0.345 C 3.5 0.343 3.5 0.342 3.5 0.341 C 3.5 0.34 3.5 0.338 3.5 0.337 C 3.5 0.336 3.5 0.334 3.5 0.333 C 3.5 0.332 3.5 0.331 3.5 0.329 C 3.5 0.328 3.5 0.327 3.5 0.326 C 3.5 0.324 3.5 0.323 3.5 0.322 C 3.5 0.321 3.5 0.319 3.5 0.318 C 3.5 0.317 3.5 0.316 3.5 0.314 C 3.5 0.313 3.5 0.312 3.5 0.311 C 3.5 0.309 3.5 0.308 3.5 0.307 C 3.5 0.306 3.5 0.304 3.5 0.303 C 3.5 0.302 3.5 0.301 3.5 0.3 C 3.5 0.298 3.5 0.297 3.5 0.296 C 3.5 0.295 3.5 0.293 3.5 0.292 C 3.5 0.291 3.5 0.29 3.5 0.289 C 3.5 0.287 3.5 0.286 3.5 0.285 C 3.5 0.284 3.5 0.283 3.5 0.281 C 3.5 0.28 3.5 0.279 3.5 0.278 C 3.5 0.277 3.5 0.276 3.5 0.274 C 3.5 0.273 3.5 0.272 3.5 0.271 C 3.5 0.27 3.5 0.268 3.5 0.267 C 3.5 0.266 3.5 0.265 3.5 0.264 C 3.5 0.263 3.5 0.261 3.5 0.26 C 3.5 0.259 3.5 0.258 3.5 0.257 C 3.5 0.256 3.5 0.255 3.5 0.253 C 3.5 0.252 3.5 0.251 3.5 0.25 C 3.5 0.249 3.5 0.248 3.5 0.247 C 3.5 0.245 3.5 0.244 3.5 0.243 C 3.5 0.242 3.5 0.241 3.5 0.24 C 3.5 0.239 3.5 0.238 3.5 0.237 C 3.5 0.235 3.5 0.234 3.5 0.233 C 3.5 0.232 3.5 0.231 3.5 0.23 C 3.5 0.229 3.5 0.228 3.5 0.227 C 3.5 0.226 3.5 0.224 3.5 0.223 C 3.5 0.222 3.5 0.221 3.5 0.22 C 3.5 0.219 3.5 0.218 3.5 0.217 C 3.5 0.216 3.5 0.215 3.5 0.214 C 3.5 0.213 3.5 0.212 3.5 0.21 C 3.5 0.209 3.5 0.208 3.5 0.207 C 3.5 0.206 3.5 0.205 3.5 0.204 C 3.5 0.203 3.5 0.202 3.5 0.201 C 3.5 0.2 3.5 0.199 3.5 0.198 C 3.5 0.197 3.5 0.196 3.5 0.195 C 3.5 0.194 3.5 0.193 3.5 0.192 C 3.5 0.191 3.5 0.19 3.5 0.189 C 3.5 0.188 3.5 0.187 3.5 0.186 C 3.5 0.185 3.5 0.184 3.5 0.183 C 3.5 0.182 3.5 0.181 3.5 0.18 C 3.5 0.179 3.5 0.178 3.5 0.177 C 3.5 0.176 3.5 0.175 3.5 0.174 C 3.5 0.173 3.5 0.172 3.5 0.171 C 3.5 0.17 3.5 0.169 3.5 0.168 C 3.5 0.167 3.5 0.166 3.5 0.165 C 3.5 0.164 3.5 0.163 3.5 0.162 C 3.5 0.161 3.5 0.16 3.5 0.159 C 3.5 0.158 3.5 0.158 3.5 0.157 C 3.5 0.156 3.5 0.155 3.5 0.154 C 3.5 0.153 3.5 0.152 3.5 0.151 C 3.5 0.15 3.5 0.149 3.5 0.148 C 3.5 0.147 3.5 0.146 3.5 0.146 C 3.5 0.145 3.5 0.144 3.5 0.143 C 3.5 0.142 3.5 0.141 3.5 0.14 C 3.5 0.139 3.5 0.138 3.5 0.137 C 3.5 0.137 3.5 0.136 3.5 0.135 C 3.5 0.134 3.5 0.133 3.5 0.132 C 3.5 0.131 3.5 0.13 3.5 0.13 C 3.5 0.129 3.5 0.128 3.5 0.127 C 3.5 0.126 3.5 0.125 3.5 0.125 C 3.5 0.124 3.5 0.123 3.5 0.122 C 3.5 0.121 3.5 0.12 3.5 0.119 C 3.5 0.119 3.5 0.118 3.5 0.117 C 3.5 0.116 3.5 0.115 3.5 0.115 C 3.5 0.114 3.5 0.113 3.5 0.112 C 3.5 0.111 3.5 0.11 3.5 0.11 C 3.5 0.109 3.5 0.108 3.5 0.107 C 3.5 0.107 3.5 0.106 3.5 0.105 C 3.5 0.104 3.5 0.103 3.5 0.103 C 3.5 0.102 3.5 0.101 3.5 0.1 C 3.5 0.1 3.5 0.099 3.5 0.098 C 3.5 0.097 3.5 0.096 3.5 0.096 C 3.5 0.095 3.5 0.094 3.5 0.093 C 3.5 0.093 3.5 0.092 3.5 0.091 C 3.5 0.091 3.5 0.09 3.5 0.089 C 3.5 0.088 3.5 0.088 3.5 0.087 C 3.5 0.086 3.5 0.085 3.5 0.085 C 3.5 0.084 3.5 0.083 3.5 0.083 C 3.5 0.082 3.5 0.081 3.5 0.081 C 3.5 0.08 3.5 0.079 3.5 0.078 C 3.5 0.078 3.5 0.077 3.5 0.076 C 3.5 0.076 3.5 0.075 3.5 0.074 C 3.5 0.074 3.5 0.073 3.5 0.072 C 3.5 0.072 3.5 0.071 3.5 0.07 C 3.5 0.07 3.5 0.069 3.5 0.068 C 3.5 0.068 3.5 0.067 3.5 0.067 C 3.5 0.066 3.5 0.065 3.5 0.065 C 3.5 0.064 3.5 0.063 3.5 0.063 C 3.5 0.062 3.5 0.061 3.5 0.061 C 3.5 0.06 3.5 0.06 3.5 0.059 C 3.5 0.058 3.5 0.058 3.5 0.057 C 3.5 0.057 3.5 0.056 3.5 0.055 C 3.5 0.055 3.5 0.054 3.5 0.054 C 3.5 0.053 3.5 0.053 3.5 0.052 C 3.5 0.051 3.5 0.051 3.5 0.05 C 3.5 0.05 3.5 0.049 3.5 0.049 C 3.5 0.048 3.5 0.048 3.5 0.047 C 3.5 0.046 3.5 0.046 3.5 0.045 C 3.5 0.045 3.5 0.044 3.5 0.044 C 3.5 0.043 3.5 0.043 3.5 0.042 C 3.5 0.042 3.5 0.041 3.5 0.041 C 3.5 0.04 3.5 0.04 3.5 0.039 C 3.5 0.039 3.5 0.038 3.5 0.038 C 3.5 0.037 3.5 0.037 3.5 0.036 C 3.5 0.036 3.5 0.035 3.5 0.035 C 3.5 0.034 3.5 0.034 3.5 0.033 C 3.5 0.033 3.5 0.033 3.5 0.032 C 3.5 0.032 3.5 0.031 3.5 0.031 C 3.5 0.03 3.5 0.03 3.5 0.029 C 3.5 0.029 3.5 0.029 3.5 0.028 C 3.5 0.028 3.5 0.027 3.5 0.027 C 3.5 0.026 3.5 0.026 3.5 0.026 C 3.5 0.025 3.5 0.025 3.5 0.024 C 3.5 0.024 3.5 0.024 3.5 0.023 C 3.5 0.023 3.5 0.022 3.5 0.022 C 3.5 0.022 3.5 0.021 3.5 0.021 C 3.5 0.021 3.5 0.02 3.5 0.02 C 3.5 0.019 3.5 0.019 3.5 0.019 C 3.5 0.018 3.5 0.018 3.5 0.018 C 3.5 0.017 3.5 0.017 3.5 0.017 C 3.5 0.016 3.5 0.016 3.5 0.016 C 3.5 0.015 3.5 0.015 3.5 0.015 C 3.5 0.014 3.5 0.014 3.5 0.014 C 3.5 0.014 3.5 0.013 3.5 0.013 C 3.5 0.013 3.5 0.012 3.5 0.012 C 3.5 0.012 3.5 0.012 3.5 0.011 C 3.5 0.011 3.5 0.011 3.5 0.01 C 3.5 0.01 3.5 0.01 3.5 0.01 C 3.5 0.009 3.5 0.009 3.5 0.009 C 3.5 0.009 3.5 0.008 3.5 0.008 C 3.5 0.008 3.5 0.008 3.5 0.008 C 3.5 0.007 3.5 0.007 3.5 0.007 C 3.5 0.007 3.5 0.006 3.5 0.006 C 3.5 0.006 3.5 0.006 3.5 0.006 C 3.5 0.005 3.5 0.005 3.5 0.005 C 3.5 0.005 3.5 0.005 3.5 0.005 C 3.5 0.004 3.5 0.004 3.5 0.004 C 3.5 0.004 3.5 0.004 3.5 0.004 C 3.5 0.003 3.5 0.003 3.5 0.003 C 3.5 0.003 3.5 0.003 3.5 0.003 C 3.5 0.003 3.5 0.002 3.5 0.002 C 3.5 0.002 3.5 0.002 3.5 0.002 C 3.5 0.002 3.5 0.002 3.5 0.002 C 3.5 0.001 3.5 0.001 3.5 0.001 C 3.5 0.001 3.5 0.001 3.5 0.001 C 3.5 0.001 3.5 0.001 3.5 0.001 C 3.5 0.001 3.5 0.001 3.5 0.001 C 3.5 0.001 3.5 0 3.5 0 C 3.5 0 3.5 0 3.5 0 C 3.5 0 3.5 0 3.5 0 C 3.5 0 3.5 0 3.5 0 C 3.5 0 3.5 0 3.5 0 C 3.5 0 3.5 0 3 0 C 2.5 0 2.5 0 2.5 0 C 2.5 0 2.5 0 2.5 0 C 2.5 0 2.5 0 2.5 0 C 2.5 0 2.5 0 2.5 0 C 2.5 0 2.5 0 2.5 0 C 2.5 0 2.5 0.001 2.5 0.001 C 2.5 0.001 2.5 0.001 2.5 0.001 C 2.5 0.001 2.5 0.001 2.5 0.001 C 2.5 0.001 2.5 0.001 2.5 0.001 C 2.5 0.001 2.5 0.001 2.5 0.002 C 2.5 0.002 2.5 0.002 2.5 0.002 C 2.5 0.002 2.5 0.002 2.5 0.002 C 2.5 0.002 2.5 0.003 2.5 0.003 C 2.5 0.003 2.5 0.003 2.5 0.003 C 2.5 0.003 2.5 0.003 2.5 0.004 C 2.5 0.004 2.5 0.004 2.5 0.004 C 2.5 0.004 2.5 0.004 2.5 0.005 C 2.5 0.005 2.5 0.005 2.5 0.005 C 2.5 0.005 2.5 0.005 2.5 0.006 C 2.5 0.006 2.5 0.006 2.5 0.006 C 2.5 0.006 2.5 0.007 2.5 0.007 C 2.5 0.007 2.5 0.007 2.5 0.008 C 2.5 0.008 2.5 0.008 2.5 0.008 C 2.5 0.008 2.5 0.009 2.5 0.009 C 2.5 0.009 2.5 0.009 2.5 0.01 C 2.5 0.01 2.5 0.01 2.5 0.01 C 2.5 0.011 2.5 0.011 2.5 0.011 C 2.5 0.012 2.5 0.012 2.5 0.012 C 2.5 0.012 2.5 0.013 2.5 0.013 C 2.5 0.013 2.5 0.014 2.5 0.014 C 2.5 0.014 2.5 0.014 2.5 0.015 C 2.5 0.015 2.5 0.015 2.5 0.016 C 2.5 0.016 2.5 0.016 2.5 0.017 C 2.5 0.017 2.5 0.017 2.5 0.018 C 2.5 0.018 2.5 0.018 2.5 0.019 C 2.5 0.019 2.5 0.019 2.5 0.02 C 2.5 0.02 2.5 0.021 2.5 0.021 C 2.5 0.021 2.5 0.022 2.5 0.022 C 2.5 0.022 2.5 0.023 2.5 0.023 C 2.5 0.024 2.5 0.024 2.5 0.024 C 2.5 0.025 2.5 0.025 2.5 0.026 C 2.5 0.026 2.5 0.026 2.5 0.027 C 2.5 0.027 2.5 0.028 2.5 0.028 C 2.5 0.029 2.5 0.029 2.5 0.029 C 2.5 0.03 2.5 0.03 2.5 0.031 C 2.5 0.031 2.5 0.032 2.5 0.032 C 2.5 0.033 2.5 0.033 2.5 0.033 C 2.5 0.034 2.5 0.034 2.5 0.035 C 2.5 0.035 2.5 0.036 2.5 0.036 C 2.5 0.037 2.5 0.037 2.5 0.038 C 2.5 0.038 2.5 0.039 2.5 0.039 C 2.5 0.04 2.5 0.04 2.5 0.041 C 2.5 0.041 2.5 0.042 2.5 0.042 C 2.5 0.043 2.5 0.043 2.5 0.044 C 2.5 0.044 2.5 0.045 2.5 0.045 C 2.5 0.046 2.5 0.046 2.5 0.047 C 2.5 0.048 2.5 0.048 2.5 0.049 C 2.5 0.049 2.5 0.05 2.5 0.05 C 2.5 0.051 2.5 0.051 2.5 0.052 C 2.5 0.053 2.5 0.053 2.5 0.054 C 2.5 0.054 2.5 0.055 2.5 0.055 C 2.5 0.056 2.5 0.057 2.5 0.057 C 2.5 0.058 2.5 0.058 2.5 0.059 C 2.5 0.06 2.5 0.06 2.5 0.061 C 2.5 0.061 2.5 0.062 2.5 0.063 C 2.5 0.063 2.5 0.064 2.5 0.065 C 2.5 0.065 2.5 0.066 2.5 0.067 C 2.5 0.067 2.5 0.068 2.5 0.068 C 2.5 0.069 2.5 0.07 2.5 0.07 C 2.5 0.071 2.5 0.072 2.5 0.072 C 2.5 0.073 2.5 0.074 2.5 0.074 C 2.5 0.075 2.5 0.076 2.5 0.076 C 2.5 0.077 2.5 0.078 2.5 0.078 C 2.5 0.079 2.5 0.08 2.5 0.081 C 2.5 0.081 2.5 0.082 2.5 0.083 C 2.5 0.083 2.5 0.084 2.5 0.085 C 2.5 0.085 2.5 0.086 2.5 0.087 C 2.5 0.088 2.5 0.088 2.5 0.089 C 2.5 0.09 2.5 0.091 2.5 0.091 C 2.5 0.092 2.5 0.093 2.5 0.093 C 2.5 0.094 2.5 0.095 2.5 0.096 C 2.5 0.096 2.5 0.097 2.5 0.098 C 2.5 0.099 2.5 0.1 2.5 0.1 C 2.5 0.101 2.5 0.102 2.5 0.103 C 2.5 0.103 2.5 0.104 2.5 0.105 C 2.5 0.106 2.5 0.107 2.5 0.107 C 2.5 0.108 2.5 0.109 2.5 0.11 C 2.5 0.11 2.5 0.111 2.5 0.112 C 2.5 0.113 2.5 0.114 2.5 0.115 C 2.5 0.115 2.5 0.116 2.5 0.117 C 2.5 0.118 2.5 0.119 2.5 0.119 C 2.5 0.12 2.5 0.121 2.5 0.122 C 2.5 0.123 2.5 0.124 2.5 0.125 C 2.5 0.125 2.5 0.126 2.5 0.127 C 2.5 0.128 2.5 0.129 2.5 0.13 C 2.5 0.13 2.5 0.131 2.5 0.132 C 2.5 0.133 2.5 0.134 2.5 0.135 C 2.5 0.136 2.5 0.137 2.5 0.137 C 2.5 0.138 2.5 0.139 2.5 0.14 C 2.5 0.141 2.5 0.142 2.5 0.143 C 2.5 0.144 2.5 0.145 2.5 0.146 C 2.5 0.146 2.5 0.147 2.5 0.148 C 2.5 0.149 2.5 0.15 2.5 0.151 C 2.5 0.152 2.5 0.153 2.5 0.154 C 2.5 0.155 2.5 0.156 2.5 0.157 C 2.5 0.158 2.5 0.158 2.5 0.159 C 2.5 0.16 2.5 0.161 2.5 0.162 C 2.5 0.163 2.5 0.164 2.5 0.165 C 2.5 0.166 2.5 0.167 2.5 0.168 C 2.5 0.169 2.5 0.17 2.5 0.171 C 2.5 0.172 2.5 0.173 2.5 0.174 C 2.5 0.175 2.5 0.176 2.5 0.177 C 2.5 0.178 2.5 0.179 2.5 0.18 C 2.5 0.181 2.5 0.182 2.5 0.183 C 2.5 0.184 2.5 0.185 2.5 0.186 C 2.5 0.187 2.5 0.188 2.5 0.189 C 2.5 0.19 2.5 0.191 2.5 0.192 C 2.5 0.193 2.5 0.194 2.5 0.195 C 2.5 0.196 2.5 0.197 2.5 0.198 C 2.5 0.199 2.5 0.2 2.5 0.201 C 2.5 0.202 2.5 0.203 2.5 0.204 C 2.5 0.205 2.5 0.206 2.5 0.207 C 2.5 0.208 2.5 0.209 2.5 0.21 C 2.5 0.212 2.5 0.213 2.5 0.214 C 2.5 0.215 2.5 0.216 2.5 0.217 C 2.5 0.218 2.5 0.219 2.5 0.22 C 2.5 0.221 2.5 0.222 2.5 0.223 C 2.5 0.224 2.5 0.226 2.5 0.227 C 2.5 0.228 2.5 0.229 2.5 0.23 C 2.5 0.231 2.5 0.232 2.5 0.233 C 2.5 0.234 2.5 0.235 2.5 0.237 C 2.5 0.238 2.5 0.239 2.5 0.24 C 2.5 0.241 2.5 0.242 2.5 0.243 C 2.5 0.244 2.5 0.245 2.5 0.247 C 2.5 0.248 2.5 0.249 2.5 0.25 C 2.5 0.251 2.5 0.252 2.5 0.253 C 2.5 0.255 2.5 0.256 2.5 0.257 C 2.5 0.258 2.5 0.259 2.5 0.26 C 2.5 0.261 2.5 0.263 2.5 0.264 C 2.5 0.265 2.5 0.266 2.5 0.267 C 2.5 0.268 2.5 0.27 2.5 0.271 C 2.5 0.272 2.5 0.273 2.5 0.274 C 2.5 0.276 2.5 0.277 2.5 0.278 C 2.5 0.279 2.5 0.28 2.5 0.281 C 2.5 0.283 2.5 0.284 2.5 0.285 C 2.5 0.286 2.5 0.287 2.5 0.289 C 2.5 0.29 2.5 0.291 2.5 0.292 C 2.5 0.293 2.5 0.295 2.5 0.296 C 2.5 0.297 2.5 0.298 2.5 0.3 C 2.5 0.301 2.5 0.302 2.5 0.303 C 2.5 0.304 2.5 0.306 2.5 0.307 C 2.5 0.308 2.5 0.309 2.5 0.311 C 2.5 0.312 2.5 0.313 2.5 0.314 C 2.5 0.316 2.5 0.317 2.5 0.318 C 2.5 0.319 2.5 0.321 2.5 0.322 C 2.5 0.323 2.5 0.324 2.5 0.326 C 2.5 0.327 2.5 0.328 2.5 0.329 C 2.5 0.331 2.5 0.332 2.5 0.333 C 2.5 0.334 2.5 0.336 2.5 0.337 C 2.5 0.338 2.5 0.34 2.5 0.341 C 2.5 0.342 2.5 0.343 2.5 0.345 C 2.5 0.346 2.5 0.347 2.5 0.349 C 2.5 0.35 2.5 0.351 2.5 0.352 C 2.5 0.354 2.5 0.355 2.5 0.356 C 2.5 0.358 2.5 0.359 2.5 0.36 C 2.5 0.362 2.5 0.363 2.5 0.364 C 2.5 0.366 2.5 0.367 2.5 0.368 C 2.5 0.369 2.5 0.371 2.5 0.372 C 2.5 0.373 2.5 0.375 2.5 0.376 C 2.5 0.377 2.5 0.379 2.5 0.38 C 2.5 0.381 2.5 0.383 2.5 0.384 C 2.5 0.385 2.5 0.387 2.5 0.388 C 2.5 0.389 2.5 0.391 2.5 0.392 C 2.5 0.394 2.5 0.395 2.5 0.396 C 2.5 0.398 2.5 0.399 2.5 0.4 C 2.5 0.402 2.5 0.403 2.5 0.404 C 2.5 0.406 2.5 0.407 2.5 0.409 C 2.5 0.41 2.5 0.411 2.5 0.413 C 2.5 0.414 2.5 0.415 2.5 0.417 C 2.5 0.418 2.5 0.42 2.5 0.421 C 2.5 0.422 2.5 0.424 2.5 0.425 C 2.5 0.426 2.5 0.428 2.5 0.429 C 2.5 0.431 2.5 0.432 2.5 0.433 C 2.5 0.435 2.5 0.436 2.5 0.438 C 2.5 0.439 2.5 0.44 2.5 0.442 C 2.5 0.443 2.5 0.445 2.5 0.446 C 2.5 0.447 2.5 0.449 2.5 0.45 C 2.5 0.452 2.5 0.453 2.5 0.455 C 2.5 0.456 2.5 0.457 2.5 0.459 C 2.5 0.46 2.5 0.462 2.5 0.463 C 2.5 0.465 2.5 0.466 2.5 0.467 C 2.5 0.469 2.5 0.47 2.5 0.472 C 2.5 0.473 2.5 0.475 2.5 0.476 C 2.5 0.478 2.5 0.479 2.5 0.48 C 2.5 0.482 2.5 0.483 2.5 0.485 C 2.5 0.486 2.5 0.488 2.5 0.489 C 2.5 0.491 2.5 0.492 2.5 0.494 C 2.5 0.495 2.5 0.496 2.5 0.498 C 2.5 0.499 2.5 0.501 2.5 0.502 C 2.5 0.504 2.5 0.505 2.5 0.507 C 2.5 0.508 2.5 0.51 2.5 0.511 C 2.5 0.513 2.5 0.514 2.5 0.516 C 2.5 0.517 2.5 0.519 2.5 0.52 C 2.5 0.522 2.5 0.523 2.5 0.524 C 2.5 0.526 2.5 0.527 2.5 0.529 C 2.5 0.53 2.5 0.532 2.5 0.533 C 2.5 0.535 2.5 0.536 2.5 0.538 C 2.5 0.539 2.5 0.541 2.5 0.542 C 2.5 0.544 2.5 0.545 2.5 0.547 C 2.5 0.549 2.5 0.55 2.5 0.552 C 2.5 0.553 2.5 0.555 2.5 0.556 C 2.5 0.558 2.5 0.559 2.5 0.561 C 2.5 0.562 2.5 0.564 2.5 0.565 C 2.5 0.567 2.5 0.568 2.5 0.57 C 2.5 0.571 2.5 0.573 2.5 0.574 C 2.5 0.576 2.5 0.577 2.5 0.579 C 2.5 0.581 2.5 0.582 2.5 0.584 C 2.5 0.585 2.5 0.587 2.5 0.588 C 2.5 0.59 2.5 0.591 2.5 0.593 C 2.5 0.594 2.5 0.596 2.5 0.597 C 2.5 0.599 2.5 0.601 2.5 0.602 C 2.5 0.604 2.5 0.605 2.5 0.607 C 2.5 0.608 2.5 0.61 2.5 0.611 C 2.5 0.613 2.5 0.615 2.5 0.616 C 2.5 0.618 2.5 0.619 2.5 0.621 C 2.5 0.622 2.5 0.624 2.5 0.626 C 2.5 0.627 2.5 0.629 2.5 0.63 C 2.5 0.632 2.5 0.633 2.5 0.635 C 2.5 0.637 2.5 0.638 2.5 0.64 C 2.5 0.641 2.5 0.643 2.5 0.645 C 2.5 0.646 2.5 0.648 2.5 0.649 C 2.5 0.651 2.5 0.652 2.5 0.654 C 2.5 0.656 2.5 0.657 2.5 0.659 C 2.5 0.66 2.5 0.662 2.5 0.664 C 2.5 0.665 2.5 0.667 2.5 0.668 C 2.5 0.67 2.5 0.672 2.5 0.673 C 2.5 0.675 2.5 0.676 2.5 0.678 C 2.5 0.68 2.5 0.681 2.5 0.683 C 2.5 0.684 2.5 0.686 2.5 0.688 C 2.5 0.689 2.5 0.691 2.5 0.692 C 2.5 0.694 2.5 0.696 2.5 0.697 C 2.5 0.699 2.5 0.701 2.5 0.702 C 2.5 0.704 2.5 0.705 2.5 0.707 C 2.5 0.709 2.5 0.71 2.5 0.712 C 2.5 0.714 2.5 0.715 2.5 0.717 C 2.5 0.718 2.5 0.72 2.5 0.722 C 2.5 0.723 2.5 0.725 2.5 0.727 C 2.5 0.728 2.5 0.73 2.5 0.731 C 2.5 0.733 2.5 0.735 2.5 0.736 C 2.5 0.738 2.5 0.74 2.5 0.741 C 2.5 0.743 2.5 0.745 2.5 0.746 C 2.5 0.748 2.5 0.749 2.5 0.751 C 2.5 0.753 2.5 0.754 2.5 0.756 C 2.5 0.758 2.5 0.759 2.5 0.761 C 2.5 0.763 2.5 0.764 2.5 0.766 C 2.5 0.768 2.5 0.769 2.5 0.771 C 2.5 0.773 2.5 0.774 2.5 0.776 C 2.5 0.778 2.5 0.779 2.5 0.781 C 2.5 0.783 2.5 0.784 2.5 0.786 C 2.5 0.788 2.5 0.789 2.5 0.791 C 2.5 0.793 2.5 0.794 2.5 0.796 C 2.5 0.798 2.5 0.799 2.5 0.801 C 2.5 0.803 2.5 0.804 2.5 0.806 C 2.5 0.808 2.5 0.809 2.5 0.811 C 2.5 0.813 2.5 0.814 2.5 0.816 C 2.5 0.818 2.5 0.819 2.5 0.821 C 2.5 0.823 2.5 0.824 2.5 0.826 C 2.5 0.828 2.5 0.829 2.5 0.831 C 2.5 0.833 2.5 0.834 2.5 0.836 C 2.5 0.838 2.5 0.839 2.5 0.841 C 2.5 0.843 2.5 0.844 2.5 0.846 C 2.5 0.848 2.5 0.85 2.5 0.851 C 2.5 0.853 2.5 0.855 2.5 0.856 C 2.5 0.858 2.5 0.86 2.5 0.861 C 2.5 0.863 2.5 0.865 2.5 0.866 C 2.5 0.868 2.5 0.87 2.5 0.871 C 2.5 0.873 2.5 0.875 2.5 0.877 C 2.5 0.878 2.5 0.88 2.5 0.882 C 2.5 0.883 2.5 0.885 2.5 0.887 C 2.5 0.888 2.5 0.89 2.5 0.892 C 2.5 0.894 2.5 0.895 2.5 0.897 C 2.5 0.899 2.5 0.9 2.5 0.902 C 2.5 0.904 2.5 0.906 2.5 0.907 C 2.5 0.909 2.5 0.911 2.5 0.912 C 2.5 0.914 2.5 0.916 2.5 0.917 C 2.5 0.919 2.5 0.921 2.5 0.923 C 2.5 0.924 2.5 0.926 2.5 0.928 C 2.5 0.929 2.5 0.931 2.5 0.933 C 2.5 0.935 2.5 0.936 2.5 0.938 C 2.5 0.94 2.5 0.941 2.5 0.943 C 2.5 0.945 2.5 0.947 2.5 0.948 C 2.5 0.95 2.5 0.952 2.5 0.953 C 2.5 0.955 2.5 0.957 2.5 0.959 C 2.5 0.96 2.5 0.962 2.5 0.964 C 2.5 0.965 2.5 0.967 2.5 0.969 C 2.5 0.971 2.5 0.972 2.5 0.974 C 2.5 0.976 2.5 0.978 2.5 0.979 C 2.5 0.981 2.5 0.983 2.5 0.984 C 2.5 0.986 2.5 0.988 2.5 0.99 C 2.5 0.991 2.5 0.993 2.5 0.995 C 2.5 0.996 2.5 0.998 2.5 1 C 2.5 1.002 2.5 1.003 2.5 1.005 C 2.5 1.007 2.5 1.009 2.5 1.01 C 2.5 1.012 2.5 1.014 2.5 1.016 C 2.5 1.017 2.5 1.019 2.5 1.021 C 2.5 1.022 2.5 1.024 2.5 1.026 C 2.5 1.028 2.5 1.029 2.5 1.031 C 2.5 1.033 2.5 1.035 2.5 1.036 C 2.5 1.038 2.5 1.04 2.5 1.041 C 2.5 1.043 2.5 1.045 2.5 1.047 C 2.5 1.048 2.5 1.05 2.5 1.052 C 2.5 1.054 2.5 1.055 2.5 1.057 C 2.5 1.059 2.5 1.061 2.5 1.062 C 2.5 1.064 2.5 1.066 2.5 1.067 C 2.5 1.069 2.5 1.071 2.5 1.073 C 2.5 1.074 2.5 1.076 2.5 1.078 C 2.5 1.08 2.5 1.081 2.5 1.083 C 2.5 1.085 2.5 1.087 2.5 1.088 C 2.5 1.09 2.5 1.092 2.5 1.094 C 2.5 1.095 2.5 1.097 2.5 1.099 C 2.5 1.1 2.5 1.102 2.5 1.104 C 2.5 1.106 2.5 1.107 2.5 1.109 C 2.5 1.111 2.5 1.113 2.5 1.114 C 2.5 1.116 2.5 1.118 2.5 1.12 C 2.5 1.121 2.5 1.123 2.5 1.125 C 2.5 1.127 2.5 1.128 2.5 1.13 C 2.5 1.132 2.5 1.134 2.5 1.135 C 2.5 1.137 2.5 1.139 2.5 1.14 C 2.5 1.142 2.5 1.144 2.5 1.146 C 2.5 1.147 2.5 1.149 2.5 1.151 C 2.5 1.153 2.5 1.154 2.5 1.156 C 2.5 1.158 2.5 1.16 2.5 1.161 C 2.5 1.163 2.5 1.165 2.5 1.167 C 2.5 1.168 2.5 1.17 2.5 1.172 C 2.5 1.174 2.5 1.175 2.5 1.177 C 2.5 1.179 2.5 1.18 2.5 1.182 C 2.5 1.184 2.5 1.186 2.5 1.187 C 2.5 1.189 2.5 1.191 2.5 1.193 C 2.5 1.194 2.5 1.196 2.5 1.198 C 2.5 1.2 2.5 1.201 2.5 1.203 C 2.5 1.205 2.5 1.207 2.5 1.208 C 2.5 1.21 2.5 1.212 2.5 1.213 C 2.5 1.215 2.5 1.217 2.5 1.219 C 2.5 1.22 2.5 1.222 2.5 1.224 C 2.5 1.226 2.5 1.227 2.5 1.229 C 2.5 1.231 2.5 1.233 2.5 1.234 C 2.5 1.236 2.5 1.238 2.5 1.239 C 2.5 1.241 2.5 1.243 2.5 1.245 C 2.5 1.246 2.5 1.248 2.5 1.25 C 2.5 1.252 2.5 1.253 2.5 1.255 C 2.5 1.257 2.5 1.259 2.5 1.26 C 2.5 1.262 2.5 1.264 2.5 1.265 C 2.5 1.267 2.5 1.269 2.5 1.271 C 2.5 1.272 2.5 1.274 2.5 1.276 C 2.5 1.278 2.5 1.279 2.5 1.281 C 2.5 1.283 2.5 1.284 2.5 1.286 C 2.5 1.288 2.5 1.29 2.5 1.291 C 2.5 1.293 2.5 1.295 2.5 1.297 C 2.5 1.298 2.5 1.3 2.5 1.302 C 2.5 1.303 2.5 1.305 2.5 1.307 C 2.5 1.309 2.5 1.31 2.5 1.312 C 2.5 1.314 2.5 1.315 2.5 1.317 C 2.5 1.319 2.5 1.321 2.5 1.322 C 2.5 1.324 2.5 1.326 2.5 1.328 C 2.5 1.329 2.5 1.331 2.5 1.333 C 2.5 1.334 2.5 1.336 2.5 1.338 C 2.5 1.34 2.5 1.341 2.5 1.343 C 2.5 1.345 2.5 1.346 2.5 1.348 C 2.5 1.35 2.5 1.352 2.5 1.353 C 2.5 1.355 2.5 1.357 2.5 1.358 C 2.5 1.36 2.5 1.362 2.5 1.364 C 2.5 1.365 2.5 1.367 2.5 1.369 C 2.5 1.37 2.5 1.372 2.5 1.374 C 2.5 1.375 2.5 1.377 2.5 1.379 C 2.5 1.381 2.5 1.382 2.5 1.384 C 2.5 1.386 2.5 1.387 2.5 1.389 C 2.5 1.391 2.5 1.393 2.5 1.394 C 2.5 1.396 2.5 1.398 2.5 1.399 C 2.5 1.401 2.5 1.403 2.5 1.404 C 2.5 1.406 2.5 1.408 2.5 1.409 C 2.5 1.411 2.5 1.413 2.5 1.415 C 2.5 1.416 2.5 1.418 2.5 1.42 C 2.5 1.421 2.5 1.423 2.5 1.425 C 2.5 1.426 2.5 1.428 2.5 1.43 C 2.5 1.431 2.5 1.433 2.5 1.435 C 2.5 1.437 2.5 1.438 2.5 1.44 C 2.5 1.442 2.5 1.443 2.5 1.445 C 2.5 1.447 2.5 1.448 2.5 1.45 C 2.5 1.452 2.5 1.453 2.5 1.455 C 2.5 1.457 2.5 1.458 2.5 1.46 C 2.5 1.462 2.5 1.463 2.5 1.465 C 2.5 1.467 2.5 1.468 2.5 1.47 C 2.5 1.472 2.5 1.473 2.5 1.475 C 2.5 1.477 2.5 1.479 2.5 1.48 C 2.5 1.482 2.5 1.484 2.5 1.485 C 2.5 1.487 2.5 1.489 2.5 1.49 C 2.5 1.492 2.5 1.493 2.5 1.495 C 2.5 1.497 2.5 1.498 2.5 1.5 C 2.5 1.502 2.5 1.503 2.5 1.505 C 2.5 1.507 2.5 1.508 2.5 1.51 C 2.5 1.512 2.5 1.513 2.5 1.515 C 2.5 1.517 2.5 1.518 2.5 1.52 C 2.5 1.522 2.5 1.523 2.5 1.525 C 2.5 1.527 2.5 1.528 2.5 1.53 C 2.5 1.532 2.5 1.533 2.5 1.535 C 2.5 1.536 2.5 1.538 2.5 1.54 C 2.5 1.541 2.5 1.543 2.5 1.545 C 2.5 1.546 2.5 1.548 2.5 1.55 C 2.5 1.551 2.5 1.553 2.5 1.554 C 2.5 1.556 2.5 1.558 2.5 1.559 C 2.5 1.561 2.5 1.563 2.5 1.564 C 2.5 1.566 2.5 1.568 2.5 1.569 C 2.5 1.571 2.5 1.572 2.5 1.574 C 2.5 1.576 2.5 1.577 2.5 1.579 C 2.5 1.581 2.5 1.582 2.5 1.584 C 2.5 1.585 2.5 1.587 2.5 1.589 C 2.5 1.59 2.5 1.592 2.5 1.593 C 2.5 1.595 2.5 1.597 2.5 1.598 C 2.5 1.6 2.5 1.601 2.5 1.603 C 2.5 1.605 2.5 1.606 2.5 1.608 C 2.5 1.61 2.5 1.611 2.5 1.613 C 2.5 1.614 2.5 1.616 2.5 1.618 C 2.5 1.619 2.5 1.621 2.5 1.622 C 2.5 1.624 2.5 1.625 2.5 1.627 C 2.5 1.629 2.5 1.63 2.5 1.632 C 2.5 1.633 2.5 1.635 2.5 1.637 C 2.5 1.638 2.5 1.64 2.5 1.641 C 2.5 1.643 2.5 1.644 2.5 1.646 C 2.5 1.648 2.5 1.649 2.5 1.651 C 2.5 1.652 2.5 1.654 2.5 1.656 C 2.5 1.657 2.5 1.659 2.5 1.66 C 2.5 1.662 2.5 1.663 2.5 1.665 C 2.5 1.666 2.5 1.668 2.5 1.67 C 2.5 1.671 2.5 1.673 2.5 1.674 C 2.5 1.676 2.5 1.677 2.5 1.679 C 2.5 1.681 2.5 1.682 2.5 1.684 C 2.5 1.685 2.5 1.687 2.5 1.688 C 2.5 1.69 2.5 1.691 2.5 1.693 C 2.5 1.694 2.5 1.696 2.5 1.698 C 2.5 1.699 2.5 1.701 2.5 1.702 C 2.5 1.704 2.5 1.705 2.5 1.707 C 2.5 1.708 2.5 1.71 2.5 1.711 C 2.5 1.713 2.5 1.714 2.5 1.716 C 2.5 1.717 2.5 1.719 2.5 1.721 C 2.5 1.722 2.5 1.724 2.5 1.725 C 2.5 1.727 2.5 1.728 2.5 1.73 C 2.5 1.731 2.5 1.733 2.5 1.734 C 2.5 1.736 2.5 1.737 2.5 1.739 C 2.5 1.74 2.5 1.742 2.5 1.743 C 2.5 1.745 2.5 1.746 2.5 1.748 C 2.5 1.749 2.5 1.751 2.5 1.752 C 2.5 1.754 2.5 1.755 2.5 1.757 C 2.5 1.758 2.5 1.76 2.5 1.761 C 2.5 1.763 2.5 1.764 2.5 1.766 C 2.5 1.767 2.5 1.769 2.5 1.77 C 2.5 1.772 2.5 1.773 2.5 1.774 C 2.5 1.776 2.5 1.777 2.5 1.779 C 2.5 1.78 2.5 1.782 2.5 1.783 C 2.5 1.785 2.5 1.786 2.5 1.788 C 2.5 1.789 2.5 1.791 2.5 1.792 C 2.5 1.794 2.5 1.795 2.5 1.796 C 2.5 1.798 2.5 1.799 2.5 1.801 C 2.5 1.802 2.5 1.804 2.5 1.805 C 2.5 1.807 2.5 1.808 2.5 1.809 C 2.5 1.811 2.5 1.812 2.5 1.814 C 2.5 1.815 2.5 1.817 2.5 1.818 C 2.5 1.82 2.5 1.821 2.5 1.822 C 2.5 1.824 2.5 1.825 2.5 1.827 C 2.5 1.828 2.5 1.829 2.5 1.831 C 2.5 1.832 2.5 1.834 2.5 1.835 C 2.5 1.837 2.5 1.838 2.5 1.839 C 2.5 1.841 2.5 1.842 2.5 1.844 C 2.5 1.845 2.5 1.846 2.5 1.848 C 2.5 1.849 2.5 1.851 2.5 1.852 C 2.5 1.853 2.5 1.855 2.5 1.856 C 2.5 1.858 2.5 1.859 2.5 1.86 C 2.5 1.862 2.5 1.863 2.5 1.865 C 2.5 1.866 2.5 1.867 2.5 1.869 C 2.5 1.87 2.5 1.871 2.5 1.873 C 2.5 1.874 2.5 1.876 2.5 1.877 C 2.5 1.878 2.5 1.88 2.5 1.881 C 2.5 1.882 2.5 1.884 2.5 1.885 C 2.5 1.886 2.5 1.888 2.5 1.889 C 2.5 1.89 2.5 1.892 2.5 1.893 C 2.5 1.895 2.5 1.896 2.5 1.897 C 2.5 1.899 2.5 1.9 2.5 1.901 C 2.5 1.903 2.5 1.904 2.5 1.905 C 2.5 1.907 2.5 1.908 2.5 1.909 C 2.5 1.911 2.5 1.912 2.5 1.913 C 2.5 1.914 2.5 1.916 2.5 1.917 C 2.5 1.918 2.5 1.92 2.5 1.921 C 2.5 1.922 2.5 1.924 2.5 1.925 C 2.5 1.926 2.5 1.928 2.5 1.929 C 2.5 1.93 2.5 1.931 2.5 1.933 C 2.5 1.934 2.5 1.935 2.5 1.937 C 2.5 1.938 2.5 1.939 2.5 1.941 C 2.5 1.942 2.5 1.943 2.5 1.944 C 2.5 1.946 2.5 1.947 2.5 1.948 C 2.5 1.949 2.5 1.951 2.5 1.952 C 2.5 1.953 2.5 1.955 2.5 1.956 C 2.5 1.957 2.5 1.958 2.5 1.96 C 2.5 1.961 2.5 1.962 2.5 1.963 C 2.5 1.965 2.5 1.966 2.5 1.967 C 2.5 1.968 2.5 1.97 2.5 1.971 C 2.5 1.972 2.5 1.973 2.5 1.975 C 2.5 1.976 2.5 1.977 2.5 1.978 C 2.5 1.979 2.5 1.981 2.5 1.982 C 2.5 1.983 2.5 1.984 2.5 1.986 C 2.5 1.987 2.5 1.988 2.5 1.989 C 2.5 1.99 2.5 1.992 2.5 1.993 C 2.5 1.994 2.5 1.995 2.5 1.996 C 2.5 1.998 2.5 1.999 2.5 2 L 3.5 2 Z M 3 0.5 L 5 0.5 L 5 -0.5 L 3 -0.5 L 3 0.5 Z M 5 0.5 C 5.828 0.5 6.5 1.172 6.5 2 L 7.5 2 C 7.5 0.619 6.381 -0.5 5 -0.5 L 5 0.5 Z M 4.5 9 C 4.5 8.384 4.819 7.605 5.189 6.927 C 5.368 6.599 5.547 6.315 5.682 6.113 C 5.749 6.013 5.804 5.933 5.843 5.879 C 5.862 5.852 5.877 5.831 5.887 5.818 C 5.892 5.811 5.895 5.807 5.897 5.803 C 5.899 5.802 5.899 5.801 5.9 5.8 C 5.9 5.8 5.9 5.8 5.9 5.8 C 5.9 5.8 5.9 5.8 5.9 5.8 C 5.9 5.8 5.9 5.8 5.9 5.8 C 5.9 5.8 5.9 5.8 5.5 5.5 C 5.1 5.2 5.1 5.2 5.1 5.2 C 5.1 5.2 5.1 5.2 5.1 5.2 C 5.1 5.201 5.099 5.201 5.099 5.201 C 5.099 5.201 5.099 5.202 5.098 5.203 C 5.097 5.204 5.096 5.206 5.094 5.208 C 5.091 5.213 5.086 5.219 5.08 5.228 C 5.067 5.244 5.05 5.268 5.028 5.299 C 4.985 5.36 4.923 5.448 4.85 5.559 C 4.703 5.778 4.507 6.088 4.311 6.448 C 3.931 7.145 3.5 8.116 3.5 9 L 4.5 9 Z M 5.5 5.5 C 5.188 5.89 5.188 5.89 5.188 5.89 C 5.188 5.89 5.187 5.89 5.187 5.89 C 5.187 5.89 5.187 5.89 5.188 5.89 C 5.188 5.89 5.188 5.891 5.188 5.891 C 5.189 5.892 5.191 5.893 5.193 5.894 C 5.196 5.898 5.203 5.903 5.211 5.909 C 5.228 5.923 5.253 5.944 5.285 5.971 C 5.35 6.026 5.443 6.106 5.556 6.208 C 5.782 6.411 6.082 6.696 6.38 7.024 C 6.68 7.354 6.968 7.716 7.178 8.074 C 7.394 8.44 7.5 8.754 7.5 9 L 8.5 9 C 8.5 8.496 8.294 7.998 8.04 7.567 C 7.782 7.128 7.445 6.709 7.12 6.351 C 6.793 5.992 6.468 5.683 6.225 5.464 C 6.103 5.355 6.002 5.267 5.93 5.207 C 5.894 5.176 5.865 5.153 5.845 5.136 C 5.835 5.128 5.827 5.122 5.822 5.117 C 5.819 5.115 5.817 5.113 5.815 5.112 C 5.814 5.111 5.814 5.111 5.813 5.11 C 5.813 5.11 5.813 5.11 5.813 5.11 C 5.813 5.11 5.813 5.11 5.813 5.11 C 5.812 5.11 5.812 5.11 5.5 5.5 Z M 7.5 9 C 7.5 9.001 7.5 9.002 7.5 9.004 C 7.5 9.005 7.5 9.006 7.5 9.007 C 7.5 9.008 7.5 9.01 7.5 9.011 C 7.5 9.012 7.5 9.013 7.5 9.014 C 7.5 9.016 7.5 9.017 7.5 9.018 C 7.5 9.019 7.5 9.021 7.5 9.022 C 7.5 9.023 7.5 9.024 7.5 9.025 C 7.5 9.027 7.5 9.028 7.5 9.029 C 7.5 9.03 7.5 9.032 7.5 9.033 C 7.5 9.034 7.5 9.035 7.5 9.037 C 7.5 9.038 7.5 9.039 7.5 9.04 C 7.5 9.042 7.5 9.043 7.5 9.044 C 7.5 9.045 7.5 9.047 7.5 9.048 C 7.5 9.049 7.5 9.051 7.5 9.052 C 7.5 9.053 7.5 9.054 7.5 9.056 C 7.5 9.057 7.5 9.058 7.5 9.059 C 7.5 9.061 7.5 9.062 7.5 9.063 C 7.5 9.065 7.5 9.066 7.5 9.067 C 7.5 9.069 7.5 9.07 7.5 9.071 C 7.5 9.072 7.5 9.074 7.5 9.075 C 7.5 9.076 7.5 9.078 7.5 9.079 C 7.5 9.08 7.5 9.082 7.5 9.083 C 7.5 9.084 7.5 9.086 7.5 9.087 C 7.5 9.088 7.5 9.089 7.5 9.091 C 7.5 9.092 7.5 9.093 7.5 9.095 C 7.5 9.096 7.5 9.097 7.5 9.099 C 7.5 9.1 7.5 9.101 7.5 9.103 C 7.5 9.104 7.5 9.105 7.5 9.107 C 7.5 9.108 7.5 9.11 7.5 9.111 C 7.5 9.112 7.5 9.114 7.5 9.115 C 7.5 9.116 7.5 9.118 7.5 9.119 C 7.5 9.12 7.5 9.122 7.5 9.123 C 7.5 9.124 7.5 9.126 7.5 9.127 C 7.5 9.129 7.5 9.13 7.5 9.131 C 7.5 9.133 7.5 9.134 7.5 9.135 C 7.5 9.137 7.5 9.138 7.5 9.14 C 7.5 9.141 7.5 9.142 7.5 9.144 C 7.5 9.145 7.5 9.147 7.5 9.148 C 7.5 9.149 7.5 9.151 7.5 9.152 C 7.5 9.154 7.5 9.155 7.5 9.156 C 7.5 9.158 7.5 9.159 7.5 9.161 C 7.5 9.162 7.5 9.163 7.5 9.165 C 7.5 9.166 7.5 9.168 7.5 9.169 C 7.5 9.171 7.5 9.172 7.5 9.173 C 7.5 9.175 7.5 9.176 7.5 9.178 C 7.5 9.179 7.5 9.18 7.5 9.182 C 7.5 9.183 7.5 9.185 7.5 9.186 C 7.5 9.188 7.5 9.189 7.5 9.191 C 7.5 9.192 7.5 9.193 7.5 9.195 C 7.5 9.196 7.5 9.198 7.5 9.199 C 7.5 9.201 7.5 9.202 7.5 9.204 C 7.5 9.205 7.5 9.206 7.5 9.208 C 7.5 9.209 7.5 9.211 7.5 9.212 C 7.5 9.214 7.5 9.215 7.5 9.217 C 7.5 9.218 7.5 9.22 7.5 9.221 C 7.5 9.223 7.5 9.224 7.5 9.226 C 7.5 9.227 7.5 9.228 7.5 9.23 C 7.5 9.231 7.5 9.233 7.5 9.234 C 7.5 9.236 7.5 9.237 7.5 9.239 C 7.5 9.24 7.5 9.242 7.5 9.243 C 7.5 9.245 7.5 9.246 7.5 9.248 C 7.5 9.249 7.5 9.251 7.5 9.252 C 7.5 9.254 7.5 9.255 7.5 9.257 C 7.5 9.258 7.5 9.26 7.5 9.261 C 7.5 9.263 7.5 9.264 7.5 9.266 C 7.5 9.267 7.5 9.269 7.5 9.27 C 7.5 9.272 7.5 9.273 7.5 9.275 C 7.5 9.276 7.5 9.278 7.5 9.279 C 7.5 9.281 7.5 9.283 7.5 9.284 C 7.5 9.286 7.5 9.287 7.5 9.289 C 7.5 9.29 7.5 9.292 7.5 9.293 C 7.5 9.295 7.5 9.296 7.5 9.298 C 7.5 9.299 7.5 9.301 7.5 9.302 C 7.5 9.304 7.5 9.306 7.5 9.307 C 7.5 9.309 7.5 9.31 7.5 9.312 C 7.5 9.313 7.5 9.315 7.5 9.316 C 7.5 9.318 7.5 9.319 7.5 9.321 C 7.5 9.323 7.5 9.324 7.5 9.326 C 7.5 9.327 7.5 9.329 7.5 9.33 C 7.5 9.332 7.5 9.334 7.5 9.335 C 7.5 9.337 7.5 9.338 7.5 9.34 C 7.5 9.341 7.5 9.343 7.5 9.344 C 7.5 9.346 7.5 9.348 7.5 9.349 C 7.5 9.351 7.5 9.352 7.5 9.354 C 7.5 9.356 7.5 9.357 7.5 9.359 C 7.5 9.36 7.5 9.362 7.5 9.363 C 7.5 9.365 7.5 9.367 7.5 9.368 C 7.5 9.37 7.5 9.371 7.5 9.373 C 7.5 9.375 7.5 9.376 7.5 9.378 C 7.5 9.379 7.5 9.381 7.5 9.382 C 7.5 9.384 7.5 9.386 7.5 9.387 C 7.5 9.389 7.5 9.39 7.5 9.392 C 7.5 9.394 7.5 9.395 7.5 9.397 C 7.5 9.399 7.5 9.4 7.5 9.402 C 7.5 9.403 7.5 9.405 7.5 9.407 C 7.5 9.408 7.5 9.41 7.5 9.411 C 7.5 9.413 7.5 9.415 7.5 9.416 C 7.5 9.418 7.5 9.419 7.5 9.421 C 7.5 9.423 7.5 9.424 7.5 9.426 C 7.5 9.428 7.5 9.429 7.5 9.431 C 7.5 9.432 7.5 9.434 7.5 9.436 C 7.5 9.437 7.5 9.439 7.5 9.441 C 7.5 9.442 7.5 9.444 7.5 9.446 C 7.5 9.447 7.5 9.449 7.5 9.45 C 7.5 9.452 7.5 9.454 7.5 9.455 C 7.5 9.457 7.5 9.459 7.5 9.46 C 7.5 9.462 7.5 9.464 7.5 9.465 C 7.5 9.467 7.5 9.468 7.5 9.47 C 7.5 9.472 7.5 9.473 7.5 9.475 C 7.5 9.477 7.5 9.478 7.5 9.48 C 7.5 9.482 7.5 9.483 7.5 9.485 C 7.5 9.487 7.5 9.488 7.5 9.49 C 7.5 9.492 7.5 9.493 7.5 9.495 C 7.5 9.497 7.5 9.498 7.5 9.5 C 7.5 9.502 7.5 9.503 7.5 9.505 C 7.5 9.507 7.5 9.508 7.5 9.51 C 7.5 9.511 7.5 9.513 7.5 9.515 C 7.5 9.516 7.5 9.518 7.5 9.52 C 7.5 9.521 7.5 9.523 7.5 9.525 C 7.5 9.527 7.5 9.528 7.5 9.53 C 7.5 9.532 7.5 9.533 7.5 9.535 C 7.5 9.537 7.5 9.538 7.5 9.54 C 7.5 9.542 7.5 9.543 7.5 9.545 C 7.5 9.547 7.5 9.548 7.5 9.55 C 7.5 9.552 7.5 9.553 7.5 9.555 C 7.5 9.557 7.5 9.558 7.5 9.56 C 7.5 9.562 7.5 9.563 7.5 9.565 C 7.5 9.567 7.5 9.569 7.5 9.57 C 7.5 9.572 7.5 9.574 7.5 9.575 C 7.5 9.577 7.5 9.579 7.5 9.58 C 7.5 9.582 7.5 9.584 7.5 9.585 C 7.5 9.587 7.5 9.589 7.5 9.591 C 7.5 9.592 7.5 9.594 7.5 9.596 C 7.5 9.597 7.5 9.599 7.5 9.601 C 7.5 9.602 7.5 9.604 7.5 9.606 C 7.5 9.607 7.5 9.609 7.5 9.611 C 7.5 9.613 7.5 9.614 7.5 9.616 C 7.5 9.618 7.5 9.619 7.5 9.621 C 7.5 9.623 7.5 9.625 7.5 9.626 C 7.5 9.628 7.5 9.63 7.5 9.631 C 7.5 9.633 7.5 9.635 7.5 9.636 C 7.5 9.638 7.5 9.64 7.5 9.642 C 7.5 9.643 7.5 9.645 7.5 9.647 C 7.5 9.648 7.5 9.65 7.5 9.652 C 7.5 9.654 7.5 9.655 7.5 9.657 C 7.5 9.659 7.5 9.66 7.5 9.662 C 7.5 9.664 7.5 9.666 7.5 9.667 C 7.5 9.669 7.5 9.671 7.5 9.672 C 7.5 9.674 7.5 9.676 7.5 9.678 C 7.5 9.679 7.5 9.681 7.5 9.683 C 7.5 9.685 7.5 9.686 7.5 9.688 C 7.5 9.69 7.5 9.691 7.5 9.693 C 7.5 9.695 7.5 9.697 7.5 9.698 C 7.5 9.7 7.5 9.702 7.5 9.703 C 7.5 9.705 7.5 9.707 7.5 9.709 C 7.5 9.71 7.5 9.712 7.5 9.714 C 7.5 9.716 7.5 9.717 7.5 9.719 C 7.5 9.721 7.5 9.722 7.5 9.724 C 7.5 9.726 7.5 9.728 7.5 9.729 C 7.5 9.731 7.5 9.733 7.5 9.735 C 7.5 9.736 7.5 9.738 7.5 9.74 C 7.5 9.741 7.5 9.743 7.5 9.745 C 7.5 9.747 7.5 9.748 7.5 9.75 C 7.5 9.752 7.5 9.754 7.5 9.755 C 7.5 9.757 7.5 9.759 7.5 9.761 C 7.5 9.762 7.5 9.764 7.5 9.766 C 7.5 9.767 7.5 9.769 7.5 9.771 C 7.5 9.773 7.5 9.774 7.5 9.776 C 7.5 9.778 7.5 9.78 7.5 9.781 C 7.5 9.783 7.5 9.785 7.5 9.787 C 7.5 9.788 7.5 9.79 7.5 9.792 C 7.5 9.793 7.5 9.795 7.5 9.797 C 7.5 9.799 7.5 9.8 7.5 9.802 C 7.5 9.804 7.5 9.806 7.5 9.807 C 7.5 9.809 7.5 9.811 7.5 9.813 C 7.5 9.814 7.5 9.816 7.5 9.818 C 7.5 9.82 7.5 9.821 7.5 9.823 C 7.5 9.825 7.5 9.826 7.5 9.828 C 7.5 9.83 7.5 9.832 7.5 9.833 C 7.5 9.835 7.5 9.837 7.5 9.839 C 7.5 9.84 7.5 9.842 7.5 9.844 C 7.5 9.846 7.5 9.847 7.5 9.849 C 7.5 9.851 7.5 9.853 7.5 9.854 C 7.5 9.856 7.5 9.858 7.5 9.86 C 7.5 9.861 7.5 9.863 7.5 9.865 C 7.5 9.866 7.5 9.868 7.5 9.87 C 7.5 9.872 7.5 9.873 7.5 9.875 C 7.5 9.877 7.5 9.879 7.5 9.88 C 7.5 9.882 7.5 9.884 7.5 9.886 C 7.5 9.887 7.5 9.889 7.5 9.891 C 7.5 9.893 7.5 9.894 7.5 9.896 C 7.5 9.898 7.5 9.9 7.5 9.901 C 7.5 9.903 7.5 9.905 7.5 9.906 C 7.5 9.908 7.5 9.91 7.5 9.912 C 7.5 9.913 7.5 9.915 7.5 9.917 C 7.5 9.919 7.5 9.92 7.5 9.922 C 7.5 9.924 7.5 9.926 7.5 9.927 C 7.5 9.929 7.5 9.931 7.5 9.933 C 7.5 9.934 7.5 9.936 7.5 9.938 C 7.5 9.939 7.5 9.941 7.5 9.943 C 7.5 9.945 7.5 9.946 7.5 9.948 C 7.5 9.95 7.5 9.952 7.5 9.953 C 7.5 9.955 7.5 9.957 7.5 9.959 C 7.5 9.96 7.5 9.962 7.5 9.964 C 7.5 9.965 7.5 9.967 7.5 9.969 C 7.5 9.971 7.5 9.972 7.5 9.974 C 7.5 9.976 7.5 9.978 7.5 9.979 C 7.5 9.981 7.5 9.983 7.5 9.984 C 7.5 9.986 7.5 9.988 7.5 9.99 C 7.5 9.991 7.5 9.993 7.5 9.995 C 7.5 9.997 7.5 9.998 7.5 10 C 7.5 10.002 7.5 10.004 7.5 10.005 C 7.5 10.007 7.5 10.009 7.5 10.01 C 7.5 10.012 7.5 10.014 7.5 10.016 C 7.5 10.017 7.5 10.019 7.5 10.021 C 7.5 10.022 7.5 10.024 7.5 10.026 C 7.5 10.028 7.5 10.029 7.5 10.031 C 7.5 10.033 7.5 10.035 7.5 10.036 C 7.5 10.038 7.5 10.04 7.5 10.041 C 7.5 10.043 7.5 10.045 7.5 10.047 C 7.5 10.048 7.5 10.05 7.5 10.052 C 7.5 10.053 7.5 10.055 7.5 10.057 C 7.5 10.059 7.5 10.06 7.5 10.062 C 7.5 10.064 7.5 10.065 7.5 10.067 C 7.5 10.069 7.5 10.071 7.5 10.072 C 7.5 10.074 7.5 10.076 7.5 10.077 C 7.5 10.079 7.5 10.081 7.5 10.083 C 7.5 10.084 7.5 10.086 7.5 10.088 C 7.5 10.089 7.5 10.091 7.5 10.093 C 7.5 10.094 7.5 10.096 7.5 10.098 C 7.5 10.1 7.5 10.101 7.5 10.103 C 7.5 10.105 7.5 10.106 7.5 10.108 C 7.5 10.11 7.5 10.112 7.5 10.113 C 7.5 10.115 7.5 10.117 7.5 10.118 C 7.5 10.12 7.5 10.122 7.5 10.123 C 7.5 10.125 7.5 10.127 7.5 10.129 C 7.5 10.13 7.5 10.132 7.5 10.134 C 7.5 10.135 7.5 10.137 7.5 10.139 C 7.5 10.14 7.5 10.142 7.5 10.144 C 7.5 10.145 7.5 10.147 7.5 10.149 C 7.5 10.15 7.5 10.152 7.5 10.154 C 7.5 10.156 7.5 10.157 7.5 10.159 C 7.5 10.161 7.5 10.162 7.5 10.164 C 7.5 10.166 7.5 10.167 7.5 10.169 C 7.5 10.171 7.5 10.172 7.5 10.174 C 7.5 10.176 7.5 10.177 7.5 10.179 C 7.5 10.181 7.5 10.182 7.5 10.184 C 7.5 10.186 7.5 10.187 7.5 10.189 C 7.5 10.191 7.5 10.192 7.5 10.194 C 7.5 10.196 7.5 10.197 7.5 10.199 C 7.5 10.201 7.5 10.202 7.5 10.204 C 7.5 10.206 7.5 10.207 7.5 10.209 C 7.5 10.211 7.5 10.212 7.5 10.214 C 7.5 10.216 7.5 10.217 7.5 10.219 C 7.5 10.221 7.5 10.222 7.5 10.224 C 7.5 10.226 7.5 10.227 7.5 10.229 C 7.5 10.231 7.5 10.232 7.5 10.234 C 7.5 10.236 7.5 10.237 7.5 10.239 C 7.5 10.241 7.5 10.242 7.5 10.244 C 7.5 10.246 7.5 10.247 7.5 10.249 C 7.5 10.251 7.5 10.252 7.5 10.254 C 7.5 10.255 7.5 10.257 7.5 10.259 C 7.5 10.26 7.5 10.262 7.5 10.264 C 7.5 10.265 7.5 10.267 7.5 10.269 C 7.5 10.27 7.5 10.272 7.5 10.273 C 7.5 10.275 7.5 10.277 7.5 10.278 C 7.5 10.28 7.5 10.282 7.5 10.283 C 7.5 10.285 7.5 10.286 7.5 10.288 C 7.5 10.29 7.5 10.291 7.5 10.293 C 7.5 10.295 7.5 10.296 7.5 10.298 C 7.5 10.299 7.5 10.301 7.5 10.303 C 7.5 10.304 7.5 10.306 7.5 10.308 C 7.5 10.309 7.5 10.311 7.5 10.312 C 7.5 10.314 7.5 10.316 7.5 10.317 C 7.5 10.319 7.5 10.32 7.5 10.322 C 7.5 10.324 7.5 10.325 7.5 10.327 C 7.5 10.328 7.5 10.33 7.5 10.332 C 7.5 10.333 7.5 10.335 7.5 10.336 C 7.5 10.338 7.5 10.34 7.5 10.341 C 7.5 10.343 7.5 10.344 7.5 10.346 C 7.5 10.348 7.5 10.349 7.5 10.351 C 7.5 10.352 7.5 10.354 7.5 10.355 C 7.5 10.357 7.5 10.359 7.5 10.36 C 7.5 10.362 7.5 10.363 7.5 10.365 C 7.5 10.367 7.5 10.368 7.5 10.37 C 7.5 10.371 7.5 10.373 7.5 10.374 C 7.5 10.376 7.5 10.378 7.5 10.379 C 7.5 10.381 7.5 10.382 7.5 10.384 C 7.5 10.385 7.5 10.387 7.5 10.389 C 7.5 10.39 7.5 10.392 7.5 10.393 C 7.5 10.395 7.5 10.396 7.5 10.398 C 7.5 10.399 7.5 10.401 7.5 10.403 C 7.5 10.404 7.5 10.406 7.5 10.407 C 7.5 10.409 7.5 10.41 7.5 10.412 C 7.5 10.413 7.5 10.415 7.5 10.416 C 7.5 10.418 7.5 10.419 7.5 10.421 C 7.5 10.423 7.5 10.424 7.5 10.426 C 7.5 10.427 7.5 10.429 7.5 10.43 C 7.5 10.432 7.5 10.433 7.5 10.435 C 7.5 10.436 7.5 10.438 7.5 10.439 C 7.5 10.441 7.5 10.442 7.5 10.444 C 7.5 10.445 7.5 10.447 7.5 10.448 C 7.5 10.45 7.5 10.451 7.5 10.453 C 7.5 10.455 7.5 10.456 7.5 10.458 C 7.5 10.459 7.5 10.461 7.5 10.462 C 7.5 10.464 7.5 10.465 7.5 10.467 C 7.5 10.468 7.5 10.47 7.5 10.471 C 7.5 10.473 7.5 10.474 7.5 10.476 C 7.5 10.477 7.5 10.478 7.5 10.48 C 7.5 10.481 7.5 10.483 7.5 10.484 C 7.5 10.486 7.5 10.487 7.5 10.489 C 7.5 10.49 7.5 10.492 7.5 10.493 C 7.5 10.495 7.5 10.496 7.5 10.498 C 7.5 10.499 7.5 10.501 7.5 10.502 C 7.5 10.504 7.5 10.505 7.5 10.506 C 7.5 10.508 7.5 10.509 7.5 10.511 C 7.5 10.512 7.5 10.514 7.5 10.515 C 7.5 10.517 7.5 10.518 7.5 10.52 C 7.5 10.521 7.5 10.522 7.5 10.524 C 7.5 10.525 7.5 10.527 7.5 10.528 C 7.5 10.53 7.5 10.531 7.5 10.533 C 7.5 10.534 7.5 10.535 7.5 10.537 C 7.5 10.538 7.5 10.54 7.5 10.541 C 7.5 10.543 7.5 10.544 7.5 10.545 C 7.5 10.547 7.5 10.548 7.5 10.55 C 7.5 10.551 7.5 10.553 7.5 10.554 C 7.5 10.555 7.5 10.557 7.5 10.558 C 7.5 10.56 7.5 10.561 7.5 10.562 C 7.5 10.564 7.5 10.565 7.5 10.567 C 7.5 10.568 7.5 10.569 7.5 10.571 C 7.5 10.572 7.5 10.574 7.5 10.575 C 7.5 10.576 7.5 10.578 7.5 10.579 C 7.5 10.58 7.5 10.582 7.5 10.583 C 7.5 10.585 7.5 10.586 7.5 10.587 C 7.5 10.589 7.5 10.59 7.5 10.591 C 7.5 10.593 7.5 10.594 7.5 10.596 C 7.5 10.597 7.5 10.598 7.5 10.6 C 7.5 10.601 7.5 10.602 7.5 10.604 C 7.5 10.605 7.5 10.606 7.5 10.608 C 7.5 10.609 7.5 10.611 7.5 10.612 C 7.5 10.613 7.5 10.615 7.5 10.616 C 7.5 10.617 7.5 10.619 7.5 10.62 C 7.5 10.621 7.5 10.623 7.5 10.624 C 7.5 10.625 7.5 10.627 7.5 10.628 C 7.5 10.629 7.5 10.631 7.5 10.632 C 7.5 10.633 7.5 10.634 7.5 10.636 C 7.5 10.637 7.5 10.638 7.5 10.64 C 7.5 10.641 7.5 10.642 7.5 10.644 C 7.5 10.645 7.5 10.646 7.5 10.648 C 7.5 10.649 7.5 10.65 7.5 10.651 C 7.5 10.653 7.5 10.654 7.5 10.655 C 7.5 10.657 7.5 10.658 7.5 10.659 C 7.5 10.66 7.5 10.662 7.5 10.663 C 7.5 10.664 7.5 10.666 7.5 10.667 C 7.5 10.668 7.5 10.669 7.5 10.671 C 7.5 10.672 7.5 10.673 7.5 10.674 C 7.5 10.676 7.5 10.677 7.5 10.678 C 7.5 10.679 7.5 10.681 7.5 10.682 C 7.5 10.683 7.5 10.684 7.5 10.686 C 7.5 10.687 7.5 10.688 7.5 10.689 C 7.5 10.691 7.5 10.692 7.5 10.693 C 7.5 10.694 7.5 10.696 7.5 10.697 C 7.5 10.698 7.5 10.699 7.5 10.7 C 7.5 10.702 7.5 10.703 7.5 10.704 C 7.5 10.705 7.5 10.707 7.5 10.708 C 7.5 10.709 7.5 10.71 7.5 10.711 C 7.5 10.713 7.5 10.714 7.5 10.715 C 7.5 10.716 7.5 10.717 7.5 10.719 C 7.5 10.72 7.5 10.721 7.5 10.722 C 7.5 10.723 7.5 10.724 7.5 10.726 C 7.5 10.727 7.5 10.728 7.5 10.729 C 7.5 10.73 7.5 10.732 7.5 10.733 C 7.5 10.734 7.5 10.735 7.5 10.736 C 7.5 10.737 7.5 10.739 7.5 10.74 C 7.5 10.741 7.5 10.742 7.5 10.743 C 7.5 10.744 7.5 10.745 7.5 10.747 C 7.5 10.748 7.5 10.749 7.5 10.75 C 7.5 10.751 7.5 10.752 7.5 10.753 C 7.5 10.755 7.5 10.756 7.5 10.757 C 7.5 10.758 7.5 10.759 7.5 10.76 C 7.5 10.761 7.5 10.762 7.5 10.763 C 7.5 10.765 7.5 10.766 7.5 10.767 C 7.5 10.768 7.5 10.769 7.5 10.77 C 7.5 10.771 7.5 10.772 7.5 10.773 C 7.5 10.774 7.5 10.776 7.5 10.777 C 7.5 10.778 7.5 10.779 7.5 10.78 C 7.5 10.781 7.5 10.782 7.5 10.783 C 7.5 10.784 7.5 10.785 7.5 10.786 C 7.5 10.787 7.5 10.788 7.5 10.79 C 7.5 10.791 7.5 10.792 7.5 10.793 C 7.5 10.794 7.5 10.795 7.5 10.796 C 7.5 10.797 7.5 10.798 7.5 10.799 C 7.5 10.8 7.5 10.801 7.5 10.802 C 7.5 10.803 7.5 10.804 7.5 10.805 C 7.5 10.806 7.5 10.807 7.5 10.808 C 7.5 10.809 7.5 10.81 7.5 10.811 C 7.5 10.812 7.5 10.813 7.5 10.814 C 7.5 10.815 7.5 10.816 7.5 10.817 C 7.5 10.818 7.5 10.819 7.5 10.82 C 7.5 10.821 7.5 10.822 7.5 10.823 C 7.5 10.824 7.5 10.825 7.5 10.826 C 7.5 10.827 7.5 10.828 7.5 10.829 C 7.5 10.83 7.5 10.831 7.5 10.832 C 7.5 10.833 7.5 10.834 7.5 10.835 C 7.5 10.836 7.5 10.837 7.5 10.838 C 7.5 10.839 7.5 10.84 7.5 10.841 C 7.5 10.842 7.5 10.842 7.5 10.843 C 7.5 10.844 7.5 10.845 7.5 10.846 C 7.5 10.847 7.5 10.848 7.5 10.849 C 7.5 10.85 7.5 10.851 7.5 10.852 C 7.5 10.853 7.5 10.854 7.5 10.854 C 7.5 10.855 7.5 10.856 7.5 10.857 C 7.5 10.858 7.5 10.859 7.5 10.86 C 7.5 10.861 7.5 10.862 7.5 10.863 C 7.5 10.863 7.5 10.864 7.5 10.865 C 7.5 10.866 7.5 10.867 7.5 10.868 C 7.5 10.869 7.5 10.87 7.5 10.87 C 7.5 10.871 7.5 10.872 7.5 10.873 C 7.5 10.874 7.5 10.875 7.5 10.875 C 7.5 10.876 7.5 10.877 7.5 10.878 C 7.5 10.879 7.5 10.88 7.5 10.881 C 7.5 10.881 7.5 10.882 7.5 10.883 C 7.5 10.884 7.5 10.885 7.5 10.885 C 7.5 10.886 7.5 10.887 7.5 10.888 C 7.5 10.889 7.5 10.89 7.5 10.89 C 7.5 10.891 7.5 10.892 7.5 10.893 C 7.5 10.893 7.5 10.894 7.5 10.895 C 7.5 10.896 7.5 10.897 7.5 10.897 C 7.5 10.898 7.5 10.899 7.5 10.9 C 7.5 10.9 7.5 10.901 7.5 10.902 C 7.5 10.903 7.5 10.904 7.5 10.904 C 7.5 10.905 7.5 10.906 7.5 10.907 C 7.5 10.907 7.5 10.908 7.5 10.909 C 7.5 10.909 7.5 10.91 7.5 10.911 C 7.5 10.912 7.5 10.912 7.5 10.913 C 7.5 10.914 7.5 10.915 7.5 10.915 C 7.5 10.916 7.5 10.917 7.5 10.917 C 7.5 10.918 7.5 10.919 7.5 10.919 C 7.5 10.92 7.5 10.921 7.5 10.922 C 7.5 10.922 7.5 10.923 7.5 10.924 C 7.5 10.924 7.5 10.925 7.5 10.926 C 7.5 10.926 7.5 10.927 7.5 10.928 C 7.5 10.928 7.5 10.929 7.5 10.93 C 7.5 10.93 7.5 10.931 7.5 10.932 C 7.5 10.932 7.5 10.933 7.5 10.933 C 7.5 10.934 7.5 10.935 7.5 10.935 C 7.5 10.936 7.5 10.937 7.5 10.937 C 7.5 10.938 7.5 10.939 7.5 10.939 C 7.5 10.94 7.5 10.94 7.5 10.941 C 7.5 10.942 7.5 10.942 7.5 10.943 C 7.5 10.943 7.5 10.944 7.5 10.945 C 7.5 10.945 7.5 10.946 7.5 10.946 C 7.5 10.947 7.5 10.947 7.5 10.948 C 7.5 10.949 7.5 10.949 7.5 10.95 C 7.5 10.95 7.5 10.951 7.5 10.951 C 7.5 10.952 7.5 10.952 7.5 10.953 C 7.5 10.954 7.5 10.954 7.5 10.955 C 7.5 10.955 7.5 10.956 7.5 10.956 C 7.5 10.957 7.5 10.957 7.5 10.958 C 7.5 10.958 7.5 10.959 7.5 10.959 C 7.5 10.96 7.5 10.96 7.5 10.961 C 7.5 10.961 7.5 10.962 7.5 10.962 C 7.5 10.963 7.5 10.963 7.5 10.964 C 7.5 10.964 7.5 10.965 7.5 10.965 C 7.5 10.966 7.5 10.966 7.5 10.967 C 7.5 10.967 7.5 10.967 7.5 10.968 C 7.5 10.968 7.5 10.969 7.5 10.969 C 7.5 10.97 7.5 10.97 7.5 10.971 C 7.5 10.971 7.5 10.971 7.5 10.972 C 7.5 10.972 7.5 10.973 7.5 10.973 C 7.5 10.974 7.5 10.974 7.5 10.974 C 7.5 10.975 7.5 10.975 7.5 10.976 C 7.5 10.976 7.5 10.976 7.5 10.977 C 7.5 10.977 7.5 10.978 7.5 10.978 C 7.5 10.978 7.5 10.979 7.5 10.979 C 7.5 10.979 7.5 10.98 7.5 10.98 C 7.5 10.981 7.5 10.981 7.5 10.981 C 7.5 10.982 7.5 10.982 7.5 10.982 C 7.5 10.983 7.5 10.983 7.5 10.983 C 7.5 10.984 7.5 10.984 7.5 10.984 C 7.5 10.985 7.5 10.985 7.5 10.985 C 7.5 10.986 7.5 10.986 7.5 10.986 C 7.5 10.986 7.5 10.987 7.5 10.987 C 7.5 10.987 7.5 10.988 7.5 10.988 C 7.5 10.988 7.5 10.988 7.5 10.989 C 7.5 10.989 7.5 10.989 7.5 10.99 C 7.5 10.99 7.5 10.99 7.5 10.99 C 7.5 10.991 7.5 10.991 7.5 10.991 C 7.5 10.991 7.5 10.992 7.5 10.992 C 7.5 10.992 7.5 10.992 7.5 10.992 C 7.5 10.993 7.5 10.993 7.5 10.993 C 7.5 10.993 7.5 10.994 7.5 10.994 C 7.5 10.994 7.5 10.994 7.5 10.994 C 7.5 10.995 7.5 10.995 7.5 10.995 C 7.5 10.995 7.5 10.995 7.5 10.995 C 7.5 10.996 7.5 10.996 7.5 10.996 C 7.5 10.996 7.5 10.996 7.5 10.996 C 7.5 10.997 7.5 10.997 7.5 10.997 C 7.5 10.997 7.5 10.997 7.5 10.997 C 7.5 10.997 7.5 10.998 7.5 10.998 C 7.5 10.998 7.5 10.998 7.5 10.998 C 7.5 10.998 7.5 10.998 7.5 10.998 C 7.5 10.999 7.5 10.999 7.5 10.999 C 7.5 10.999 7.5 10.999 7.5 10.999 C 7.5 10.999 7.5 10.999 7.5 10.999 C 7.5 10.999 7.5 10.999 7.5 10.999 C 7.5 10.999 7.5 11 7.5 11 C 7.5 11 7.5 11 7.5 11 C 7.5 11 7.5 11 7.5 11 C 7.5 11 7.5 11 7.5 11 C 7.5 11 7.5 11 7.5 11 C 7.5 11 7.5 11 8 11 C 8.5 11 8.5 11 8.5 11 C 8.5 11 8.5 11 8.5 11 C 8.5 11 8.5 11 8.5 11 C 8.5 11 8.5 11 8.5 11 C 8.5 11 8.5 11 8.5 11 C 8.5 11 8.5 10.999 8.5 10.999 C 8.5 10.999 8.5 10.999 8.5 10.999 C 8.5 10.999 8.5 10.999 8.5 10.999 C 8.5 10.999 8.5 10.999 8.5 10.999 C 8.5 10.999 8.5 10.999 8.5 10.998 C 8.5 10.998 8.5 10.998 8.5 10.998 C 8.5 10.998 8.5 10.998 8.5 10.998 C 8.5 10.998 8.5 10.997 8.5 10.997 C 8.5 10.997 8.5 10.997 8.5 10.997 C 8.5 10.997 8.5 10.997 8.5 10.996 C 8.5 10.996 8.5 10.996 8.5 10.996 C 8.5 10.996 8.5 10.996 8.5 10.995 C 8.5 10.995 8.5 10.995 8.5 10.995 C 8.5 10.995 8.5 10.995 8.5 10.994 C 8.5 10.994 8.5 10.994 8.5 10.994 C 8.5 10.994 8.5 10.993 8.5 10.993 C 8.5 10.993 8.5 10.993 8.5 10.992 C 8.5 10.992 8.5 10.992 8.5 10.992 C 8.5 10.992 8.5 10.991 8.5 10.991 C 8.5 10.991 8.5 10.991 8.5 10.99 C 8.5 10.99 8.5 10.99 8.5 10.99 C 8.5 10.989 8.5 10.989 8.5 10.989 C 8.5 10.988 8.5 10.988 8.5 10.988 C 8.5 10.988 8.5 10.987 8.5 10.987 C 8.5 10.987 8.5 10.986 8.5 10.986 C 8.5 10.986 8.5 10.986 8.5 10.985 C 8.5 10.985 8.5 10.985 8.5 10.984 C 8.5 10.984 8.5 10.984 8.5 10.983 C 8.5 10.983 8.5 10.983 8.5 10.982 C 8.5 10.982 8.5 10.982 8.5 10.981 C 8.5 10.981 8.5 10.981 8.5 10.98 C 8.5 10.98 8.5 10.979 8.5 10.979 C 8.5 10.979 8.5 10.978 8.5 10.978 C 8.5 10.978 8.5 10.977 8.5 10.977 C 8.5 10.976 8.5 10.976 8.5 10.976 C 8.5 10.975 8.5 10.975 8.5 10.974 C 8.5 10.974 8.5 10.974 8.5 10.973 C 8.5 10.973 8.5 10.972 8.5 10.972 C 8.5 10.971 8.5 10.971 8.5 10.971 C 8.5 10.97 8.5 10.97 8.5 10.969 C 8.5 10.969 8.5 10.968 8.5 10.968 C 8.5 10.967 8.5 10.967 8.5 10.967 C 8.5 10.966 8.5 10.966 8.5 10.965 C 8.5 10.965 8.5 10.964 8.5 10.964 C 8.5 10.963 8.5 10.963 8.5 10.962 C 8.5 10.962 8.5 10.961 8.5 10.961 C 8.5 10.96 8.5 10.96 8.5 10.959 C 8.5 10.959 8.5 10.958 8.5 10.958 C 8.5 10.957 8.5 10.957 8.5 10.956 C 8.5 10.956 8.5 10.955 8.5 10.955 C 8.5 10.954 8.5 10.954 8.5 10.953 C 8.5 10.952 8.5 10.952 8.5 10.951 C 8.5 10.951 8.5 10.95 8.5 10.95 C 8.5 10.949 8.5 10.949 8.5 10.948 C 8.5 10.947 8.5 10.947 8.5 10.946 C 8.5 10.946 8.5 10.945 8.5 10.945 C 8.5 10.944 8.5 10.943 8.5 10.943 C 8.5 10.942 8.5 10.942 8.5 10.941 C 8.5 10.94 8.5 10.94 8.5 10.939 C 8.5 10.939 8.5 10.938 8.5 10.937 C 8.5 10.937 8.5 10.936 8.5 10.935 C 8.5 10.935 8.5 10.934 8.5 10.933 C 8.5 10.933 8.5 10.932 8.5 10.932 C 8.5 10.931 8.5 10.93 8.5 10.93 C 8.5 10.929 8.5 10.928 8.5 10.928 C 8.5 10.927 8.5 10.926 8.5 10.926 C 8.5 10.925 8.5 10.924 8.5 10.924 C 8.5 10.923 8.5 10.922 8.5 10.922 C 8.5 10.921 8.5 10.92 8.5 10.919 C 8.5 10.919 8.5 10.918 8.5 10.917 C 8.5 10.917 8.5 10.916 8.5 10.915 C 8.5 10.915 8.5 10.914 8.5 10.913 C 8.5 10.912 8.5 10.912 8.5 10.911 C 8.5 10.91 8.5 10.909 8.5 10.909 C 8.5 10.908 8.5 10.907 8.5 10.907 C 8.5 10.906 8.5 10.905 8.5 10.904 C 8.5 10.904 8.5 10.903 8.5 10.902 C 8.5 10.901 8.5 10.9 8.5 10.9 C 8.5 10.899 8.5 10.898 8.5 10.897 C 8.5 10.897 8.5 10.896 8.5 10.895 C 8.5 10.894 8.5 10.893 8.5 10.893 C 8.5 10.892 8.5 10.891 8.5 10.89 C 8.5 10.89 8.5 10.889 8.5 10.888 C 8.5 10.887 8.5 10.886 8.5 10.885 C 8.5 10.885 8.5 10.884 8.5 10.883 C 8.5 10.882 8.5 10.881 8.5 10.881 C 8.5 10.88 8.5 10.879 8.5 10.878 C 8.5 10.877 8.5 10.876 8.5 10.875 C 8.5 10.875 8.5 10.874 8.5 10.873 C 8.5 10.872 8.5 10.871 8.5 10.87 C 8.5 10.87 8.5 10.869 8.5 10.868 C 8.5 10.867 8.5 10.866 8.5 10.865 C 8.5 10.864 8.5 10.863 8.5 10.863 C 8.5 10.862 8.5 10.861 8.5 10.86 C 8.5 10.859 8.5 10.858 8.5 10.857 C 8.5 10.856 8.5 10.855 8.5 10.854 C 8.5 10.854 8.5 10.853 8.5 10.852 C 8.5 10.851 8.5 10.85 8.5 10.849 C 8.5 10.848 8.5 10.847 8.5 10.846 C 8.5 10.845 8.5 10.844 8.5 10.843 C 8.5 10.842 8.5 10.842 8.5 10.841 C 8.5 10.84 8.5 10.839 8.5 10.838 C 8.5 10.837 8.5 10.836 8.5 10.835 C 8.5 10.834 8.5 10.833 8.5 10.832 C 8.5 10.831 8.5 10.83 8.5 10.829 C 8.5 10.828 8.5 10.827 8.5 10.826 C 8.5 10.825 8.5 10.824 8.5 10.823 C 8.5 10.822 8.5 10.821 8.5 10.82 C 8.5 10.819 8.5 10.818 8.5 10.817 C 8.5 10.816 8.5 10.815 8.5 10.814 C 8.5 10.813 8.5 10.812 8.5 10.811 C 8.5 10.81 8.5 10.809 8.5 10.808 C 8.5 10.807 8.5 10.806 8.5 10.805 C 8.5 10.804 8.5 10.803 8.5 10.802 C 8.5 10.801 8.5 10.8 8.5 10.799 C 8.5 10.798 8.5 10.797 8.5 10.796 C 8.5 10.795 8.5 10.794 8.5 10.793 C 8.5 10.792 8.5 10.791 8.5 10.79 C 8.5 10.788 8.5 10.787 8.5 10.786 C 8.5 10.785 8.5 10.784 8.5 10.783 C 8.5 10.782 8.5 10.781 8.5 10.78 C 8.5 10.779 8.5 10.778 8.5 10.777 C 8.5 10.776 8.5 10.774 8.5 10.773 C 8.5 10.772 8.5 10.771 8.5 10.77 C 8.5 10.769 8.5 10.768 8.5 10.767 C 8.5 10.766 8.5 10.765 8.5 10.763 C 8.5 10.762 8.5 10.761 8.5 10.76 C 8.5 10.759 8.5 10.758 8.5 10.757 C 8.5 10.756 8.5 10.755 8.5 10.753 C 8.5 10.752 8.5 10.751 8.5 10.75 C 8.5 10.749 8.5 10.748 8.5 10.747 C 8.5 10.745 8.5 10.744 8.5 10.743 C 8.5 10.742 8.5 10.741 8.5 10.74 C 8.5 10.739 8.5 10.737 8.5 10.736 C 8.5 10.735 8.5 10.734 8.5 10.733 C 8.5 10.732 8.5 10.73 8.5 10.729 C 8.5 10.728 8.5 10.727 8.5 10.726 C 8.5 10.724 8.5 10.723 8.5 10.722 C 8.5 10.721 8.5 10.72 8.5 10.719 C 8.5 10.717 8.5 10.716 8.5 10.715 C 8.5 10.714 8.5 10.713 8.5 10.711 C 8.5 10.71 8.5 10.709 8.5 10.708 C 8.5 10.707 8.5 10.705 8.5 10.704 C 8.5 10.703 8.5 10.702 8.5 10.7 C 8.5 10.699 8.5 10.698 8.5 10.697 C 8.5 10.696 8.5 10.694 8.5 10.693 C 8.5 10.692 8.5 10.691 8.5 10.689 C 8.5 10.688 8.5 10.687 8.5 10.686 C 8.5 10.684 8.5 10.683 8.5 10.682 C 8.5 10.681 8.5 10.679 8.5 10.678 C 8.5 10.677 8.5 10.676 8.5 10.674 C 8.5 10.673 8.5 10.672 8.5 10.671 C 8.5 10.669 8.5 10.668 8.5 10.667 C 8.5 10.666 8.5 10.664 8.5 10.663 C 8.5 10.662 8.5 10.66 8.5 10.659 C 8.5 10.658 8.5 10.657 8.5 10.655 C 8.5 10.654 8.5 10.653 8.5 10.651 C 8.5 10.65 8.5 10.649 8.5 10.648 C 8.5 10.646 8.5 10.645 8.5 10.644 C 8.5 10.642 8.5 10.641 8.5 10.64 C 8.5 10.638 8.5 10.637 8.5 10.636 C 8.5 10.634 8.5 10.633 8.5 10.632 C 8.5 10.631 8.5 10.629 8.5 10.628 C 8.5 10.627 8.5 10.625 8.5 10.624 C 8.5 10.623 8.5 10.621 8.5 10.62 C 8.5 10.619 8.5 10.617 8.5 10.616 C 8.5 10.615 8.5 10.613 8.5 10.612 C 8.5 10.611 8.5 10.609 8.5 10.608 C 8.5 10.606 8.5 10.605 8.5 10.604 C 8.5 10.602 8.5 10.601 8.5 10.6 C 8.5 10.598 8.5 10.597 8.5 10.596 C 8.5 10.594 8.5 10.593 8.5 10.591 C 8.5 10.59 8.5 10.589 8.5 10.587 C 8.5 10.586 8.5 10.585 8.5 10.583 C 8.5 10.582 8.5 10.58 8.5 10.579 C 8.5 10.578 8.5 10.576 8.5 10.575 C 8.5 10.574 8.5 10.572 8.5 10.571 C 8.5 10.569 8.5 10.568 8.5 10.567 C 8.5 10.565 8.5 10.564 8.5 10.562 C 8.5 10.561 8.5 10.56 8.5 10.558 C 8.5 10.557 8.5 10.555 8.5 10.554 C 8.5 10.553 8.5 10.551 8.5 10.55 C 8.5 10.548 8.5 10.547 8.5 10.545 C 8.5 10.544 8.5 10.543 8.5 10.541 C 8.5 10.54 8.5 10.538 8.5 10.537 C 8.5 10.535 8.5 10.534 8.5 10.533 C 8.5 10.531 8.5 10.53 8.5 10.528 C 8.5 10.527 8.5 10.525 8.5 10.524 C 8.5 10.522 8.5 10.521 8.5 10.52 C 8.5 10.518 8.5 10.517 8.5 10.515 C 8.5 10.514 8.5 10.512 8.5 10.511 C 8.5 10.509 8.5 10.508 8.5 10.506 C 8.5 10.505 8.5 10.504 8.5 10.502 C 8.5 10.501 8.5 10.499 8.5 10.498 C 8.5 10.496 8.5 10.495 8.5 10.493 C 8.5 10.492 8.5 10.49 8.5 10.489 C 8.5 10.487 8.5 10.486 8.5 10.484 C 8.5 10.483 8.5 10.481 8.5 10.48 C 8.5 10.478 8.5 10.477 8.5 10.476 C 8.5 10.474 8.5 10.473 8.5 10.471 C 8.5 10.47 8.5 10.468 8.5 10.467 C 8.5 10.465 8.5 10.464 8.5 10.462 C 8.5 10.461 8.5 10.459 8.5 10.458 C 8.5 10.456 8.5 10.455 8.5 10.453 C 8.5 10.451 8.5 10.45 8.5 10.448 C 8.5 10.447 8.5 10.445 8.5 10.444 C 8.5 10.442 8.5 10.441 8.5 10.439 C 8.5 10.438 8.5 10.436 8.5 10.435 C 8.5 10.433 8.5 10.432 8.5 10.43 C 8.5 10.429 8.5 10.427 8.5 10.426 C 8.5 10.424 8.5 10.423 8.5 10.421 C 8.5 10.419 8.5 10.418 8.5 10.416 C 8.5 10.415 8.5 10.413 8.5 10.412 C 8.5 10.41 8.5 10.409 8.5 10.407 C 8.5 10.406 8.5 10.404 8.5 10.403 C 8.5 10.401 8.5 10.399 8.5 10.398 C 8.5 10.396 8.5 10.395 8.5 10.393 C 8.5 10.392 8.5 10.39 8.5 10.389 C 8.5 10.387 8.5 10.385 8.5 10.384 C 8.5 10.382 8.5 10.381 8.5 10.379 C 8.5 10.378 8.5 10.376 8.5 10.374 C 8.5 10.373 8.5 10.371 8.5 10.37 C 8.5 10.368 8.5 10.367 8.5 10.365 C 8.5 10.363 8.5 10.362 8.5 10.36 C 8.5 10.359 8.5 10.357 8.5 10.355 C 8.5 10.354 8.5 10.352 8.5 10.351 C 8.5 10.349 8.5 10.348 8.5 10.346 C 8.5 10.344 8.5 10.343 8.5 10.341 C 8.5 10.34 8.5 10.338 8.5 10.336 C 8.5 10.335 8.5 10.333 8.5 10.332 C 8.5 10.33 8.5 10.328 8.5 10.327 C 8.5 10.325 8.5 10.324 8.5 10.322 C 8.5 10.32 8.5 10.319 8.5 10.317 C 8.5 10.316 8.5 10.314 8.5 10.312 C 8.5 10.311 8.5 10.309 8.5 10.308 C 8.5 10.306 8.5 10.304 8.5 10.303 C 8.5 10.301 8.5 10.299 8.5 10.298 C 8.5 10.296 8.5 10.295 8.5 10.293 C 8.5 10.291 8.5 10.29 8.5 10.288 C 8.5 10.286 8.5 10.285 8.5 10.283 C 8.5 10.282 8.5 10.28 8.5 10.278 C 8.5 10.277 8.5 10.275 8.5 10.273 C 8.5 10.272 8.5 10.27 8.5 10.269 C 8.5 10.267 8.5 10.265 8.5 10.264 C 8.5 10.262 8.5 10.26 8.5 10.259 C 8.5 10.257 8.5 10.255 8.5 10.254 C 8.5 10.252 8.5 10.251 8.5 10.249 C 8.5 10.247 8.5 10.246 8.5 10.244 C 8.5 10.242 8.5 10.241 8.5 10.239 C 8.5 10.237 8.5 10.236 8.5 10.234 C 8.5 10.232 8.5 10.231 8.5 10.229 C 8.5 10.227 8.5 10.226 8.5 10.224 C 8.5 10.222 8.5 10.221 8.5 10.219 C 8.5 10.217 8.5 10.216 8.5 10.214 C 8.5 10.212 8.5 10.211 8.5 10.209 C 8.5 10.207 8.5 10.206 8.5 10.204 C 8.5 10.202 8.5 10.201 8.5 10.199 C 8.5 10.197 8.5 10.196 8.5 10.194 C 8.5 10.192 8.5 10.191 8.5 10.189 C 8.5 10.187 8.5 10.186 8.5 10.184 C 8.5 10.182 8.5 10.181 8.5 10.179 C 8.5 10.177 8.5 10.176 8.5 10.174 C 8.5 10.172 8.5 10.171 8.5 10.169 C 8.5 10.167 8.5 10.166 8.5 10.164 C 8.5 10.162 8.5 10.161 8.5 10.159 C 8.5 10.157 8.5 10.156 8.5 10.154 C 8.5 10.152 8.5 10.15 8.5 10.149 C 8.5 10.147 8.5 10.145 8.5 10.144 C 8.5 10.142 8.5 10.14 8.5 10.139 C 8.5 10.137 8.5 10.135 8.5 10.134 C 8.5 10.132 8.5 10.13 8.5 10.129 C 8.5 10.127 8.5 10.125 8.5 10.123 C 8.5 10.122 8.5 10.12 8.5 10.118 C 8.5 10.117 8.5 10.115 8.5 10.113 C 8.5 10.112 8.5 10.11 8.5 10.108 C 8.5 10.106 8.5 10.105 8.5 10.103 C 8.5 10.101 8.5 10.1 8.5 10.098 C 8.5 10.096 8.5 10.094 8.5 10.093 C 8.5 10.091 8.5 10.089 8.5 10.088 C 8.5 10.086 8.5 10.084 8.5 10.083 C 8.5 10.081 8.5 10.079 8.5 10.077 C 8.5 10.076 8.5 10.074 8.5 10.072 C 8.5 10.071 8.5 10.069 8.5 10.067 C 8.5 10.065 8.5 10.064 8.5 10.062 C 8.5 10.06 8.5 10.059 8.5 10.057 C 8.5 10.055 8.5 10.053 8.5 10.052 C 8.5 10.05 8.5 10.048 8.5 10.047 C 8.5 10.045 8.5 10.043 8.5 10.041 C 8.5 10.04 8.5 10.038 8.5 10.036 C 8.5 10.035 8.5 10.033 8.5 10.031 C 8.5 10.029 8.5 10.028 8.5 10.026 C 8.5 10.024 8.5 10.022 8.5 10.021 C 8.5 10.019 8.5 10.017 8.5 10.016 C 8.5 10.014 8.5 10.012 8.5 10.01 C 8.5 10.009 8.5 10.007 8.5 10.005 C 8.5 10.004 8.5 10.002 8.5 10 C 8.5 9.998 8.5 9.997 8.5 9.995 C 8.5 9.993 8.5 9.991 8.5 9.99 C 8.5 9.988 8.5 9.986 8.5 9.984 C 8.5 9.983 8.5 9.981 8.5 9.979 C 8.5 9.978 8.5 9.976 8.5 9.974 C 8.5 9.972 8.5 9.971 8.5 9.969 C 8.5 9.967 8.5 9.965 8.5 9.964 C 8.5 9.962 8.5 9.96 8.5 9.959 C 8.5 9.957 8.5 9.955 8.5 9.953 C 8.5 9.952 8.5 9.95 8.5 9.948 C 8.5 9.946 8.5 9.945 8.5 9.943 C 8.5 9.941 8.5 9.939 8.5 9.938 C 8.5 9.936 8.5 9.934 8.5 9.933 C 8.5 9.931 8.5 9.929 8.5 9.927 C 8.5 9.926 8.5 9.924 8.5 9.922 C 8.5 9.92 8.5 9.919 8.5 9.917 C 8.5 9.915 8.5 9.913 8.5 9.912 C 8.5 9.91 8.5 9.908 8.5 9.906 C 8.5 9.905 8.5 9.903 8.5 9.901 C 8.5 9.9 8.5 9.898 8.5 9.896 C 8.5 9.894 8.5 9.893 8.5 9.891 C 8.5 9.889 8.5 9.887 8.5 9.886 C 8.5 9.884 8.5 9.882 8.5 9.88 C 8.5 9.879 8.5 9.877 8.5 9.875 C 8.5 9.873 8.5 9.872 8.5 9.87 C 8.5 9.868 8.5 9.866 8.5 9.865 C 8.5 9.863 8.5 9.861 8.5 9.86 C 8.5 9.858 8.5 9.856 8.5 9.854 C 8.5 9.853 8.5 9.851 8.5 9.849 C 8.5 9.847 8.5 9.846 8.5 9.844 C 8.5 9.842 8.5 9.84 8.5 9.839 C 8.5 9.837 8.5 9.835 8.5 9.833 C 8.5 9.832 8.5 9.83 8.5 9.828 C 8.5 9.826 8.5 9.825 8.5 9.823 C 8.5 9.821 8.5 9.82 8.5 9.818 C 8.5 9.816 8.5 9.814 8.5 9.813 C 8.5 9.811 8.5 9.809 8.5 9.807 C 8.5 9.806 8.5 9.804 8.5 9.802 C 8.5 9.8 8.5 9.799 8.5 9.797 C 8.5 9.795 8.5 9.793 8.5 9.792 C 8.5 9.79 8.5 9.788 8.5 9.787 C 8.5 9.785 8.5 9.783 8.5 9.781 C 8.5 9.78 8.5 9.778 8.5 9.776 C 8.5 9.774 8.5 9.773 8.5 9.771 C 8.5 9.769 8.5 9.767 8.5 9.766 C 8.5 9.764 8.5 9.762 8.5 9.761 C 8.5 9.759 8.5 9.757 8.5 9.755 C 8.5 9.754 8.5 9.752 8.5 9.75 C 8.5 9.748 8.5 9.747 8.5 9.745 C 8.5 9.743 8.5 9.741 8.5 9.74 C 8.5 9.738 8.5 9.736 8.5 9.735 C 8.5 9.733 8.5 9.731 8.5 9.729 C 8.5 9.728 8.5 9.726 8.5 9.724 C 8.5 9.722 8.5 9.721 8.5 9.719 C 8.5 9.717 8.5 9.716 8.5 9.714 C 8.5 9.712 8.5 9.71 8.5 9.709 C 8.5 9.707 8.5 9.705 8.5 9.703 C 8.5 9.702 8.5 9.7 8.5 9.698 C 8.5 9.697 8.5 9.695 8.5 9.693 C 8.5 9.691 8.5 9.69 8.5 9.688 C 8.5 9.686 8.5 9.685 8.5 9.683 C 8.5 9.681 8.5 9.679 8.5 9.678 C 8.5 9.676 8.5 9.674 8.5 9.672 C 8.5 9.671 8.5 9.669 8.5 9.667 C 8.5 9.666 8.5 9.664 8.5 9.662 C 8.5 9.66 8.5 9.659 8.5 9.657 C 8.5 9.655 8.5 9.654 8.5 9.652 C 8.5 9.65 8.5 9.648 8.5 9.647 C 8.5 9.645 8.5 9.643 8.5 9.642 C 8.5 9.64 8.5 9.638 8.5 9.636 C 8.5 9.635 8.5 9.633 8.5 9.631 C 8.5 9.63 8.5 9.628 8.5 9.626 C 8.5 9.625 8.5 9.623 8.5 9.621 C 8.5 9.619 8.5 9.618 8.5 9.616 C 8.5 9.614 8.5 9.613 8.5 9.611 C 8.5 9.609 8.5 9.607 8.5 9.606 C 8.5 9.604 8.5 9.602 8.5 9.601 C 8.5 9.599 8.5 9.597 8.5 9.596 C 8.5 9.594 8.5 9.592 8.5 9.591 C 8.5 9.589 8.5 9.587 8.5 9.585 C 8.5 9.584 8.5 9.582 8.5 9.58 C 8.5 9.579 8.5 9.577 8.5 9.575 C 8.5 9.574 8.5 9.572 8.5 9.57 C 8.5 9.569 8.5 9.567 8.5 9.565 C 8.5 9.563 8.5 9.562 8.5 9.56 C 8.5 9.558 8.5 9.557 8.5 9.555 C 8.5 9.553 8.5 9.552 8.5 9.55 C 8.5 9.548 8.5 9.547 8.5 9.545 C 8.5 9.543 8.5 9.542 8.5 9.54 C 8.5 9.538 8.5 9.537 8.5 9.535 C 8.5 9.533 8.5 9.532 8.5 9.53 C 8.5 9.528 8.5 9.527 8.5 9.525 C 8.5 9.523 8.5 9.521 8.5 9.52 C 8.5 9.518 8.5 9.516 8.5 9.515 C 8.5 9.513 8.5 9.511 8.5 9.51 C 8.5 9.508 8.5 9.507 8.5 9.505 C 8.5 9.503 8.5 9.502 8.5 9.5 C 8.5 9.498 8.5 9.497 8.5 9.495 C 8.5 9.493 8.5 9.492 8.5 9.49 C 8.5 9.488 8.5 9.487 8.5 9.485 C 8.5 9.483 8.5 9.482 8.5 9.48 C 8.5 9.478 8.5 9.477 8.5 9.475 C 8.5 9.473 8.5 9.472 8.5 9.47 C 8.5 9.468 8.5 9.467 8.5 9.465 C 8.5 9.464 8.5 9.462 8.5 9.46 C 8.5 9.459 8.5 9.457 8.5 9.455 C 8.5 9.454 8.5 9.452 8.5 9.45 C 8.5 9.449 8.5 9.447 8.5 9.446 C 8.5 9.444 8.5 9.442 8.5 9.441 C 8.5 9.439 8.5 9.437 8.5 9.436 C 8.5 9.434 8.5 9.432 8.5 9.431 C 8.5 9.429 8.5 9.428 8.5 9.426 C 8.5 9.424 8.5 9.423 8.5 9.421 C 8.5 9.419 8.5 9.418 8.5 9.416 C 8.5 9.415 8.5 9.413 8.5 9.411 C 8.5 9.41 8.5 9.408 8.5 9.407 C 8.5 9.405 8.5 9.403 8.5 9.402 C 8.5 9.4 8.5 9.399 8.5 9.397 C 8.5 9.395 8.5 9.394 8.5 9.392 C 8.5 9.39 8.5 9.389 8.5 9.387 C 8.5 9.386 8.5 9.384 8.5 9.382 C 8.5 9.381 8.5 9.379 8.5 9.378 C 8.5 9.376 8.5 9.375 8.5 9.373 C 8.5 9.371 8.5 9.37 8.5 9.368 C 8.5 9.367 8.5 9.365 8.5 9.363 C 8.5 9.362 8.5 9.36 8.5 9.359 C 8.5 9.357 8.5 9.356 8.5 9.354 C 8.5 9.352 8.5 9.351 8.5 9.349 C 8.5 9.348 8.5 9.346 8.5 9.344 C 8.5 9.343 8.5 9.341 8.5 9.34 C 8.5 9.338 8.5 9.337 8.5 9.335 C 8.5 9.334 8.5 9.332 8.5 9.33 C 8.5 9.329 8.5 9.327 8.5 9.326 C 8.5 9.324 8.5 9.323 8.5 9.321 C 8.5 9.319 8.5 9.318 8.5 9.316 C 8.5 9.315 8.5 9.313 8.5 9.312 C 8.5 9.31 8.5 9.309 8.5 9.307 C 8.5 9.306 8.5 9.304 8.5 9.302 C 8.5 9.301 8.5 9.299 8.5 9.298 C 8.5 9.296 8.5 9.295 8.5 9.293 C 8.5 9.292 8.5 9.29 8.5 9.289 C 8.5 9.287 8.5 9.286 8.5 9.284 C 8.5 9.283 8.5 9.281 8.5 9.279 C 8.5 9.278 8.5 9.276 8.5 9.275 C 8.5 9.273 8.5 9.272 8.5 9.27 C 8.5 9.269 8.5 9.267 8.5 9.266 C 8.5 9.264 8.5 9.263 8.5 9.261 C 8.5 9.26 8.5 9.258 8.5 9.257 C 8.5 9.255 8.5 9.254 8.5 9.252 C 8.5 9.251 8.5 9.249 8.5 9.248 C 8.5 9.246 8.5 9.245 8.5 9.243 C 8.5 9.242 8.5 9.24 8.5 9.239 C 8.5 9.237 8.5 9.236 8.5 9.234 C 8.5 9.233 8.5 9.231 8.5 9.23 C 8.5 9.228 8.5 9.227 8.5 9.226 C 8.5 9.224 8.5 9.223 8.5 9.221 C 8.5 9.22 8.5 9.218 8.5 9.217 C 8.5 9.215 8.5 9.214 8.5 9.212 C 8.5 9.211 8.5 9.209 8.5 9.208 C 8.5 9.206 8.5 9.205 8.5 9.204 C 8.5 9.202 8.5 9.201 8.5 9.199 C 8.5 9.198 8.5 9.196 8.5 9.195 C 8.5 9.193 8.5 9.192 8.5 9.191 C 8.5 9.189 8.5 9.188 8.5 9.186 C 8.5 9.185 8.5 9.183 8.5 9.182 C 8.5 9.18 8.5 9.179 8.5 9.178 C 8.5 9.176 8.5 9.175 8.5 9.173 C 8.5 9.172 8.5 9.171 8.5 9.169 C 8.5 9.168 8.5 9.166 8.5 9.165 C 8.5 9.163 8.5 9.162 8.5 9.161 C 8.5 9.159 8.5 9.158 8.5 9.156 C 8.5 9.155 8.5 9.154 8.5 9.152 C 8.5 9.151 8.5 9.149 8.5 9.148 C 8.5 9.147 8.5 9.145 8.5 9.144 C 8.5 9.142 8.5 9.141 8.5 9.14 C 8.5 9.138 8.5 9.137 8.5 9.135 C 8.5 9.134 8.5 9.133 8.5 9.131 C 8.5 9.13 8.5 9.129 8.5 9.127 C 8.5 9.126 8.5 9.124 8.5 9.123 C 8.5 9.122 8.5 9.12 8.5 9.119 C 8.5 9.118 8.5 9.116 8.5 9.115 C 8.5 9.114 8.5 9.112 8.5 9.111 C 8.5 9.11 8.5 9.108 8.5 9.107 C 8.5 9.105 8.5 9.104 8.5 9.103 C 8.5 9.101 8.5 9.1 8.5 9.099 C 8.5 9.097 8.5 9.096 8.5 9.095 C 8.5 9.093 8.5 9.092 8.5 9.091 C 8.5 9.089 8.5 9.088 8.5 9.087 C 8.5 9.086 8.5 9.084 8.5 9.083 C 8.5 9.082 8.5 9.08 8.5 9.079 C 8.5 9.078 8.5 9.076 8.5 9.075 C 8.5 9.074 8.5 9.072 8.5 9.071 C 8.5 9.07 8.5 9.069 8.5 9.067 C 8.5 9.066 8.5 9.065 8.5 9.063 C 8.5 9.062 8.5 9.061 8.5 9.059 C 8.5 9.058 8.5 9.057 8.5 9.056 C 8.5 9.054 8.5 9.053 8.5 9.052 C 8.5 9.051 8.5 9.049 8.5 9.048 C 8.5 9.047 8.5 9.045 8.5 9.044 C 8.5 9.043 8.5 9.042 8.5 9.04 C 8.5 9.039 8.5 9.038 8.5 9.037 C 8.5 9.035 8.5 9.034 8.5 9.033 C 8.5 9.032 8.5 9.03 8.5 9.029 C 8.5 9.028 8.5 9.027 8.5 9.025 C 8.5 9.024 8.5 9.023 8.5 9.022 C 8.5 9.021 8.5 9.019 8.5 9.018 C 8.5 9.017 8.5 9.016 8.5 9.014 C 8.5 9.013 8.5 9.012 8.5 9.011 C 8.5 9.01 8.5 9.008 8.5 9.007 C 8.5 9.006 8.5 9.005 8.5 9.004 C 8.5 9.002 8.5 9.001 8.5 9 L 7.5 9 Z M 8 10.5 L 6 10.5 L 6 11.5 L 8 11.5 L 8 10.5 Z M 6 10.5 C 5.172 10.5 4.5 9.828 4.5 9 L 3.5 9 C 3.5 10.381 4.619 11.5 6 11.5 L 6 10.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "hardware-gamepad": "<g transform=\"translate(1.688,3.83)\"><path d=\"M 1.812 0.67 L 1.682 0.187 L 1.417 0.258 L 1.335 0.52 L 1.812 0.67 Z M 0 6.431 L -0.477 6.281 L -0.521 6.42 L -0.483 6.561 L 0 6.431 Z M 4.312 0 L 4.441 -0.483 L 4.312 -0.518 L 4.182 -0.483 L 4.312 0 Z M 5.812 0.402 L 5.682 0.885 L 5.746 0.902 L 5.812 0.902 L 5.812 0.402 Z M 3.312 6.67 L 3.182 6.187 L 2.904 6.262 L 2.829 6.54 L 3.312 6.67 Z M 2.776 8.67 L 2.905 9.153 L 3.184 9.078 L 3.259 8.799 L 2.776 8.67 Z M 6.312 5.866 L 6.441 5.383 L 6.312 5.348 L 6.183 5.383 L 6.312 5.866 Z M 1.776 8.938 L 1.587 9.401 L 1.743 9.464 L 1.905 9.421 L 1.776 8.938 Z M 10.812 0.67 L 11.289 0.52 L 11.207 0.258 L 10.942 0.187 L 10.812 0.67 Z M 12.624 6.431 L 13.107 6.561 L 13.145 6.42 L 13.101 6.281 L 12.624 6.431 Z M 8.313 0 L 8.442 -0.483 L 8.313 -0.518 L 8.183 -0.483 L 8.313 0 Z M 6.812 0.402 L 6.812 0.902 L 6.878 0.902 L 6.942 0.885 L 6.812 0.402 Z M 9.312 6.67 L 9.795 6.54 L 9.72 6.262 L 9.442 6.187 L 9.312 6.67 Z M 9.848 8.67 L 9.365 8.799 L 9.44 9.078 L 9.719 9.153 L 9.848 8.67 Z M 10.848 8.938 L 10.719 9.421 L 10.881 9.464 L 11.037 9.401 L 10.848 8.938 Z M 0.536 8.431 L 0.053 8.561 L 0.117 8.8 L 0.347 8.894 L 0.536 8.431 Z M 12.088 8.431 L 12.277 8.894 L 12.507 8.8 L 12.571 8.561 L 12.088 8.431 Z M 3.312 2.17 L 3.312 5.17 L 4.312 5.17 L 4.312 2.17 L 3.312 2.17 Z M 2.312 4.17 L 5.312 4.17 L 5.312 3.17 L 2.312 3.17 L 2.312 4.17 Z M 1.335 0.52 L -0.477 6.281 L 0.477 6.581 L 2.289 0.82 L 1.335 0.52 Z M 4.182 0.483 L 5.682 0.885 L 5.941 -0.081 L 4.441 -0.483 L 4.182 0.483 Z M 1.941 1.153 L 4.441 0.483 L 4.182 -0.483 L 1.682 0.187 L 1.941 1.153 Z M 2.829 6.54 L 2.293 8.54 L 3.259 8.799 L 3.795 6.799 L 2.829 6.54 Z M 3.441 7.153 L 6.441 6.349 L 6.183 5.383 L 3.182 6.187 L 3.441 7.153 Z M 1.905 9.421 L 2.905 9.153 L 2.646 8.187 L 1.646 8.455 L 1.905 9.421 Z M 10.335 0.82 L 12.147 6.581 L 13.101 6.281 L 11.289 0.52 L 10.335 0.82 Z M 8.183 -0.483 L 6.683 -0.081 L 6.942 0.885 L 8.442 0.483 L 8.183 -0.483 Z M 10.942 0.187 L 8.442 -0.483 L 8.183 0.483 L 10.683 1.153 L 10.942 0.187 Z M 8.829 6.799 L 9.365 8.799 L 10.331 8.54 L 9.795 6.54 L 8.829 6.799 Z M 9.442 6.187 L 6.441 5.383 L 6.183 6.349 L 9.183 7.153 L 9.442 6.187 Z M 10.978 8.455 L 9.978 8.187 L 9.719 9.153 L 10.719 9.421 L 10.978 8.455 Z M 5.812 0.902 L 6.812 0.902 L 6.812 -0.098 L 5.812 -0.098 L 5.812 0.902 Z M -0.483 6.561 L 0.053 8.561 L 1.019 8.302 L 0.483 6.302 L -0.483 6.561 Z M 1.965 8.475 L 0.725 7.968 L 0.347 8.894 L 1.587 9.401 L 1.965 8.475 Z M 12.141 6.302 L 11.605 8.302 L 12.571 8.561 L 13.107 6.561 L 12.141 6.302 Z M 11.037 9.401 L 12.277 8.894 L 11.899 7.968 L 10.659 8.475 L 11.037 9.401 Z M 9.312 3.67 C 9.312 3.946 9.088 4.17 8.812 4.17 L 8.812 5.17 C 9.64 5.17 10.312 4.498 10.312 3.67 L 9.312 3.67 Z M 8.812 4.17 C 8.536 4.17 8.312 3.946 8.312 3.67 L 7.312 3.67 C 7.312 4.498 7.984 5.17 8.812 5.17 L 8.812 4.17 Z M 8.312 3.67 C 8.312 3.394 8.536 3.17 8.812 3.17 L 8.812 2.17 C 7.984 2.17 7.312 2.841 7.312 3.67 L 8.312 3.67 Z M 8.812 3.17 C 9.088 3.17 9.312 3.394 9.312 3.67 L 10.312 3.67 C 10.312 2.841 9.64 2.17 8.812 2.17 L 8.812 3.17 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "hardware-gpu": "<g transform=\"translate(1.5,1.5)\"><path d=\"M 13 13 L 13 13.5 L 13.5 13.5 L 13.5 13 L 13 13 Z M 0 13 L -0.5 13 L -0.5 13.5 L 0 13.5 L 0 13 Z M 13 0 L 13.5 0 L 13.5 -0.5 L 13 -0.5 L 13 0 Z M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 0.5 5 L 1 5 L 1 4.5 L 0.5 4.5 L 0.5 5 Z M 0.5 8 L 0.5 8.5 L 1 8.5 L 1 8 L 0.5 8 Z M 5 0.5 L 4.5 0.5 L 4.5 1 L 5 1 L 5 0.5 Z M 8 0.5 L 8 1 L 8.5 1 L 8.5 0.5 L 8 0.5 Z M 12.5 5 L 12.5 4.5 L 12 4.5 L 12 5 L 12.5 5 Z M 12.5 8 L 12 8 L 12 8.5 L 12.5 8.5 L 12.5 8 Z M 5 12.5 L 5 12 L 4.5 12 L 4.5 12.5 L 5 12.5 Z M 8 12.5 L 8.5 12.5 L 8.5 12 L 8 12 L 8 12.5 Z M 3 3 L 3 2.5 L 2.5 2.5 L 2.5 3 L 3 3 Z M 4 3 L 4.5 3 L 4.5 2.5 L 4 2.5 L 4 3 Z M 4 4 L 4 4.5 L 4.5 4.5 L 4.5 4 L 4 4 Z M 3 4 L 2.5 4 L 2.5 4.5 L 3 4.5 L 3 4 Z M 3 6 L 3 5.5 L 2.5 5.5 L 2.5 6 L 3 6 Z M 4 6 L 4.5 6 L 4.5 5.5 L 4 5.5 L 4 6 Z M 4 7 L 4 7.5 L 4.5 7.5 L 4.5 7 L 4 7 Z M 3 7 L 2.5 7 L 2.5 7.5 L 3 7.5 L 3 7 Z M 3 9 L 3 8.5 L 2.5 8.5 L 2.5 9 L 3 9 Z M 4 9 L 4.5 9 L 4.5 8.5 L 4 8.5 L 4 9 Z M 4 10 L 4 10.5 L 4.5 10.5 L 4.5 10 L 4 10 Z M 3 10 L 2.5 10 L 2.5 10.5 L 3 10.5 L 3 10 Z M 6 3 L 6 2.5 L 5.5 2.5 L 5.5 3 L 6 3 Z M 7 3 L 7.5 3 L 7.5 2.5 L 7 2.5 L 7 3 Z M 7 4 L 7 4.5 L 7.5 4.5 L 7.5 4 L 7 4 Z M 6 4 L 5.5 4 L 5.5 4.5 L 6 4.5 L 6 4 Z M 6 6 L 6 5.5 L 5.5 5.5 L 5.5 6 L 6 6 Z M 7 6 L 7.5 6 L 7.5 5.5 L 7 5.5 L 7 6 Z M 7 7 L 7 7.5 L 7.5 7.5 L 7.5 7 L 7 7 Z M 6 7 L 5.5 7 L 5.5 7.5 L 6 7.5 L 6 7 Z M 6 9 L 6 8.5 L 5.5 8.5 L 5.5 9 L 6 9 Z M 7 9 L 7.5 9 L 7.5 8.5 L 7 8.5 L 7 9 Z M 7 10 L 7 10.5 L 7.5 10.5 L 7.5 10 L 7 10 Z M 6 10 L 5.5 10 L 5.5 10.5 L 6 10.5 L 6 10 Z M 9 3 L 9 2.5 L 8.5 2.5 L 8.5 3 L 9 3 Z M 10 3 L 10.5 3 L 10.5 2.5 L 10 2.5 L 10 3 Z M 10 4 L 10 4.5 L 10.5 4.5 L 10.5 4 L 10 4 Z M 9 4 L 8.5 4 L 8.5 4.5 L 9 4.5 L 9 4 Z M 9 6 L 9 5.5 L 8.5 5.5 L 8.5 6 L 9 6 Z M 10 6 L 10.5 6 L 10.5 5.5 L 10 5.5 L 10 6 Z M 10 7 L 10 7.5 L 10.5 7.5 L 10.5 7 L 10 7 Z M 9 7 L 8.5 7 L 8.5 7.5 L 9 7.5 L 9 7 Z M 9 9 L 9 8.5 L 8.5 8.5 L 8.5 9 L 9 9 Z M 10 9 L 10.5 9 L 10.5 8.5 L 10 8.5 L 10 9 Z M 10 10 L 10 10.5 L 10.5 10.5 L 10.5 10 L 10 10 Z M 9 10 L 8.5 10 L 8.5 10.5 L 9 10.5 L 9 10 Z M 13 12.5 L 0 12.5 L 0 13.5 L 13 13.5 L 13 12.5 Z M 12.5 0 L 12.5 13 L 13.5 13 L 13.5 0 L 12.5 0 Z M 0 0.5 L 13 0.5 L 13 -0.5 L 0 -0.5 L 0 0.5 Z M 0.5 13 L 0.5 0 L -0.5 0 L -0.5 13 L 0.5 13 Z M 0 5.5 L 0.5 5.5 L 0.5 4.5 L 0 4.5 L 0 5.5 Z M 0 5 L 0 8 L 1 8 L 1 5 L 0 5 Z M 0.5 7.5 L 0 7.5 L 0 8.5 L 0.5 8.5 L 0.5 7.5 Z M 4.5 0 L 4.5 0.5 L 5.5 0.5 L 5.5 0 L 4.5 0 Z M 5 1 L 8 1 L 8 0 L 5 0 L 5 1 Z M 8.5 0.5 L 8.5 0 L 7.5 0 L 7.5 0.5 L 8.5 0.5 Z M 13 4.5 L 12.5 4.5 L 12.5 5.5 L 13 5.5 L 13 4.5 Z M 12 5 L 12 8 L 13 8 L 13 5 L 12 5 Z M 12.5 8.5 L 13 8.5 L 13 7.5 L 12.5 7.5 L 12.5 8.5 Z M 5.5 13 L 5.5 12.5 L 4.5 12.5 L 4.5 13 L 5.5 13 Z M 5 13 L 8 13 L 8 12 L 5 12 L 5 13 Z M 7.5 12.5 L 7.5 13 L 8.5 13 L 8.5 12.5 L 7.5 12.5 Z M 3 3.5 L 4 3.5 L 4 2.5 L 3 2.5 L 3 3.5 Z M 3.5 3 L 3.5 4 L 4.5 4 L 4.5 3 L 3.5 3 Z M 4 3.5 L 3 3.5 L 3 4.5 L 4 4.5 L 4 3.5 Z M 3.5 4 L 3.5 3 L 2.5 3 L 2.5 4 L 3.5 4 Z M 3 6.5 L 4 6.5 L 4 5.5 L 3 5.5 L 3 6.5 Z M 3.5 6 L 3.5 7 L 4.5 7 L 4.5 6 L 3.5 6 Z M 4 6.5 L 3 6.5 L 3 7.5 L 4 7.5 L 4 6.5 Z M 3.5 7 L 3.5 6 L 2.5 6 L 2.5 7 L 3.5 7 Z M 3 9.5 L 4 9.5 L 4 8.5 L 3 8.5 L 3 9.5 Z M 3.5 9 L 3.5 10 L 4.5 10 L 4.5 9 L 3.5 9 Z M 4 9.5 L 3 9.5 L 3 10.5 L 4 10.5 L 4 9.5 Z M 3.5 10 L 3.5 9 L 2.5 9 L 2.5 10 L 3.5 10 Z M 6 3.5 L 7 3.5 L 7 2.5 L 6 2.5 L 6 3.5 Z M 6.5 3 L 6.5 4 L 7.5 4 L 7.5 3 L 6.5 3 Z M 7 3.5 L 6 3.5 L 6 4.5 L 7 4.5 L 7 3.5 Z M 6.5 4 L 6.5 3 L 5.5 3 L 5.5 4 L 6.5 4 Z M 6 6.5 L 7 6.5 L 7 5.5 L 6 5.5 L 6 6.5 Z M 6.5 6 L 6.5 7 L 7.5 7 L 7.5 6 L 6.5 6 Z M 7 6.5 L 6 6.5 L 6 7.5 L 7 7.5 L 7 6.5 Z M 6.5 7 L 6.5 6 L 5.5 6 L 5.5 7 L 6.5 7 Z M 6 9.5 L 7 9.5 L 7 8.5 L 6 8.5 L 6 9.5 Z M 6.5 9 L 6.5 10 L 7.5 10 L 7.5 9 L 6.5 9 Z M 7 9.5 L 6 9.5 L 6 10.5 L 7 10.5 L 7 9.5 Z M 6.5 10 L 6.5 9 L 5.5 9 L 5.5 10 L 6.5 10 Z M 9 3.5 L 10 3.5 L 10 2.5 L 9 2.5 L 9 3.5 Z M 9.5 3 L 9.5 4 L 10.5 4 L 10.5 3 L 9.5 3 Z M 10 3.5 L 9 3.5 L 9 4.5 L 10 4.5 L 10 3.5 Z M 9.5 4 L 9.5 3 L 8.5 3 L 8.5 4 L 9.5 4 Z M 9 6.5 L 10 6.5 L 10 5.5 L 9 5.5 L 9 6.5 Z M 9.5 6 L 9.5 7 L 10.5 7 L 10.5 6 L 9.5 6 Z M 10 6.5 L 9 6.5 L 9 7.5 L 10 7.5 L 10 6.5 Z M 9.5 7 L 9.5 6 L 8.5 6 L 8.5 7 L 9.5 7 Z M 9 9.5 L 10 9.5 L 10 8.5 L 9 8.5 L 9 9.5 Z M 9.5 9 L 9.5 10 L 10.5 10 L 10.5 9 L 9.5 9 Z M 10 9.5 L 9 9.5 L 9 10.5 L 10 10.5 L 10 9.5 Z M 9.5 10 L 9.5 9 L 8.5 9 L 8.5 10 L 9.5 10 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "hardware-gpu-card": "<g transform=\"translate(1,4.5)\"><path d=\"M 1.5 0 L 2 0 L 2 -0.5 L 1.5 -0.5 L 1.5 0 Z M 12.5 1 L 13 1 L 13 0.5 L 12.5 0.5 L 12.5 1 Z M 3.5 6 L 4 6 L 4 5.5 L 3.5 5.5 L 3.5 6 Z M 3.5 7 L 3 7 L 3 7.5 L 3.5 7.5 L 3.5 7 Z M 7.5 7 L 7.5 7.5 L 8 7.5 L 8 7 L 7.5 7 Z M 7.5 6 L 7.5 5.5 L 7 5.5 L 7 6 L 7.5 6 Z M 12.5 6 L 12.5 6.5 L 13 6.5 L 13 6 L 12.5 6 Z M 0 0.5 L 1.5 0.5 L 1.5 -0.5 L 0 -0.5 L 0 0.5 Z M 1 0 L 1 7.5 L 2 7.5 L 2 0 L 1 0 Z M 12.5 0.5 L 1.5 0.5 L 1.5 1.5 L 12.5 1.5 L 12.5 0.5 Z M 1.5 6.5 L 3.5 6.5 L 3.5 5.5 L 1.5 5.5 L 1.5 6.5 Z M 3 6 L 3 7 L 4 7 L 4 6 L 3 6 Z M 3.5 7.5 L 7.5 7.5 L 7.5 6.5 L 3.5 6.5 L 3.5 7.5 Z M 8 7 L 8 6 L 7 6 L 7 7 L 8 7 Z M 7.5 6.5 L 12.5 6.5 L 12.5 5.5 L 7.5 5.5 L 7.5 6.5 Z M 13 6 L 13 1 L 12 1 L 12 6 L 13 6 Z M 10.5 3.5 C 10.5 3.776 10.276 4 10 4 L 10 5 C 10.828 5 11.5 4.328 11.5 3.5 L 10.5 3.5 Z M 10 4 C 9.724 4 9.5 3.776 9.5 3.5 L 8.5 3.5 C 8.5 4.328 9.172 5 10 5 L 10 4 Z M 9.5 3.5 C 9.5 3.224 9.724 3 10 3 L 10 2 C 9.172 2 8.5 2.672 8.5 3.5 L 9.5 3.5 Z M 10 3 C 10.276 3 10.5 3.224 10.5 3.5 L 11.5 3.5 C 11.5 2.672 10.828 2 10 2 L 10 3 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "hardware-gpu-card-multi": "<g transform=\"translate(0,2.5)\"><path d=\"M 1.5 0 L 2 0 L 2 -0.5 L 1.5 -0.5 L 1.5 0 Z M 12.5 1 L 13 1 L 13 0.5 L 12.5 0.5 L 12.5 1 Z M 3.5 6 L 4 6 L 4 5.5 L 3.5 5.5 L 3.5 6 Z M 3.5 7 L 3 7 L 3 7.5 L 3.5 7.5 L 3.5 7 Z M 7.5 7 L 7.5 7.5 L 8 7.5 L 8 7 L 7.5 7 Z M 7.5 6 L 7.5 5.5 L 7 5.5 L 7 6 L 7.5 6 Z M 12.5 6 L 12.5 6.5 L 13 6.5 L 13 6 L 12.5 6 Z M 9.5 9 L 9.5 9.5 L 10 9.5 L 10 9 L 9.5 9 Z M 9.5 8 L 9.5 7.5 L 9 7.5 L 9 8 L 9.5 8 Z M 14.5 8 L 14.5 8.5 L 15 8.5 L 15 8 L 14.5 8 Z M 0 0.5 L 1.5 0.5 L 1.5 -0.5 L 0 -0.5 L 0 0.5 Z M 1 0 L 1 7.5 L 2 7.5 L 2 0 L 1 0 Z M 12.5 0.5 L 1.5 0.5 L 1.5 1.5 L 12.5 1.5 L 12.5 0.5 Z M 1.5 6.5 L 3.5 6.5 L 3.5 5.5 L 1.5 5.5 L 1.5 6.5 Z M 3 6 L 3 7 L 4 7 L 4 6 L 3 6 Z M 3.5 7.5 L 7.5 7.5 L 7.5 6.5 L 3.5 6.5 L 3.5 7.5 Z M 8 7 L 8 6 L 7 6 L 7 7 L 8 7 Z M 7.5 6.5 L 12.5 6.5 L 12.5 5.5 L 7.5 5.5 L 7.5 6.5 Z M 13 6 L 13 1 L 12 1 L 12 6 L 13 6 Z M 10.5 3.5 C 10.5 3.776 10.276 4 10 4 L 10 5 C 10.828 5 11.5 4.328 11.5 3.5 L 10.5 3.5 Z M 10 4 C 9.724 4 9.5 3.776 9.5 3.5 L 8.5 3.5 C 8.5 4.328 9.172 5 10 5 L 10 4 Z M 9.5 3.5 C 9.5 3.224 9.724 3 10 3 L 10 2 C 9.172 2 8.5 2.672 8.5 3.5 L 9.5 3.5 Z M 10 3 C 10.276 3 10.5 3.224 10.5 3.5 L 11.5 3.5 C 11.5 2.672 10.828 2 10 2 L 10 3 Z M 10 9 L 10 8 L 9 8 L 9 9 L 10 9 Z M 15 8 L 15 3 L 14 3 L 14 8 L 15 8 Z M 9.5 8.5 L 14.5 8.5 L 14.5 7.5 L 9.5 7.5 L 9.5 8.5 Z M 5 9.5 L 9.5 9.5 L 9.5 8.5 L 5 8.5 L 5 9.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "hardware-headset": "<g transform=\"translate(2.5,2.5)\"><path d=\"M 11 11 L 11 11.5 L 11.5 11.5 L 11.5 11 L 11 11 Z M 0.5 5.5 C 0.5 3.955 1.007 2.714 1.861 1.861 C 2.714 1.007 3.955 0.5 5.5 0.5 L 5.5 -0.5 C 3.731 -0.5 2.222 0.086 1.154 1.154 C 0.086 2.222 -0.5 3.731 -0.5 5.5 L 0.5 5.5 Z M 5.5 0.5 C 7.045 0.5 8.286 1.007 9.139 1.861 C 9.993 2.714 10.5 3.955 10.5 5.5 L 11.5 5.5 C 11.5 3.731 10.914 2.222 9.846 1.154 C 8.778 0.086 7.269 -0.5 5.5 -0.5 L 5.5 0.5 Z M 10.5 8.5 L 10.5 11 L 11.5 11 L 11.5 8.5 L 10.5 8.5 Z M 11 10.5 L 6.5 10.5 L 6.5 11.5 L 11 11.5 L 11 10.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(0,0)\"><path d=\"M 2 0 L 2.5 0 L 2.5 -0.5 L 2 -0.5 L 2 0 Z M 0.5 0 L 0.5 -0.5 L 0.14 -0.5 L 0.026 -0.158 L 0.5 0 Z M 0 1.5 L -0.474 1.342 L -0.527 1.5 L -0.474 1.658 L 0 1.5 Z M 0.5 3 L 0.026 3.158 L 0.14 3.5 L 0.5 3.5 L 0.5 3 Z M 2 3 L 2 3.5 L 2.5 3.5 L 2.5 3 L 2 3 Z M 2 -0.5 L 0.5 -0.5 L 0.5 0.5 L 2 0.5 L 2 -0.5 Z M -0.474 1.658 L 0.026 3.158 L 0.974 2.842 L 0.474 1.342 L -0.474 1.658 Z M 0.5 3.5 L 2 3.5 L 2 2.5 L 0.5 2.5 L 0.5 3.5 Z M 2.5 3 L 2.5 0 L 1.5 0 L 1.5 3 L 2.5 3 Z M 0.474 1.658 L 0.974 0.158 L 0.026 -0.158 L -0.474 1.342 L 0.474 1.658 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(7.5,13)\"><path d=\"M 1 0 L 1.5 0 L 1.5 -0.5 L 1 -0.5 L 1 0 Z M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 0 1 L -0.5 1 L -0.5 1.5 L 0 1.5 L 0 1 Z M 1 1 L 1 1.5 L 1.5 1.5 L 1.5 1 L 1 1 Z M -0.5 0 L -0.5 1 L 0.5 1 L 0.5 0 L -0.5 0 Z M 0 1.5 L 1 1.5 L 1 0.5 L 0 0.5 L 0 1.5 Z M 1.5 1 L 1.5 0 L 0.5 0 L 0.5 1 L 1.5 1 Z M 1 -0.5 L 0 -0.5 L 0 0.5 L 1 0.5 L 1 -0.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g><g transform=\"translate(1.5,7.5)\"><path d=\"M 2 0 L 2.5 0 L 2.5 -0.5 L 2 -0.5 L 2 0 Z M 0.5 0 L 0.5 -0.5 L 0.14 -0.5 L 0.026 -0.158 L 0.5 0 Z M 0 1.5 L -0.474 1.342 L -0.527 1.5 L -0.474 1.658 L 0 1.5 Z M 0.5 3 L 0.026 3.158 L 0.14 3.5 L 0.5 3.5 L 0.5 3 Z M 2 3 L 2 3.5 L 2.5 3.5 L 2.5 3 L 2 3 Z M 2 -0.5 L 0.5 -0.5 L 0.5 0.5 L 2 0.5 L 2 -0.5 Z M -0.474 1.658 L 0.026 3.158 L 0.974 2.842 L 0.474 1.342 L -0.474 1.658 Z M 0.5 3.5 L 2 3.5 L 2 2.5 L 0.5 2.5 L 0.5 3.5 Z M 2.5 3 L 2.5 0 L 1.5 0 L 1.5 3 L 2.5 3 Z M 0.474 1.658 L 0.974 0.158 L 0.026 -0.158 L -0.474 1.342 L 0.474 1.658 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "hardware-keyboard": "<g transform=\"translate(1.5,3.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 13 0 L 13.5 0 L 13.5 -0.5 L 13 -0.5 L 13 0 Z M 13 8 L 13 8.5 L 13.5 8.5 L 13.5 8 L 13 8 Z M 0 8 L -0.5 8 L -0.5 8.5 L 0 8.5 L 0 8 Z M 1.5 2.5 L 2.5 2.5 L 2.5 1.5 L 1.5 1.5 L 1.5 2.5 Z M 0 0.5 L 13 0.5 L 13 -0.5 L 0 -0.5 L 0 0.5 Z M 12.5 0 L 12.5 8 L 13.5 8 L 13.5 0 L 12.5 0 Z M 13 7.5 L 0 7.5 L 0 8.5 L 13 8.5 L 13 7.5 Z M 0.5 8 L 0.5 0 L -0.5 0 L -0.5 8 L 0.5 8 Z M 1.5 4.5 L 2.5 4.5 L 2.5 3.5 L 1.5 3.5 L 1.5 4.5 Z M 1.5 6.5 L 2.5 6.5 L 2.5 5.5 L 1.5 5.5 L 1.5 6.5 Z M 4.5 2.5 L 5.5 2.5 L 5.5 1.5 L 4.5 1.5 L 4.5 2.5 Z M 3 2.5 L 4 2.5 L 4 1.5 L 3 1.5 L 3 2.5 Z M 3 4.5 L 4 4.5 L 4 3.5 L 3 3.5 L 3 4.5 Z M 6 2.5 L 7 2.5 L 7 1.5 L 6 1.5 L 6 2.5 Z M 6 4.5 L 7 4.5 L 7 3.5 L 6 3.5 L 6 4.5 Z M 9 2.5 L 10 2.5 L 10 1.5 L 9 1.5 L 9 2.5 Z M 9 4.5 L 10 4.5 L 10 3.5 L 9 3.5 L 9 4.5 Z M 7.5 2.5 L 8.5 2.5 L 8.5 1.5 L 7.5 1.5 L 7.5 2.5 Z M 4.5 4.5 L 5.5 4.5 L 5.5 3.5 L 4.5 3.5 L 4.5 4.5 Z M 3.5 6.5 L 9.5 6.5 L 9.5 5.5 L 3.5 5.5 L 3.5 6.5 Z M 10.5 2.5 L 11.5 2.5 L 11.5 1.5 L 10.5 1.5 L 10.5 2.5 Z M 7.5 4.5 L 8.5 4.5 L 8.5 3.5 L 7.5 3.5 L 7.5 4.5 Z M 10.5 4.5 L 11.5 4.5 L 11.5 3.5 L 10.5 3.5 L 10.5 4.5 Z M 10.5 6.5 L 11.5 6.5 L 11.5 5.5 L 10.5 5.5 L 10.5 6.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "hardware-laptop": "<g transform=\"translate(2,3.5)\"><path d=\"M 1.5 0 L 1.5 -0.5 L 1 -0.5 L 1 0 L 1.5 0 Z M 10.5 0 L 11 0 L 11 -0.5 L 10.5 -0.5 L 10.5 0 Z M 10.5 6 L 10.5 6.5 L 11 6.5 L 11 6 L 10.5 6 Z M 1.5 6 L 1 6 L 1 6.5 L 1.5 6.5 L 1.5 6 Z M 1.5 0.5 L 10.5 0.5 L 10.5 -0.5 L 1.5 -0.5 L 1.5 0.5 Z M 10 0 L 10 6 L 11 6 L 11 0 L 10 0 Z M 2 6 L 2 0 L 1 0 L 1 6 L 2 6 Z M 10.5 5.5 L 1.5 5.5 L 1.5 6.5 L 10.5 6.5 L 10.5 5.5 Z M 0 8.5 L 12 8.5 L 12 7.5 L 0 7.5 L 0 8.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "hardware-lightning": "<g transform=\"translate(4.624,2.5)\"><path d=\"M 0 7 L -0.483 6.871 L -0.652 7.5 L 0 7.5 L 0 7 Z M 1.876 0 L 1.876 -0.5 L 1.492 -0.5 L 1.393 -0.129 L 1.876 0 Z M 2.876 11 L 2.376 11 L 2.376 12.883 L 3.31 11.248 L 2.876 11 Z M 6.876 4 L 7.31 4.248 L 7.738 3.5 L 6.876 3.5 L 6.876 4 Z M 3.804 4 L 3.321 3.871 L 3.152 4.5 L 3.804 4.5 L 3.804 4 Z M 4.876 0 L 5.359 0.129 L 5.527 -0.5 L 4.876 -0.5 L 4.876 0 Z M 2.876 7 L 3.376 7 L 3.376 6.5 L 2.876 6.5 L 2.876 7 Z M 0.483 7.129 L 2.359 0.129 L 1.393 -0.129 L -0.483 6.871 L 0.483 7.129 Z M 6.442 3.752 L 2.442 10.752 L 3.31 11.248 L 7.31 4.248 L 6.442 3.752 Z M 3.804 4.5 L 6.876 4.5 L 6.876 3.5 L 3.804 3.5 L 3.804 4.5 Z M 4.393 -0.129 L 3.321 3.871 L 4.287 4.129 L 5.359 0.129 L 4.393 -0.129 Z M 1.876 0.5 L 4.876 0.5 L 4.876 -0.5 L 1.876 -0.5 L 1.876 0.5 Z M 3.376 11 L 3.376 7 L 2.376 7 L 2.376 11 L 3.376 11 Z M 2.876 6.5 L 0 6.5 L 0 7.5 L 2.876 7.5 L 2.876 6.5 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>",
  "hardware-mac": "<g transform=\"translate(2.5,3.5)\"><path d=\"M 0 0 L 0 -0.5 L -0.5 -0.5 L -0.5 0 L 0 0 Z M 11 0 L 11.5 0 L 11.5 -0.5 L 11 -0.5 L 11 0 Z M 11 7 L 11 7.5 L 11.5 7.5 L 11.5 7 L 11 7 Z M 0 7 L -0.5 7 L -0.5 7.5 L 0 7.5 L 0 7 Z M 0 0.5 L 11 0.5 L 11 -0.5 L 0 -0.5 L 0 0.5 Z M 10.5 0 L 10.5 7 L 11.5 7 L 11.5 0 L 10.5 0 Z M 11 6.5 L 0 6.5 L 0 7.5 L 11 7.5 L 11 6.5 Z M 0.5 7 L 0.5 0 L -0.5 0 L -0.5 7 L 0.5 7 Z M 0 5.5 L 11 5.5 L 11 4.5 L 0 4.5 L 0 5.5 Z M 4.198 9.129 L 4.734 7.129 L 3.768 6.871 L 3.232 8.871 L 4.198 9.129 Z M 2.5 9.5 L 8.5 9.5 L 8.5 8.5 L 2.5 8.5 L 2.5 9.5 Z M 7.769 8.871 L 7.233 6.871 L 6.267 7.129 L 6.803 9.129 L 7.769 8.871 Z\" fill=\"currentColor\" fill-rule=\"nonzero\" /></g>"
};

/**
 * NVIDIA Icon. Renders a line icon from the official GUI set by `name`
 * (e.g. "hardware-gpu", "common-magnifying-glass", "av-play"). Inherits color
 * via currentColor; set `size` in px. `iconNames` lists everything available.
 */
function Icon({
  name,
  size = 20,
  color = "currentColor",
  title,
  style,
  ...rest
}) {
  const inner = ICONS[name];
  if (!inner) return null;
  return /*#__PURE__*/React.createElement("svg", _extends({
    viewBox: "0 0 16 16",
    width: size,
    height: size,
    fill: "none",
    role: "img",
    "aria-label": title || name,
    style: {
      display: "inline-block",
      color,
      flex: "none",
      ...style
    },
    dangerouslySetInnerHTML: {
      __html: inner
    }
  }, rest));
}
const iconNames = Object.keys(ICONS);
Object.assign(__ds_scope, { Icon, iconNames });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/icons/Icon.jsx", error: String((e && e.message) || e) }); }

// components/feedback/Banner.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
// NVIDIA / KUI inline status Banner (alert). Rounded card with a status-colored
// left accent, a status icon, title + message, and optional action / dismiss.
// Status colors resolve through semantic tokens, so they adapt in Kaizen mode.
const STATUS = {
  info: {
    icon: "common-info-circle",
    color: "var(--nv-info)"
  },
  success: {
    icon: "common-check-circle",
    color: "var(--nv-success)"
  },
  warning: {
    icon: "common-warning",
    color: "var(--nv-warning)"
  },
  error: {
    icon: "common-close-circle",
    color: "var(--nv-danger)"
  }
};
function Banner({
  status = "info",
  title,
  children,
  action,
  onClose,
  style,
  ...rest
}) {
  const s = STATUS[status] || STATUS.info;
  return /*#__PURE__*/React.createElement("div", _extends({
    role: "status",
    style: {
      display: "flex",
      gap: 12,
      alignItems: "flex-start",
      padding: "12px 14px",
      borderRadius: "var(--radius-sm)",
      border: "1px solid var(--border-subtle)",
      borderLeft: `3px solid ${s.color}`,
      background: "var(--surface-card)",
      boxShadow: "var(--shadow-xs)",
      fontFamily: "var(--font-sans)",
      ...style
    }
  }, rest), /*#__PURE__*/React.createElement("span", {
    style: {
      color: s.color,
      flex: "none",
      display: "inline-flex",
      marginTop: 1
    }
  }, /*#__PURE__*/React.createElement(__ds_scope.Icon, {
    name: s.icon,
    size: 18
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      flex: 1,
      minWidth: 0
    }
  }, title && /*#__PURE__*/React.createElement("div", {
    style: {
      fontWeight: 600,
      fontSize: 14,
      lineHeight: 1.35,
      color: "var(--text-primary)",
      marginBottom: children ? 3 : 0
    }
  }, title), children && /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 13,
      lineHeight: 1.45,
      color: "var(--text-secondary)"
    }
  }, children), action && /*#__PURE__*/React.createElement("div", {
    style: {
      marginTop: 10,
      display: "flex",
      gap: 8
    }
  }, action)), onClose && /*#__PURE__*/React.createElement("button", {
    type: "button",
    onClick: onClose,
    "aria-label": "Dismiss",
    style: {
      flex: "none",
      border: "none",
      background: "none",
      cursor: "pointer",
      padding: 2,
      margin: "-2px -2px 0 0",
      color: "var(--text-tertiary)",
      display: "inline-flex",
      borderRadius: "var(--radius-xs)"
    }
  }, /*#__PURE__*/React.createElement(__ds_scope.Icon, {
    name: "common-close",
    size: 14
  })));
}
Object.assign(__ds_scope, { Banner });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/feedback/Banner.jsx", error: String((e && e.message) || e) }); }

// components/navigation/Breadcrumb.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
// NVIDIA / KUI Breadcrumb — hierarchical path of links with a chevron separator.
// The last item is rendered as the current page (non-link).
function Chevron() {
  return /*#__PURE__*/React.createElement("svg", {
    width: "12",
    height: "12",
    viewBox: "0 0 16 16",
    "aria-hidden": "true",
    style: {
      flex: "none",
      color: "var(--text-tertiary)"
    }
  }, /*#__PURE__*/React.createElement("path", {
    d: "M 6 3.5 L 10.5 8 L 6 12.5",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: "1.4",
    strokeLinecap: "round",
    strokeLinejoin: "round"
  }));
}
function Breadcrumb({
  items = [],
  separator,
  onNavigate,
  style,
  ...rest
}) {
  return /*#__PURE__*/React.createElement("nav", _extends({
    "aria-label": "Breadcrumb",
    style: style
  }, rest), /*#__PURE__*/React.createElement("ol", {
    style: {
      display: "flex",
      flexWrap: "wrap",
      alignItems: "center",
      gap: 8,
      listStyle: "none",
      margin: 0,
      padding: 0,
      fontFamily: "var(--font-sans)",
      fontSize: 14,
      lineHeight: 1.4
    }
  }, items.map((it, i) => {
    const last = i === items.length - 1;
    return /*#__PURE__*/React.createElement("li", {
      key: i,
      style: {
        display: "inline-flex",
        alignItems: "center",
        gap: 8
      }
    }, last ? /*#__PURE__*/React.createElement("span", {
      "aria-current": "page",
      style: {
        color: "var(--text-primary)",
        fontWeight: 500
      }
    }, it.label) : /*#__PURE__*/React.createElement("a", {
      href: it.href || "#",
      onClick: onNavigate ? e => {
        e.preventDefault();
        onNavigate(it, i);
      } : undefined,
      style: {
        color: "var(--text-secondary)",
        textDecoration: "none"
      }
    }, it.label), !last && (separator !== undefined ? separator : /*#__PURE__*/React.createElement(Chevron, null)));
  })));
}
Object.assign(__ds_scope, { Breadcrumb });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/navigation/Breadcrumb.jsx", error: String((e && e.message) || e) }); }

// components/navigation/Tabs.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * NVIDIA Tabs — underline tab bar. The active tab carries a green underline
 * bar and stronger ink. Controlled via `value`/`onChange` or uncontrolled.
 */
function Tabs({
  items = [],
  value,
  defaultValue,
  onChange,
  style,
  ...rest
}) {
  const norm = items.map(i => typeof i === "string" ? {
    value: i,
    label: i
  } : i);
  const [internal, setInternal] = React.useState(defaultValue ?? norm[0]?.value);
  const active = value !== undefined ? value : internal;
  const select = v => {
    if (value === undefined) setInternal(v);
    onChange && onChange(v);
  };
  return /*#__PURE__*/React.createElement("div", _extends({
    role: "tablist",
    style: {
      display: "flex",
      gap: 4,
      borderBottom: "1px solid var(--border-default)",
      ...style
    }
  }, rest), norm.map(t => {
    const on = t.value === active;
    return /*#__PURE__*/React.createElement("button", {
      key: t.value,
      role: "tab",
      "aria-selected": on,
      onClick: () => select(t.value),
      style: {
        position: "relative",
        appearance: "none",
        background: "transparent",
        border: "none",
        cursor: "pointer",
        padding: "10px 14px",
        marginBottom: -1,
        fontFamily: "var(--font-sans)",
        fontSize: "var(--fs-sm)",
        fontWeight: on ? "var(--fw-semibold)" : "var(--fw-medium)",
        color: on ? "var(--text-primary)" : "var(--text-secondary)",
        display: "inline-flex",
        alignItems: "center",
        gap: 7,
        transition: "color var(--dur-fast)"
      }
    }, t.label, t.count != null && /*#__PURE__*/React.createElement("span", {
      style: {
        fontSize: "var(--fs-2xs)",
        fontWeight: "var(--fw-semibold)",
        color: on ? "var(--nv-green-700)" : "var(--text-tertiary)",
        background: on ? "var(--nv-green-100)" : "var(--nv-gray-100)",
        borderRadius: "var(--radius-pill)",
        padding: "1px 7px"
      }
    }, t.count), /*#__PURE__*/React.createElement("span", {
      style: {
        position: "absolute",
        left: 0,
        right: 0,
        bottom: 0,
        height: 2,
        background: on ? "var(--nv-green)" : "transparent",
        borderRadius: "2px 2px 0 0"
      }
    }));
  }));
}
Object.assign(__ds_scope, { Tabs });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/navigation/Tabs.jsx", error: String((e && e.message) || e) }); }

// components/overlay/Tooltip.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
// NVIDIA / KUI Tooltip — compact dark label shown on hover/focus of its child.
// Wrap a single trigger element; `label` is the tip text. placement: top/bottom/left/right.
function Tooltip({
  label,
  placement = "top",
  children,
  style,
  ...rest
}) {
  const [show, setShow] = React.useState(false);
  const base = {
    position: "absolute",
    zIndex: "var(--z-overlay, 1000)",
    background: "var(--nv-gray-900)",
    color: "var(--nv-white)",
    fontFamily: "var(--font-sans)",
    fontSize: 12,
    lineHeight: 1.3,
    fontWeight: 500,
    padding: "6px 8px",
    borderRadius: "var(--radius-xs)",
    boxShadow: "var(--shadow-md)",
    whiteSpace: "nowrap",
    pointerEvents: "none"
  };
  const place = {
    top: {
      bottom: "100%",
      left: "50%",
      transform: "translateX(-50%)",
      marginBottom: 6
    },
    bottom: {
      top: "100%",
      left: "50%",
      transform: "translateX(-50%)",
      marginTop: 6
    },
    left: {
      right: "100%",
      top: "50%",
      transform: "translateY(-50%)",
      marginRight: 6
    },
    right: {
      left: "100%",
      top: "50%",
      transform: "translateY(-50%)",
      marginLeft: 6
    }
  }[placement] || {};
  return /*#__PURE__*/React.createElement("span", _extends({
    style: {
      position: "relative",
      display: "inline-flex",
      ...style
    },
    onMouseEnter: () => setShow(true),
    onMouseLeave: () => setShow(false),
    onFocus: () => setShow(true),
    onBlur: () => setShow(false)
  }, rest), children, show && label && /*#__PURE__*/React.createElement("span", {
    role: "tooltip",
    style: {
      ...base,
      ...place
    }
  }, label));
}
Object.assign(__ds_scope, { Tooltip });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/overlay/Tooltip.jsx", error: String((e && e.message) || e) }); }

// decks/edm-stack/deck-stage.js
try { (() => {
// @ds-adherence-ignore -- omelette starter scaffold (raw elements/hex/px by design)
/* BEGIN USAGE */
/**
 * <deck-stage> — reusable web component for HTML decks.
 *
 * Handles:
 *  (a) speaker notes — reads <script type="application/json" id="speaker-notes">
 *      and posts {slideIndexChanged: N} to the parent window on nav.
 *  (b) keyboard navigation — ←/→, PgUp/PgDn, Space, Home/End, number keys.
 *      On touch devices, tapping the left/right half of the stage goes
 *      prev/next — taps on links, buttons and other interactive slide
 *      content are left alone.
 *  (c) press R to reset to slide 0 (with a tasteful keyboard hint).
 *  (d) bottom-center overlay showing slide count + hints, fades out on idle.
 *  (e) auto-scaling — inner canvas is a fixed design size (default 1920×1080)
 *      scaled with `transform: scale()` to fit the viewport, letterboxed.
 *      Set the `noscale` attribute to render at authored size (1:1) — the
 *      PPTX exporter sets this so its DOM capture sees unscaled geometry.
 *  (f) print — `@media print` lays every slide out as its own page at the
 *      design size, so the browser's Print → Save as PDF produces a clean
 *      one-page-per-slide PDF with no extra setup.
 *  (g) thumbnail rail — resizable left-hand column of per-slide thumbnails
 *      (static clones). Click to navigate; ↑/↓ with a thumbnail focused to
 *      step between slides; drag to reorder; right-click for
 *      Skip / Move up / Move down / Duplicate / Delete (Delete opens a
 *      Cancel/Delete confirm dialog). Drag the rail's right edge to resize;
 *      width persists to
 *      localStorage. Skipped slides carry `data-deck-skip`, are dimmed in
 *      the rail, omitted from prev/next navigation, and hidden at print.
 *      The rail is suppressed in presenting mode, in the host's Preview
 *      mode (ViewerMode='none'), on `noscale`, on narrow viewports
 *      (≤640px), and via the `no-rail` attribute. Rail mutations dispatch
 *      a `deckchange`
 *      CustomEvent on the element: detail = {action, from, to, slide}.
 *
 * Slides are HIDDEN, not unmounted. Non-active slides stay in the DOM with
 * `visibility: hidden` + `opacity: 0`, so their state (videos, iframes,
 * form inputs, React trees) is preserved across navigation.
 *
 * Lifecycle event — the component dispatches a `slidechange` CustomEvent on
 * itself whenever the active slide changes (including the initial mount).
 * The event bubbles and composes out of shadow DOM, so you can listen on
 * the <deck-stage> element or on document:
 *
 *   document.querySelector('deck-stage').addEventListener('slidechange', (e) => {
 *     e.detail.index         // new 0-based index
 *     e.detail.previousIndex // previous index, or -1 on init
 *     e.detail.total         // total slide count
 *     e.detail.slide         // the new active slide element
 *     e.detail.previousSlide // the prior slide element, or null on init
 *     e.detail.reason        // 'init' | 'keyboard' | 'click' | 'tap' | 'api'
 *   });
 *
 * Persistence: none at the deck level. The host app keeps the current slide
 * in its own URL (?slide=) and re-delivers it via location.hash on load, so a
 * bare load with no hash always starts at slide 1.
 *
 * Usage:
 *   <style>deck-stage:not(:defined){visibility:hidden}</style>
 *   <deck-stage width="1920" height="1080">
 *     <section data-label="Title">...</section>
 *     <section data-label="Agenda">...</section>
 *   </deck-stage>
 *   <script src="deck-stage.js"></script>
 *
 * The :not(:defined) rule prevents a flash of the first slide at its
 * authored styles before this script runs and attaches the shadow root.
 *
 * Slides are the direct element children of <deck-stage>. Each slide is
 * automatically tagged with:
 *   - data-screen-label="NN Label"   (1-indexed, for comment flow)
 *   - data-om-validate="no_overflowing_text,no_overlapping_text,slide_sized_text"
 *
 * Speaker notes stay in sync because the component posts {slideIndexChanged: N}
 * to the parent — just include the #speaker-notes script tag if asked for notes.
 *
 * Authoring guidance:
 *   - Write slide bodies as static HTML inside <deck-stage>, with sizing via
 *     CSS custom properties in a <style> block rather than JS constants.
 *     Static slide markup is what lets the user click a heading in edit mode
 *     and retype it directly; a slide rendered through <script type="text/babel">,
 *     React, or a loop over a JS array has to round-trip every tweak through a
 *     chat message instead. Reach for script-generated slides only when the
 *     content genuinely needs interactive behaviour static HTML can't express.
 *   - Do NOT set position/inset/width/height on the slide <section> elements —
 *     the component absolutely positions every slotted child for you.
 *   - Entrance animations: make the visible end-state the base style and
 *     animate *from* hidden, so print and reduced-motion show content.
 *     Gate the animation on [data-deck-active] and the motion query, e.g.
 *     `@media (prefers-reduced-motion:no-preference){ [data-deck-active] .x{animation:fade-in .5s both} }`.
 *     Avoid infinite decorative loops on slide content.
 */
/* END USAGE */

(() => {
  const DESIGN_W_DEFAULT = 1920;
  const DESIGN_H_DEFAULT = 1080;
  const OVERLAY_HIDE_MS = 1800;
  const VALIDATE_ATTR = 'no_overflowing_text,no_overlapping_text,slide_sized_text';
  const FINE_POINTER_MQ = matchMedia('(hover: hover) and (pointer: fine)');
  const NARROW_MQ = matchMedia('(max-width: 640px)');
  // Slide-authored controls that should keep a tap instead of it navigating.
  const INTERACTIVE_SEL = 'a[href], button, input, select, textarea, summary, label, video[controls], audio[controls], [role="button"], [onclick], [tabindex]:not([tabindex^="-"]), [contenteditable]:not([contenteditable="false" i])';
  const pad2 = n => String(n).padStart(2, '0');

  // Label precedence: data-label → data-screen-label (number stripped) → first heading → "Slide".
  const getSlideLabel = el => {
    const explicit = el.getAttribute('data-label');
    if (explicit) return explicit;
    const existing = el.getAttribute('data-screen-label');
    if (existing) return existing.replace(/^\s*\d+\s*/, '').trim() || existing;
    const h = el.querySelector('h1, h2, h3, [data-title]');
    const t = h && (h.textContent || '').trim().slice(0, 40);
    if (t) return t;
    return 'Slide';
  };
  const stylesheet = `
    :host {
      position: fixed;
      inset: 0;
      display: block;
      background: #000;
      color: #fff;
      font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif;
      overflow: hidden;
      -webkit-tap-highlight-color: transparent;
    }
    /* connectedCallback holds this until document.fonts.ready (capped 2s) so
     * the first visible paint has the deck's real typography + final rail
     * layout. opacity (not visibility) so the active slide can't un-hide
     * itself via the ::slotted([data-deck-active]) visibility:visible rule.
     * Only the stage/rail hide — the black :host background stays, so the
     * iframe doesn't flash the page's default white. */
    :host([data-fonts-pending]) .stage,
    :host([data-fonts-pending]) .rail { opacity: 0; pointer-events: none; }

    .stage {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .canvas {
      position: relative;
      transform-origin: center center;
      flex-shrink: 0;
      background: #fff;
      will-change: transform;
    }

    /* Slides live in light DOM (via <slot>) so authored CSS still applies.
       We absolutely position each slotted child to stack them. */
    ::slotted(*) {
      position: absolute !important;
      inset: 0 !important;
      width: 100% !important;
      height: 100% !important;
      box-sizing: border-box !important;
      overflow: hidden;
      opacity: 0;
      pointer-events: none;
      visibility: hidden;
    }
    ::slotted([data-deck-active]) {
      opacity: 1;
      pointer-events: auto;
      visibility: visible;
    }

    .overlay {
      position: fixed;
      left: 50%;
      bottom: 22px;
      transform: translate(-50%, 6px) scale(0.92);
      filter: blur(6px);
      display: flex;
      align-items: center;
      gap: 4px;
      padding: 4px;
      background: #000;
      color: #fff;
      border-radius: 999px;
      font-size: 12px;
      font-feature-settings: "tnum" 1;
      letter-spacing: 0.01em;
      opacity: 0;
      pointer-events: none;
      transition: opacity 260ms ease, transform 260ms cubic-bezier(.2,.8,.2,1), filter 260ms ease;
      transform-origin: center bottom;
      z-index: 2147483000;
      user-select: none;
    }
    .overlay[data-visible] {
      opacity: 1;
      pointer-events: auto;
      transform: translate(-50%, 0) scale(1);
      filter: blur(0);
    }

    .btn {
      appearance: none;
      -webkit-appearance: none;
      background: transparent;
      border: 0;
      margin: 0;
      padding: 0;
      color: inherit;
      font: inherit;
      cursor: default;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      height: 28px;
      min-width: 28px;
      border-radius: 999px;
      color: rgba(255,255,255,0.72);
      transition: background 140ms ease, color 140ms ease;
      -webkit-tap-highlight-color: transparent;
    }
    .btn:hover { background: rgba(255,255,255,0.12); color: #fff; }
    .btn:active { background: rgba(255,255,255,0.18); }
    .btn:focus { outline: none; }
    .btn:focus-visible { outline: none; }
    .btn::-moz-focus-inner { border: 0; }
    .btn svg { width: 14px; height: 14px; display: block; }
    .btn.reset {
      font-size: 11px;
      font-weight: 500;
      letter-spacing: 0.02em;
      padding: 0 10px 0 12px;
      gap: 6px;
      color: rgba(255,255,255,0.72);
    }
    .btn.reset .kbd {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 16px;
      height: 16px;
      padding: 0 4px;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      font-size: 10px;
      line-height: 1;
      color: rgba(255,255,255,0.88);
      background: rgba(255,255,255,0.12);
      border-radius: 4px;
    }

    .count {
      font-variant-numeric: tabular-nums;
      color: #fff;
      font-weight: 500;
      padding: 0 8px;
      min-width: 42px;
      text-align: center;
      font-size: 12px;
    }
    .count .sep { color: rgba(255,255,255,0.45); margin: 0 3px; font-weight: 400; }
    .count .total { color: rgba(255,255,255,0.55); }

    .divider {
      width: 1px;
      height: 14px;
      background: rgba(255,255,255,0.18);
      margin: 0 2px;
    }

    /* ── Thumbnail rail ──────────────────────────────────────────────────
       Fixed column on the left; each thumbnail is a static deep-clone of
       the light-DOM slide scaled into a 16:9 (or design-aspect) frame. The
       stage re-fits around it (see _fit); hidden during present / noscale
       / print so capture geometry and fullscreen output are unchanged. */
    .rail {
      position: fixed;
      left: 0;
      top: 0;
      bottom: 0;
      width: var(--deck-rail-w, 188px);
      background: #141414;
      border-right: 1px solid rgba(255,255,255,0.08);
      overflow-y: auto;
      overflow-x: hidden;
      padding: 12px 10px;
      box-sizing: border-box;
      display: flex;
      flex-direction: column;
      gap: 12px;
      z-index: 2147482500;
      scrollbar-width: thin;
      scrollbar-color: rgba(255,255,255,0.18) transparent;
    }
    .rail::-webkit-scrollbar { width: 8px; }
    .rail::-webkit-scrollbar-track { background: transparent; margin: 2px; }
    .rail::-webkit-scrollbar-thumb {
      background: rgba(255,255,255,0.18);
      border-radius: 4px;
      border: 2px solid transparent;
      background-clip: content-box;
    }
    .rail::-webkit-scrollbar-thumb:hover {
      background: rgba(255,255,255,0.28);
      border: 2px solid transparent;
      background-clip: content-box;
    }
    :host([no-rail]) .rail,
    :host([noscale]) .rail { display: none; }
    .rail[data-presenting] { display: none; }
    @media (max-width: 640px) {
      .rail, .rail-resize { display: none; }
    }
    /* User-driven show/hide (the TweaksPanel toggle) slides instead of
       popping. Transitions are gated on :host([data-rail-anim]) — set only
       for the 200ms around the toggle — so window-resize and rail-width
       drag (which also call _fit) don't lag behind the cursor. */
    .rail[data-user-hidden] { transform: translateX(-100%); }
    :host([data-rail-anim]) .rail { transition: transform 200ms cubic-bezier(.3,.7,.4,1); }
    :host([data-rail-anim]) .stage { transition: left 200ms cubic-bezier(.3,.7,.4,1); }
    :host([data-rail-anim]) .canvas { transition: transform 200ms cubic-bezier(.3,.7,.4,1); }
    /* transition shorthand replaces rather than merges — repeat the base
       .overlay opacity/transform/filter transitions so visibility changes
       during the 200ms toggle window still fade instead of popping. */
    :host([data-rail-anim]) .overlay {
      transition: margin-left 200ms cubic-bezier(.3,.7,.4,1),
                  opacity 260ms ease,
                  transform 260ms cubic-bezier(.2,.8,.2,1),
                  filter 260ms ease;
    }

    .thumb {
      position: relative;
      display: flex;
      align-items: flex-start;
      gap: 8px;
      cursor: pointer;
      user-select: none;
    }
    .thumb .num {
      width: 16px;
      flex-shrink: 0;
      font-size: 11px;
      font-weight: 500;
      text-align: right;
      color: rgba(255,255,255,0.55);
      padding-top: 2px;
      font-variant-numeric: tabular-nums;
    }
    .thumb .frame {
      position: relative;
      flex: 1;
      min-width: 0;
      aspect-ratio: var(--deck-aspect);
      background: #fff;
      border-radius: 4px;
      outline: 2px solid transparent;
      outline-offset: 0;
      overflow: hidden;
      transition: outline-color 120ms ease;
    }
    .thumb:hover .frame { outline-color: rgba(255,255,255,0.25); }
    .thumb { outline: none; }
    .thumb:focus-visible .frame { outline-color: rgba(255,255,255,0.5); }
    .thumb[data-current] .num { color: #fff; }
    .thumb[data-current] .frame { outline-color: #D97757; }
    .thumb[data-dragging] { opacity: 0.35; }
    .thumb::before {
      content: '';
      position: absolute;
      left: 24px;
      right: 0;
      height: 3px;
      border-radius: 2px;
      background: #D97757;
      opacity: 0;
      pointer-events: none;
    }
    .thumb[data-drop="before"]::before { top: -8px; opacity: 1; }
    .thumb[data-drop="after"]::before { bottom: -8px; opacity: 1; }
    .thumb[data-skip] .frame { opacity: 0.35; }
    .thumb[data-skip] .frame::after {
      content: 'Skipped';
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(0,0,0,0.45);
      color: #fff;
      font-size: 10px;
      font-weight: 500;
      letter-spacing: 0.04em;
    }

    .ctxmenu {
      position: fixed;
      min-width: 150px;
      padding: 4px;
      background: #242424;
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 7px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.45);
      z-index: 2147483100;
      display: none;
      font-size: 12px;
    }
    .ctxmenu[data-open] { display: block; }
    .ctxmenu button {
      display: block;
      width: 100%;
      appearance: none;
      border: 0;
      background: transparent;
      color: #e8e8e8;
      font: inherit;
      text-align: left;
      padding: 6px 10px;
      border-radius: 4px;
      cursor: pointer;
    }
    .ctxmenu button:hover:not(:disabled) { background: rgba(255,255,255,0.08); }
    .ctxmenu button:disabled { opacity: 0.35; cursor: default; }
    .ctxmenu hr {
      border: 0;
      border-top: 1px solid rgba(255,255,255,0.1);
      margin: 4px 2px;
    }

    .rail-resize {
      position: fixed;
      left: calc(var(--deck-rail-w, 188px) - 3px);
      top: 0;
      bottom: 0;
      width: 6px;
      cursor: col-resize;
      z-index: 2147482600;
      touch-action: none;
    }
    .rail-resize:hover,
    .rail-resize[data-dragging] { background: rgba(255,255,255,0.12); }
    :host([no-rail]) .rail-resize,
    :host([noscale]) .rail-resize,
    .rail[data-presenting] + .rail-resize,
    .rail[data-user-hidden] + .rail-resize { display: none; }

    /* Delete-confirm popup — matches the SPA's ConfirmDialog layout
       (title + message body, depressed footer with Cancel / Delete). */
    .confirm-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.45);
      z-index: 2147483200;
      display: none;
      align-items: center;
      justify-content: center;
    }
    .confirm-backdrop[data-open] { display: flex; }
    .confirm {
      width: 320px;
      max-width: calc(100vw - 32px);
      background: #2a2a2a;
      color: #e8e8e8;
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 12px;
      box-shadow: 0 12px 32px rgba(0,0,0,0.5);
      overflow: hidden;
      font-family: inherit;
      animation: deck-confirm-in 0.18s ease;
    }
    @keyframes deck-confirm-in {
      from { opacity: 0; transform: scale(0.96); }
      to { opacity: 1; transform: scale(1); }
    }
    .confirm .body { padding: 20px 20px 16px; }
    .confirm .title { font-size: 14px; font-weight: 600; margin-bottom: 4px; }
    .confirm .msg { font-size: 13px; line-height: 1.5; color: rgba(255,255,255,0.65); }
    .confirm .footer {
      padding: 14px 20px;
      background: #1f1f1f;
      border-top: 1px solid rgba(255,255,255,0.08);
      display: flex;
      justify-content: flex-end;
      gap: 8px;
    }
    .confirm button {
      appearance: none;
      font: inherit;
      font-size: 13px;
      font-weight: 500;
      padding: 8px 16px;
      border-radius: 8px;
      cursor: pointer;
    }
    .confirm .cancel {
      background: transparent;
      border: 0;
      color: rgba(255,255,255,0.8);
    }
    .confirm .cancel:hover { background: rgba(255,255,255,0.08); }
    .confirm .danger {
      background: #c96442;
      border: 1px solid rgba(0,0,0,0.15);
      color: #fff;
      box-shadow: 0 1px 3px rgba(166,50,68,0.3), 0 2px 6px rgba(166,50,68,0.18);
    }
    .confirm .danger:hover { background: #b5563a; }

    /* ── Print: one page per slide, no chrome ────────────────────────────
       The screen layout stacks every slide at inset:0 inside a scaled
       canvas; for print we want them in document flow at the authored
       design size so the browser paginates one slide per sheet. The
       @page size is set from the width/height attributes via the inline
       <style id="deck-stage-print-page"> that connectedCallback injects
       into <head> (the @page at-rule has no effect inside shadow DOM). */
    @media print {
      :host {
        position: static;
        inset: auto;
        background: none;
        overflow: visible;
        color: inherit;
      }
      .stage { position: static; display: block; }
      .canvas {
        transform: none !important;
        width: auto !important;
        height: auto !important;
        background: none;
        will-change: auto;
      }
      ::slotted(*) {
        position: relative !important;
        inset: auto !important;
        width: var(--deck-design-w) !important;
        height: var(--deck-design-h) !important;
        box-sizing: border-box !important;
        opacity: 1 !important;
        visibility: visible !important;
        pointer-events: auto;
        break-after: page;
        page-break-after: always;
        break-inside: avoid;
        overflow: hidden;
      }
      /* :last-child alone isn't enough once data-deck-skip hides the
         trailing slide(s) — the last *visible* slide still carries
         break-after:page and prints a blank sheet. _markLastVisible()
         maintains data-deck-last-visible on the last non-skipped slide. */
      ::slotted(*:last-child),
      ::slotted([data-deck-last-visible]) {
        break-after: auto;
        page-break-after: auto;
      }
      ::slotted([data-deck-skip]) { display: none !important; }
      .overlay, .rail, .rail-resize, .ctxmenu, .confirm-backdrop { display: none !important; }
    }
  `;
  class DeckStage extends HTMLElement {
    static get observedAttributes() {
      return ['width', 'height', 'noscale', 'no-rail'];
    }
    constructor() {
      super();
      this._root = this.attachShadow({
        mode: 'open'
      });
      this._index = 0;
      this._slides = [];
      this._notes = [];
      this._hideTimer = null;
      this._mouseIdleTimer = null;
      this._menuIndex = -1;
      this._onKey = this._onKey.bind(this);
      this._onResize = this._onResize.bind(this);
      this._onSlotChange = this._onSlotChange.bind(this);
      this._onMouseMove = this._onMouseMove.bind(this);
      this._onTap = this._onTap.bind(this);
      this._onMessage = this._onMessage.bind(this);
      // Capture-phase close so a click anywhere dismisses the menu, but
      // ignore clicks that land inside the menu itself — otherwise the
      // capture handler runs before the menu's own (bubble) handler and
      // clears _menuIndex out from under it.
      this._onDocClick = e => {
        if (this._menu && e.composedPath && e.composedPath().includes(this._menu)) return;
        this._closeMenu();
      };
    }
    get designWidth() {
      return parseInt(this.getAttribute('width'), 10) || DESIGN_W_DEFAULT;
    }
    get designHeight() {
      return parseInt(this.getAttribute('height'), 10) || DESIGN_H_DEFAULT;
    }
    connectedCallback() {
      // Presenter-view popup loads deckUrl?_snthumb=...#N for its prev/cur/
      // next thumbnails — the rail has no business rendering inside those
      // (wrong scale, and it offsets the stage so the thumb shows a gutter).
      if (/[?&]_snthumb=/.test(location.search)) this.setAttribute('no-rail', '');
      this._render();
      this._loadNotes();
      this._syncPrintPageRule();
      window.addEventListener('keydown', this._onKey);
      window.addEventListener('resize', this._onResize);
      window.addEventListener('mousemove', this._onMouseMove, {
        passive: true
      });
      window.addEventListener('message', this._onMessage);
      window.addEventListener('click', this._onDocClick, true);
      this.addEventListener('click', this._onTap);
      // Print lays every slide out as its own page, so [data-deck-active]-
      // gated entrance styles need the attribute on every slide (not just
      // the current one) or their content prints at the hidden base style.
      // The transient freeze style lands BEFORE the attributes so any
      // attribute-keyed transition fires at 0s (changing transition-
      // duration after a transition has started doesn't affect it).
      this._onBeforePrint = () => {
        if (this._freezeStyle) this._freezeStyle.remove();
        this._freezeStyle = document.createElement('style');
        this._freezeStyle.textContent = '*,*::before,*::after{transition-duration:0s !important}';
        document.head.appendChild(this._freezeStyle);
        this._slides.forEach(s => s.setAttribute('data-deck-active', ''));
      };
      this._onAfterPrint = () => {
        this._applyIndex({
          showOverlay: false,
          broadcast: false
        });
        if (this._freezeStyle) {
          this._freezeStyle.remove();
          this._freezeStyle = null;
        }
      };
      window.addEventListener('beforeprint', this._onBeforePrint);
      window.addEventListener('afterprint', this._onAfterPrint);
      // Initial collection + layout happens via slotchange, which fires on mount.
      this._enableRail();
      // Hold the stage hidden until webfonts are ready so the first visible
      // paint has the deck's real typography — the :not(:defined) guard in
      // the page HTML only covers custom-element upgrade, not font load.
      // Capped so a 404'd font URL can't blank the deck indefinitely.
      this.setAttribute('data-fonts-pending', '');
      const reveal = () => this.removeAttribute('data-fonts-pending');
      // rAF first: fonts.ready is a pre-resolved promise until layout has
      // resolved the slotted text's font-family and pushed a FontFace into
      // 'loading'. Reading it here in connectedCallback (parse-time) would
      // settle the race in a microtask before any font fetch starts.
      requestAnimationFrame(() => {
        Promise.race([document.fonts ? document.fonts.ready : Promise.resolve(), new Promise(r => setTimeout(r, 2000))]).then(reveal, reveal);
      });
    }
    _enableRail() {
      // Idempotent — older host builds still post __omelette_rail_enabled.
      // no-rail guard keeps the observers/stylesheet walk off the cheap path
      // for presenter-popup thumbnail iframes (up to 9 per view).
      if (this._railEnabled || this.hasAttribute('no-rail')) return;
      this._railEnabled = true;
      // Per-viewer preference — restored alongside rail width. Default on;
      // only a stored '0' (from the TweaksPanel toggle) hides it.
      this._railVisible = true;
      try {
        if (localStorage.getItem('deck-stage.railVisible') === '0') this._railVisible = false;
      } catch (e) {}
      // Live thumbnail updates: watch the light-DOM slides for content
      // edits and re-clone just the affected thumb(s), debounced. Ignore
      // the data-deck-* / data-screen-label / data-om-validate attributes
      // this component itself writes so nav and skip don't trigger
      // spurious refreshes.
      const OWN_ATTRS = /^data-(deck-|screen-label$|om-validate$)/;
      this._liveDirty = new Set();
      this._liveObserver = new MutationObserver(records => {
        for (const r of records) {
          if (r.type === 'attributes' && OWN_ATTRS.test(r.attributeName || '')) continue;
          let n = r.target;
          while (n && n.parentElement !== this) n = n.parentElement;
          if (n && this._slideSet && this._slideSet.has(n)) this._liveDirty.add(n);
        }
        if (this._liveDirty.size && !this._liveTimer) {
          this._liveTimer = setTimeout(() => {
            this._liveTimer = null;
            this._liveDirty.forEach(s => this._refreshThumb(s));
            this._liveDirty.clear();
          }, 200);
        }
      });
      this._liveObserver.observe(this, {
        subtree: true,
        childList: true,
        characterData: true,
        attributes: true
      });
      // Lazy thumbnail materialization — clone the slide only when its
      // frame scrolls into (or near) the rail viewport. rootMargin gives
      // ~4 thumbs of pre-load so fast scrolling doesn't flash blanks.
      this._railObserver = new IntersectionObserver(entries => {
        entries.forEach(e => {
          if (e.isIntersecting && e.target.__deckThumb) {
            this._materialize(e.target.__deckThumb);
          }
        });
      }, {
        root: this._rail,
        rootMargin: '400px 0px'
      });
      // Tweaks typically change CSS vars / attrs OUTSIDE <deck-stage>
      // (on <html>, <body>, a wrapper div, or a <style> tag), which
      // _liveObserver can't see. Re-snapshot author CSS (constructable
      // sheet is shared by reference, so one replaceSync updates every
      // thumb shadow root) and re-sync each thumb host's attrs + custom
      // properties. In-slide DOM mutations are _liveObserver's job.
      // Debounced so slider drags don't thrash.
      this._onTweakChange = () => {
        clearTimeout(this._tweakTimer);
        this._tweakTimer = setTimeout(() => {
          this._snapshotAuthorCss();
          // One getComputedStyle for the whole batch — each
          // getPropertyValue read below reuses the same computed style
          // as long as nothing invalidates layout between thumbs.
          const cs = getComputedStyle(this);
          (this._thumbs || []).forEach(t => {
            if (t.host) this._syncThumbHostAttrs(t.host, cs);
          });
        }, 120);
      };
      window.addEventListener('tweakchange', this._onTweakChange);
      this._snapshotAuthorCss();
      // Build the rail now that it's enabled — slotchange already fired,
      // so _renderRail's early-return skipped the initial build.
      this._syncRailHidden();
      this._renderRail();
      this._fit();
    }

    /** Snapshot document stylesheets into a constructable sheet that each
     *  thumbnail's nested shadow root adopts — so author CSS styles the
     *  cloned slide content without touching this component's chrome.
     *  Cross-origin sheets throw on .cssRules — skip them. Re-callable:
     *  the existing constructable sheet is reused via replaceSync so every
     *  already-adopted shadow root picks up the fresh CSS without re-adopt. */
    _snapshotAuthorCss() {
      // :root in an adopted sheet inside a shadow root matches nothing
      // (only the document root qualifies), so author rules like
      // `:root[data-voice="modern"] .serif` never reach the clones.
      // Rewrite :root → :host and mirror <html>'s data-*/class/lang onto
      // each thumb host (see _syncThumbHostAttrs) so the same selectors
      // match inside the thumbnail's shadow tree.
      const authorCss = Array.from(document.styleSheets).map(sh => {
        try {
          return Array.from(sh.cssRules).map(r => r.cssText).join('\n');
        } catch (e) {
          return '';
        }
      }).join('\n')
      // The shadow host is featureless outside the functional :host(...)
      // form, so any compound on :root — [attr], .class, #id, :pseudo —
      // must become :host(<compound>) not :host<compound>. Same for the
      // html type selector (Tailwind class-strategy dark mode emits
      // html.dark; Pico uses html[data-theme]), which has nothing to
      // match inside the thumb's shadow tree.
      .replace(/:root((?:\[[^\]]*\]|[.#][-\w]+|:[-\w]+(?:\([^)]*\))?)+)/g, ':host($1)').replace(/:root\b/g, ':host').replace(/(^|[\s,>~+(}])html((?:\[[^\]]*\]|[.#][-\w]+|:[-\w]+(?:\([^)]*\))?)+)(?![-\w])/g, '$1:host($2)').replace(/(^|[\s,>~+(}])html(?![-\w])/g, '$1:host');
      // Every custom property the author references. _syncThumbHostAttrs
      // mirrors each one's *computed* value at <deck-stage> onto the
      // thumb host so the live value wins over the :host default above
      // regardless of which ancestor the tweak wrote to (<html>, <body>,
      // a wrapper div, or the deck-stage element itself all inherit
      // down to getComputedStyle(this)).
      this._authorVars = new Set(authorCss.match(/--[\w-]+/g) || []);
      try {
        if (!this._adoptedSheet) this._adoptedSheet = new CSSStyleSheet();
        this._adoptedSheet.replaceSync(authorCss);
      } catch (e) {
        this._adoptedSheet = null;
        this._authorCss = authorCss;
      }
    }
    _syncThumbHostAttrs(host, cs) {
      const de = document.documentElement;
      // setAttribute overwrites but can't delete — an attr removed from
      // <html> (toggleAttribute off, classList emptied) would linger on
      // the host and :host([data-*]) / :host(.foo) rules would keep
      // matching. Remove stale mirrored attrs first; iterate backward
      // because removeAttribute mutates the live NamedNodeMap.
      for (let i = host.attributes.length - 1; i >= 0; i--) {
        const n = host.attributes[i].name;
        if ((n.startsWith('data-') || n === 'class' || n === 'lang') && !de.hasAttribute(n)) {
          host.removeAttribute(n);
        }
      }
      for (const a of de.attributes) {
        if (a.name.startsWith('data-') || a.name === 'class' || a.name === 'lang') {
          host.setAttribute(a.name, a.value);
        }
      }
      // The :root→:host rewrite in _snapshotAuthorCss pins each custom
      // property to its stylesheet default on the thumb host, shadowing
      // the live value that would otherwise inherit. Tweaks can write the
      // live value on any ancestor — <html>, <body>, a wrapper div, the
      // deck-stage element — so read it as the *computed* value at
      // <deck-stage> (which sees the whole inheritance chain) rather than
      // trying to guess which element the author wrote to. Inline on the
      // host beats the :host{} rule. remove-stale covers vars dropped
      // from the stylesheet between snapshots.
      const vars = this._authorVars || new Set();
      for (let i = host.style.length - 1; i >= 0; i--) {
        const p = host.style[i];
        if (p.startsWith('--') && !vars.has(p)) host.style.removeProperty(p);
      }
      const live = cs || getComputedStyle(this);
      vars.forEach(p => {
        const v = live.getPropertyValue(p);
        if (v) host.style.setProperty(p, v.trim());else host.style.removeProperty(p);
      });
    }
    disconnectedCallback() {
      window.removeEventListener('keydown', this._onKey);
      window.removeEventListener('resize', this._onResize);
      window.removeEventListener('mousemove', this._onMouseMove);
      window.removeEventListener('message', this._onMessage);
      window.removeEventListener('click', this._onDocClick, true);
      window.removeEventListener('beforeprint', this._onBeforePrint);
      window.removeEventListener('afterprint', this._onAfterPrint);
      if (this._freezeStyle) {
        this._freezeStyle.remove();
        this._freezeStyle = null;
      }
      this.removeEventListener('click', this._onTap);
      if (this._hideTimer) clearTimeout(this._hideTimer);
      if (this._mouseIdleTimer) clearTimeout(this._mouseIdleTimer);
      if (this._liveTimer) clearTimeout(this._liveTimer);
      if (this._tweakTimer) clearTimeout(this._tweakTimer);
      if (this._railAnimTimer) clearTimeout(this._railAnimTimer);
      if (this._scaleRaf) cancelAnimationFrame(this._scaleRaf);
      if (this._liveObserver) this._liveObserver.disconnect();
      if (this._railObserver) this._railObserver.disconnect();
      if (this._onTweakChange) window.removeEventListener('tweakchange', this._onTweakChange);
    }
    attributeChangedCallback() {
      if (this._canvas) {
        this._canvas.style.width = this.designWidth + 'px';
        this._canvas.style.height = this.designHeight + 'px';
        this._canvas.style.setProperty('--deck-design-w', this.designWidth + 'px');
        this._canvas.style.setProperty('--deck-design-h', this.designHeight + 'px');
        if (this._rail) {
          this._rail.style.setProperty('--deck-aspect', this.designWidth + '/' + this.designHeight);
        }
        this._fit();
        this._scaleThumbs();
        this._syncPrintPageRule();
      }
    }
    _render() {
      const style = document.createElement('style');
      style.textContent = stylesheet;
      const stage = document.createElement('div');
      stage.className = 'stage';
      const canvas = document.createElement('div');
      canvas.className = 'canvas';
      canvas.style.width = this.designWidth + 'px';
      canvas.style.height = this.designHeight + 'px';
      canvas.style.setProperty('--deck-design-w', this.designWidth + 'px');
      canvas.style.setProperty('--deck-design-h', this.designHeight + 'px');
      const slot = document.createElement('slot');
      slot.addEventListener('slotchange', this._onSlotChange);
      canvas.appendChild(slot);
      stage.appendChild(canvas);

      // Overlay: compact, solid black, with clickable controls.
      const overlay = document.createElement('div');
      overlay.className = 'overlay export-hidden';
      overlay.setAttribute('role', 'toolbar');
      overlay.setAttribute('aria-label', 'Deck controls');
      overlay.setAttribute('data-omelette-chrome', '');
      overlay.innerHTML = `
        <button class="btn prev" type="button" aria-label="Previous slide" title="Previous (←)">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10 3L5 8l5 5"/></svg>
        </button>
        <span class="count" aria-live="polite"><span class="current">1</span><span class="sep">/</span><span class="total">1</span></span>
        <button class="btn next" type="button" aria-label="Next slide" title="Next (→)">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 3l5 5-5 5"/></svg>
        </button>
        <span class="divider"></span>
        <button class="btn reset" type="button" aria-label="Reset to first slide" title="Reset (R)">Reset<span class="kbd">R</span></button>
      `;
      overlay.querySelector('.prev').addEventListener('click', () => this._advance(-1, 'click'));
      overlay.querySelector('.next').addEventListener('click', () => this._advance(1, 'click'));
      overlay.querySelector('.reset').addEventListener('click', () => this._go(0, 'click'));

      // Thumbnail rail + context menu. Thumbnails are populated in
      // _renderRail() after _collectSlides().
      const rail = document.createElement('div');
      rail.className = 'rail export-hidden';
      rail.setAttribute('data-omelette-chrome', '');
      rail.style.setProperty('--deck-aspect', this.designWidth + '/' + this.designHeight);
      // Edge auto-scroll while dragging a thumb near the rail's top/bottom
      // so off-screen drop targets are reachable. Native dragover fires
      // continuously while the pointer is stationary, so a per-event nudge
      // (ramped by edge proximity) is enough — no rAF loop needed.
      rail.addEventListener('dragover', e => {
        if (this._dragFrom == null) return;
        const r = rail.getBoundingClientRect();
        const EDGE = 40;
        const dt = e.clientY - r.top;
        const db = r.bottom - e.clientY;
        if (dt < EDGE) rail.scrollTop -= Math.ceil((EDGE - dt) / 3);else if (db < EDGE) rail.scrollTop += Math.ceil((EDGE - db) / 3);
      });
      const menu = document.createElement('div');
      menu.className = 'ctxmenu export-hidden';
      menu.setAttribute('data-omelette-chrome', '');
      menu.innerHTML = `
        <button type="button" data-act="skip">Skip slide</button>
        <button type="button" data-act="up">Move up</button>
        <button type="button" data-act="down">Move down</button>
        <button type="button" data-act="duplicate">Duplicate slide</button>
        <hr>
        <button type="button" data-act="delete">Delete slide</button>
      `;
      menu.addEventListener('click', e => {
        const act = e.target && e.target.getAttribute && e.target.getAttribute('data-act');
        if (!act) return;
        const i = this._menuIndex;
        this._closeMenu();
        if (act === 'skip') this._toggleSkip(i);else if (act === 'up') this._moveSlide(i, i - 1);else if (act === 'down') this._moveSlide(i, i + 1);else if (act === 'duplicate') this._duplicateSlide(i);else if (act === 'delete') this._openConfirm(i);
      });
      menu.addEventListener('contextmenu', e => e.preventDefault());

      // Rail resize handle — drag to set --deck-rail-w, persisted to
      // localStorage so the width survives reloads.
      const resize = document.createElement('div');
      resize.className = 'rail-resize export-hidden';
      resize.setAttribute('data-omelette-chrome', '');
      resize.addEventListener('pointerdown', e => {
        e.preventDefault();
        resize.setPointerCapture(e.pointerId);
        resize.setAttribute('data-dragging', '');
        const move = ev => this._setRailWidth(ev.clientX);
        const up = () => {
          resize.removeEventListener('pointermove', move);
          resize.removeEventListener('pointerup', up);
          resize.removeEventListener('pointercancel', up);
          resize.removeAttribute('data-dragging');
          try {
            localStorage.setItem('deck-stage.railWidth', String(this._railPx));
          } catch (err) {}
        };
        resize.addEventListener('pointermove', move);
        resize.addEventListener('pointerup', up);
        resize.addEventListener('pointercancel', up);
      });

      // Delete-confirm dialog — mirrors the SPA's ConfirmDialog layout.
      const confirm = document.createElement('div');
      confirm.className = 'confirm-backdrop export-hidden';
      confirm.setAttribute('data-omelette-chrome', '');
      confirm.innerHTML = `
        <div class="confirm" role="dialog" aria-modal="true">
          <div class="body">
            <div class="title">Delete slide?</div>
            <div class="msg">This slide will be removed from the deck.</div>
          </div>
          <div class="footer">
            <button type="button" class="cancel">Cancel</button>
            <button type="button" class="danger">Delete</button>
          </div>
        </div>
      `;
      confirm.addEventListener('click', e => {
        if (e.target === confirm) this._closeConfirm();
      });
      confirm.querySelector('.cancel').addEventListener('click', () => this._closeConfirm());
      confirm.querySelector('.danger').addEventListener('click', () => {
        const i = this._confirmIndex;
        this._closeConfirm();
        this._deleteSlide(i);
      });
      this._root.append(style, rail, resize, stage, overlay, menu, confirm);
      this._canvas = canvas;
      this._stage = stage;
      this._slot = slot;
      this._overlay = overlay;
      this._rail = rail;
      this._resize = resize;
      this._menu = menu;
      this._confirm = confirm;
      this._countEl = overlay.querySelector('.current');
      this._totalEl = overlay.querySelector('.total');

      // Restore persisted rail width.
      let rw = 188;
      try {
        const s = localStorage.getItem('deck-stage.railWidth');
        if (s) rw = parseInt(s, 10) || rw;
      } catch (err) {}
      this._setRailWidth(rw);
      this._syncRailHidden();
    }
    _setRailWidth(px) {
      const w = Math.max(120, Math.min(360, Math.round(px)));
      this._railPx = w;
      this.style.setProperty('--deck-rail-w', w + 'px');
      this._fit();
      // _scaleThumbs forces a sync layout (frame.offsetWidth) then writes
      // N transforms. During a resize drag this runs per-pointermove;
      // coalesce to one per frame.
      if (!this._scaleRaf) {
        this._scaleRaf = requestAnimationFrame(() => {
          this._scaleRaf = null;
          this._scaleThumbs();
        });
      }
    }

    /** @page must live in the document stylesheet — it's a no-op inside
     *  shadow DOM. Inject/update a single <head> style tag so the print
     *  sheet matches the design size and Save-as-PDF yields one slide per
     *  page with no margins. */
    _syncPrintPageRule() {
      const id = 'deck-stage-print-page';
      let tag = document.getElementById(id);
      if (!tag) {
        tag = document.createElement('style');
        tag.id = id;
        document.head.appendChild(tag);
      }
      tag.textContent = '@page { size: ' + this.designWidth + 'px ' + this.designHeight + 'px; margin: 0; } ' + '@media print { html, body { margin: 0 !important; padding: 0 !important; background: none !important; overflow: visible !important; height: auto !important; } ' + '* { -webkit-print-color-adjust: exact; print-color-adjust: exact; } ' +
      // Jump authored animations/transitions to their end state so print
      // never captures mid-entrance — pairs with the beforeprint handler
      // in connectedCallback that sets data-deck-active on every slide.
      '*, *::before, *::after { animation-delay: -99s !important; animation-duration: .001s !important; ' + 'animation-iteration-count: 1 !important; animation-fill-mode: both !important; ' + 'animation-play-state: running !important; transition-duration: 0s !important; } }';
    }
    _onSlotChange() {
      // Rail mutations (delete/move/duplicate) already reconcile synchronously and
      // emit slidechange with reason 'api'; skip the async slotchange that
      // would otherwise re-broadcast with reason 'init'.
      if (this._squelchSlotChange) {
        this._squelchSlotChange = false;
        return;
      }
      this._collectSlides();
      this._restoreIndex();
      this._applyIndex({
        showOverlay: false,
        broadcast: true,
        reason: 'init'
      });
      this._fit();
    }
    _collectSlides() {
      const assigned = this._slot.assignedElements({
        flatten: true
      });
      this._slides = assigned.filter(el => {
        // Skip template/style/script nodes even if someone slots them.
        const tag = el.tagName;
        return tag !== 'TEMPLATE' && tag !== 'SCRIPT' && tag !== 'STYLE';
      });
      this._slideSet = new Set(this._slides);
      this._slides.forEach((slide, i) => {
        const n = i + 1;
        slide.setAttribute('data-screen-label', `${pad2(n)} ${getSlideLabel(slide)}`);

        // Validation attribute for comment flow / auto-checks.
        if (!slide.hasAttribute('data-om-validate')) {
          slide.setAttribute('data-om-validate', VALIDATE_ATTR);
        }
        slide.setAttribute('data-deck-slide', String(i));
      });
      if (this._totalEl) this._totalEl.textContent = String(this._slides.length || 1);
      if (this._index >= this._slides.length) this._index = Math.max(0, this._slides.length - 1);
      this._markLastVisible();
      this._renderRail();
    }

    /** Tag the last non-skipped slide so print CSS can drop its
     *  break-after (see the @media print comment above — :last-child
     *  alone matches a hidden skipped slide). */
    _markLastVisible() {
      let last = null;
      this._slides.forEach(s => {
        s.removeAttribute('data-deck-last-visible');
        if (!s.hasAttribute('data-deck-skip')) last = s;
      });
      if (last) last.setAttribute('data-deck-last-visible', '');
    }
    _loadNotes() {
      const tag = document.getElementById('speaker-notes');
      if (!tag) {
        this._notes = [];
        return;
      }
      try {
        const parsed = JSON.parse(tag.textContent || '[]');
        if (Array.isArray(parsed)) this._notes = parsed;
      } catch (e) {
        console.warn('[deck-stage] Failed to parse #speaker-notes JSON:', e);
        this._notes = [];
      }
    }
    _restoreIndex() {
      // The host's ?slide= param is delivered as a #<int> hash (1-indexed) on
      // the iframe src. No hash → slide 1; the deck itself keeps no position
      // state across loads.
      const h = (location.hash || '').match(/^#(\d+)$/);
      if (h) {
        const n = parseInt(h[1], 10) - 1;
        if (n >= 0 && n < this._slides.length) this._index = n;
      }
    }
    _applyIndex({
      showOverlay = true,
      broadcast = true,
      reason = 'init'
    } = {}) {
      if (!this._slides.length) return;
      const prev = this._prevIndex == null ? -1 : this._prevIndex;
      const curr = this._index;
      // Keep the iframe's own hash in sync so an in-iframe location.reload()
      // (reload banner path in viewer-handle.ts) lands on the current slide,
      // not the stale deep-link hash from initial load.
      try {
        history.replaceState(null, '', '#' + (curr + 1));
      } catch (e) {}
      this._slides.forEach((s, i) => {
        if (i === curr) s.setAttribute('data-deck-active', '');else s.removeAttribute('data-deck-active');
      });
      if (this._countEl) this._countEl.textContent = String(curr + 1);
      // Follow-scroll on every navigation (init deep-link, keyboard, click,
      // tap, external goTo) — the only time we *don't* want the rail to
      // track current is after a rail-internal mutation, where _renderRail
      // has already restored the user's scroll position and yanking back to
      // current would undo it.
      this._syncRail(reason !== 'mutation');
      if (broadcast) {
        // (1) Legacy: host-window postMessage for speaker-notes renderers.
        try {
          window.postMessage({
            slideIndexChanged: curr,
            deckTotal: this._slides.length,
            deckSkipped: this._skippedIndices()
          }, '*');
        } catch (e) {}

        // (2) In-page CustomEvent on the <deck-stage> element itself.
        //     Bubbles and composes out of shadow DOM so slide code can listen:
        //       document.querySelector('deck-stage').addEventListener('slidechange', e => {
        //         e.detail.index, e.detail.previousIndex, e.detail.total, e.detail.slide, e.detail.reason
        //       });
        const detail = {
          index: curr,
          previousIndex: prev,
          total: this._slides.length,
          slide: this._slides[curr] || null,
          previousSlide: prev >= 0 ? this._slides[prev] || null : null,
          reason: reason // 'init' | 'keyboard' | 'click' | 'tap' | 'api'
        };
        this.dispatchEvent(new CustomEvent('slidechange', {
          detail,
          bubbles: true,
          composed: true
        }));
      }
      this._prevIndex = curr;
      if (showOverlay) this._flashOverlay();
    }
    _flashOverlay() {
      // Host posts __omelette_presenting while in fullscreen/tab presentation
      // mode — suppress the nav footer entirely (both hover and slide-change
      // flash) so the audience sees clean slides.
      if (!this._overlay || this._presenting) return;
      this._overlay.setAttribute('data-visible', '');
      if (this._hideTimer) clearTimeout(this._hideTimer);
      this._hideTimer = setTimeout(() => {
        this._overlay.removeAttribute('data-visible');
      }, OVERLAY_HIDE_MS);
    }
    _railWidth() {
      // State-based, no offsetWidth: the first _fit() can run before the
      // rail has had layout on some load paths, and a 0 there paints the
      // slide full-width for one frame before the post-slotchange _fit()
      // corrects it.
      if (!this._railEnabled || !this._railVisible || this.hasAttribute('no-rail') || this.hasAttribute('noscale') || this._presenting || this._previewMode || NARROW_MQ.matches) return 0;
      return this._railPx || 0;
    }
    _fit() {
      if (!this._canvas) return;
      const stage = this._canvas.parentElement;
      // PPTX export sets noscale so the DOM capture sees authored-size
      // geometry — the scaled canvas is in shadow DOM, so the exporter's
      // resetTransformSelector can't reach .canvas.style.transform directly.
      if (this.hasAttribute('noscale')) {
        this._canvas.style.transform = 'none';
        if (stage) stage.style.left = '0';
        if (this._overlay) this._overlay.style.marginLeft = '0';
        return;
      }
      const rw = this._railWidth();
      if (stage) stage.style.left = rw + 'px';
      // Overlay is centred on the viewport via left:50% + translate(-50%);
      // marginLeft shifts the centre by rw/2 so it lands in the middle of
      // the [rw, innerWidth] stage region.
      if (this._overlay) this._overlay.style.marginLeft = rw / 2 + 'px';
      const vw = window.innerWidth - rw;
      const vh = window.innerHeight;
      const s = Math.min(vw / this.designWidth, vh / this.designHeight);
      this._canvas.style.transform = `scale(${s})`;
    }
    _onResize() {
      this._fit();
      // Crossing the narrow-viewport breakpoint reveals the rail — rerun the
      // thumbnail scale the same way _setRailWidth does.
      if (!this._scaleRaf) {
        this._scaleRaf = requestAnimationFrame(() => {
          this._scaleRaf = null;
          this._scaleThumbs();
        });
      }
    }
    _onMouseMove() {
      // Keep overlay visible while mouse moves; hide after idle.
      this._flashOverlay();
    }
    _onMessage(e) {
      const d = e.data;
      if (d && typeof d.__omelette_presenting === 'boolean') {
        this._presenting = d.__omelette_presenting;
        if (this._presenting && this._overlay) {
          this._overlay.removeAttribute('data-visible');
          if (this._hideTimer) clearTimeout(this._hideTimer);
        }
        this._syncRailHidden();
        this._closeMenu();
        this._closeConfirm();
        this._fit();
        this._scaleThumbs();
      }
      // Host's Preview segment (ViewerMode='none'): the rail's drag-reorder /
      // right-click skip-delete affordances are editing chrome, so hide it
      // while the user is just looking at the deck. Same hard-hide path as
      // presenting; independent of the user's _railVisible preference so
      // returning to Edit restores whatever they had.
      if (d && typeof d.__omelette_preview_mode === 'boolean') {
        if (d.__omelette_preview_mode === this._previewMode) return;
        this._previewMode = d.__omelette_preview_mode;
        this._syncRailHidden();
        this._closeMenu();
        this._closeConfirm();
        this._fit();
        this._scaleThumbs();
      }
      // Per-viewer show/hide, driven by the TweaksPanel's auto-injected
      // "Thumbnail rail" toggle (or any author script). Independent of
      // whether the Tweaks panel itself is open — closing the panel
      // doesn't change rail visibility. Persists alongside rail width.
      if (d && d.type === '__deck_rail_visible' && typeof d.on === 'boolean') {
        if (d.on === this._railVisible) return;
        this._railVisible = d.on;
        try {
          localStorage.setItem('deck-stage.railVisible', d.on ? '1' : '0');
        } catch (e) {}
        // Arm the transition, commit it, then flip state — otherwise the
        // browser coalesces both writes and nothing animates on show.
        this.setAttribute('data-rail-anim', '');
        void (this._rail && this._rail.offsetHeight);
        this._syncRailHidden();
        this._fit();
        this._scaleThumbs();
        clearTimeout(this._railAnimTimer);
        this._railAnimTimer = setTimeout(() => this.removeAttribute('data-rail-anim'), 220);
      }
      if (d && d.type === '__omelette_rail_enabled') this._enableRail();
    }
    _syncRailHidden() {
      if (!this._rail) return;
      // data-presenting is the hard hide (display:none) for flag-off,
      // presentation mode, and the host's Preview segment — instant, no
      // transition. data-user-hidden is the soft hide (translateX(-100%))
      // for the viewer's rail toggle, so show/hide slides under
      // :host([data-rail-anim]).
      const hard = !this._railEnabled || this._presenting || this._previewMode;
      if (hard) this._rail.setAttribute('data-presenting', '');else this._rail.removeAttribute('data-presenting');
      if (!this._railVisible) this._rail.setAttribute('data-user-hidden', '');else this._rail.removeAttribute('data-user-hidden');
      // translateX hide leaves thumbs (tabIndex=0) in the tab order —
      // inert keeps them unfocusable while the rail is off-screen.
      this._rail.inert = hard || !this._railVisible;
    }
    _onTap(e) {
      // Touch-only — keyboard + the overlay toolbar cover nav on desktop.
      if (FINE_POINTER_MQ.matches) return;
      // Only taps that land on the stage (slide content or letterbox); the
      // overlay / rail / menus are siblings with their own click handlers.
      const path = e.composedPath();
      if (!this._stage || !path.includes(this._stage)) return;
      // Let interactive slide content keep the tap. composedPath (not
      // e.target.closest) so we see through open shadow roots — a <button>
      // inside a slide-authored custom element retargets e.target to the
      // host but still appears in the composed path.
      if (e.defaultPrevented) return;
      for (const n of path) {
        if (n === this._stage) break;
        if (n.matches && n.matches(INTERACTIVE_SEL)) return;
      }
      e.preventDefault();
      const rw = this._railWidth();
      const mid = rw + (window.innerWidth - rw) / 2;
      this._advance(e.clientX < mid ? -1 : 1, 'tap');
    }
    _onKey(e) {
      // Ignore when the user is typing.
      const t = e.target;
      if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
      // Confirm dialog swallows nav keys while open; Escape cancels. Enter
      // is left to the focused button's native activation so Tab→Cancel
      // →Enter activates Cancel, not the window-level confirm path.
      if (this._confirm && this._confirm.hasAttribute('data-open')) {
        if (e.key === 'Escape') {
          this._closeConfirm();
          e.preventDefault();
        }
        return;
      }
      if (e.key === 'Escape' && this._menu && this._menu.hasAttribute('data-open')) {
        this._closeMenu();
        e.preventDefault();
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const key = e.key;
      let handled = true;
      if (key === 'ArrowRight' || key === 'PageDown' || key === ' ' || key === 'Spacebar') {
        this._advance(1, 'keyboard');
      } else if (key === 'ArrowLeft' || key === 'PageUp') {
        this._advance(-1, 'keyboard');
      } else if (key === 'Home') {
        this._go(0, 'keyboard');
      } else if (key === 'End') {
        this._go(this._slides.length - 1, 'keyboard');
      } else if (key === 'r' || key === 'R') {
        this._go(0, 'keyboard');
      } else if (/^[0-9]$/.test(key)) {
        // 1..9 jump to that slide; 0 jumps to 10.
        const n = key === '0' ? 9 : parseInt(key, 10) - 1;
        if (n < this._slides.length) this._go(n, 'keyboard');
      } else {
        handled = false;
      }
      if (handled) {
        e.preventDefault();
        this._flashOverlay();
      }
    }
    _go(i, reason = 'api') {
      if (!this._slides.length) return;
      const clamped = Math.max(0, Math.min(this._slides.length - 1, i));
      if (clamped === this._index) {
        this._flashOverlay();
        return;
      }
      this._index = clamped;
      this._applyIndex({
        showOverlay: true,
        broadcast: true,
        reason
      });
    }

    /** Step forward/back skipping any slide marked data-deck-skip. Falls
     *  back to _go's clamp-at-ends behaviour (flash overlay) when there's
     *  nothing further in that direction. */
    _advance(dir, reason) {
      if (!this._slides.length) return;
      let i = this._index + dir;
      while (i >= 0 && i < this._slides.length && this._slides[i].hasAttribute('data-deck-skip')) {
        i += dir;
      }
      if (i < 0 || i >= this._slides.length) {
        this._flashOverlay();
        return;
      }
      this._go(i, reason);
    }

    // ── Thumbnail rail ────────────────────────────────────────────────────
    //
    // Thumbs are keyed by slide element and reused across _renderRail()
    // calls, so a reorder/delete is an O(changed) DOM shuffle instead of an
    // O(N) teardown-and-re-clone. Each thumb starts as a lightweight shell
    // (num + empty frame); the clone is materialized lazily by an
    // IntersectionObserver when the frame scrolls into (or near) view, so
    // only visible-ish slides pay the clone + image-decode cost.

    _renderRail() {
      if (!this._rail || !this._railEnabled) {
        this._thumbs = [];
        return;
      }
      // FLIP: record each *materialized* thumb's top before the reconcile.
      // Off-screen (non-materialized) thumbs don't need the animation and
      // skipping their getBoundingClientRect saves a forced layout per
      // off-screen thumb on large decks.
      const prevTops = new Map();
      (this._thumbs || []).forEach(({
        thumb,
        slide,
        host
      }) => {
        if (host) prevTops.set(slide, thumb.getBoundingClientRect().top);
      });
      const st = this._rail.scrollTop;

      // Reconcile: reuse thumbs that already exist for a slide, create
      // shells for new slides, drop thumbs for removed slides.
      const bySlide = new Map();
      (this._thumbs || []).forEach(t => bySlide.set(t.slide, t));
      const next = [];
      this._slides.forEach(slide => {
        let t = bySlide.get(slide);
        if (t) bySlide.delete(slide);else t = this._makeThumb(slide);
        next.push(t);
      });
      // Orphans — slides removed since last render.
      bySlide.forEach(t => {
        if (this._railObserver) this._railObserver.unobserve(t.frame);
        t.thumb.remove();
      });
      // Put thumbs into document order to match _slides. insertBefore on
      // an already-correctly-placed node is a no-op, so this is cheap
      // when nothing moved.
      next.forEach((t, i) => {
        const want = t.thumb;
        const at = this._rail.children[i];
        if (at !== want) this._rail.insertBefore(want, at || null);
        t.i = i;
        t.num.textContent = String(i + 1);
        if (t.slide.hasAttribute('data-deck-skip')) t.thumb.setAttribute('data-skip', '');else t.thumb.removeAttribute('data-skip');
      });
      this._thumbs = next;
      this._rail.scrollTop = st;
      if (prevTops.size) {
        const moved = [];
        this._thumbs.forEach(({
          thumb,
          slide
        }) => {
          const old = prevTops.get(slide);
          if (old == null) return;
          const dy = old - thumb.getBoundingClientRect().top;
          if (Math.abs(dy) < 1) return;
          thumb.style.transition = 'none';
          thumb.style.transform = `translateY(${dy}px)`;
          moved.push(thumb);
        });
        if (moved.length) {
          // Commit the inverted positions before flipping the transition
          // on — otherwise the browser coalesces both style writes and
          // nothing animates.
          void this._rail.offsetHeight;
          moved.forEach(t => {
            t.style.transition = 'transform 180ms cubic-bezier(.2,.7,.3,1)';
            t.style.transform = '';
          });
          setTimeout(() => moved.forEach(t => {
            t.style.transition = '';
          }), 220);
        }
      }
      requestAnimationFrame(() => this._scaleThumbs());
      this._syncRail(false);
    }

    /** Create a lightweight thumb shell for one slide. The clone is
     *  materialized later by the IntersectionObserver. Event handlers
     *  look up the thumb's *current* index (via _thumbs.indexOf) so the
     *  same element can be reused across reorders. */
    _makeThumb(slide) {
      const thumb = document.createElement('div');
      thumb.className = 'thumb';
      thumb.tabIndex = 0;
      const num = document.createElement('div');
      num.className = 'num';
      const frame = document.createElement('div');
      frame.className = 'frame';
      thumb.append(num, frame);
      const entry = {
        thumb,
        num,
        frame,
        slide,
        clone: null,
        host: null,
        i: -1
      };
      // entry.i is refreshed on every _renderRail reconcile pass, so
      // handlers read the thumb's current position without an O(N) scan.
      const idx = () => entry.i;
      thumb.addEventListener('click', () => this._go(idx(), 'click'));
      // ↑/↓ step through the rail when a thumb has focus. _go clamps at the
      // ends and _applyIndex→_syncRail scrolls the new current thumb into
      // view; we move focus to it (preventScroll — _syncRail already
      // scrolled) so a held key walks the whole list. stopPropagation keeps
      // this out of the window-level _onKey nav handler.
      thumb.addEventListener('keydown', e => {
        if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
        if (e.metaKey || e.ctrlKey || e.altKey) return;
        e.preventDefault();
        e.stopPropagation();
        this._go(idx() + (e.key === 'ArrowDown' ? 1 : -1), 'keyboard');
        const cur = this._thumbs && this._thumbs[this._index];
        if (cur) cur.thumb.focus({
          preventScroll: true
        });
      });
      thumb.addEventListener('contextmenu', e => {
        e.preventDefault();
        this._openMenu(idx(), e.clientX, e.clientY);
      });
      thumb.draggable = true;
      thumb.addEventListener('dragstart', e => {
        this._dragFrom = idx();
        thumb.setAttribute('data-dragging', '');
        e.dataTransfer.effectAllowed = 'move';
        try {
          e.dataTransfer.setData('text/plain', String(this._dragFrom));
        } catch (err) {}
      });
      thumb.addEventListener('dragend', () => {
        thumb.removeAttribute('data-dragging');
        this._clearDrop();
        this._dragFrom = null;
      });
      thumb.addEventListener('dragover', e => {
        if (this._dragFrom == null) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        const r = thumb.getBoundingClientRect();
        this._setDrop(idx(), e.clientY < r.top + r.height / 2 ? 'before' : 'after');
      });
      thumb.addEventListener('drop', e => {
        if (this._dragFrom == null) return;
        e.preventDefault();
        const i = idx();
        const r = thumb.getBoundingClientRect();
        let to = e.clientY >= r.top + r.height / 2 ? i + 1 : i;
        if (this._dragFrom < to) to--;
        const from = this._dragFrom;
        this._clearDrop();
        this._dragFrom = null;
        if (to !== from) this._moveSlide(from, to);
      });
      if (this._railObserver) this._railObserver.observe(frame);
      frame.__deckThumb = entry;
      return entry;
    }

    /** Lazily build the clone for a thumb that has scrolled into view. */
    _materialize(entry) {
      if (entry.host) return;
      const dw = this.designWidth,
        dh = this.designHeight;
      let clone = entry.slide.cloneNode(true);
      clone.removeAttribute('id');
      clone.removeAttribute('data-deck-active');
      clone.querySelectorAll('[id]').forEach(el => el.removeAttribute('id'));
      // Neuter heavy media; replace <video> with its poster so the box
      // keeps a visual. <iframe>/<audio> become empty placeholders.
      clone.querySelectorAll('iframe, audio, object, embed').forEach(el => {
        el.removeAttribute('src');
        el.removeAttribute('srcdoc');
        el.removeAttribute('data');
        el.innerHTML = '';
      });
      clone.querySelectorAll('video').forEach(el => {
        if (!el.poster) {
          el.removeAttribute('src');
          el.innerHTML = '';
          return;
        }
        const img = document.createElement('img');
        img.src = el.poster;
        img.alt = '';
        img.style.cssText = el.style.cssText + ';object-fit:cover;width:100%;height:100%;';
        img.className = el.className;
        el.replaceWith(img);
      });
      // Images: defer decode and let the browser pick the smallest
      // srcset candidate for the ~140px thumb. Same-URL clones reuse the
      // slide's decoded bitmap (URL-keyed cache), so the remaining cost
      // is paint/composite — lazy+async keeps that off the main thread.
      clone.querySelectorAll('img').forEach(el => {
        el.loading = 'lazy';
        el.decoding = 'async';
        if (el.srcset) el.sizes = (this._railPx || 188) + 'px';
      });
      // Custom elements inside the slide would have their
      // connectedCallback fire when the clone is appended. Replace them
      // with inert boxes so a component-heavy deck doesn't run N copies
      // of each component's mount logic in the rail. Children are
      // preserved so layout-wrapper elements (<my-column><h2>…</h2>)
      // still show their authored content; the querySelectorAll NodeList
      // is static, so nested custom elements in the moved subtree are
      // still visited on later iterations.
      const neuter = el => {
        const box = document.createElement('div');
        box.style.cssText = (el.getAttribute('style') || '') + ';background:rgba(0,0,0,0.06);border:1px dashed rgba(0,0,0,0.15);';
        box.className = el.className;
        // Preserve theming/i18n hooks so [data-*] / :lang() / [dir]
        // descendant selectors still match the neutered root.
        for (const a of el.attributes) {
          const n = a.name;
          if (n.startsWith('data-') || n.startsWith('aria-') || n === 'lang' || n === 'dir' || n === 'role' || n === 'title') {
            box.setAttribute(n, a.value);
          }
        }
        while (el.firstChild) box.appendChild(el.firstChild);
        return box;
      };
      // querySelectorAll('*') returns descendants only — a custom-element
      // slide root (<my-slide>…</my-slide>) would slip through and upgrade
      // on append. Swap the root first.
      if (clone.tagName.includes('-')) clone = neuter(clone);
      clone.querySelectorAll('*').forEach(el => {
        if (el.tagName.includes('-')) el.replaceWith(neuter(el));
      });
      clone.style.cssText += ';position:absolute;top:0;left:0;transform-origin:0 0;' + 'pointer-events:none;width:' + dw + 'px;height:' + dh + 'px;' + 'box-sizing:border-box;overflow:hidden;visibility:visible;opacity:1;';
      const host = document.createElement('div');
      host.style.cssText = 'position:absolute;inset:0;';
      this._syncThumbHostAttrs(host);
      const sr = host.attachShadow({
        mode: 'open'
      });
      if (this._adoptedSheet) sr.adoptedStyleSheets = [this._adoptedSheet];else {
        const st = document.createElement('style');
        st.textContent = this._authorCss || '';
        sr.appendChild(st);
      }
      sr.appendChild(clone);
      entry.frame.appendChild(host);
      entry.host = host;
      entry.clone = clone;
      if (this._thumbScale) clone.style.transform = 'scale(' + this._thumbScale + ')';
      // Once materialized the IO callback is a no-op early-return —
      // unobserve so scroll doesn't keep firing it.
      if (this._railObserver) this._railObserver.unobserve(entry.frame);
    }

    /** Re-clone a single thumb (live-update path). No-op if the thumb
     *  hasn't been materialized yet — it'll pick up current content when
     *  it scrolls into view. */
    _refreshThumb(slide) {
      const entry = (this._thumbs || []).find(t => t.slide === slide);
      if (!entry || !entry.host) return;
      entry.host.remove();
      entry.host = entry.clone = null;
      this._materialize(entry);
    }
    _scaleThumbs() {
      if (!this._thumbs || !this._thumbs.length) return;
      // Every frame is the same width; if it reads 0 the rail is
      // display:none (noscale / no-rail / presenting / print) — leave the
      // clones as-is and re-run when the rail is revealed.
      const fw = this._thumbs[0].frame.offsetWidth;
      if (!fw) return;
      this._thumbScale = fw / this.designWidth;
      this._thumbs.forEach(({
        clone
      }) => {
        if (clone) clone.style.transform = 'scale(' + this._thumbScale + ')';
      });
    }
    _setDrop(i, where) {
      // dragover fires at pointer-event rate; touch only the previous
      // and new target rather than sweeping all N thumbs.
      const t = this._thumbs && this._thumbs[i];
      if (this._dropOn && this._dropOn !== t) {
        this._dropOn.thumb.removeAttribute('data-drop');
      }
      if (t) t.thumb.setAttribute('data-drop', where);
      this._dropOn = t || null;
    }
    _clearDrop() {
      if (this._dropOn) this._dropOn.thumb.removeAttribute('data-drop');
      this._dropOn = null;
    }
    _syncRail(follow) {
      if (!this._thumbs) return;
      this._thumbs.forEach(({
        thumb
      }, i) => {
        if (i === this._index) {
          thumb.setAttribute('data-current', '');
          if (follow && typeof thumb.scrollIntoView === 'function') {
            thumb.scrollIntoView({
              block: 'nearest'
            });
          }
        } else {
          thumb.removeAttribute('data-current');
        }
      });
    }
    _openMenu(i, x, y) {
      if (!this._menu) return;
      this._menuIndex = i;
      const slide = this._slides[i];
      const skip = slide && slide.hasAttribute('data-deck-skip');
      this._menu.querySelector('[data-act="skip"]').textContent = skip ? 'Unskip slide' : 'Skip slide';
      this._menu.querySelector('[data-act="up"]').disabled = i <= 0;
      this._menu.querySelector('[data-act="down"]').disabled = i >= this._slides.length - 1;
      this._menu.querySelector('[data-act="delete"]').disabled = this._slides.length <= 1;
      // Place, then clamp to viewport after it's measurable.
      this._menu.style.left = x + 'px';
      this._menu.style.top = y + 'px';
      this._menu.setAttribute('data-open', '');
      const r = this._menu.getBoundingClientRect();
      const nx = Math.min(x, window.innerWidth - r.width - 4);
      const ny = Math.min(y, window.innerHeight - r.height - 4);
      this._menu.style.left = Math.max(4, nx) + 'px';
      this._menu.style.top = Math.max(4, ny) + 'px';
    }
    _closeMenu() {
      if (this._menu) this._menu.removeAttribute('data-open');
      this._menuIndex = -1;
    }
    _openConfirm(i) {
      if (!this._confirm) return;
      this._confirmIndex = i;
      this._confirm.querySelector('.title').textContent = 'Delete slide ' + (i + 1) + '?';
      this._confirm.setAttribute('data-open', '');
      const btn = this._confirm.querySelector('.danger');
      if (btn && btn.focus) btn.focus();
    }
    _closeConfirm() {
      if (this._confirm) this._confirm.removeAttribute('data-open');
      this._confirmIndex = -1;
    }
    _emitDeckChange(detail) {
      this.dispatchEvent(new CustomEvent('deckchange', {
        detail,
        bubbles: true,
        composed: true
      }));
    }
    _deleteSlide(i) {
      const slide = this._slides[i];
      if (!slide || this._slides.length <= 1) return;
      const wasCurrent = i === this._index;
      if (i < this._index || wasCurrent && i === this._slides.length - 1) this._index--;
      this._squelchSlotChange = true;
      slide.remove();
      this._emitDeckChange({
        action: 'delete',
        from: i,
        slide
      });
      this._collectSlides();
      this._applyIndex({
        showOverlay: true,
        broadcast: true,
        reason: 'mutation'
      });
    }
    _duplicateSlide(i) {
      const slide = this._slides[i];
      if (!slide) return;
      const copy = slide.cloneNode(true);
      // Strip ids so the document stays valid (no duplicate-id collisions
      // with the original). Same treatment _materialize gives rail clones.
      copy.removeAttribute('id');
      copy.querySelectorAll('[id]').forEach(el => el.removeAttribute('id'));
      // Insert after the original and make the copy active so it's the one
      // on screen. _collectSlides re-derives data-screen-label / data-deck-*
      // attrs, so the cloned values are overwritten.
      this._index = i + 1;
      this._squelchSlotChange = true;
      this.insertBefore(copy, slide.nextSibling);
      this._emitDeckChange({
        action: 'duplicate',
        from: i,
        to: i + 1,
        slide: copy
      });
      this._collectSlides();
      this._applyIndex({
        showOverlay: true,
        broadcast: true,
        reason: 'mutation'
      });
    }
    _toggleSkip(i) {
      const slide = this._slides[i];
      if (!slide) return;
      const on = !slide.hasAttribute('data-deck-skip');
      if (on) slide.setAttribute('data-deck-skip', '');else slide.removeAttribute('data-deck-skip');
      if (this._thumbs && this._thumbs[i]) {
        if (on) this._thumbs[i].thumb.setAttribute('data-skip', '');else this._thumbs[i].thumb.removeAttribute('data-skip');
      }
      this._markLastVisible();
      this._emitDeckChange({
        action: on ? 'skip' : 'unskip',
        from: i,
        slide
      });
      // Re-broadcast so the presenter popup's prev/next thumbnails re-pick
      // the nearest non-skipped slide without waiting for a nav event.
      try {
        window.postMessage({
          slideIndexChanged: this._index,
          deckTotal: this._slides.length,
          deckSkipped: this._skippedIndices()
        }, '*');
      } catch (e) {}
    }
    _skippedIndices() {
      const out = [];
      for (let i = 0; i < this._slides.length; i++) {
        if (this._slides[i].hasAttribute('data-deck-skip')) out.push(i);
      }
      return out;
    }
    _moveSlide(i, j) {
      if (j < 0 || j >= this._slides.length || j === i) return;
      const slide = this._slides[i];
      const ref = j < i ? this._slides[j] : this._slides[j].nextSibling;
      // Track the active slide across the reorder so the same content
      // stays on screen.
      const cur = this._index;
      if (cur === i) this._index = j;else if (i < cur && j >= cur) this._index = cur - 1;else if (i > cur && j <= cur) this._index = cur + 1;
      this._squelchSlotChange = true;
      this.insertBefore(slide, ref);
      this._emitDeckChange({
        action: 'move',
        from: i,
        to: j,
        slide
      });
      this._collectSlides();
      this._applyIndex({
        showOverlay: false,
        broadcast: true,
        reason: 'mutation'
      });
    }

    // Public API ------------------------------------------------------------

    /** Current slide index (0-based). */
    get index() {
      return this._index;
    }
    /** Total slide count. */
    get length() {
      return this._slides.length;
    }
    /** Programmatically navigate. */
    goTo(i) {
      this._go(i, 'api');
    }
    next() {
      this._advance(1, 'api');
    }
    prev() {
      this._advance(-1, 'api');
    }
    reset() {
      this._go(0, 'api');
    }
  }
  if (!customElements.get('deck-stage')) {
    customElements.define('deck-stage', DeckStage);
  }
})();
})(); } catch (e) { __ds_ns.__errors.push({ path: "decks/edm-stack/deck-stage.js", error: String((e && e.message) || e) }); }

// decks/pdmt-boards-systems/deck-stage.js
try { (() => {
// @ds-adherence-ignore -- omelette starter scaffold (raw elements/hex/px by design)
/* BEGIN USAGE */
/**
 * <deck-stage> — reusable web component for HTML decks.
 *
 * Handles:
 *  (a) speaker notes — reads <script type="application/json" id="speaker-notes">
 *      and posts {slideIndexChanged: N} to the parent window on nav.
 *  (b) keyboard navigation — ←/→, PgUp/PgDn, Space, Home/End, number keys.
 *      On touch devices, tapping the left/right half of the stage goes
 *      prev/next — taps on links, buttons and other interactive slide
 *      content are left alone.
 *  (c) press R to reset to slide 0 (with a tasteful keyboard hint).
 *  (d) bottom-center overlay showing slide count + hints, fades out on idle.
 *  (e) auto-scaling — inner canvas is a fixed design size (default 1920×1080)
 *      scaled with `transform: scale()` to fit the viewport, letterboxed.
 *      Set the `noscale` attribute to render at authored size (1:1) — the
 *      PPTX exporter sets this so its DOM capture sees unscaled geometry.
 *  (f) print — `@media print` lays every slide out as its own page at the
 *      design size, so the browser's Print → Save as PDF produces a clean
 *      one-page-per-slide PDF with no extra setup.
 *  (g) thumbnail rail — resizable left-hand column of per-slide thumbnails
 *      (static clones). Click to navigate; ↑/↓ with a thumbnail focused to
 *      step between slides; drag to reorder; right-click for
 *      Skip / Move up / Move down / Duplicate / Delete (Delete opens a
 *      Cancel/Delete confirm dialog). Drag the rail's right edge to resize;
 *      width persists to
 *      localStorage. Skipped slides carry `data-deck-skip`, are dimmed in
 *      the rail, omitted from prev/next navigation, and hidden at print.
 *      The rail is suppressed in presenting mode, in the host's Preview
 *      mode (ViewerMode='none'), on `noscale`, on narrow viewports
 *      (≤640px), and via the `no-rail` attribute. Rail mutations dispatch
 *      a `deckchange`
 *      CustomEvent on the element: detail = {action, from, to, slide}.
 *
 * Slides are HIDDEN, not unmounted. Non-active slides stay in the DOM with
 * `visibility: hidden` + `opacity: 0`, so their state (videos, iframes,
 * form inputs, React trees) is preserved across navigation.
 *
 * Lifecycle event — the component dispatches a `slidechange` CustomEvent on
 * itself whenever the active slide changes (including the initial mount).
 * The event bubbles and composes out of shadow DOM, so you can listen on
 * the <deck-stage> element or on document:
 *
 *   document.querySelector('deck-stage').addEventListener('slidechange', (e) => {
 *     e.detail.index         // new 0-based index
 *     e.detail.previousIndex // previous index, or -1 on init
 *     e.detail.total         // total slide count
 *     e.detail.slide         // the new active slide element
 *     e.detail.previousSlide // the prior slide element, or null on init
 *     e.detail.reason        // 'init' | 'keyboard' | 'click' | 'tap' | 'api'
 *   });
 *
 * Persistence: none at the deck level. The host app keeps the current slide
 * in its own URL (?slide=) and re-delivers it via location.hash on load, so a
 * bare load with no hash always starts at slide 1.
 *
 * Usage:
 *   <style>deck-stage:not(:defined){visibility:hidden}</style>
 *   <deck-stage width="1920" height="1080">
 *     <section data-label="Title">...</section>
 *     <section data-label="Agenda">...</section>
 *   </deck-stage>
 *   <script src="deck-stage.js"></script>
 *
 * The :not(:defined) rule prevents a flash of the first slide at its
 * authored styles before this script runs and attaches the shadow root.
 *
 * Slides are the direct element children of <deck-stage>. Each slide is
 * automatically tagged with:
 *   - data-screen-label="NN Label"   (1-indexed, for comment flow)
 *   - data-om-validate="no_overflowing_text,no_overlapping_text,slide_sized_text"
 *
 * Speaker notes stay in sync because the component posts {slideIndexChanged: N}
 * to the parent — just include the #speaker-notes script tag if asked for notes.
 *
 * Authoring guidance:
 *   - Write slide bodies as static HTML inside <deck-stage>, with sizing via
 *     CSS custom properties in a <style> block rather than JS constants.
 *     Static slide markup is what lets the user click a heading in edit mode
 *     and retype it directly; a slide rendered through <script type="text/babel">,
 *     React, or a loop over a JS array has to round-trip every tweak through a
 *     chat message instead. Reach for script-generated slides only when the
 *     content genuinely needs interactive behaviour static HTML can't express.
 *   - Do NOT set position/inset/width/height on the slide <section> elements —
 *     the component absolutely positions every slotted child for you.
 *   - Entrance animations: make the visible end-state the base style and
 *     animate *from* hidden, so print and reduced-motion show content.
 *     Gate the animation on [data-deck-active] and the motion query, e.g.
 *     `@media (prefers-reduced-motion:no-preference){ [data-deck-active] .x{animation:fade-in .5s both} }`.
 *     Avoid infinite decorative loops on slide content.
 */
/* END USAGE */

(() => {
  const DESIGN_W_DEFAULT = 1920;
  const DESIGN_H_DEFAULT = 1080;
  const OVERLAY_HIDE_MS = 1800;
  const VALIDATE_ATTR = 'no_overflowing_text,no_overlapping_text,slide_sized_text';
  const FINE_POINTER_MQ = matchMedia('(hover: hover) and (pointer: fine)');
  const NARROW_MQ = matchMedia('(max-width: 640px)');
  // Slide-authored controls that should keep a tap instead of it navigating.
  const INTERACTIVE_SEL = 'a[href], button, input, select, textarea, summary, label, video[controls], audio[controls], [role="button"], [onclick], [tabindex]:not([tabindex^="-"]), [contenteditable]:not([contenteditable="false" i])';
  const pad2 = n => String(n).padStart(2, '0');

  // Label precedence: data-label → data-screen-label (number stripped) → first heading → "Slide".
  const getSlideLabel = el => {
    const explicit = el.getAttribute('data-label');
    if (explicit) return explicit;
    const existing = el.getAttribute('data-screen-label');
    if (existing) return existing.replace(/^\s*\d+\s*/, '').trim() || existing;
    const h = el.querySelector('h1, h2, h3, [data-title]');
    const t = h && (h.textContent || '').trim().slice(0, 40);
    if (t) return t;
    return 'Slide';
  };
  const stylesheet = `
    :host {
      position: fixed;
      inset: 0;
      display: block;
      background: #000;
      color: #fff;
      font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif;
      overflow: hidden;
      -webkit-tap-highlight-color: transparent;
    }
    /* connectedCallback holds this until document.fonts.ready (capped 2s) so
     * the first visible paint has the deck's real typography + final rail
     * layout. opacity (not visibility) so the active slide can't un-hide
     * itself via the ::slotted([data-deck-active]) visibility:visible rule.
     * Only the stage/rail hide — the black :host background stays, so the
     * iframe doesn't flash the page's default white. */
    :host([data-fonts-pending]) .stage,
    :host([data-fonts-pending]) .rail { opacity: 0; pointer-events: none; }

    .stage {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .canvas {
      position: relative;
      transform-origin: center center;
      flex-shrink: 0;
      background: #fff;
      will-change: transform;
    }

    /* Slides live in light DOM (via <slot>) so authored CSS still applies.
       We absolutely position each slotted child to stack them. */
    ::slotted(*) {
      position: absolute !important;
      inset: 0 !important;
      width: 100% !important;
      height: 100% !important;
      box-sizing: border-box !important;
      overflow: hidden;
      opacity: 0;
      pointer-events: none;
      visibility: hidden;
    }
    ::slotted([data-deck-active]) {
      opacity: 1;
      pointer-events: auto;
      visibility: visible;
    }

    .overlay {
      position: fixed;
      left: 50%;
      bottom: 22px;
      transform: translate(-50%, 6px) scale(0.92);
      filter: blur(6px);
      display: flex;
      align-items: center;
      gap: 4px;
      padding: 4px;
      background: #000;
      color: #fff;
      border-radius: 999px;
      font-size: 12px;
      font-feature-settings: "tnum" 1;
      letter-spacing: 0.01em;
      opacity: 0;
      pointer-events: none;
      transition: opacity 260ms ease, transform 260ms cubic-bezier(.2,.8,.2,1), filter 260ms ease;
      transform-origin: center bottom;
      z-index: 2147483000;
      user-select: none;
    }
    .overlay[data-visible] {
      opacity: 1;
      pointer-events: auto;
      transform: translate(-50%, 0) scale(1);
      filter: blur(0);
    }

    .btn {
      appearance: none;
      -webkit-appearance: none;
      background: transparent;
      border: 0;
      margin: 0;
      padding: 0;
      color: inherit;
      font: inherit;
      cursor: default;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      height: 28px;
      min-width: 28px;
      border-radius: 999px;
      color: rgba(255,255,255,0.72);
      transition: background 140ms ease, color 140ms ease;
      -webkit-tap-highlight-color: transparent;
    }
    .btn:hover { background: rgba(255,255,255,0.12); color: #fff; }
    .btn:active { background: rgba(255,255,255,0.18); }
    .btn:focus { outline: none; }
    .btn:focus-visible { outline: none; }
    .btn::-moz-focus-inner { border: 0; }
    .btn svg { width: 14px; height: 14px; display: block; }
    .btn.reset {
      font-size: 11px;
      font-weight: 500;
      letter-spacing: 0.02em;
      padding: 0 10px 0 12px;
      gap: 6px;
      color: rgba(255,255,255,0.72);
    }
    .btn.reset .kbd {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 16px;
      height: 16px;
      padding: 0 4px;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      font-size: 10px;
      line-height: 1;
      color: rgba(255,255,255,0.88);
      background: rgba(255,255,255,0.12);
      border-radius: 4px;
    }

    .count {
      font-variant-numeric: tabular-nums;
      color: #fff;
      font-weight: 500;
      padding: 0 8px;
      min-width: 42px;
      text-align: center;
      font-size: 12px;
    }
    .count .sep { color: rgba(255,255,255,0.45); margin: 0 3px; font-weight: 400; }
    .count .total { color: rgba(255,255,255,0.55); }

    .divider {
      width: 1px;
      height: 14px;
      background: rgba(255,255,255,0.18);
      margin: 0 2px;
    }

    /* ── Thumbnail rail ──────────────────────────────────────────────────
       Fixed column on the left; each thumbnail is a static deep-clone of
       the light-DOM slide scaled into a 16:9 (or design-aspect) frame. The
       stage re-fits around it (see _fit); hidden during present / noscale
       / print so capture geometry and fullscreen output are unchanged. */
    .rail {
      position: fixed;
      left: 0;
      top: 0;
      bottom: 0;
      width: var(--deck-rail-w, 188px);
      background: #141414;
      border-right: 1px solid rgba(255,255,255,0.08);
      overflow-y: auto;
      overflow-x: hidden;
      padding: 12px 10px;
      box-sizing: border-box;
      display: flex;
      flex-direction: column;
      gap: 12px;
      z-index: 2147482500;
      scrollbar-width: thin;
      scrollbar-color: rgba(255,255,255,0.18) transparent;
    }
    .rail::-webkit-scrollbar { width: 8px; }
    .rail::-webkit-scrollbar-track { background: transparent; margin: 2px; }
    .rail::-webkit-scrollbar-thumb {
      background: rgba(255,255,255,0.18);
      border-radius: 4px;
      border: 2px solid transparent;
      background-clip: content-box;
    }
    .rail::-webkit-scrollbar-thumb:hover {
      background: rgba(255,255,255,0.28);
      border: 2px solid transparent;
      background-clip: content-box;
    }
    :host([no-rail]) .rail,
    :host([noscale]) .rail { display: none; }
    .rail[data-presenting] { display: none; }
    @media (max-width: 640px) {
      .rail, .rail-resize { display: none; }
    }
    /* User-driven show/hide (the TweaksPanel toggle) slides instead of
       popping. Transitions are gated on :host([data-rail-anim]) — set only
       for the 200ms around the toggle — so window-resize and rail-width
       drag (which also call _fit) don't lag behind the cursor. */
    .rail[data-user-hidden] { transform: translateX(-100%); }
    :host([data-rail-anim]) .rail { transition: transform 200ms cubic-bezier(.3,.7,.4,1); }
    :host([data-rail-anim]) .stage { transition: left 200ms cubic-bezier(.3,.7,.4,1); }
    :host([data-rail-anim]) .canvas { transition: transform 200ms cubic-bezier(.3,.7,.4,1); }
    /* transition shorthand replaces rather than merges — repeat the base
       .overlay opacity/transform/filter transitions so visibility changes
       during the 200ms toggle window still fade instead of popping. */
    :host([data-rail-anim]) .overlay {
      transition: margin-left 200ms cubic-bezier(.3,.7,.4,1),
                  opacity 260ms ease,
                  transform 260ms cubic-bezier(.2,.8,.2,1),
                  filter 260ms ease;
    }

    .thumb {
      position: relative;
      display: flex;
      align-items: flex-start;
      gap: 8px;
      cursor: pointer;
      user-select: none;
    }
    .thumb .num {
      width: 16px;
      flex-shrink: 0;
      font-size: 11px;
      font-weight: 500;
      text-align: right;
      color: rgba(255,255,255,0.55);
      padding-top: 2px;
      font-variant-numeric: tabular-nums;
    }
    .thumb .frame {
      position: relative;
      flex: 1;
      min-width: 0;
      aspect-ratio: var(--deck-aspect);
      background: #fff;
      border-radius: 4px;
      outline: 2px solid transparent;
      outline-offset: 0;
      overflow: hidden;
      transition: outline-color 120ms ease;
    }
    .thumb:hover .frame { outline-color: rgba(255,255,255,0.25); }
    .thumb { outline: none; }
    .thumb:focus-visible .frame { outline-color: rgba(255,255,255,0.5); }
    .thumb[data-current] .num { color: #fff; }
    .thumb[data-current] .frame { outline-color: #D97757; }
    .thumb[data-dragging] { opacity: 0.35; }
    .thumb::before {
      content: '';
      position: absolute;
      left: 24px;
      right: 0;
      height: 3px;
      border-radius: 2px;
      background: #D97757;
      opacity: 0;
      pointer-events: none;
    }
    .thumb[data-drop="before"]::before { top: -8px; opacity: 1; }
    .thumb[data-drop="after"]::before { bottom: -8px; opacity: 1; }
    .thumb[data-skip] .frame { opacity: 0.35; }
    .thumb[data-skip] .frame::after {
      content: 'Skipped';
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(0,0,0,0.45);
      color: #fff;
      font-size: 10px;
      font-weight: 500;
      letter-spacing: 0.04em;
    }

    .ctxmenu {
      position: fixed;
      min-width: 150px;
      padding: 4px;
      background: #242424;
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 7px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.45);
      z-index: 2147483100;
      display: none;
      font-size: 12px;
    }
    .ctxmenu[data-open] { display: block; }
    .ctxmenu button {
      display: block;
      width: 100%;
      appearance: none;
      border: 0;
      background: transparent;
      color: #e8e8e8;
      font: inherit;
      text-align: left;
      padding: 6px 10px;
      border-radius: 4px;
      cursor: pointer;
    }
    .ctxmenu button:hover:not(:disabled) { background: rgba(255,255,255,0.08); }
    .ctxmenu button:disabled { opacity: 0.35; cursor: default; }
    .ctxmenu hr {
      border: 0;
      border-top: 1px solid rgba(255,255,255,0.1);
      margin: 4px 2px;
    }

    .rail-resize {
      position: fixed;
      left: calc(var(--deck-rail-w, 188px) - 3px);
      top: 0;
      bottom: 0;
      width: 6px;
      cursor: col-resize;
      z-index: 2147482600;
      touch-action: none;
    }
    .rail-resize:hover,
    .rail-resize[data-dragging] { background: rgba(255,255,255,0.12); }
    :host([no-rail]) .rail-resize,
    :host([noscale]) .rail-resize,
    .rail[data-presenting] + .rail-resize,
    .rail[data-user-hidden] + .rail-resize { display: none; }

    /* Delete-confirm popup — matches the SPA's ConfirmDialog layout
       (title + message body, depressed footer with Cancel / Delete). */
    .confirm-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.45);
      z-index: 2147483200;
      display: none;
      align-items: center;
      justify-content: center;
    }
    .confirm-backdrop[data-open] { display: flex; }
    .confirm {
      width: 320px;
      max-width: calc(100vw - 32px);
      background: #2a2a2a;
      color: #e8e8e8;
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 12px;
      box-shadow: 0 12px 32px rgba(0,0,0,0.5);
      overflow: hidden;
      font-family: inherit;
      animation: deck-confirm-in 0.18s ease;
    }
    @keyframes deck-confirm-in {
      from { opacity: 0; transform: scale(0.96); }
      to { opacity: 1; transform: scale(1); }
    }
    .confirm .body { padding: 20px 20px 16px; }
    .confirm .title { font-size: 14px; font-weight: 600; margin-bottom: 4px; }
    .confirm .msg { font-size: 13px; line-height: 1.5; color: rgba(255,255,255,0.65); }
    .confirm .footer {
      padding: 14px 20px;
      background: #1f1f1f;
      border-top: 1px solid rgba(255,255,255,0.08);
      display: flex;
      justify-content: flex-end;
      gap: 8px;
    }
    .confirm button {
      appearance: none;
      font: inherit;
      font-size: 13px;
      font-weight: 500;
      padding: 8px 16px;
      border-radius: 8px;
      cursor: pointer;
    }
    .confirm .cancel {
      background: transparent;
      border: 0;
      color: rgba(255,255,255,0.8);
    }
    .confirm .cancel:hover { background: rgba(255,255,255,0.08); }
    .confirm .danger {
      background: #c96442;
      border: 1px solid rgba(0,0,0,0.15);
      color: #fff;
      box-shadow: 0 1px 3px rgba(166,50,68,0.3), 0 2px 6px rgba(166,50,68,0.18);
    }
    .confirm .danger:hover { background: #b5563a; }

    /* ── Print: one page per slide, no chrome ────────────────────────────
       The screen layout stacks every slide at inset:0 inside a scaled
       canvas; for print we want them in document flow at the authored
       design size so the browser paginates one slide per sheet. The
       @page size is set from the width/height attributes via the inline
       <style id="deck-stage-print-page"> that connectedCallback injects
       into <head> (the @page at-rule has no effect inside shadow DOM). */
    @media print {
      :host {
        position: static;
        inset: auto;
        background: none;
        overflow: visible;
        color: inherit;
      }
      .stage { position: static; display: block; }
      .canvas {
        transform: none !important;
        width: auto !important;
        height: auto !important;
        background: none;
        will-change: auto;
      }
      ::slotted(*) {
        position: relative !important;
        inset: auto !important;
        width: var(--deck-design-w) !important;
        height: var(--deck-design-h) !important;
        box-sizing: border-box !important;
        opacity: 1 !important;
        visibility: visible !important;
        pointer-events: auto;
        break-after: page;
        page-break-after: always;
        break-inside: avoid;
        overflow: hidden;
      }
      /* :last-child alone isn't enough once data-deck-skip hides the
         trailing slide(s) — the last *visible* slide still carries
         break-after:page and prints a blank sheet. _markLastVisible()
         maintains data-deck-last-visible on the last non-skipped slide. */
      ::slotted(*:last-child),
      ::slotted([data-deck-last-visible]) {
        break-after: auto;
        page-break-after: auto;
      }
      ::slotted([data-deck-skip]) { display: none !important; }
      .overlay, .rail, .rail-resize, .ctxmenu, .confirm-backdrop { display: none !important; }
    }
  `;
  class DeckStage extends HTMLElement {
    static get observedAttributes() {
      return ['width', 'height', 'noscale', 'no-rail'];
    }
    constructor() {
      super();
      this._root = this.attachShadow({
        mode: 'open'
      });
      this._index = 0;
      this._slides = [];
      this._notes = [];
      this._hideTimer = null;
      this._mouseIdleTimer = null;
      this._menuIndex = -1;
      this._onKey = this._onKey.bind(this);
      this._onResize = this._onResize.bind(this);
      this._onSlotChange = this._onSlotChange.bind(this);
      this._onMouseMove = this._onMouseMove.bind(this);
      this._onTap = this._onTap.bind(this);
      this._onMessage = this._onMessage.bind(this);
      // Capture-phase close so a click anywhere dismisses the menu, but
      // ignore clicks that land inside the menu itself — otherwise the
      // capture handler runs before the menu's own (bubble) handler and
      // clears _menuIndex out from under it.
      this._onDocClick = e => {
        if (this._menu && e.composedPath && e.composedPath().includes(this._menu)) return;
        this._closeMenu();
      };
    }
    get designWidth() {
      return parseInt(this.getAttribute('width'), 10) || DESIGN_W_DEFAULT;
    }
    get designHeight() {
      return parseInt(this.getAttribute('height'), 10) || DESIGN_H_DEFAULT;
    }
    connectedCallback() {
      // Presenter-view popup loads deckUrl?_snthumb=...#N for its prev/cur/
      // next thumbnails — the rail has no business rendering inside those
      // (wrong scale, and it offsets the stage so the thumb shows a gutter).
      if (/[?&]_snthumb=/.test(location.search)) this.setAttribute('no-rail', '');
      this._render();
      this._loadNotes();
      this._syncPrintPageRule();
      window.addEventListener('keydown', this._onKey);
      window.addEventListener('resize', this._onResize);
      window.addEventListener('mousemove', this._onMouseMove, {
        passive: true
      });
      window.addEventListener('message', this._onMessage);
      window.addEventListener('click', this._onDocClick, true);
      this.addEventListener('click', this._onTap);
      // Print lays every slide out as its own page, so [data-deck-active]-
      // gated entrance styles need the attribute on every slide (not just
      // the current one) or their content prints at the hidden base style.
      // The transient freeze style lands BEFORE the attributes so any
      // attribute-keyed transition fires at 0s (changing transition-
      // duration after a transition has started doesn't affect it).
      this._onBeforePrint = () => {
        if (this._freezeStyle) this._freezeStyle.remove();
        this._freezeStyle = document.createElement('style');
        this._freezeStyle.textContent = '*,*::before,*::after{transition-duration:0s !important}';
        document.head.appendChild(this._freezeStyle);
        this._slides.forEach(s => s.setAttribute('data-deck-active', ''));
      };
      this._onAfterPrint = () => {
        this._applyIndex({
          showOverlay: false,
          broadcast: false
        });
        if (this._freezeStyle) {
          this._freezeStyle.remove();
          this._freezeStyle = null;
        }
      };
      window.addEventListener('beforeprint', this._onBeforePrint);
      window.addEventListener('afterprint', this._onAfterPrint);
      // Initial collection + layout happens via slotchange, which fires on mount.
      this._enableRail();
      // Hold the stage hidden until webfonts are ready so the first visible
      // paint has the deck's real typography — the :not(:defined) guard in
      // the page HTML only covers custom-element upgrade, not font load.
      // Capped so a 404'd font URL can't blank the deck indefinitely.
      this.setAttribute('data-fonts-pending', '');
      const reveal = () => this.removeAttribute('data-fonts-pending');
      // rAF first: fonts.ready is a pre-resolved promise until layout has
      // resolved the slotted text's font-family and pushed a FontFace into
      // 'loading'. Reading it here in connectedCallback (parse-time) would
      // settle the race in a microtask before any font fetch starts.
      requestAnimationFrame(() => {
        Promise.race([document.fonts ? document.fonts.ready : Promise.resolve(), new Promise(r => setTimeout(r, 2000))]).then(reveal, reveal);
      });
    }
    _enableRail() {
      // Idempotent — older host builds still post __omelette_rail_enabled.
      // no-rail guard keeps the observers/stylesheet walk off the cheap path
      // for presenter-popup thumbnail iframes (up to 9 per view).
      if (this._railEnabled || this.hasAttribute('no-rail')) return;
      this._railEnabled = true;
      // Per-viewer preference — restored alongside rail width. Default on;
      // only a stored '0' (from the TweaksPanel toggle) hides it.
      this._railVisible = true;
      try {
        if (localStorage.getItem('deck-stage.railVisible') === '0') this._railVisible = false;
      } catch (e) {}
      // Live thumbnail updates: watch the light-DOM slides for content
      // edits and re-clone just the affected thumb(s), debounced. Ignore
      // the data-deck-* / data-screen-label / data-om-validate attributes
      // this component itself writes so nav and skip don't trigger
      // spurious refreshes.
      const OWN_ATTRS = /^data-(deck-|screen-label$|om-validate$)/;
      this._liveDirty = new Set();
      this._liveObserver = new MutationObserver(records => {
        for (const r of records) {
          if (r.type === 'attributes' && OWN_ATTRS.test(r.attributeName || '')) continue;
          let n = r.target;
          while (n && n.parentElement !== this) n = n.parentElement;
          if (n && this._slideSet && this._slideSet.has(n)) this._liveDirty.add(n);
        }
        if (this._liveDirty.size && !this._liveTimer) {
          this._liveTimer = setTimeout(() => {
            this._liveTimer = null;
            this._liveDirty.forEach(s => this._refreshThumb(s));
            this._liveDirty.clear();
          }, 200);
        }
      });
      this._liveObserver.observe(this, {
        subtree: true,
        childList: true,
        characterData: true,
        attributes: true
      });
      // Lazy thumbnail materialization — clone the slide only when its
      // frame scrolls into (or near) the rail viewport. rootMargin gives
      // ~4 thumbs of pre-load so fast scrolling doesn't flash blanks.
      this._railObserver = new IntersectionObserver(entries => {
        entries.forEach(e => {
          if (e.isIntersecting && e.target.__deckThumb) {
            this._materialize(e.target.__deckThumb);
          }
        });
      }, {
        root: this._rail,
        rootMargin: '400px 0px'
      });
      // Tweaks typically change CSS vars / attrs OUTSIDE <deck-stage>
      // (on <html>, <body>, a wrapper div, or a <style> tag), which
      // _liveObserver can't see. Re-snapshot author CSS (constructable
      // sheet is shared by reference, so one replaceSync updates every
      // thumb shadow root) and re-sync each thumb host's attrs + custom
      // properties. In-slide DOM mutations are _liveObserver's job.
      // Debounced so slider drags don't thrash.
      this._onTweakChange = () => {
        clearTimeout(this._tweakTimer);
        this._tweakTimer = setTimeout(() => {
          this._snapshotAuthorCss();
          // One getComputedStyle for the whole batch — each
          // getPropertyValue read below reuses the same computed style
          // as long as nothing invalidates layout between thumbs.
          const cs = getComputedStyle(this);
          (this._thumbs || []).forEach(t => {
            if (t.host) this._syncThumbHostAttrs(t.host, cs);
          });
        }, 120);
      };
      window.addEventListener('tweakchange', this._onTweakChange);
      this._snapshotAuthorCss();
      // Build the rail now that it's enabled — slotchange already fired,
      // so _renderRail's early-return skipped the initial build.
      this._syncRailHidden();
      this._renderRail();
      this._fit();
    }

    /** Snapshot document stylesheets into a constructable sheet that each
     *  thumbnail's nested shadow root adopts — so author CSS styles the
     *  cloned slide content without touching this component's chrome.
     *  Cross-origin sheets throw on .cssRules — skip them. Re-callable:
     *  the existing constructable sheet is reused via replaceSync so every
     *  already-adopted shadow root picks up the fresh CSS without re-adopt. */
    _snapshotAuthorCss() {
      // :root in an adopted sheet inside a shadow root matches nothing
      // (only the document root qualifies), so author rules like
      // `:root[data-voice="modern"] .serif` never reach the clones.
      // Rewrite :root → :host and mirror <html>'s data-*/class/lang onto
      // each thumb host (see _syncThumbHostAttrs) so the same selectors
      // match inside the thumbnail's shadow tree.
      const authorCss = Array.from(document.styleSheets).map(sh => {
        try {
          return Array.from(sh.cssRules).map(r => r.cssText).join('\n');
        } catch (e) {
          return '';
        }
      }).join('\n')
      // The shadow host is featureless outside the functional :host(...)
      // form, so any compound on :root — [attr], .class, #id, :pseudo —
      // must become :host(<compound>) not :host<compound>. Same for the
      // html type selector (Tailwind class-strategy dark mode emits
      // html.dark; Pico uses html[data-theme]), which has nothing to
      // match inside the thumb's shadow tree.
      .replace(/:root((?:\[[^\]]*\]|[.#][-\w]+|:[-\w]+(?:\([^)]*\))?)+)/g, ':host($1)').replace(/:root\b/g, ':host').replace(/(^|[\s,>~+(}])html((?:\[[^\]]*\]|[.#][-\w]+|:[-\w]+(?:\([^)]*\))?)+)(?![-\w])/g, '$1:host($2)').replace(/(^|[\s,>~+(}])html(?![-\w])/g, '$1:host');
      // Every custom property the author references. _syncThumbHostAttrs
      // mirrors each one's *computed* value at <deck-stage> onto the
      // thumb host so the live value wins over the :host default above
      // regardless of which ancestor the tweak wrote to (<html>, <body>,
      // a wrapper div, or the deck-stage element itself all inherit
      // down to getComputedStyle(this)).
      this._authorVars = new Set(authorCss.match(/--[\w-]+/g) || []);
      try {
        if (!this._adoptedSheet) this._adoptedSheet = new CSSStyleSheet();
        this._adoptedSheet.replaceSync(authorCss);
      } catch (e) {
        this._adoptedSheet = null;
        this._authorCss = authorCss;
      }
    }
    _syncThumbHostAttrs(host, cs) {
      const de = document.documentElement;
      // setAttribute overwrites but can't delete — an attr removed from
      // <html> (toggleAttribute off, classList emptied) would linger on
      // the host and :host([data-*]) / :host(.foo) rules would keep
      // matching. Remove stale mirrored attrs first; iterate backward
      // because removeAttribute mutates the live NamedNodeMap.
      for (let i = host.attributes.length - 1; i >= 0; i--) {
        const n = host.attributes[i].name;
        if ((n.startsWith('data-') || n === 'class' || n === 'lang') && !de.hasAttribute(n)) {
          host.removeAttribute(n);
        }
      }
      for (const a of de.attributes) {
        if (a.name.startsWith('data-') || a.name === 'class' || a.name === 'lang') {
          host.setAttribute(a.name, a.value);
        }
      }
      // The :root→:host rewrite in _snapshotAuthorCss pins each custom
      // property to its stylesheet default on the thumb host, shadowing
      // the live value that would otherwise inherit. Tweaks can write the
      // live value on any ancestor — <html>, <body>, a wrapper div, the
      // deck-stage element — so read it as the *computed* value at
      // <deck-stage> (which sees the whole inheritance chain) rather than
      // trying to guess which element the author wrote to. Inline on the
      // host beats the :host{} rule. remove-stale covers vars dropped
      // from the stylesheet between snapshots.
      const vars = this._authorVars || new Set();
      for (let i = host.style.length - 1; i >= 0; i--) {
        const p = host.style[i];
        if (p.startsWith('--') && !vars.has(p)) host.style.removeProperty(p);
      }
      const live = cs || getComputedStyle(this);
      vars.forEach(p => {
        const v = live.getPropertyValue(p);
        if (v) host.style.setProperty(p, v.trim());else host.style.removeProperty(p);
      });
    }
    disconnectedCallback() {
      window.removeEventListener('keydown', this._onKey);
      window.removeEventListener('resize', this._onResize);
      window.removeEventListener('mousemove', this._onMouseMove);
      window.removeEventListener('message', this._onMessage);
      window.removeEventListener('click', this._onDocClick, true);
      window.removeEventListener('beforeprint', this._onBeforePrint);
      window.removeEventListener('afterprint', this._onAfterPrint);
      if (this._freezeStyle) {
        this._freezeStyle.remove();
        this._freezeStyle = null;
      }
      this.removeEventListener('click', this._onTap);
      if (this._hideTimer) clearTimeout(this._hideTimer);
      if (this._mouseIdleTimer) clearTimeout(this._mouseIdleTimer);
      if (this._liveTimer) clearTimeout(this._liveTimer);
      if (this._tweakTimer) clearTimeout(this._tweakTimer);
      if (this._railAnimTimer) clearTimeout(this._railAnimTimer);
      if (this._scaleRaf) cancelAnimationFrame(this._scaleRaf);
      if (this._liveObserver) this._liveObserver.disconnect();
      if (this._railObserver) this._railObserver.disconnect();
      if (this._onTweakChange) window.removeEventListener('tweakchange', this._onTweakChange);
    }
    attributeChangedCallback() {
      if (this._canvas) {
        this._canvas.style.width = this.designWidth + 'px';
        this._canvas.style.height = this.designHeight + 'px';
        this._canvas.style.setProperty('--deck-design-w', this.designWidth + 'px');
        this._canvas.style.setProperty('--deck-design-h', this.designHeight + 'px');
        if (this._rail) {
          this._rail.style.setProperty('--deck-aspect', this.designWidth + '/' + this.designHeight);
        }
        this._fit();
        this._scaleThumbs();
        this._syncPrintPageRule();
      }
    }
    _render() {
      const style = document.createElement('style');
      style.textContent = stylesheet;
      const stage = document.createElement('div');
      stage.className = 'stage';
      const canvas = document.createElement('div');
      canvas.className = 'canvas';
      canvas.style.width = this.designWidth + 'px';
      canvas.style.height = this.designHeight + 'px';
      canvas.style.setProperty('--deck-design-w', this.designWidth + 'px');
      canvas.style.setProperty('--deck-design-h', this.designHeight + 'px');
      const slot = document.createElement('slot');
      slot.addEventListener('slotchange', this._onSlotChange);
      canvas.appendChild(slot);
      stage.appendChild(canvas);

      // Overlay: compact, solid black, with clickable controls.
      const overlay = document.createElement('div');
      overlay.className = 'overlay export-hidden';
      overlay.setAttribute('role', 'toolbar');
      overlay.setAttribute('aria-label', 'Deck controls');
      overlay.setAttribute('data-omelette-chrome', '');
      overlay.innerHTML = `
        <button class="btn prev" type="button" aria-label="Previous slide" title="Previous (←)">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10 3L5 8l5 5"/></svg>
        </button>
        <span class="count" aria-live="polite"><span class="current">1</span><span class="sep">/</span><span class="total">1</span></span>
        <button class="btn next" type="button" aria-label="Next slide" title="Next (→)">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 3l5 5-5 5"/></svg>
        </button>
        <span class="divider"></span>
        <button class="btn reset" type="button" aria-label="Reset to first slide" title="Reset (R)">Reset<span class="kbd">R</span></button>
      `;
      overlay.querySelector('.prev').addEventListener('click', () => this._advance(-1, 'click'));
      overlay.querySelector('.next').addEventListener('click', () => this._advance(1, 'click'));
      overlay.querySelector('.reset').addEventListener('click', () => this._go(0, 'click'));

      // Thumbnail rail + context menu. Thumbnails are populated in
      // _renderRail() after _collectSlides().
      const rail = document.createElement('div');
      rail.className = 'rail export-hidden';
      rail.setAttribute('data-omelette-chrome', '');
      rail.style.setProperty('--deck-aspect', this.designWidth + '/' + this.designHeight);
      // Edge auto-scroll while dragging a thumb near the rail's top/bottom
      // so off-screen drop targets are reachable. Native dragover fires
      // continuously while the pointer is stationary, so a per-event nudge
      // (ramped by edge proximity) is enough — no rAF loop needed.
      rail.addEventListener('dragover', e => {
        if (this._dragFrom == null) return;
        const r = rail.getBoundingClientRect();
        const EDGE = 40;
        const dt = e.clientY - r.top;
        const db = r.bottom - e.clientY;
        if (dt < EDGE) rail.scrollTop -= Math.ceil((EDGE - dt) / 3);else if (db < EDGE) rail.scrollTop += Math.ceil((EDGE - db) / 3);
      });
      const menu = document.createElement('div');
      menu.className = 'ctxmenu export-hidden';
      menu.setAttribute('data-omelette-chrome', '');
      menu.innerHTML = `
        <button type="button" data-act="skip">Skip slide</button>
        <button type="button" data-act="up">Move up</button>
        <button type="button" data-act="down">Move down</button>
        <button type="button" data-act="duplicate">Duplicate slide</button>
        <hr>
        <button type="button" data-act="delete">Delete slide</button>
      `;
      menu.addEventListener('click', e => {
        const act = e.target && e.target.getAttribute && e.target.getAttribute('data-act');
        if (!act) return;
        const i = this._menuIndex;
        this._closeMenu();
        if (act === 'skip') this._toggleSkip(i);else if (act === 'up') this._moveSlide(i, i - 1);else if (act === 'down') this._moveSlide(i, i + 1);else if (act === 'duplicate') this._duplicateSlide(i);else if (act === 'delete') this._openConfirm(i);
      });
      menu.addEventListener('contextmenu', e => e.preventDefault());

      // Rail resize handle — drag to set --deck-rail-w, persisted to
      // localStorage so the width survives reloads.
      const resize = document.createElement('div');
      resize.className = 'rail-resize export-hidden';
      resize.setAttribute('data-omelette-chrome', '');
      resize.addEventListener('pointerdown', e => {
        e.preventDefault();
        resize.setPointerCapture(e.pointerId);
        resize.setAttribute('data-dragging', '');
        const move = ev => this._setRailWidth(ev.clientX);
        const up = () => {
          resize.removeEventListener('pointermove', move);
          resize.removeEventListener('pointerup', up);
          resize.removeEventListener('pointercancel', up);
          resize.removeAttribute('data-dragging');
          try {
            localStorage.setItem('deck-stage.railWidth', String(this._railPx));
          } catch (err) {}
        };
        resize.addEventListener('pointermove', move);
        resize.addEventListener('pointerup', up);
        resize.addEventListener('pointercancel', up);
      });

      // Delete-confirm dialog — mirrors the SPA's ConfirmDialog layout.
      const confirm = document.createElement('div');
      confirm.className = 'confirm-backdrop export-hidden';
      confirm.setAttribute('data-omelette-chrome', '');
      confirm.innerHTML = `
        <div class="confirm" role="dialog" aria-modal="true">
          <div class="body">
            <div class="title">Delete slide?</div>
            <div class="msg">This slide will be removed from the deck.</div>
          </div>
          <div class="footer">
            <button type="button" class="cancel">Cancel</button>
            <button type="button" class="danger">Delete</button>
          </div>
        </div>
      `;
      confirm.addEventListener('click', e => {
        if (e.target === confirm) this._closeConfirm();
      });
      confirm.querySelector('.cancel').addEventListener('click', () => this._closeConfirm());
      confirm.querySelector('.danger').addEventListener('click', () => {
        const i = this._confirmIndex;
        this._closeConfirm();
        this._deleteSlide(i);
      });
      this._root.append(style, rail, resize, stage, overlay, menu, confirm);
      this._canvas = canvas;
      this._stage = stage;
      this._slot = slot;
      this._overlay = overlay;
      this._rail = rail;
      this._resize = resize;
      this._menu = menu;
      this._confirm = confirm;
      this._countEl = overlay.querySelector('.current');
      this._totalEl = overlay.querySelector('.total');

      // Restore persisted rail width.
      let rw = 188;
      try {
        const s = localStorage.getItem('deck-stage.railWidth');
        if (s) rw = parseInt(s, 10) || rw;
      } catch (err) {}
      this._setRailWidth(rw);
      this._syncRailHidden();
    }
    _setRailWidth(px) {
      const w = Math.max(120, Math.min(360, Math.round(px)));
      this._railPx = w;
      this.style.setProperty('--deck-rail-w', w + 'px');
      this._fit();
      // _scaleThumbs forces a sync layout (frame.offsetWidth) then writes
      // N transforms. During a resize drag this runs per-pointermove;
      // coalesce to one per frame.
      if (!this._scaleRaf) {
        this._scaleRaf = requestAnimationFrame(() => {
          this._scaleRaf = null;
          this._scaleThumbs();
        });
      }
    }

    /** @page must live in the document stylesheet — it's a no-op inside
     *  shadow DOM. Inject/update a single <head> style tag so the print
     *  sheet matches the design size and Save-as-PDF yields one slide per
     *  page with no margins. */
    _syncPrintPageRule() {
      const id = 'deck-stage-print-page';
      let tag = document.getElementById(id);
      if (!tag) {
        tag = document.createElement('style');
        tag.id = id;
        document.head.appendChild(tag);
      }
      tag.textContent = '@page { size: ' + this.designWidth + 'px ' + this.designHeight + 'px; margin: 0; } ' + '@media print { html, body { margin: 0 !important; padding: 0 !important; background: none !important; overflow: visible !important; height: auto !important; } ' + '* { -webkit-print-color-adjust: exact; print-color-adjust: exact; } ' +
      // Jump authored animations/transitions to their end state so print
      // never captures mid-entrance — pairs with the beforeprint handler
      // in connectedCallback that sets data-deck-active on every slide.
      '*, *::before, *::after { animation-delay: -99s !important; animation-duration: .001s !important; ' + 'animation-iteration-count: 1 !important; animation-fill-mode: both !important; ' + 'animation-play-state: running !important; transition-duration: 0s !important; } }';
    }
    _onSlotChange() {
      // Rail mutations (delete/move/duplicate) already reconcile synchronously and
      // emit slidechange with reason 'api'; skip the async slotchange that
      // would otherwise re-broadcast with reason 'init'.
      if (this._squelchSlotChange) {
        this._squelchSlotChange = false;
        return;
      }
      this._collectSlides();
      this._restoreIndex();
      this._applyIndex({
        showOverlay: false,
        broadcast: true,
        reason: 'init'
      });
      this._fit();
    }
    _collectSlides() {
      const assigned = this._slot.assignedElements({
        flatten: true
      });
      this._slides = assigned.filter(el => {
        // Skip template/style/script nodes even if someone slots them.
        const tag = el.tagName;
        return tag !== 'TEMPLATE' && tag !== 'SCRIPT' && tag !== 'STYLE';
      });
      this._slideSet = new Set(this._slides);
      this._slides.forEach((slide, i) => {
        const n = i + 1;
        slide.setAttribute('data-screen-label', `${pad2(n)} ${getSlideLabel(slide)}`);

        // Validation attribute for comment flow / auto-checks.
        if (!slide.hasAttribute('data-om-validate')) {
          slide.setAttribute('data-om-validate', VALIDATE_ATTR);
        }
        slide.setAttribute('data-deck-slide', String(i));
      });
      if (this._totalEl) this._totalEl.textContent = String(this._slides.length || 1);
      if (this._index >= this._slides.length) this._index = Math.max(0, this._slides.length - 1);
      this._markLastVisible();
      this._renderRail();
    }

    /** Tag the last non-skipped slide so print CSS can drop its
     *  break-after (see the @media print comment above — :last-child
     *  alone matches a hidden skipped slide). */
    _markLastVisible() {
      let last = null;
      this._slides.forEach(s => {
        s.removeAttribute('data-deck-last-visible');
        if (!s.hasAttribute('data-deck-skip')) last = s;
      });
      if (last) last.setAttribute('data-deck-last-visible', '');
    }
    _loadNotes() {
      const tag = document.getElementById('speaker-notes');
      if (!tag) {
        this._notes = [];
        return;
      }
      try {
        const parsed = JSON.parse(tag.textContent || '[]');
        if (Array.isArray(parsed)) this._notes = parsed;
      } catch (e) {
        console.warn('[deck-stage] Failed to parse #speaker-notes JSON:', e);
        this._notes = [];
      }
    }
    _restoreIndex() {
      // The host's ?slide= param is delivered as a #<int> hash (1-indexed) on
      // the iframe src. No hash → slide 1; the deck itself keeps no position
      // state across loads.
      const h = (location.hash || '').match(/^#(\d+)$/);
      if (h) {
        const n = parseInt(h[1], 10) - 1;
        if (n >= 0 && n < this._slides.length) this._index = n;
      }
    }
    _applyIndex({
      showOverlay = true,
      broadcast = true,
      reason = 'init'
    } = {}) {
      if (!this._slides.length) return;
      const prev = this._prevIndex == null ? -1 : this._prevIndex;
      const curr = this._index;
      // Keep the iframe's own hash in sync so an in-iframe location.reload()
      // (reload banner path in viewer-handle.ts) lands on the current slide,
      // not the stale deep-link hash from initial load.
      try {
        history.replaceState(null, '', '#' + (curr + 1));
      } catch (e) {}
      this._slides.forEach((s, i) => {
        if (i === curr) s.setAttribute('data-deck-active', '');else s.removeAttribute('data-deck-active');
      });
      if (this._countEl) this._countEl.textContent = String(curr + 1);
      // Follow-scroll on every navigation (init deep-link, keyboard, click,
      // tap, external goTo) — the only time we *don't* want the rail to
      // track current is after a rail-internal mutation, where _renderRail
      // has already restored the user's scroll position and yanking back to
      // current would undo it.
      this._syncRail(reason !== 'mutation');
      if (broadcast) {
        // (1) Legacy: host-window postMessage for speaker-notes renderers.
        try {
          window.postMessage({
            slideIndexChanged: curr,
            deckTotal: this._slides.length,
            deckSkipped: this._skippedIndices()
          }, '*');
        } catch (e) {}

        // (2) In-page CustomEvent on the <deck-stage> element itself.
        //     Bubbles and composes out of shadow DOM so slide code can listen:
        //       document.querySelector('deck-stage').addEventListener('slidechange', e => {
        //         e.detail.index, e.detail.previousIndex, e.detail.total, e.detail.slide, e.detail.reason
        //       });
        const detail = {
          index: curr,
          previousIndex: prev,
          total: this._slides.length,
          slide: this._slides[curr] || null,
          previousSlide: prev >= 0 ? this._slides[prev] || null : null,
          reason: reason // 'init' | 'keyboard' | 'click' | 'tap' | 'api'
        };
        this.dispatchEvent(new CustomEvent('slidechange', {
          detail,
          bubbles: true,
          composed: true
        }));
      }
      this._prevIndex = curr;
      if (showOverlay) this._flashOverlay();
    }
    _flashOverlay() {
      // Host posts __omelette_presenting while in fullscreen/tab presentation
      // mode — suppress the nav footer entirely (both hover and slide-change
      // flash) so the audience sees clean slides.
      if (!this._overlay || this._presenting) return;
      this._overlay.setAttribute('data-visible', '');
      if (this._hideTimer) clearTimeout(this._hideTimer);
      this._hideTimer = setTimeout(() => {
        this._overlay.removeAttribute('data-visible');
      }, OVERLAY_HIDE_MS);
    }
    _railWidth() {
      // State-based, no offsetWidth: the first _fit() can run before the
      // rail has had layout on some load paths, and a 0 there paints the
      // slide full-width for one frame before the post-slotchange _fit()
      // corrects it.
      if (!this._railEnabled || !this._railVisible || this.hasAttribute('no-rail') || this.hasAttribute('noscale') || this._presenting || this._previewMode || NARROW_MQ.matches) return 0;
      return this._railPx || 0;
    }
    _fit() {
      if (!this._canvas) return;
      const stage = this._canvas.parentElement;
      // PPTX export sets noscale so the DOM capture sees authored-size
      // geometry — the scaled canvas is in shadow DOM, so the exporter's
      // resetTransformSelector can't reach .canvas.style.transform directly.
      if (this.hasAttribute('noscale')) {
        this._canvas.style.transform = 'none';
        if (stage) stage.style.left = '0';
        if (this._overlay) this._overlay.style.marginLeft = '0';
        return;
      }
      const rw = this._railWidth();
      if (stage) stage.style.left = rw + 'px';
      // Overlay is centred on the viewport via left:50% + translate(-50%);
      // marginLeft shifts the centre by rw/2 so it lands in the middle of
      // the [rw, innerWidth] stage region.
      if (this._overlay) this._overlay.style.marginLeft = rw / 2 + 'px';
      const vw = window.innerWidth - rw;
      const vh = window.innerHeight;
      const s = Math.min(vw / this.designWidth, vh / this.designHeight);
      this._canvas.style.transform = `scale(${s})`;
    }
    _onResize() {
      this._fit();
      // Crossing the narrow-viewport breakpoint reveals the rail — rerun the
      // thumbnail scale the same way _setRailWidth does.
      if (!this._scaleRaf) {
        this._scaleRaf = requestAnimationFrame(() => {
          this._scaleRaf = null;
          this._scaleThumbs();
        });
      }
    }
    _onMouseMove() {
      // Keep overlay visible while mouse moves; hide after idle.
      this._flashOverlay();
    }
    _onMessage(e) {
      const d = e.data;
      if (d && typeof d.__omelette_presenting === 'boolean') {
        this._presenting = d.__omelette_presenting;
        if (this._presenting && this._overlay) {
          this._overlay.removeAttribute('data-visible');
          if (this._hideTimer) clearTimeout(this._hideTimer);
        }
        this._syncRailHidden();
        this._closeMenu();
        this._closeConfirm();
        this._fit();
        this._scaleThumbs();
      }
      // Host's Preview segment (ViewerMode='none'): the rail's drag-reorder /
      // right-click skip-delete affordances are editing chrome, so hide it
      // while the user is just looking at the deck. Same hard-hide path as
      // presenting; independent of the user's _railVisible preference so
      // returning to Edit restores whatever they had.
      if (d && typeof d.__omelette_preview_mode === 'boolean') {
        if (d.__omelette_preview_mode === this._previewMode) return;
        this._previewMode = d.__omelette_preview_mode;
        this._syncRailHidden();
        this._closeMenu();
        this._closeConfirm();
        this._fit();
        this._scaleThumbs();
      }
      // Per-viewer show/hide, driven by the TweaksPanel's auto-injected
      // "Thumbnail rail" toggle (or any author script). Independent of
      // whether the Tweaks panel itself is open — closing the panel
      // doesn't change rail visibility. Persists alongside rail width.
      if (d && d.type === '__deck_rail_visible' && typeof d.on === 'boolean') {
        if (d.on === this._railVisible) return;
        this._railVisible = d.on;
        try {
          localStorage.setItem('deck-stage.railVisible', d.on ? '1' : '0');
        } catch (e) {}
        // Arm the transition, commit it, then flip state — otherwise the
        // browser coalesces both writes and nothing animates on show.
        this.setAttribute('data-rail-anim', '');
        void (this._rail && this._rail.offsetHeight);
        this._syncRailHidden();
        this._fit();
        this._scaleThumbs();
        clearTimeout(this._railAnimTimer);
        this._railAnimTimer = setTimeout(() => this.removeAttribute('data-rail-anim'), 220);
      }
      if (d && d.type === '__omelette_rail_enabled') this._enableRail();
    }
    _syncRailHidden() {
      if (!this._rail) return;
      // data-presenting is the hard hide (display:none) for flag-off,
      // presentation mode, and the host's Preview segment — instant, no
      // transition. data-user-hidden is the soft hide (translateX(-100%))
      // for the viewer's rail toggle, so show/hide slides under
      // :host([data-rail-anim]).
      const hard = !this._railEnabled || this._presenting || this._previewMode;
      if (hard) this._rail.setAttribute('data-presenting', '');else this._rail.removeAttribute('data-presenting');
      if (!this._railVisible) this._rail.setAttribute('data-user-hidden', '');else this._rail.removeAttribute('data-user-hidden');
      // translateX hide leaves thumbs (tabIndex=0) in the tab order —
      // inert keeps them unfocusable while the rail is off-screen.
      this._rail.inert = hard || !this._railVisible;
    }
    _onTap(e) {
      // Touch-only — keyboard + the overlay toolbar cover nav on desktop.
      if (FINE_POINTER_MQ.matches) return;
      // Only taps that land on the stage (slide content or letterbox); the
      // overlay / rail / menus are siblings with their own click handlers.
      const path = e.composedPath();
      if (!this._stage || !path.includes(this._stage)) return;
      // Let interactive slide content keep the tap. composedPath (not
      // e.target.closest) so we see through open shadow roots — a <button>
      // inside a slide-authored custom element retargets e.target to the
      // host but still appears in the composed path.
      if (e.defaultPrevented) return;
      for (const n of path) {
        if (n === this._stage) break;
        if (n.matches && n.matches(INTERACTIVE_SEL)) return;
      }
      e.preventDefault();
      const rw = this._railWidth();
      const mid = rw + (window.innerWidth - rw) / 2;
      this._advance(e.clientX < mid ? -1 : 1, 'tap');
    }
    _onKey(e) {
      // Ignore when the user is typing.
      const t = e.target;
      if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
      // Confirm dialog swallows nav keys while open; Escape cancels. Enter
      // is left to the focused button's native activation so Tab→Cancel
      // →Enter activates Cancel, not the window-level confirm path.
      if (this._confirm && this._confirm.hasAttribute('data-open')) {
        if (e.key === 'Escape') {
          this._closeConfirm();
          e.preventDefault();
        }
        return;
      }
      if (e.key === 'Escape' && this._menu && this._menu.hasAttribute('data-open')) {
        this._closeMenu();
        e.preventDefault();
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const key = e.key;
      let handled = true;
      if (key === 'ArrowRight' || key === 'PageDown' || key === ' ' || key === 'Spacebar') {
        this._advance(1, 'keyboard');
      } else if (key === 'ArrowLeft' || key === 'PageUp') {
        this._advance(-1, 'keyboard');
      } else if (key === 'Home') {
        this._go(0, 'keyboard');
      } else if (key === 'End') {
        this._go(this._slides.length - 1, 'keyboard');
      } else if (key === 'r' || key === 'R') {
        this._go(0, 'keyboard');
      } else if (/^[0-9]$/.test(key)) {
        // 1..9 jump to that slide; 0 jumps to 10.
        const n = key === '0' ? 9 : parseInt(key, 10) - 1;
        if (n < this._slides.length) this._go(n, 'keyboard');
      } else {
        handled = false;
      }
      if (handled) {
        e.preventDefault();
        this._flashOverlay();
      }
    }
    _go(i, reason = 'api') {
      if (!this._slides.length) return;
      const clamped = Math.max(0, Math.min(this._slides.length - 1, i));
      if (clamped === this._index) {
        this._flashOverlay();
        return;
      }
      this._index = clamped;
      this._applyIndex({
        showOverlay: true,
        broadcast: true,
        reason
      });
    }

    /** Step forward/back skipping any slide marked data-deck-skip. Falls
     *  back to _go's clamp-at-ends behaviour (flash overlay) when there's
     *  nothing further in that direction. */
    _advance(dir, reason) {
      if (!this._slides.length) return;
      let i = this._index + dir;
      while (i >= 0 && i < this._slides.length && this._slides[i].hasAttribute('data-deck-skip')) {
        i += dir;
      }
      if (i < 0 || i >= this._slides.length) {
        this._flashOverlay();
        return;
      }
      this._go(i, reason);
    }

    // ── Thumbnail rail ────────────────────────────────────────────────────
    //
    // Thumbs are keyed by slide element and reused across _renderRail()
    // calls, so a reorder/delete is an O(changed) DOM shuffle instead of an
    // O(N) teardown-and-re-clone. Each thumb starts as a lightweight shell
    // (num + empty frame); the clone is materialized lazily by an
    // IntersectionObserver when the frame scrolls into (or near) view, so
    // only visible-ish slides pay the clone + image-decode cost.

    _renderRail() {
      if (!this._rail || !this._railEnabled) {
        this._thumbs = [];
        return;
      }
      // FLIP: record each *materialized* thumb's top before the reconcile.
      // Off-screen (non-materialized) thumbs don't need the animation and
      // skipping their getBoundingClientRect saves a forced layout per
      // off-screen thumb on large decks.
      const prevTops = new Map();
      (this._thumbs || []).forEach(({
        thumb,
        slide,
        host
      }) => {
        if (host) prevTops.set(slide, thumb.getBoundingClientRect().top);
      });
      const st = this._rail.scrollTop;

      // Reconcile: reuse thumbs that already exist for a slide, create
      // shells for new slides, drop thumbs for removed slides.
      const bySlide = new Map();
      (this._thumbs || []).forEach(t => bySlide.set(t.slide, t));
      const next = [];
      this._slides.forEach(slide => {
        let t = bySlide.get(slide);
        if (t) bySlide.delete(slide);else t = this._makeThumb(slide);
        next.push(t);
      });
      // Orphans — slides removed since last render.
      bySlide.forEach(t => {
        if (this._railObserver) this._railObserver.unobserve(t.frame);
        t.thumb.remove();
      });
      // Put thumbs into document order to match _slides. insertBefore on
      // an already-correctly-placed node is a no-op, so this is cheap
      // when nothing moved.
      next.forEach((t, i) => {
        const want = t.thumb;
        const at = this._rail.children[i];
        if (at !== want) this._rail.insertBefore(want, at || null);
        t.i = i;
        t.num.textContent = String(i + 1);
        if (t.slide.hasAttribute('data-deck-skip')) t.thumb.setAttribute('data-skip', '');else t.thumb.removeAttribute('data-skip');
      });
      this._thumbs = next;
      this._rail.scrollTop = st;
      if (prevTops.size) {
        const moved = [];
        this._thumbs.forEach(({
          thumb,
          slide
        }) => {
          const old = prevTops.get(slide);
          if (old == null) return;
          const dy = old - thumb.getBoundingClientRect().top;
          if (Math.abs(dy) < 1) return;
          thumb.style.transition = 'none';
          thumb.style.transform = `translateY(${dy}px)`;
          moved.push(thumb);
        });
        if (moved.length) {
          // Commit the inverted positions before flipping the transition
          // on — otherwise the browser coalesces both style writes and
          // nothing animates.
          void this._rail.offsetHeight;
          moved.forEach(t => {
            t.style.transition = 'transform 180ms cubic-bezier(.2,.7,.3,1)';
            t.style.transform = '';
          });
          setTimeout(() => moved.forEach(t => {
            t.style.transition = '';
          }), 220);
        }
      }
      requestAnimationFrame(() => this._scaleThumbs());
      this._syncRail(false);
    }

    /** Create a lightweight thumb shell for one slide. The clone is
     *  materialized later by the IntersectionObserver. Event handlers
     *  look up the thumb's *current* index (via _thumbs.indexOf) so the
     *  same element can be reused across reorders. */
    _makeThumb(slide) {
      const thumb = document.createElement('div');
      thumb.className = 'thumb';
      thumb.tabIndex = 0;
      const num = document.createElement('div');
      num.className = 'num';
      const frame = document.createElement('div');
      frame.className = 'frame';
      thumb.append(num, frame);
      const entry = {
        thumb,
        num,
        frame,
        slide,
        clone: null,
        host: null,
        i: -1
      };
      // entry.i is refreshed on every _renderRail reconcile pass, so
      // handlers read the thumb's current position without an O(N) scan.
      const idx = () => entry.i;
      thumb.addEventListener('click', () => this._go(idx(), 'click'));
      // ↑/↓ step through the rail when a thumb has focus. _go clamps at the
      // ends and _applyIndex→_syncRail scrolls the new current thumb into
      // view; we move focus to it (preventScroll — _syncRail already
      // scrolled) so a held key walks the whole list. stopPropagation keeps
      // this out of the window-level _onKey nav handler.
      thumb.addEventListener('keydown', e => {
        if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
        if (e.metaKey || e.ctrlKey || e.altKey) return;
        e.preventDefault();
        e.stopPropagation();
        this._go(idx() + (e.key === 'ArrowDown' ? 1 : -1), 'keyboard');
        const cur = this._thumbs && this._thumbs[this._index];
        if (cur) cur.thumb.focus({
          preventScroll: true
        });
      });
      thumb.addEventListener('contextmenu', e => {
        e.preventDefault();
        this._openMenu(idx(), e.clientX, e.clientY);
      });
      thumb.draggable = true;
      thumb.addEventListener('dragstart', e => {
        this._dragFrom = idx();
        thumb.setAttribute('data-dragging', '');
        e.dataTransfer.effectAllowed = 'move';
        try {
          e.dataTransfer.setData('text/plain', String(this._dragFrom));
        } catch (err) {}
      });
      thumb.addEventListener('dragend', () => {
        thumb.removeAttribute('data-dragging');
        this._clearDrop();
        this._dragFrom = null;
      });
      thumb.addEventListener('dragover', e => {
        if (this._dragFrom == null) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        const r = thumb.getBoundingClientRect();
        this._setDrop(idx(), e.clientY < r.top + r.height / 2 ? 'before' : 'after');
      });
      thumb.addEventListener('drop', e => {
        if (this._dragFrom == null) return;
        e.preventDefault();
        const i = idx();
        const r = thumb.getBoundingClientRect();
        let to = e.clientY >= r.top + r.height / 2 ? i + 1 : i;
        if (this._dragFrom < to) to--;
        const from = this._dragFrom;
        this._clearDrop();
        this._dragFrom = null;
        if (to !== from) this._moveSlide(from, to);
      });
      if (this._railObserver) this._railObserver.observe(frame);
      frame.__deckThumb = entry;
      return entry;
    }

    /** Lazily build the clone for a thumb that has scrolled into view. */
    _materialize(entry) {
      if (entry.host) return;
      const dw = this.designWidth,
        dh = this.designHeight;
      let clone = entry.slide.cloneNode(true);
      clone.removeAttribute('id');
      clone.removeAttribute('data-deck-active');
      clone.querySelectorAll('[id]').forEach(el => el.removeAttribute('id'));
      // Neuter heavy media; replace <video> with its poster so the box
      // keeps a visual. <iframe>/<audio> become empty placeholders.
      clone.querySelectorAll('iframe, audio, object, embed').forEach(el => {
        el.removeAttribute('src');
        el.removeAttribute('srcdoc');
        el.removeAttribute('data');
        el.innerHTML = '';
      });
      clone.querySelectorAll('video').forEach(el => {
        if (!el.poster) {
          el.removeAttribute('src');
          el.innerHTML = '';
          return;
        }
        const img = document.createElement('img');
        img.src = el.poster;
        img.alt = '';
        img.style.cssText = el.style.cssText + ';object-fit:cover;width:100%;height:100%;';
        img.className = el.className;
        el.replaceWith(img);
      });
      // Images: defer decode and let the browser pick the smallest
      // srcset candidate for the ~140px thumb. Same-URL clones reuse the
      // slide's decoded bitmap (URL-keyed cache), so the remaining cost
      // is paint/composite — lazy+async keeps that off the main thread.
      clone.querySelectorAll('img').forEach(el => {
        el.loading = 'lazy';
        el.decoding = 'async';
        if (el.srcset) el.sizes = (this._railPx || 188) + 'px';
      });
      // Custom elements inside the slide would have their
      // connectedCallback fire when the clone is appended. Replace them
      // with inert boxes so a component-heavy deck doesn't run N copies
      // of each component's mount logic in the rail. Children are
      // preserved so layout-wrapper elements (<my-column><h2>…</h2>)
      // still show their authored content; the querySelectorAll NodeList
      // is static, so nested custom elements in the moved subtree are
      // still visited on later iterations.
      const neuter = el => {
        const box = document.createElement('div');
        box.style.cssText = (el.getAttribute('style') || '') + ';background:rgba(0,0,0,0.06);border:1px dashed rgba(0,0,0,0.15);';
        box.className = el.className;
        // Preserve theming/i18n hooks so [data-*] / :lang() / [dir]
        // descendant selectors still match the neutered root.
        for (const a of el.attributes) {
          const n = a.name;
          if (n.startsWith('data-') || n.startsWith('aria-') || n === 'lang' || n === 'dir' || n === 'role' || n === 'title') {
            box.setAttribute(n, a.value);
          }
        }
        while (el.firstChild) box.appendChild(el.firstChild);
        return box;
      };
      // querySelectorAll('*') returns descendants only — a custom-element
      // slide root (<my-slide>…</my-slide>) would slip through and upgrade
      // on append. Swap the root first.
      if (clone.tagName.includes('-')) clone = neuter(clone);
      clone.querySelectorAll('*').forEach(el => {
        if (el.tagName.includes('-')) el.replaceWith(neuter(el));
      });
      clone.style.cssText += ';position:absolute;top:0;left:0;transform-origin:0 0;' + 'pointer-events:none;width:' + dw + 'px;height:' + dh + 'px;' + 'box-sizing:border-box;overflow:hidden;visibility:visible;opacity:1;';
      const host = document.createElement('div');
      host.style.cssText = 'position:absolute;inset:0;';
      this._syncThumbHostAttrs(host);
      const sr = host.attachShadow({
        mode: 'open'
      });
      if (this._adoptedSheet) sr.adoptedStyleSheets = [this._adoptedSheet];else {
        const st = document.createElement('style');
        st.textContent = this._authorCss || '';
        sr.appendChild(st);
      }
      sr.appendChild(clone);
      entry.frame.appendChild(host);
      entry.host = host;
      entry.clone = clone;
      if (this._thumbScale) clone.style.transform = 'scale(' + this._thumbScale + ')';
      // Once materialized the IO callback is a no-op early-return —
      // unobserve so scroll doesn't keep firing it.
      if (this._railObserver) this._railObserver.unobserve(entry.frame);
    }

    /** Re-clone a single thumb (live-update path). No-op if the thumb
     *  hasn't been materialized yet — it'll pick up current content when
     *  it scrolls into view. */
    _refreshThumb(slide) {
      const entry = (this._thumbs || []).find(t => t.slide === slide);
      if (!entry || !entry.host) return;
      entry.host.remove();
      entry.host = entry.clone = null;
      this._materialize(entry);
    }
    _scaleThumbs() {
      if (!this._thumbs || !this._thumbs.length) return;
      // Every frame is the same width; if it reads 0 the rail is
      // display:none (noscale / no-rail / presenting / print) — leave the
      // clones as-is and re-run when the rail is revealed.
      const fw = this._thumbs[0].frame.offsetWidth;
      if (!fw) return;
      this._thumbScale = fw / this.designWidth;
      this._thumbs.forEach(({
        clone
      }) => {
        if (clone) clone.style.transform = 'scale(' + this._thumbScale + ')';
      });
    }
    _setDrop(i, where) {
      // dragover fires at pointer-event rate; touch only the previous
      // and new target rather than sweeping all N thumbs.
      const t = this._thumbs && this._thumbs[i];
      if (this._dropOn && this._dropOn !== t) {
        this._dropOn.thumb.removeAttribute('data-drop');
      }
      if (t) t.thumb.setAttribute('data-drop', where);
      this._dropOn = t || null;
    }
    _clearDrop() {
      if (this._dropOn) this._dropOn.thumb.removeAttribute('data-drop');
      this._dropOn = null;
    }
    _syncRail(follow) {
      if (!this._thumbs) return;
      this._thumbs.forEach(({
        thumb
      }, i) => {
        if (i === this._index) {
          thumb.setAttribute('data-current', '');
          if (follow && typeof thumb.scrollIntoView === 'function') {
            thumb.scrollIntoView({
              block: 'nearest'
            });
          }
        } else {
          thumb.removeAttribute('data-current');
        }
      });
    }
    _openMenu(i, x, y) {
      if (!this._menu) return;
      this._menuIndex = i;
      const slide = this._slides[i];
      const skip = slide && slide.hasAttribute('data-deck-skip');
      this._menu.querySelector('[data-act="skip"]').textContent = skip ? 'Unskip slide' : 'Skip slide';
      this._menu.querySelector('[data-act="up"]').disabled = i <= 0;
      this._menu.querySelector('[data-act="down"]').disabled = i >= this._slides.length - 1;
      this._menu.querySelector('[data-act="delete"]').disabled = this._slides.length <= 1;
      // Place, then clamp to viewport after it's measurable.
      this._menu.style.left = x + 'px';
      this._menu.style.top = y + 'px';
      this._menu.setAttribute('data-open', '');
      const r = this._menu.getBoundingClientRect();
      const nx = Math.min(x, window.innerWidth - r.width - 4);
      const ny = Math.min(y, window.innerHeight - r.height - 4);
      this._menu.style.left = Math.max(4, nx) + 'px';
      this._menu.style.top = Math.max(4, ny) + 'px';
    }
    _closeMenu() {
      if (this._menu) this._menu.removeAttribute('data-open');
      this._menuIndex = -1;
    }
    _openConfirm(i) {
      if (!this._confirm) return;
      this._confirmIndex = i;
      this._confirm.querySelector('.title').textContent = 'Delete slide ' + (i + 1) + '?';
      this._confirm.setAttribute('data-open', '');
      const btn = this._confirm.querySelector('.danger');
      if (btn && btn.focus) btn.focus();
    }
    _closeConfirm() {
      if (this._confirm) this._confirm.removeAttribute('data-open');
      this._confirmIndex = -1;
    }
    _emitDeckChange(detail) {
      this.dispatchEvent(new CustomEvent('deckchange', {
        detail,
        bubbles: true,
        composed: true
      }));
    }
    _deleteSlide(i) {
      const slide = this._slides[i];
      if (!slide || this._slides.length <= 1) return;
      const wasCurrent = i === this._index;
      if (i < this._index || wasCurrent && i === this._slides.length - 1) this._index--;
      this._squelchSlotChange = true;
      slide.remove();
      this._emitDeckChange({
        action: 'delete',
        from: i,
        slide
      });
      this._collectSlides();
      this._applyIndex({
        showOverlay: true,
        broadcast: true,
        reason: 'mutation'
      });
    }
    _duplicateSlide(i) {
      const slide = this._slides[i];
      if (!slide) return;
      const copy = slide.cloneNode(true);
      // Strip ids so the document stays valid (no duplicate-id collisions
      // with the original). Same treatment _materialize gives rail clones.
      copy.removeAttribute('id');
      copy.querySelectorAll('[id]').forEach(el => el.removeAttribute('id'));
      // Insert after the original and make the copy active so it's the one
      // on screen. _collectSlides re-derives data-screen-label / data-deck-*
      // attrs, so the cloned values are overwritten.
      this._index = i + 1;
      this._squelchSlotChange = true;
      this.insertBefore(copy, slide.nextSibling);
      this._emitDeckChange({
        action: 'duplicate',
        from: i,
        to: i + 1,
        slide: copy
      });
      this._collectSlides();
      this._applyIndex({
        showOverlay: true,
        broadcast: true,
        reason: 'mutation'
      });
    }
    _toggleSkip(i) {
      const slide = this._slides[i];
      if (!slide) return;
      const on = !slide.hasAttribute('data-deck-skip');
      if (on) slide.setAttribute('data-deck-skip', '');else slide.removeAttribute('data-deck-skip');
      if (this._thumbs && this._thumbs[i]) {
        if (on) this._thumbs[i].thumb.setAttribute('data-skip', '');else this._thumbs[i].thumb.removeAttribute('data-skip');
      }
      this._markLastVisible();
      this._emitDeckChange({
        action: on ? 'skip' : 'unskip',
        from: i,
        slide
      });
      // Re-broadcast so the presenter popup's prev/next thumbnails re-pick
      // the nearest non-skipped slide without waiting for a nav event.
      try {
        window.postMessage({
          slideIndexChanged: this._index,
          deckTotal: this._slides.length,
          deckSkipped: this._skippedIndices()
        }, '*');
      } catch (e) {}
    }
    _skippedIndices() {
      const out = [];
      for (let i = 0; i < this._slides.length; i++) {
        if (this._slides[i].hasAttribute('data-deck-skip')) out.push(i);
      }
      return out;
    }
    _moveSlide(i, j) {
      if (j < 0 || j >= this._slides.length || j === i) return;
      const slide = this._slides[i];
      const ref = j < i ? this._slides[j] : this._slides[j].nextSibling;
      // Track the active slide across the reorder so the same content
      // stays on screen.
      const cur = this._index;
      if (cur === i) this._index = j;else if (i < cur && j >= cur) this._index = cur - 1;else if (i > cur && j <= cur) this._index = cur + 1;
      this._squelchSlotChange = true;
      this.insertBefore(slide, ref);
      this._emitDeckChange({
        action: 'move',
        from: i,
        to: j,
        slide
      });
      this._collectSlides();
      this._applyIndex({
        showOverlay: false,
        broadcast: true,
        reason: 'mutation'
      });
    }

    // Public API ------------------------------------------------------------

    /** Current slide index (0-based). */
    get index() {
      return this._index;
    }
    /** Total slide count. */
    get length() {
      return this._slides.length;
    }
    /** Programmatically navigate. */
    goTo(i) {
      this._go(i, 'api');
    }
    next() {
      this._advance(1, 'api');
    }
    prev() {
      this._advance(-1, 'api');
    }
    reset() {
      this._go(0, 'api');
    }
  }
  if (!customElements.get('deck-stage')) {
    customElements.define('deck-stage', DeckStage);
  }
})();
})(); } catch (e) { __ds_ns.__errors.push({ path: "decks/pdmt-boards-systems/deck-stage.js", error: String((e && e.message) || e) }); }

// slides/deck-stage.js
try { (() => {
// @ds-adherence-ignore -- omelette starter scaffold (raw elements/hex/px by design)
/* BEGIN USAGE */
/**
 * <deck-stage> — reusable web component for HTML decks.
 *
 * Handles:
 *  (a) speaker notes — reads <script type="application/json" id="speaker-notes">
 *      and posts {slideIndexChanged: N} to the parent window on nav.
 *  (b) keyboard navigation — ←/→, PgUp/PgDn, Space, Home/End, number keys.
 *      On touch devices, tapping the left/right half of the stage goes
 *      prev/next — taps on links, buttons and other interactive slide
 *      content are left alone.
 *  (c) press R to reset to slide 0 (with a tasteful keyboard hint).
 *  (d) bottom-center overlay showing slide count + hints, fades out on idle.
 *  (e) auto-scaling — inner canvas is a fixed design size (default 1920×1080)
 *      scaled with `transform: scale()` to fit the viewport, letterboxed.
 *      Set the `noscale` attribute to render at authored size (1:1) — the
 *      PPTX exporter sets this so its DOM capture sees unscaled geometry.
 *  (f) print — `@media print` lays every slide out as its own page at the
 *      design size, so the browser's Print → Save as PDF produces a clean
 *      one-page-per-slide PDF with no extra setup.
 *  (g) thumbnail rail — resizable left-hand column of per-slide thumbnails
 *      (static clones). Click to navigate; ↑/↓ with a thumbnail focused to
 *      step between slides; drag to reorder; right-click for
 *      Skip / Move up / Move down / Duplicate / Delete (Delete opens a
 *      Cancel/Delete confirm dialog). Drag the rail's right edge to resize;
 *      width persists to
 *      localStorage. Skipped slides carry `data-deck-skip`, are dimmed in
 *      the rail, omitted from prev/next navigation, and hidden at print.
 *      The rail is suppressed in presenting mode, in the host's Preview
 *      mode (ViewerMode='none'), on `noscale`, on narrow viewports
 *      (≤640px), and via the `no-rail` attribute. Rail mutations dispatch
 *      a `deckchange`
 *      CustomEvent on the element: detail = {action, from, to, slide}.
 *
 * Slides are HIDDEN, not unmounted. Non-active slides stay in the DOM with
 * `visibility: hidden` + `opacity: 0`, so their state (videos, iframes,
 * form inputs, React trees) is preserved across navigation.
 *
 * Lifecycle event — the component dispatches a `slidechange` CustomEvent on
 * itself whenever the active slide changes (including the initial mount).
 * The event bubbles and composes out of shadow DOM, so you can listen on
 * the <deck-stage> element or on document:
 *
 *   document.querySelector('deck-stage').addEventListener('slidechange', (e) => {
 *     e.detail.index         // new 0-based index
 *     e.detail.previousIndex // previous index, or -1 on init
 *     e.detail.total         // total slide count
 *     e.detail.slide         // the new active slide element
 *     e.detail.previousSlide // the prior slide element, or null on init
 *     e.detail.reason        // 'init' | 'keyboard' | 'click' | 'tap' | 'api'
 *   });
 *
 * Persistence: none at the deck level. The host app keeps the current slide
 * in its own URL (?slide=) and re-delivers it via location.hash on load, so a
 * bare load with no hash always starts at slide 1.
 *
 * Usage:
 *   <style>deck-stage:not(:defined){visibility:hidden}</style>
 *   <deck-stage width="1920" height="1080">
 *     <section data-label="Title">...</section>
 *     <section data-label="Agenda">...</section>
 *   </deck-stage>
 *   <script src="deck-stage.js"></script>
 *
 * The :not(:defined) rule prevents a flash of the first slide at its
 * authored styles before this script runs and attaches the shadow root.
 *
 * Slides are the direct element children of <deck-stage>. Each slide is
 * automatically tagged with:
 *   - data-screen-label="NN Label"   (1-indexed, for comment flow)
 *   - data-om-validate="no_overflowing_text,no_overlapping_text,slide_sized_text"
 *
 * Speaker notes stay in sync because the component posts {slideIndexChanged: N}
 * to the parent — just include the #speaker-notes script tag if asked for notes.
 *
 * Authoring guidance:
 *   - Write slide bodies as static HTML inside <deck-stage>, with sizing via
 *     CSS custom properties in a <style> block rather than JS constants.
 *     Static slide markup is what lets the user click a heading in edit mode
 *     and retype it directly; a slide rendered through <script type="text/babel">,
 *     React, or a loop over a JS array has to round-trip every tweak through a
 *     chat message instead. Reach for script-generated slides only when the
 *     content genuinely needs interactive behaviour static HTML can't express.
 *   - Do NOT set position/inset/width/height on the slide <section> elements —
 *     the component absolutely positions every slotted child for you.
 *   - Entrance animations: make the visible end-state the base style and
 *     animate *from* hidden, so print and reduced-motion show content.
 *     Gate the animation on [data-deck-active] and the motion query, e.g.
 *     `@media (prefers-reduced-motion:no-preference){ [data-deck-active] .x{animation:fade-in .5s both} }`.
 *     Avoid infinite decorative loops on slide content.
 */
/* END USAGE */

(() => {
  const DESIGN_W_DEFAULT = 1920;
  const DESIGN_H_DEFAULT = 1080;
  const OVERLAY_HIDE_MS = 1800;
  const VALIDATE_ATTR = 'no_overflowing_text,no_overlapping_text,slide_sized_text';
  const FINE_POINTER_MQ = matchMedia('(hover: hover) and (pointer: fine)');
  const NARROW_MQ = matchMedia('(max-width: 640px)');
  // Slide-authored controls that should keep a tap instead of it navigating.
  const INTERACTIVE_SEL = 'a[href], button, input, select, textarea, summary, label, video[controls], audio[controls], [role="button"], [onclick], [tabindex]:not([tabindex^="-"]), [contenteditable]:not([contenteditable="false" i])';
  const pad2 = n => String(n).padStart(2, '0');

  // Label precedence: data-label → data-screen-label (number stripped) → first heading → "Slide".
  const getSlideLabel = el => {
    const explicit = el.getAttribute('data-label');
    if (explicit) return explicit;
    const existing = el.getAttribute('data-screen-label');
    if (existing) return existing.replace(/^\s*\d+\s*/, '').trim() || existing;
    const h = el.querySelector('h1, h2, h3, [data-title]');
    const t = h && (h.textContent || '').trim().slice(0, 40);
    if (t) return t;
    return 'Slide';
  };
  const stylesheet = `
    :host {
      position: fixed;
      inset: 0;
      display: block;
      background: #000;
      color: #fff;
      font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif;
      overflow: hidden;
      -webkit-tap-highlight-color: transparent;
    }
    /* connectedCallback holds this until document.fonts.ready (capped 2s) so
     * the first visible paint has the deck's real typography + final rail
     * layout. opacity (not visibility) so the active slide can't un-hide
     * itself via the ::slotted([data-deck-active]) visibility:visible rule.
     * Only the stage/rail hide — the black :host background stays, so the
     * iframe doesn't flash the page's default white. */
    :host([data-fonts-pending]) .stage,
    :host([data-fonts-pending]) .rail { opacity: 0; pointer-events: none; }

    .stage {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .canvas {
      position: relative;
      transform-origin: center center;
      flex-shrink: 0;
      background: #fff;
      will-change: transform;
    }

    /* Slides live in light DOM (via <slot>) so authored CSS still applies.
       We absolutely position each slotted child to stack them. */
    ::slotted(*) {
      position: absolute !important;
      inset: 0 !important;
      width: 100% !important;
      height: 100% !important;
      box-sizing: border-box !important;
      overflow: hidden;
      opacity: 0;
      pointer-events: none;
      visibility: hidden;
    }
    ::slotted([data-deck-active]) {
      opacity: 1;
      pointer-events: auto;
      visibility: visible;
    }

    .overlay {
      position: fixed;
      left: 50%;
      bottom: 22px;
      transform: translate(-50%, 6px) scale(0.92);
      filter: blur(6px);
      display: flex;
      align-items: center;
      gap: 4px;
      padding: 4px;
      background: #000;
      color: #fff;
      border-radius: 999px;
      font-size: 12px;
      font-feature-settings: "tnum" 1;
      letter-spacing: 0.01em;
      opacity: 0;
      pointer-events: none;
      transition: opacity 260ms ease, transform 260ms cubic-bezier(.2,.8,.2,1), filter 260ms ease;
      transform-origin: center bottom;
      z-index: 2147483000;
      user-select: none;
    }
    .overlay[data-visible] {
      opacity: 1;
      pointer-events: auto;
      transform: translate(-50%, 0) scale(1);
      filter: blur(0);
    }

    .btn {
      appearance: none;
      -webkit-appearance: none;
      background: transparent;
      border: 0;
      margin: 0;
      padding: 0;
      color: inherit;
      font: inherit;
      cursor: default;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      height: 28px;
      min-width: 28px;
      border-radius: 999px;
      color: rgba(255,255,255,0.72);
      transition: background 140ms ease, color 140ms ease;
      -webkit-tap-highlight-color: transparent;
    }
    .btn:hover { background: rgba(255,255,255,0.12); color: #fff; }
    .btn:active { background: rgba(255,255,255,0.18); }
    .btn:focus { outline: none; }
    .btn:focus-visible { outline: none; }
    .btn::-moz-focus-inner { border: 0; }
    .btn svg { width: 14px; height: 14px; display: block; }
    .btn.reset {
      font-size: 11px;
      font-weight: 500;
      letter-spacing: 0.02em;
      padding: 0 10px 0 12px;
      gap: 6px;
      color: rgba(255,255,255,0.72);
    }
    .btn.reset .kbd {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 16px;
      height: 16px;
      padding: 0 4px;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      font-size: 10px;
      line-height: 1;
      color: rgba(255,255,255,0.88);
      background: rgba(255,255,255,0.12);
      border-radius: 4px;
    }

    .count {
      font-variant-numeric: tabular-nums;
      color: #fff;
      font-weight: 500;
      padding: 0 8px;
      min-width: 42px;
      text-align: center;
      font-size: 12px;
    }
    .count .sep { color: rgba(255,255,255,0.45); margin: 0 3px; font-weight: 400; }
    .count .total { color: rgba(255,255,255,0.55); }

    .divider {
      width: 1px;
      height: 14px;
      background: rgba(255,255,255,0.18);
      margin: 0 2px;
    }

    /* ── Thumbnail rail ──────────────────────────────────────────────────
       Fixed column on the left; each thumbnail is a static deep-clone of
       the light-DOM slide scaled into a 16:9 (or design-aspect) frame. The
       stage re-fits around it (see _fit); hidden during present / noscale
       / print so capture geometry and fullscreen output are unchanged. */
    .rail {
      position: fixed;
      left: 0;
      top: 0;
      bottom: 0;
      width: var(--deck-rail-w, 188px);
      background: #141414;
      border-right: 1px solid rgba(255,255,255,0.08);
      overflow-y: auto;
      overflow-x: hidden;
      padding: 12px 10px;
      box-sizing: border-box;
      display: flex;
      flex-direction: column;
      gap: 12px;
      z-index: 2147482500;
      scrollbar-width: thin;
      scrollbar-color: rgba(255,255,255,0.18) transparent;
    }
    .rail::-webkit-scrollbar { width: 8px; }
    .rail::-webkit-scrollbar-track { background: transparent; margin: 2px; }
    .rail::-webkit-scrollbar-thumb {
      background: rgba(255,255,255,0.18);
      border-radius: 4px;
      border: 2px solid transparent;
      background-clip: content-box;
    }
    .rail::-webkit-scrollbar-thumb:hover {
      background: rgba(255,255,255,0.28);
      border: 2px solid transparent;
      background-clip: content-box;
    }
    :host([no-rail]) .rail,
    :host([noscale]) .rail { display: none; }
    .rail[data-presenting] { display: none; }
    @media (max-width: 640px) {
      .rail, .rail-resize { display: none; }
    }
    /* User-driven show/hide (the TweaksPanel toggle) slides instead of
       popping. Transitions are gated on :host([data-rail-anim]) — set only
       for the 200ms around the toggle — so window-resize and rail-width
       drag (which also call _fit) don't lag behind the cursor. */
    .rail[data-user-hidden] { transform: translateX(-100%); }
    :host([data-rail-anim]) .rail { transition: transform 200ms cubic-bezier(.3,.7,.4,1); }
    :host([data-rail-anim]) .stage { transition: left 200ms cubic-bezier(.3,.7,.4,1); }
    :host([data-rail-anim]) .canvas { transition: transform 200ms cubic-bezier(.3,.7,.4,1); }
    /* transition shorthand replaces rather than merges — repeat the base
       .overlay opacity/transform/filter transitions so visibility changes
       during the 200ms toggle window still fade instead of popping. */
    :host([data-rail-anim]) .overlay {
      transition: margin-left 200ms cubic-bezier(.3,.7,.4,1),
                  opacity 260ms ease,
                  transform 260ms cubic-bezier(.2,.8,.2,1),
                  filter 260ms ease;
    }

    .thumb {
      position: relative;
      display: flex;
      align-items: flex-start;
      gap: 8px;
      cursor: pointer;
      user-select: none;
    }
    .thumb .num {
      width: 16px;
      flex-shrink: 0;
      font-size: 11px;
      font-weight: 500;
      text-align: right;
      color: rgba(255,255,255,0.55);
      padding-top: 2px;
      font-variant-numeric: tabular-nums;
    }
    .thumb .frame {
      position: relative;
      flex: 1;
      min-width: 0;
      aspect-ratio: var(--deck-aspect);
      background: #fff;
      border-radius: 4px;
      outline: 2px solid transparent;
      outline-offset: 0;
      overflow: hidden;
      transition: outline-color 120ms ease;
    }
    .thumb:hover .frame { outline-color: rgba(255,255,255,0.25); }
    .thumb { outline: none; }
    .thumb:focus-visible .frame { outline-color: rgba(255,255,255,0.5); }
    .thumb[data-current] .num { color: #fff; }
    .thumb[data-current] .frame { outline-color: #D97757; }
    .thumb[data-dragging] { opacity: 0.35; }
    .thumb::before {
      content: '';
      position: absolute;
      left: 24px;
      right: 0;
      height: 3px;
      border-radius: 2px;
      background: #D97757;
      opacity: 0;
      pointer-events: none;
    }
    .thumb[data-drop="before"]::before { top: -8px; opacity: 1; }
    .thumb[data-drop="after"]::before { bottom: -8px; opacity: 1; }
    .thumb[data-skip] .frame { opacity: 0.35; }
    .thumb[data-skip] .frame::after {
      content: 'Skipped';
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(0,0,0,0.45);
      color: #fff;
      font-size: 10px;
      font-weight: 500;
      letter-spacing: 0.04em;
    }

    .ctxmenu {
      position: fixed;
      min-width: 150px;
      padding: 4px;
      background: #242424;
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 7px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.45);
      z-index: 2147483100;
      display: none;
      font-size: 12px;
    }
    .ctxmenu[data-open] { display: block; }
    .ctxmenu button {
      display: block;
      width: 100%;
      appearance: none;
      border: 0;
      background: transparent;
      color: #e8e8e8;
      font: inherit;
      text-align: left;
      padding: 6px 10px;
      border-radius: 4px;
      cursor: pointer;
    }
    .ctxmenu button:hover:not(:disabled) { background: rgba(255,255,255,0.08); }
    .ctxmenu button:disabled { opacity: 0.35; cursor: default; }
    .ctxmenu hr {
      border: 0;
      border-top: 1px solid rgba(255,255,255,0.1);
      margin: 4px 2px;
    }

    .rail-resize {
      position: fixed;
      left: calc(var(--deck-rail-w, 188px) - 3px);
      top: 0;
      bottom: 0;
      width: 6px;
      cursor: col-resize;
      z-index: 2147482600;
      touch-action: none;
    }
    .rail-resize:hover,
    .rail-resize[data-dragging] { background: rgba(255,255,255,0.12); }
    :host([no-rail]) .rail-resize,
    :host([noscale]) .rail-resize,
    .rail[data-presenting] + .rail-resize,
    .rail[data-user-hidden] + .rail-resize { display: none; }

    /* Delete-confirm popup — matches the SPA's ConfirmDialog layout
       (title + message body, depressed footer with Cancel / Delete). */
    .confirm-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.45);
      z-index: 2147483200;
      display: none;
      align-items: center;
      justify-content: center;
    }
    .confirm-backdrop[data-open] { display: flex; }
    .confirm {
      width: 320px;
      max-width: calc(100vw - 32px);
      background: #2a2a2a;
      color: #e8e8e8;
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 12px;
      box-shadow: 0 12px 32px rgba(0,0,0,0.5);
      overflow: hidden;
      font-family: inherit;
      animation: deck-confirm-in 0.18s ease;
    }
    @keyframes deck-confirm-in {
      from { opacity: 0; transform: scale(0.96); }
      to { opacity: 1; transform: scale(1); }
    }
    .confirm .body { padding: 20px 20px 16px; }
    .confirm .title { font-size: 14px; font-weight: 600; margin-bottom: 4px; }
    .confirm .msg { font-size: 13px; line-height: 1.5; color: rgba(255,255,255,0.65); }
    .confirm .footer {
      padding: 14px 20px;
      background: #1f1f1f;
      border-top: 1px solid rgba(255,255,255,0.08);
      display: flex;
      justify-content: flex-end;
      gap: 8px;
    }
    .confirm button {
      appearance: none;
      font: inherit;
      font-size: 13px;
      font-weight: 500;
      padding: 8px 16px;
      border-radius: 8px;
      cursor: pointer;
    }
    .confirm .cancel {
      background: transparent;
      border: 0;
      color: rgba(255,255,255,0.8);
    }
    .confirm .cancel:hover { background: rgba(255,255,255,0.08); }
    .confirm .danger {
      background: #c96442;
      border: 1px solid rgba(0,0,0,0.15);
      color: #fff;
      box-shadow: 0 1px 3px rgba(166,50,68,0.3), 0 2px 6px rgba(166,50,68,0.18);
    }
    .confirm .danger:hover { background: #b5563a; }

    /* ── Print: one page per slide, no chrome ────────────────────────────
       The screen layout stacks every slide at inset:0 inside a scaled
       canvas; for print we want them in document flow at the authored
       design size so the browser paginates one slide per sheet. The
       @page size is set from the width/height attributes via the inline
       <style id="deck-stage-print-page"> that connectedCallback injects
       into <head> (the @page at-rule has no effect inside shadow DOM). */
    @media print {
      :host {
        position: static;
        inset: auto;
        background: none;
        overflow: visible;
        color: inherit;
      }
      .stage { position: static; display: block; }
      .canvas {
        transform: none !important;
        width: auto !important;
        height: auto !important;
        background: none;
        will-change: auto;
      }
      ::slotted(*) {
        position: relative !important;
        inset: auto !important;
        width: var(--deck-design-w) !important;
        height: var(--deck-design-h) !important;
        box-sizing: border-box !important;
        opacity: 1 !important;
        visibility: visible !important;
        pointer-events: auto;
        break-after: page;
        page-break-after: always;
        break-inside: avoid;
        overflow: hidden;
      }
      /* :last-child alone isn't enough once data-deck-skip hides the
         trailing slide(s) — the last *visible* slide still carries
         break-after:page and prints a blank sheet. _markLastVisible()
         maintains data-deck-last-visible on the last non-skipped slide. */
      ::slotted(*:last-child),
      ::slotted([data-deck-last-visible]) {
        break-after: auto;
        page-break-after: auto;
      }
      ::slotted([data-deck-skip]) { display: none !important; }
      .overlay, .rail, .rail-resize, .ctxmenu, .confirm-backdrop { display: none !important; }
    }
  `;
  class DeckStage extends HTMLElement {
    static get observedAttributes() {
      return ['width', 'height', 'noscale', 'no-rail'];
    }
    constructor() {
      super();
      this._root = this.attachShadow({
        mode: 'open'
      });
      this._index = 0;
      this._slides = [];
      this._notes = [];
      this._hideTimer = null;
      this._mouseIdleTimer = null;
      this._menuIndex = -1;
      this._onKey = this._onKey.bind(this);
      this._onResize = this._onResize.bind(this);
      this._onSlotChange = this._onSlotChange.bind(this);
      this._onMouseMove = this._onMouseMove.bind(this);
      this._onTap = this._onTap.bind(this);
      this._onMessage = this._onMessage.bind(this);
      // Capture-phase close so a click anywhere dismisses the menu, but
      // ignore clicks that land inside the menu itself — otherwise the
      // capture handler runs before the menu's own (bubble) handler and
      // clears _menuIndex out from under it.
      this._onDocClick = e => {
        if (this._menu && e.composedPath && e.composedPath().includes(this._menu)) return;
        this._closeMenu();
      };
    }
    get designWidth() {
      return parseInt(this.getAttribute('width'), 10) || DESIGN_W_DEFAULT;
    }
    get designHeight() {
      return parseInt(this.getAttribute('height'), 10) || DESIGN_H_DEFAULT;
    }
    connectedCallback() {
      // Presenter-view popup loads deckUrl?_snthumb=...#N for its prev/cur/
      // next thumbnails — the rail has no business rendering inside those
      // (wrong scale, and it offsets the stage so the thumb shows a gutter).
      if (/[?&]_snthumb=/.test(location.search)) this.setAttribute('no-rail', '');
      this._render();
      this._loadNotes();
      this._syncPrintPageRule();
      window.addEventListener('keydown', this._onKey);
      window.addEventListener('resize', this._onResize);
      window.addEventListener('mousemove', this._onMouseMove, {
        passive: true
      });
      window.addEventListener('message', this._onMessage);
      window.addEventListener('click', this._onDocClick, true);
      this.addEventListener('click', this._onTap);
      // Print lays every slide out as its own page, so [data-deck-active]-
      // gated entrance styles need the attribute on every slide (not just
      // the current one) or their content prints at the hidden base style.
      // The transient freeze style lands BEFORE the attributes so any
      // attribute-keyed transition fires at 0s (changing transition-
      // duration after a transition has started doesn't affect it).
      this._onBeforePrint = () => {
        if (this._freezeStyle) this._freezeStyle.remove();
        this._freezeStyle = document.createElement('style');
        this._freezeStyle.textContent = '*,*::before,*::after{transition-duration:0s !important}';
        document.head.appendChild(this._freezeStyle);
        this._slides.forEach(s => s.setAttribute('data-deck-active', ''));
      };
      this._onAfterPrint = () => {
        this._applyIndex({
          showOverlay: false,
          broadcast: false
        });
        if (this._freezeStyle) {
          this._freezeStyle.remove();
          this._freezeStyle = null;
        }
      };
      window.addEventListener('beforeprint', this._onBeforePrint);
      window.addEventListener('afterprint', this._onAfterPrint);
      // Initial collection + layout happens via slotchange, which fires on mount.
      this._enableRail();
      // Hold the stage hidden until webfonts are ready so the first visible
      // paint has the deck's real typography — the :not(:defined) guard in
      // the page HTML only covers custom-element upgrade, not font load.
      // Capped so a 404'd font URL can't blank the deck indefinitely.
      this.setAttribute('data-fonts-pending', '');
      const reveal = () => this.removeAttribute('data-fonts-pending');
      // rAF first: fonts.ready is a pre-resolved promise until layout has
      // resolved the slotted text's font-family and pushed a FontFace into
      // 'loading'. Reading it here in connectedCallback (parse-time) would
      // settle the race in a microtask before any font fetch starts.
      requestAnimationFrame(() => {
        Promise.race([document.fonts ? document.fonts.ready : Promise.resolve(), new Promise(r => setTimeout(r, 2000))]).then(reveal, reveal);
      });
    }
    _enableRail() {
      // Idempotent — older host builds still post __omelette_rail_enabled.
      // no-rail guard keeps the observers/stylesheet walk off the cheap path
      // for presenter-popup thumbnail iframes (up to 9 per view).
      if (this._railEnabled || this.hasAttribute('no-rail')) return;
      this._railEnabled = true;
      // Per-viewer preference — restored alongside rail width. Default on;
      // only a stored '0' (from the TweaksPanel toggle) hides it.
      this._railVisible = true;
      try {
        if (localStorage.getItem('deck-stage.railVisible') === '0') this._railVisible = false;
      } catch (e) {}
      // Live thumbnail updates: watch the light-DOM slides for content
      // edits and re-clone just the affected thumb(s), debounced. Ignore
      // the data-deck-* / data-screen-label / data-om-validate attributes
      // this component itself writes so nav and skip don't trigger
      // spurious refreshes.
      const OWN_ATTRS = /^data-(deck-|screen-label$|om-validate$)/;
      this._liveDirty = new Set();
      this._liveObserver = new MutationObserver(records => {
        for (const r of records) {
          if (r.type === 'attributes' && OWN_ATTRS.test(r.attributeName || '')) continue;
          let n = r.target;
          while (n && n.parentElement !== this) n = n.parentElement;
          if (n && this._slideSet && this._slideSet.has(n)) this._liveDirty.add(n);
        }
        if (this._liveDirty.size && !this._liveTimer) {
          this._liveTimer = setTimeout(() => {
            this._liveTimer = null;
            this._liveDirty.forEach(s => this._refreshThumb(s));
            this._liveDirty.clear();
          }, 200);
        }
      });
      this._liveObserver.observe(this, {
        subtree: true,
        childList: true,
        characterData: true,
        attributes: true
      });
      // Lazy thumbnail materialization — clone the slide only when its
      // frame scrolls into (or near) the rail viewport. rootMargin gives
      // ~4 thumbs of pre-load so fast scrolling doesn't flash blanks.
      this._railObserver = new IntersectionObserver(entries => {
        entries.forEach(e => {
          if (e.isIntersecting && e.target.__deckThumb) {
            this._materialize(e.target.__deckThumb);
          }
        });
      }, {
        root: this._rail,
        rootMargin: '400px 0px'
      });
      // Tweaks typically change CSS vars / attrs OUTSIDE <deck-stage>
      // (on <html>, <body>, a wrapper div, or a <style> tag), which
      // _liveObserver can't see. Re-snapshot author CSS (constructable
      // sheet is shared by reference, so one replaceSync updates every
      // thumb shadow root) and re-sync each thumb host's attrs + custom
      // properties. In-slide DOM mutations are _liveObserver's job.
      // Debounced so slider drags don't thrash.
      this._onTweakChange = () => {
        clearTimeout(this._tweakTimer);
        this._tweakTimer = setTimeout(() => {
          this._snapshotAuthorCss();
          // One getComputedStyle for the whole batch — each
          // getPropertyValue read below reuses the same computed style
          // as long as nothing invalidates layout between thumbs.
          const cs = getComputedStyle(this);
          (this._thumbs || []).forEach(t => {
            if (t.host) this._syncThumbHostAttrs(t.host, cs);
          });
        }, 120);
      };
      window.addEventListener('tweakchange', this._onTweakChange);
      this._snapshotAuthorCss();
      // Build the rail now that it's enabled — slotchange already fired,
      // so _renderRail's early-return skipped the initial build.
      this._syncRailHidden();
      this._renderRail();
      this._fit();
    }

    /** Snapshot document stylesheets into a constructable sheet that each
     *  thumbnail's nested shadow root adopts — so author CSS styles the
     *  cloned slide content without touching this component's chrome.
     *  Cross-origin sheets throw on .cssRules — skip them. Re-callable:
     *  the existing constructable sheet is reused via replaceSync so every
     *  already-adopted shadow root picks up the fresh CSS without re-adopt. */
    _snapshotAuthorCss() {
      // :root in an adopted sheet inside a shadow root matches nothing
      // (only the document root qualifies), so author rules like
      // `:root[data-voice="modern"] .serif` never reach the clones.
      // Rewrite :root → :host and mirror <html>'s data-*/class/lang onto
      // each thumb host (see _syncThumbHostAttrs) so the same selectors
      // match inside the thumbnail's shadow tree.
      const authorCss = Array.from(document.styleSheets).map(sh => {
        try {
          return Array.from(sh.cssRules).map(r => r.cssText).join('\n');
        } catch (e) {
          return '';
        }
      }).join('\n')
      // The shadow host is featureless outside the functional :host(...)
      // form, so any compound on :root — [attr], .class, #id, :pseudo —
      // must become :host(<compound>) not :host<compound>. Same for the
      // html type selector (Tailwind class-strategy dark mode emits
      // html.dark; Pico uses html[data-theme]), which has nothing to
      // match inside the thumb's shadow tree.
      .replace(/:root((?:\[[^\]]*\]|[.#][-\w]+|:[-\w]+(?:\([^)]*\))?)+)/g, ':host($1)').replace(/:root\b/g, ':host').replace(/(^|[\s,>~+(}])html((?:\[[^\]]*\]|[.#][-\w]+|:[-\w]+(?:\([^)]*\))?)+)(?![-\w])/g, '$1:host($2)').replace(/(^|[\s,>~+(}])html(?![-\w])/g, '$1:host');
      // Every custom property the author references. _syncThumbHostAttrs
      // mirrors each one's *computed* value at <deck-stage> onto the
      // thumb host so the live value wins over the :host default above
      // regardless of which ancestor the tweak wrote to (<html>, <body>,
      // a wrapper div, or the deck-stage element itself all inherit
      // down to getComputedStyle(this)).
      this._authorVars = new Set(authorCss.match(/--[\w-]+/g) || []);
      try {
        if (!this._adoptedSheet) this._adoptedSheet = new CSSStyleSheet();
        this._adoptedSheet.replaceSync(authorCss);
      } catch (e) {
        this._adoptedSheet = null;
        this._authorCss = authorCss;
      }
    }
    _syncThumbHostAttrs(host, cs) {
      const de = document.documentElement;
      // setAttribute overwrites but can't delete — an attr removed from
      // <html> (toggleAttribute off, classList emptied) would linger on
      // the host and :host([data-*]) / :host(.foo) rules would keep
      // matching. Remove stale mirrored attrs first; iterate backward
      // because removeAttribute mutates the live NamedNodeMap.
      for (let i = host.attributes.length - 1; i >= 0; i--) {
        const n = host.attributes[i].name;
        if ((n.startsWith('data-') || n === 'class' || n === 'lang') && !de.hasAttribute(n)) {
          host.removeAttribute(n);
        }
      }
      for (const a of de.attributes) {
        if (a.name.startsWith('data-') || a.name === 'class' || a.name === 'lang') {
          host.setAttribute(a.name, a.value);
        }
      }
      // The :root→:host rewrite in _snapshotAuthorCss pins each custom
      // property to its stylesheet default on the thumb host, shadowing
      // the live value that would otherwise inherit. Tweaks can write the
      // live value on any ancestor — <html>, <body>, a wrapper div, the
      // deck-stage element — so read it as the *computed* value at
      // <deck-stage> (which sees the whole inheritance chain) rather than
      // trying to guess which element the author wrote to. Inline on the
      // host beats the :host{} rule. remove-stale covers vars dropped
      // from the stylesheet between snapshots.
      const vars = this._authorVars || new Set();
      for (let i = host.style.length - 1; i >= 0; i--) {
        const p = host.style[i];
        if (p.startsWith('--') && !vars.has(p)) host.style.removeProperty(p);
      }
      const live = cs || getComputedStyle(this);
      vars.forEach(p => {
        const v = live.getPropertyValue(p);
        if (v) host.style.setProperty(p, v.trim());else host.style.removeProperty(p);
      });
    }
    disconnectedCallback() {
      window.removeEventListener('keydown', this._onKey);
      window.removeEventListener('resize', this._onResize);
      window.removeEventListener('mousemove', this._onMouseMove);
      window.removeEventListener('message', this._onMessage);
      window.removeEventListener('click', this._onDocClick, true);
      window.removeEventListener('beforeprint', this._onBeforePrint);
      window.removeEventListener('afterprint', this._onAfterPrint);
      if (this._freezeStyle) {
        this._freezeStyle.remove();
        this._freezeStyle = null;
      }
      this.removeEventListener('click', this._onTap);
      if (this._hideTimer) clearTimeout(this._hideTimer);
      if (this._mouseIdleTimer) clearTimeout(this._mouseIdleTimer);
      if (this._liveTimer) clearTimeout(this._liveTimer);
      if (this._tweakTimer) clearTimeout(this._tweakTimer);
      if (this._railAnimTimer) clearTimeout(this._railAnimTimer);
      if (this._scaleRaf) cancelAnimationFrame(this._scaleRaf);
      if (this._liveObserver) this._liveObserver.disconnect();
      if (this._railObserver) this._railObserver.disconnect();
      if (this._onTweakChange) window.removeEventListener('tweakchange', this._onTweakChange);
    }
    attributeChangedCallback() {
      if (this._canvas) {
        this._canvas.style.width = this.designWidth + 'px';
        this._canvas.style.height = this.designHeight + 'px';
        this._canvas.style.setProperty('--deck-design-w', this.designWidth + 'px');
        this._canvas.style.setProperty('--deck-design-h', this.designHeight + 'px');
        if (this._rail) {
          this._rail.style.setProperty('--deck-aspect', this.designWidth + '/' + this.designHeight);
        }
        this._fit();
        this._scaleThumbs();
        this._syncPrintPageRule();
      }
    }
    _render() {
      const style = document.createElement('style');
      style.textContent = stylesheet;
      const stage = document.createElement('div');
      stage.className = 'stage';
      const canvas = document.createElement('div');
      canvas.className = 'canvas';
      canvas.style.width = this.designWidth + 'px';
      canvas.style.height = this.designHeight + 'px';
      canvas.style.setProperty('--deck-design-w', this.designWidth + 'px');
      canvas.style.setProperty('--deck-design-h', this.designHeight + 'px');
      const slot = document.createElement('slot');
      slot.addEventListener('slotchange', this._onSlotChange);
      canvas.appendChild(slot);
      stage.appendChild(canvas);

      // Overlay: compact, solid black, with clickable controls.
      const overlay = document.createElement('div');
      overlay.className = 'overlay export-hidden';
      overlay.setAttribute('role', 'toolbar');
      overlay.setAttribute('aria-label', 'Deck controls');
      overlay.setAttribute('data-omelette-chrome', '');
      overlay.innerHTML = `
        <button class="btn prev" type="button" aria-label="Previous slide" title="Previous (←)">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10 3L5 8l5 5"/></svg>
        </button>
        <span class="count" aria-live="polite"><span class="current">1</span><span class="sep">/</span><span class="total">1</span></span>
        <button class="btn next" type="button" aria-label="Next slide" title="Next (→)">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 3l5 5-5 5"/></svg>
        </button>
        <span class="divider"></span>
        <button class="btn reset" type="button" aria-label="Reset to first slide" title="Reset (R)">Reset<span class="kbd">R</span></button>
      `;
      overlay.querySelector('.prev').addEventListener('click', () => this._advance(-1, 'click'));
      overlay.querySelector('.next').addEventListener('click', () => this._advance(1, 'click'));
      overlay.querySelector('.reset').addEventListener('click', () => this._go(0, 'click'));

      // Thumbnail rail + context menu. Thumbnails are populated in
      // _renderRail() after _collectSlides().
      const rail = document.createElement('div');
      rail.className = 'rail export-hidden';
      rail.setAttribute('data-omelette-chrome', '');
      rail.style.setProperty('--deck-aspect', this.designWidth + '/' + this.designHeight);
      // Edge auto-scroll while dragging a thumb near the rail's top/bottom
      // so off-screen drop targets are reachable. Native dragover fires
      // continuously while the pointer is stationary, so a per-event nudge
      // (ramped by edge proximity) is enough — no rAF loop needed.
      rail.addEventListener('dragover', e => {
        if (this._dragFrom == null) return;
        const r = rail.getBoundingClientRect();
        const EDGE = 40;
        const dt = e.clientY - r.top;
        const db = r.bottom - e.clientY;
        if (dt < EDGE) rail.scrollTop -= Math.ceil((EDGE - dt) / 3);else if (db < EDGE) rail.scrollTop += Math.ceil((EDGE - db) / 3);
      });
      const menu = document.createElement('div');
      menu.className = 'ctxmenu export-hidden';
      menu.setAttribute('data-omelette-chrome', '');
      menu.innerHTML = `
        <button type="button" data-act="skip">Skip slide</button>
        <button type="button" data-act="up">Move up</button>
        <button type="button" data-act="down">Move down</button>
        <button type="button" data-act="duplicate">Duplicate slide</button>
        <hr>
        <button type="button" data-act="delete">Delete slide</button>
      `;
      menu.addEventListener('click', e => {
        const act = e.target && e.target.getAttribute && e.target.getAttribute('data-act');
        if (!act) return;
        const i = this._menuIndex;
        this._closeMenu();
        if (act === 'skip') this._toggleSkip(i);else if (act === 'up') this._moveSlide(i, i - 1);else if (act === 'down') this._moveSlide(i, i + 1);else if (act === 'duplicate') this._duplicateSlide(i);else if (act === 'delete') this._openConfirm(i);
      });
      menu.addEventListener('contextmenu', e => e.preventDefault());

      // Rail resize handle — drag to set --deck-rail-w, persisted to
      // localStorage so the width survives reloads.
      const resize = document.createElement('div');
      resize.className = 'rail-resize export-hidden';
      resize.setAttribute('data-omelette-chrome', '');
      resize.addEventListener('pointerdown', e => {
        e.preventDefault();
        resize.setPointerCapture(e.pointerId);
        resize.setAttribute('data-dragging', '');
        const move = ev => this._setRailWidth(ev.clientX);
        const up = () => {
          resize.removeEventListener('pointermove', move);
          resize.removeEventListener('pointerup', up);
          resize.removeEventListener('pointercancel', up);
          resize.removeAttribute('data-dragging');
          try {
            localStorage.setItem('deck-stage.railWidth', String(this._railPx));
          } catch (err) {}
        };
        resize.addEventListener('pointermove', move);
        resize.addEventListener('pointerup', up);
        resize.addEventListener('pointercancel', up);
      });

      // Delete-confirm dialog — mirrors the SPA's ConfirmDialog layout.
      const confirm = document.createElement('div');
      confirm.className = 'confirm-backdrop export-hidden';
      confirm.setAttribute('data-omelette-chrome', '');
      confirm.innerHTML = `
        <div class="confirm" role="dialog" aria-modal="true">
          <div class="body">
            <div class="title">Delete slide?</div>
            <div class="msg">This slide will be removed from the deck.</div>
          </div>
          <div class="footer">
            <button type="button" class="cancel">Cancel</button>
            <button type="button" class="danger">Delete</button>
          </div>
        </div>
      `;
      confirm.addEventListener('click', e => {
        if (e.target === confirm) this._closeConfirm();
      });
      confirm.querySelector('.cancel').addEventListener('click', () => this._closeConfirm());
      confirm.querySelector('.danger').addEventListener('click', () => {
        const i = this._confirmIndex;
        this._closeConfirm();
        this._deleteSlide(i);
      });
      this._root.append(style, rail, resize, stage, overlay, menu, confirm);
      this._canvas = canvas;
      this._stage = stage;
      this._slot = slot;
      this._overlay = overlay;
      this._rail = rail;
      this._resize = resize;
      this._menu = menu;
      this._confirm = confirm;
      this._countEl = overlay.querySelector('.current');
      this._totalEl = overlay.querySelector('.total');

      // Restore persisted rail width.
      let rw = 188;
      try {
        const s = localStorage.getItem('deck-stage.railWidth');
        if (s) rw = parseInt(s, 10) || rw;
      } catch (err) {}
      this._setRailWidth(rw);
      this._syncRailHidden();
    }
    _setRailWidth(px) {
      const w = Math.max(120, Math.min(360, Math.round(px)));
      this._railPx = w;
      this.style.setProperty('--deck-rail-w', w + 'px');
      this._fit();
      // _scaleThumbs forces a sync layout (frame.offsetWidth) then writes
      // N transforms. During a resize drag this runs per-pointermove;
      // coalesce to one per frame.
      if (!this._scaleRaf) {
        this._scaleRaf = requestAnimationFrame(() => {
          this._scaleRaf = null;
          this._scaleThumbs();
        });
      }
    }

    /** @page must live in the document stylesheet — it's a no-op inside
     *  shadow DOM. Inject/update a single <head> style tag so the print
     *  sheet matches the design size and Save-as-PDF yields one slide per
     *  page with no margins. */
    _syncPrintPageRule() {
      const id = 'deck-stage-print-page';
      let tag = document.getElementById(id);
      if (!tag) {
        tag = document.createElement('style');
        tag.id = id;
        document.head.appendChild(tag);
      }
      tag.textContent = '@page { size: ' + this.designWidth + 'px ' + this.designHeight + 'px; margin: 0; } ' + '@media print { html, body { margin: 0 !important; padding: 0 !important; background: none !important; overflow: visible !important; height: auto !important; } ' + '* { -webkit-print-color-adjust: exact; print-color-adjust: exact; } ' +
      // Jump authored animations/transitions to their end state so print
      // never captures mid-entrance — pairs with the beforeprint handler
      // in connectedCallback that sets data-deck-active on every slide.
      '*, *::before, *::after { animation-delay: -99s !important; animation-duration: .001s !important; ' + 'animation-iteration-count: 1 !important; animation-fill-mode: both !important; ' + 'animation-play-state: running !important; transition-duration: 0s !important; } }';
    }
    _onSlotChange() {
      // Rail mutations (delete/move/duplicate) already reconcile synchronously and
      // emit slidechange with reason 'api'; skip the async slotchange that
      // would otherwise re-broadcast with reason 'init'.
      if (this._squelchSlotChange) {
        this._squelchSlotChange = false;
        return;
      }
      this._collectSlides();
      this._restoreIndex();
      this._applyIndex({
        showOverlay: false,
        broadcast: true,
        reason: 'init'
      });
      this._fit();
    }
    _collectSlides() {
      const assigned = this._slot.assignedElements({
        flatten: true
      });
      this._slides = assigned.filter(el => {
        // Skip template/style/script nodes even if someone slots them.
        const tag = el.tagName;
        return tag !== 'TEMPLATE' && tag !== 'SCRIPT' && tag !== 'STYLE';
      });
      this._slideSet = new Set(this._slides);
      this._slides.forEach((slide, i) => {
        const n = i + 1;
        slide.setAttribute('data-screen-label', `${pad2(n)} ${getSlideLabel(slide)}`);

        // Validation attribute for comment flow / auto-checks.
        if (!slide.hasAttribute('data-om-validate')) {
          slide.setAttribute('data-om-validate', VALIDATE_ATTR);
        }
        slide.setAttribute('data-deck-slide', String(i));
      });
      if (this._totalEl) this._totalEl.textContent = String(this._slides.length || 1);
      if (this._index >= this._slides.length) this._index = Math.max(0, this._slides.length - 1);
      this._markLastVisible();
      this._renderRail();
    }

    /** Tag the last non-skipped slide so print CSS can drop its
     *  break-after (see the @media print comment above — :last-child
     *  alone matches a hidden skipped slide). */
    _markLastVisible() {
      let last = null;
      this._slides.forEach(s => {
        s.removeAttribute('data-deck-last-visible');
        if (!s.hasAttribute('data-deck-skip')) last = s;
      });
      if (last) last.setAttribute('data-deck-last-visible', '');
    }
    _loadNotes() {
      const tag = document.getElementById('speaker-notes');
      if (!tag) {
        this._notes = [];
        return;
      }
      try {
        const parsed = JSON.parse(tag.textContent || '[]');
        if (Array.isArray(parsed)) this._notes = parsed;
      } catch (e) {
        console.warn('[deck-stage] Failed to parse #speaker-notes JSON:', e);
        this._notes = [];
      }
    }
    _restoreIndex() {
      // The host's ?slide= param is delivered as a #<int> hash (1-indexed) on
      // the iframe src. No hash → slide 1; the deck itself keeps no position
      // state across loads.
      const h = (location.hash || '').match(/^#(\d+)$/);
      if (h) {
        const n = parseInt(h[1], 10) - 1;
        if (n >= 0 && n < this._slides.length) this._index = n;
      }
    }
    _applyIndex({
      showOverlay = true,
      broadcast = true,
      reason = 'init'
    } = {}) {
      if (!this._slides.length) return;
      const prev = this._prevIndex == null ? -1 : this._prevIndex;
      const curr = this._index;
      // Keep the iframe's own hash in sync so an in-iframe location.reload()
      // (reload banner path in viewer-handle.ts) lands on the current slide,
      // not the stale deep-link hash from initial load.
      try {
        history.replaceState(null, '', '#' + (curr + 1));
      } catch (e) {}
      this._slides.forEach((s, i) => {
        if (i === curr) s.setAttribute('data-deck-active', '');else s.removeAttribute('data-deck-active');
      });
      if (this._countEl) this._countEl.textContent = String(curr + 1);
      // Follow-scroll on every navigation (init deep-link, keyboard, click,
      // tap, external goTo) — the only time we *don't* want the rail to
      // track current is after a rail-internal mutation, where _renderRail
      // has already restored the user's scroll position and yanking back to
      // current would undo it.
      this._syncRail(reason !== 'mutation');
      if (broadcast) {
        // (1) Legacy: host-window postMessage for speaker-notes renderers.
        try {
          window.postMessage({
            slideIndexChanged: curr,
            deckTotal: this._slides.length,
            deckSkipped: this._skippedIndices()
          }, '*');
        } catch (e) {}

        // (2) In-page CustomEvent on the <deck-stage> element itself.
        //     Bubbles and composes out of shadow DOM so slide code can listen:
        //       document.querySelector('deck-stage').addEventListener('slidechange', e => {
        //         e.detail.index, e.detail.previousIndex, e.detail.total, e.detail.slide, e.detail.reason
        //       });
        const detail = {
          index: curr,
          previousIndex: prev,
          total: this._slides.length,
          slide: this._slides[curr] || null,
          previousSlide: prev >= 0 ? this._slides[prev] || null : null,
          reason: reason // 'init' | 'keyboard' | 'click' | 'tap' | 'api'
        };
        this.dispatchEvent(new CustomEvent('slidechange', {
          detail,
          bubbles: true,
          composed: true
        }));
      }
      this._prevIndex = curr;
      if (showOverlay) this._flashOverlay();
    }
    _flashOverlay() {
      // Host posts __omelette_presenting while in fullscreen/tab presentation
      // mode — suppress the nav footer entirely (both hover and slide-change
      // flash) so the audience sees clean slides.
      if (!this._overlay || this._presenting) return;
      this._overlay.setAttribute('data-visible', '');
      if (this._hideTimer) clearTimeout(this._hideTimer);
      this._hideTimer = setTimeout(() => {
        this._overlay.removeAttribute('data-visible');
      }, OVERLAY_HIDE_MS);
    }
    _railWidth() {
      // State-based, no offsetWidth: the first _fit() can run before the
      // rail has had layout on some load paths, and a 0 there paints the
      // slide full-width for one frame before the post-slotchange _fit()
      // corrects it.
      if (!this._railEnabled || !this._railVisible || this.hasAttribute('no-rail') || this.hasAttribute('noscale') || this._presenting || this._previewMode || NARROW_MQ.matches) return 0;
      return this._railPx || 0;
    }
    _fit() {
      if (!this._canvas) return;
      const stage = this._canvas.parentElement;
      // PPTX export sets noscale so the DOM capture sees authored-size
      // geometry — the scaled canvas is in shadow DOM, so the exporter's
      // resetTransformSelector can't reach .canvas.style.transform directly.
      if (this.hasAttribute('noscale')) {
        this._canvas.style.transform = 'none';
        if (stage) stage.style.left = '0';
        if (this._overlay) this._overlay.style.marginLeft = '0';
        return;
      }
      const rw = this._railWidth();
      if (stage) stage.style.left = rw + 'px';
      // Overlay is centred on the viewport via left:50% + translate(-50%);
      // marginLeft shifts the centre by rw/2 so it lands in the middle of
      // the [rw, innerWidth] stage region.
      if (this._overlay) this._overlay.style.marginLeft = rw / 2 + 'px';
      const vw = window.innerWidth - rw;
      const vh = window.innerHeight;
      const s = Math.min(vw / this.designWidth, vh / this.designHeight);
      this._canvas.style.transform = `scale(${s})`;
    }
    _onResize() {
      this._fit();
      // Crossing the narrow-viewport breakpoint reveals the rail — rerun the
      // thumbnail scale the same way _setRailWidth does.
      if (!this._scaleRaf) {
        this._scaleRaf = requestAnimationFrame(() => {
          this._scaleRaf = null;
          this._scaleThumbs();
        });
      }
    }
    _onMouseMove() {
      // Keep overlay visible while mouse moves; hide after idle.
      this._flashOverlay();
    }
    _onMessage(e) {
      const d = e.data;
      if (d && typeof d.__omelette_presenting === 'boolean') {
        this._presenting = d.__omelette_presenting;
        if (this._presenting && this._overlay) {
          this._overlay.removeAttribute('data-visible');
          if (this._hideTimer) clearTimeout(this._hideTimer);
        }
        this._syncRailHidden();
        this._closeMenu();
        this._closeConfirm();
        this._fit();
        this._scaleThumbs();
      }
      // Host's Preview segment (ViewerMode='none'): the rail's drag-reorder /
      // right-click skip-delete affordances are editing chrome, so hide it
      // while the user is just looking at the deck. Same hard-hide path as
      // presenting; independent of the user's _railVisible preference so
      // returning to Edit restores whatever they had.
      if (d && typeof d.__omelette_preview_mode === 'boolean') {
        if (d.__omelette_preview_mode === this._previewMode) return;
        this._previewMode = d.__omelette_preview_mode;
        this._syncRailHidden();
        this._closeMenu();
        this._closeConfirm();
        this._fit();
        this._scaleThumbs();
      }
      // Per-viewer show/hide, driven by the TweaksPanel's auto-injected
      // "Thumbnail rail" toggle (or any author script). Independent of
      // whether the Tweaks panel itself is open — closing the panel
      // doesn't change rail visibility. Persists alongside rail width.
      if (d && d.type === '__deck_rail_visible' && typeof d.on === 'boolean') {
        if (d.on === this._railVisible) return;
        this._railVisible = d.on;
        try {
          localStorage.setItem('deck-stage.railVisible', d.on ? '1' : '0');
        } catch (e) {}
        // Arm the transition, commit it, then flip state — otherwise the
        // browser coalesces both writes and nothing animates on show.
        this.setAttribute('data-rail-anim', '');
        void (this._rail && this._rail.offsetHeight);
        this._syncRailHidden();
        this._fit();
        this._scaleThumbs();
        clearTimeout(this._railAnimTimer);
        this._railAnimTimer = setTimeout(() => this.removeAttribute('data-rail-anim'), 220);
      }
      if (d && d.type === '__omelette_rail_enabled') this._enableRail();
    }
    _syncRailHidden() {
      if (!this._rail) return;
      // data-presenting is the hard hide (display:none) for flag-off,
      // presentation mode, and the host's Preview segment — instant, no
      // transition. data-user-hidden is the soft hide (translateX(-100%))
      // for the viewer's rail toggle, so show/hide slides under
      // :host([data-rail-anim]).
      const hard = !this._railEnabled || this._presenting || this._previewMode;
      if (hard) this._rail.setAttribute('data-presenting', '');else this._rail.removeAttribute('data-presenting');
      if (!this._railVisible) this._rail.setAttribute('data-user-hidden', '');else this._rail.removeAttribute('data-user-hidden');
      // translateX hide leaves thumbs (tabIndex=0) in the tab order —
      // inert keeps them unfocusable while the rail is off-screen.
      this._rail.inert = hard || !this._railVisible;
    }
    _onTap(e) {
      // Touch-only — keyboard + the overlay toolbar cover nav on desktop.
      if (FINE_POINTER_MQ.matches) return;
      // Only taps that land on the stage (slide content or letterbox); the
      // overlay / rail / menus are siblings with their own click handlers.
      const path = e.composedPath();
      if (!this._stage || !path.includes(this._stage)) return;
      // Let interactive slide content keep the tap. composedPath (not
      // e.target.closest) so we see through open shadow roots — a <button>
      // inside a slide-authored custom element retargets e.target to the
      // host but still appears in the composed path.
      if (e.defaultPrevented) return;
      for (const n of path) {
        if (n === this._stage) break;
        if (n.matches && n.matches(INTERACTIVE_SEL)) return;
      }
      e.preventDefault();
      const rw = this._railWidth();
      const mid = rw + (window.innerWidth - rw) / 2;
      this._advance(e.clientX < mid ? -1 : 1, 'tap');
    }
    _onKey(e) {
      // Ignore when the user is typing.
      const t = e.target;
      if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
      // Confirm dialog swallows nav keys while open; Escape cancels. Enter
      // is left to the focused button's native activation so Tab→Cancel
      // →Enter activates Cancel, not the window-level confirm path.
      if (this._confirm && this._confirm.hasAttribute('data-open')) {
        if (e.key === 'Escape') {
          this._closeConfirm();
          e.preventDefault();
        }
        return;
      }
      if (e.key === 'Escape' && this._menu && this._menu.hasAttribute('data-open')) {
        this._closeMenu();
        e.preventDefault();
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const key = e.key;
      let handled = true;
      if (key === 'ArrowRight' || key === 'PageDown' || key === ' ' || key === 'Spacebar') {
        this._advance(1, 'keyboard');
      } else if (key === 'ArrowLeft' || key === 'PageUp') {
        this._advance(-1, 'keyboard');
      } else if (key === 'Home') {
        this._go(0, 'keyboard');
      } else if (key === 'End') {
        this._go(this._slides.length - 1, 'keyboard');
      } else if (key === 'r' || key === 'R') {
        this._go(0, 'keyboard');
      } else if (/^[0-9]$/.test(key)) {
        // 1..9 jump to that slide; 0 jumps to 10.
        const n = key === '0' ? 9 : parseInt(key, 10) - 1;
        if (n < this._slides.length) this._go(n, 'keyboard');
      } else {
        handled = false;
      }
      if (handled) {
        e.preventDefault();
        this._flashOverlay();
      }
    }
    _go(i, reason = 'api') {
      if (!this._slides.length) return;
      const clamped = Math.max(0, Math.min(this._slides.length - 1, i));
      if (clamped === this._index) {
        this._flashOverlay();
        return;
      }
      this._index = clamped;
      this._applyIndex({
        showOverlay: true,
        broadcast: true,
        reason
      });
    }

    /** Step forward/back skipping any slide marked data-deck-skip. Falls
     *  back to _go's clamp-at-ends behaviour (flash overlay) when there's
     *  nothing further in that direction. */
    _advance(dir, reason) {
      if (!this._slides.length) return;
      let i = this._index + dir;
      while (i >= 0 && i < this._slides.length && this._slides[i].hasAttribute('data-deck-skip')) {
        i += dir;
      }
      if (i < 0 || i >= this._slides.length) {
        this._flashOverlay();
        return;
      }
      this._go(i, reason);
    }

    // ── Thumbnail rail ────────────────────────────────────────────────────
    //
    // Thumbs are keyed by slide element and reused across _renderRail()
    // calls, so a reorder/delete is an O(changed) DOM shuffle instead of an
    // O(N) teardown-and-re-clone. Each thumb starts as a lightweight shell
    // (num + empty frame); the clone is materialized lazily by an
    // IntersectionObserver when the frame scrolls into (or near) view, so
    // only visible-ish slides pay the clone + image-decode cost.

    _renderRail() {
      if (!this._rail || !this._railEnabled) {
        this._thumbs = [];
        return;
      }
      // FLIP: record each *materialized* thumb's top before the reconcile.
      // Off-screen (non-materialized) thumbs don't need the animation and
      // skipping their getBoundingClientRect saves a forced layout per
      // off-screen thumb on large decks.
      const prevTops = new Map();
      (this._thumbs || []).forEach(({
        thumb,
        slide,
        host
      }) => {
        if (host) prevTops.set(slide, thumb.getBoundingClientRect().top);
      });
      const st = this._rail.scrollTop;

      // Reconcile: reuse thumbs that already exist for a slide, create
      // shells for new slides, drop thumbs for removed slides.
      const bySlide = new Map();
      (this._thumbs || []).forEach(t => bySlide.set(t.slide, t));
      const next = [];
      this._slides.forEach(slide => {
        let t = bySlide.get(slide);
        if (t) bySlide.delete(slide);else t = this._makeThumb(slide);
        next.push(t);
      });
      // Orphans — slides removed since last render.
      bySlide.forEach(t => {
        if (this._railObserver) this._railObserver.unobserve(t.frame);
        t.thumb.remove();
      });
      // Put thumbs into document order to match _slides. insertBefore on
      // an already-correctly-placed node is a no-op, so this is cheap
      // when nothing moved.
      next.forEach((t, i) => {
        const want = t.thumb;
        const at = this._rail.children[i];
        if (at !== want) this._rail.insertBefore(want, at || null);
        t.i = i;
        t.num.textContent = String(i + 1);
        if (t.slide.hasAttribute('data-deck-skip')) t.thumb.setAttribute('data-skip', '');else t.thumb.removeAttribute('data-skip');
      });
      this._thumbs = next;
      this._rail.scrollTop = st;
      if (prevTops.size) {
        const moved = [];
        this._thumbs.forEach(({
          thumb,
          slide
        }) => {
          const old = prevTops.get(slide);
          if (old == null) return;
          const dy = old - thumb.getBoundingClientRect().top;
          if (Math.abs(dy) < 1) return;
          thumb.style.transition = 'none';
          thumb.style.transform = `translateY(${dy}px)`;
          moved.push(thumb);
        });
        if (moved.length) {
          // Commit the inverted positions before flipping the transition
          // on — otherwise the browser coalesces both style writes and
          // nothing animates.
          void this._rail.offsetHeight;
          moved.forEach(t => {
            t.style.transition = 'transform 180ms cubic-bezier(.2,.7,.3,1)';
            t.style.transform = '';
          });
          setTimeout(() => moved.forEach(t => {
            t.style.transition = '';
          }), 220);
        }
      }
      requestAnimationFrame(() => this._scaleThumbs());
      this._syncRail(false);
    }

    /** Create a lightweight thumb shell for one slide. The clone is
     *  materialized later by the IntersectionObserver. Event handlers
     *  look up the thumb's *current* index (via _thumbs.indexOf) so the
     *  same element can be reused across reorders. */
    _makeThumb(slide) {
      const thumb = document.createElement('div');
      thumb.className = 'thumb';
      thumb.tabIndex = 0;
      const num = document.createElement('div');
      num.className = 'num';
      const frame = document.createElement('div');
      frame.className = 'frame';
      thumb.append(num, frame);
      const entry = {
        thumb,
        num,
        frame,
        slide,
        clone: null,
        host: null,
        i: -1
      };
      // entry.i is refreshed on every _renderRail reconcile pass, so
      // handlers read the thumb's current position without an O(N) scan.
      const idx = () => entry.i;
      thumb.addEventListener('click', () => this._go(idx(), 'click'));
      // ↑/↓ step through the rail when a thumb has focus. _go clamps at the
      // ends and _applyIndex→_syncRail scrolls the new current thumb into
      // view; we move focus to it (preventScroll — _syncRail already
      // scrolled) so a held key walks the whole list. stopPropagation keeps
      // this out of the window-level _onKey nav handler.
      thumb.addEventListener('keydown', e => {
        if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
        if (e.metaKey || e.ctrlKey || e.altKey) return;
        e.preventDefault();
        e.stopPropagation();
        this._go(idx() + (e.key === 'ArrowDown' ? 1 : -1), 'keyboard');
        const cur = this._thumbs && this._thumbs[this._index];
        if (cur) cur.thumb.focus({
          preventScroll: true
        });
      });
      thumb.addEventListener('contextmenu', e => {
        e.preventDefault();
        this._openMenu(idx(), e.clientX, e.clientY);
      });
      thumb.draggable = true;
      thumb.addEventListener('dragstart', e => {
        this._dragFrom = idx();
        thumb.setAttribute('data-dragging', '');
        e.dataTransfer.effectAllowed = 'move';
        try {
          e.dataTransfer.setData('text/plain', String(this._dragFrom));
        } catch (err) {}
      });
      thumb.addEventListener('dragend', () => {
        thumb.removeAttribute('data-dragging');
        this._clearDrop();
        this._dragFrom = null;
      });
      thumb.addEventListener('dragover', e => {
        if (this._dragFrom == null) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        const r = thumb.getBoundingClientRect();
        this._setDrop(idx(), e.clientY < r.top + r.height / 2 ? 'before' : 'after');
      });
      thumb.addEventListener('drop', e => {
        if (this._dragFrom == null) return;
        e.preventDefault();
        const i = idx();
        const r = thumb.getBoundingClientRect();
        let to = e.clientY >= r.top + r.height / 2 ? i + 1 : i;
        if (this._dragFrom < to) to--;
        const from = this._dragFrom;
        this._clearDrop();
        this._dragFrom = null;
        if (to !== from) this._moveSlide(from, to);
      });
      if (this._railObserver) this._railObserver.observe(frame);
      frame.__deckThumb = entry;
      return entry;
    }

    /** Lazily build the clone for a thumb that has scrolled into view. */
    _materialize(entry) {
      if (entry.host) return;
      const dw = this.designWidth,
        dh = this.designHeight;
      let clone = entry.slide.cloneNode(true);
      clone.removeAttribute('id');
      clone.removeAttribute('data-deck-active');
      clone.querySelectorAll('[id]').forEach(el => el.removeAttribute('id'));
      // Neuter heavy media; replace <video> with its poster so the box
      // keeps a visual. <iframe>/<audio> become empty placeholders.
      clone.querySelectorAll('iframe, audio, object, embed').forEach(el => {
        el.removeAttribute('src');
        el.removeAttribute('srcdoc');
        el.removeAttribute('data');
        el.innerHTML = '';
      });
      clone.querySelectorAll('video').forEach(el => {
        if (!el.poster) {
          el.removeAttribute('src');
          el.innerHTML = '';
          return;
        }
        const img = document.createElement('img');
        img.src = el.poster;
        img.alt = '';
        img.style.cssText = el.style.cssText + ';object-fit:cover;width:100%;height:100%;';
        img.className = el.className;
        el.replaceWith(img);
      });
      // Images: defer decode and let the browser pick the smallest
      // srcset candidate for the ~140px thumb. Same-URL clones reuse the
      // slide's decoded bitmap (URL-keyed cache), so the remaining cost
      // is paint/composite — lazy+async keeps that off the main thread.
      clone.querySelectorAll('img').forEach(el => {
        el.loading = 'lazy';
        el.decoding = 'async';
        if (el.srcset) el.sizes = (this._railPx || 188) + 'px';
      });
      // Custom elements inside the slide would have their
      // connectedCallback fire when the clone is appended. Replace them
      // with inert boxes so a component-heavy deck doesn't run N copies
      // of each component's mount logic in the rail. Children are
      // preserved so layout-wrapper elements (<my-column><h2>…</h2>)
      // still show their authored content; the querySelectorAll NodeList
      // is static, so nested custom elements in the moved subtree are
      // still visited on later iterations.
      const neuter = el => {
        const box = document.createElement('div');
        box.style.cssText = (el.getAttribute('style') || '') + ';background:rgba(0,0,0,0.06);border:1px dashed rgba(0,0,0,0.15);';
        box.className = el.className;
        // Preserve theming/i18n hooks so [data-*] / :lang() / [dir]
        // descendant selectors still match the neutered root.
        for (const a of el.attributes) {
          const n = a.name;
          if (n.startsWith('data-') || n.startsWith('aria-') || n === 'lang' || n === 'dir' || n === 'role' || n === 'title') {
            box.setAttribute(n, a.value);
          }
        }
        while (el.firstChild) box.appendChild(el.firstChild);
        return box;
      };
      // querySelectorAll('*') returns descendants only — a custom-element
      // slide root (<my-slide>…</my-slide>) would slip through and upgrade
      // on append. Swap the root first.
      if (clone.tagName.includes('-')) clone = neuter(clone);
      clone.querySelectorAll('*').forEach(el => {
        if (el.tagName.includes('-')) el.replaceWith(neuter(el));
      });
      clone.style.cssText += ';position:absolute;top:0;left:0;transform-origin:0 0;' + 'pointer-events:none;width:' + dw + 'px;height:' + dh + 'px;' + 'box-sizing:border-box;overflow:hidden;visibility:visible;opacity:1;';
      const host = document.createElement('div');
      host.style.cssText = 'position:absolute;inset:0;';
      this._syncThumbHostAttrs(host);
      const sr = host.attachShadow({
        mode: 'open'
      });
      if (this._adoptedSheet) sr.adoptedStyleSheets = [this._adoptedSheet];else {
        const st = document.createElement('style');
        st.textContent = this._authorCss || '';
        sr.appendChild(st);
      }
      sr.appendChild(clone);
      entry.frame.appendChild(host);
      entry.host = host;
      entry.clone = clone;
      if (this._thumbScale) clone.style.transform = 'scale(' + this._thumbScale + ')';
      // Once materialized the IO callback is a no-op early-return —
      // unobserve so scroll doesn't keep firing it.
      if (this._railObserver) this._railObserver.unobserve(entry.frame);
    }

    /** Re-clone a single thumb (live-update path). No-op if the thumb
     *  hasn't been materialized yet — it'll pick up current content when
     *  it scrolls into view. */
    _refreshThumb(slide) {
      const entry = (this._thumbs || []).find(t => t.slide === slide);
      if (!entry || !entry.host) return;
      entry.host.remove();
      entry.host = entry.clone = null;
      this._materialize(entry);
    }
    _scaleThumbs() {
      if (!this._thumbs || !this._thumbs.length) return;
      // Every frame is the same width; if it reads 0 the rail is
      // display:none (noscale / no-rail / presenting / print) — leave the
      // clones as-is and re-run when the rail is revealed.
      const fw = this._thumbs[0].frame.offsetWidth;
      if (!fw) return;
      this._thumbScale = fw / this.designWidth;
      this._thumbs.forEach(({
        clone
      }) => {
        if (clone) clone.style.transform = 'scale(' + this._thumbScale + ')';
      });
    }
    _setDrop(i, where) {
      // dragover fires at pointer-event rate; touch only the previous
      // and new target rather than sweeping all N thumbs.
      const t = this._thumbs && this._thumbs[i];
      if (this._dropOn && this._dropOn !== t) {
        this._dropOn.thumb.removeAttribute('data-drop');
      }
      if (t) t.thumb.setAttribute('data-drop', where);
      this._dropOn = t || null;
    }
    _clearDrop() {
      if (this._dropOn) this._dropOn.thumb.removeAttribute('data-drop');
      this._dropOn = null;
    }
    _syncRail(follow) {
      if (!this._thumbs) return;
      this._thumbs.forEach(({
        thumb
      }, i) => {
        if (i === this._index) {
          thumb.setAttribute('data-current', '');
          if (follow && typeof thumb.scrollIntoView === 'function') {
            thumb.scrollIntoView({
              block: 'nearest'
            });
          }
        } else {
          thumb.removeAttribute('data-current');
        }
      });
    }
    _openMenu(i, x, y) {
      if (!this._menu) return;
      this._menuIndex = i;
      const slide = this._slides[i];
      const skip = slide && slide.hasAttribute('data-deck-skip');
      this._menu.querySelector('[data-act="skip"]').textContent = skip ? 'Unskip slide' : 'Skip slide';
      this._menu.querySelector('[data-act="up"]').disabled = i <= 0;
      this._menu.querySelector('[data-act="down"]').disabled = i >= this._slides.length - 1;
      this._menu.querySelector('[data-act="delete"]').disabled = this._slides.length <= 1;
      // Place, then clamp to viewport after it's measurable.
      this._menu.style.left = x + 'px';
      this._menu.style.top = y + 'px';
      this._menu.setAttribute('data-open', '');
      const r = this._menu.getBoundingClientRect();
      const nx = Math.min(x, window.innerWidth - r.width - 4);
      const ny = Math.min(y, window.innerHeight - r.height - 4);
      this._menu.style.left = Math.max(4, nx) + 'px';
      this._menu.style.top = Math.max(4, ny) + 'px';
    }
    _closeMenu() {
      if (this._menu) this._menu.removeAttribute('data-open');
      this._menuIndex = -1;
    }
    _openConfirm(i) {
      if (!this._confirm) return;
      this._confirmIndex = i;
      this._confirm.querySelector('.title').textContent = 'Delete slide ' + (i + 1) + '?';
      this._confirm.setAttribute('data-open', '');
      const btn = this._confirm.querySelector('.danger');
      if (btn && btn.focus) btn.focus();
    }
    _closeConfirm() {
      if (this._confirm) this._confirm.removeAttribute('data-open');
      this._confirmIndex = -1;
    }
    _emitDeckChange(detail) {
      this.dispatchEvent(new CustomEvent('deckchange', {
        detail,
        bubbles: true,
        composed: true
      }));
    }
    _deleteSlide(i) {
      const slide = this._slides[i];
      if (!slide || this._slides.length <= 1) return;
      const wasCurrent = i === this._index;
      if (i < this._index || wasCurrent && i === this._slides.length - 1) this._index--;
      this._squelchSlotChange = true;
      slide.remove();
      this._emitDeckChange({
        action: 'delete',
        from: i,
        slide
      });
      this._collectSlides();
      this._applyIndex({
        showOverlay: true,
        broadcast: true,
        reason: 'mutation'
      });
    }
    _duplicateSlide(i) {
      const slide = this._slides[i];
      if (!slide) return;
      const copy = slide.cloneNode(true);
      // Strip ids so the document stays valid (no duplicate-id collisions
      // with the original). Same treatment _materialize gives rail clones.
      copy.removeAttribute('id');
      copy.querySelectorAll('[id]').forEach(el => el.removeAttribute('id'));
      // Insert after the original and make the copy active so it's the one
      // on screen. _collectSlides re-derives data-screen-label / data-deck-*
      // attrs, so the cloned values are overwritten.
      this._index = i + 1;
      this._squelchSlotChange = true;
      this.insertBefore(copy, slide.nextSibling);
      this._emitDeckChange({
        action: 'duplicate',
        from: i,
        to: i + 1,
        slide: copy
      });
      this._collectSlides();
      this._applyIndex({
        showOverlay: true,
        broadcast: true,
        reason: 'mutation'
      });
    }
    _toggleSkip(i) {
      const slide = this._slides[i];
      if (!slide) return;
      const on = !slide.hasAttribute('data-deck-skip');
      if (on) slide.setAttribute('data-deck-skip', '');else slide.removeAttribute('data-deck-skip');
      if (this._thumbs && this._thumbs[i]) {
        if (on) this._thumbs[i].thumb.setAttribute('data-skip', '');else this._thumbs[i].thumb.removeAttribute('data-skip');
      }
      this._markLastVisible();
      this._emitDeckChange({
        action: on ? 'skip' : 'unskip',
        from: i,
        slide
      });
      // Re-broadcast so the presenter popup's prev/next thumbnails re-pick
      // the nearest non-skipped slide without waiting for a nav event.
      try {
        window.postMessage({
          slideIndexChanged: this._index,
          deckTotal: this._slides.length,
          deckSkipped: this._skippedIndices()
        }, '*');
      } catch (e) {}
    }
    _skippedIndices() {
      const out = [];
      for (let i = 0; i < this._slides.length; i++) {
        if (this._slides[i].hasAttribute('data-deck-skip')) out.push(i);
      }
      return out;
    }
    _moveSlide(i, j) {
      if (j < 0 || j >= this._slides.length || j === i) return;
      const slide = this._slides[i];
      const ref = j < i ? this._slides[j] : this._slides[j].nextSibling;
      // Track the active slide across the reorder so the same content
      // stays on screen.
      const cur = this._index;
      if (cur === i) this._index = j;else if (i < cur && j >= cur) this._index = cur - 1;else if (i > cur && j <= cur) this._index = cur + 1;
      this._squelchSlotChange = true;
      this.insertBefore(slide, ref);
      this._emitDeckChange({
        action: 'move',
        from: i,
        to: j,
        slide
      });
      this._collectSlides();
      this._applyIndex({
        showOverlay: false,
        broadcast: true,
        reason: 'mutation'
      });
    }

    // Public API ------------------------------------------------------------

    /** Current slide index (0-based). */
    get index() {
      return this._index;
    }
    /** Total slide count. */
    get length() {
      return this._slides.length;
    }
    /** Programmatically navigate. */
    goTo(i) {
      this._go(i, 'api');
    }
    next() {
      this._advance(1, 'api');
    }
    prev() {
      this._advance(-1, 'api');
    }
    reset() {
      this._go(0, 'api');
    }
  }
  if (!customElements.get('deck-stage')) {
    customElements.define('deck-stage', DeckStage);
  }
})();
})(); } catch (e) { __ds_ns.__errors.push({ path: "slides/deck-stage.js", error: String((e && e.message) || e) }); }

// ui_kits/web/App.jsx
try { (() => {
// App composition — assembles the NVIDIA brand-applied landing page.
function App() {
  const [active, setActive] = React.useState("Platform");
  return /*#__PURE__*/React.createElement("div", {
    style: {
      background: "var(--surface-page)"
    }
  }, /*#__PURE__*/React.createElement(NavBar, {
    active: active,
    onNavigate: setActive
  }), /*#__PURE__*/React.createElement(Hero, null), /*#__PURE__*/React.createElement(Showcase, null), /*#__PURE__*/React.createElement(Footer, null));
}
ReactDOM.createRoot(document.getElementById("root")).render(/*#__PURE__*/React.createElement(App, null));
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/web/App.jsx", error: String((e && e.message) || e) }); }

// ui_kits/web/Footer.jsx
try { (() => {
// Developer CTA band + footer.
function Footer() {
  const {
    Button
  } = window.DesignSystem_6d5263;
  const Icon = window.UiIcon;
  const cols = [{
    h: "Platform",
    items: ["Data Center", "Cloud", "Networking", "CUDA-X"]
  }, {
    h: "Developers",
    items: ["Documentation", "CUDA Toolkit", "Forums", "Training"]
  }, {
    h: "Company",
    items: ["About", "Careers", "Newsroom", "Investors"]
  }, {
    h: "Support",
    items: ["Contact", "Drivers", "Community", "Security"]
  }];
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("section", {
    style: ftStyles.cta
  }, /*#__PURE__*/React.createElement("div", {
    style: ftStyles.ctaInner
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h2", {
    style: ftStyles.ctaTitle
  }, "Start building with NVIDIA"), /*#__PURE__*/React.createElement("p", {
    style: ftStyles.ctaSub
  }, "Free tools, SDKs, and documentation for developers.")), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      gap: 12
    }
  }, /*#__PURE__*/React.createElement(Button, {
    size: "lg",
    iconRight: /*#__PURE__*/React.createElement(Icon, {
      name: "ArrowRight",
      size: 17,
      color: "#000"
    })
  }, "Join the Developer Program")))), /*#__PURE__*/React.createElement("footer", {
    style: ftStyles.foot
  }, /*#__PURE__*/React.createElement("div", {
    style: ftStyles.footInner
  }, /*#__PURE__*/React.createElement("div", {
    style: ftStyles.footGrid
  }, /*#__PURE__*/React.createElement("div", {
    style: ftStyles.brandCol
  }, /*#__PURE__*/React.createElement("img", {
    src: "../../assets/logos/nvidia-logo-white.png",
    alt: "NVIDIA",
    style: {
      height: 22
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: ftStyles.social
  }, ["Github", "Youtube", "Linkedin", "Twitter"].map(s => /*#__PURE__*/React.createElement("a", {
    key: s,
    href: "#",
    onClick: e => e.preventDefault(),
    style: ftStyles.socialLink,
    "aria-label": s
  }, /*#__PURE__*/React.createElement(Icon, {
    name: s,
    size: 18,
    color: "var(--nv-gray-300)"
  }))))), cols.map(c => /*#__PURE__*/React.createElement("div", {
    key: c.h
  }, /*#__PURE__*/React.createElement("div", {
    style: ftStyles.colHead
  }, c.h), /*#__PURE__*/React.createElement("ul", {
    style: ftStyles.colList
  }, c.items.map(it => /*#__PURE__*/React.createElement("li", {
    key: it,
    style: {
      marginBottom: 10
    }
  }, /*#__PURE__*/React.createElement("a", {
    href: "#",
    onClick: e => e.preventDefault(),
    style: ftStyles.colLink
  }, it))))))), /*#__PURE__*/React.createElement("div", {
    style: ftStyles.legal
  }, /*#__PURE__*/React.createElement("span", null, "\xA9 2025 NVIDIA Corporation. All rights reserved."), /*#__PURE__*/React.createElement("span", {
    style: {
      display: "flex",
      gap: 20
    }
  }, /*#__PURE__*/React.createElement("a", {
    href: "#",
    onClick: e => e.preventDefault(),
    style: ftStyles.legalLink
  }, "Privacy Policy"), /*#__PURE__*/React.createElement("a", {
    href: "#",
    onClick: e => e.preventDefault(),
    style: ftStyles.legalLink
  }, "Terms of Use"))))));
}
const ftStyles = {
  cta: {
    background: "var(--nv-green)"
  },
  ctaInner: {
    maxWidth: 1280,
    margin: "0 auto",
    padding: "44px 28px",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 24,
    flexWrap: "wrap"
  },
  ctaTitle: {
    fontFamily: "var(--font-display)",
    fontWeight: "var(--fw-medium)",
    fontSize: "var(--fs-2xl)",
    color: "var(--nv-black)",
    margin: 0,
    letterSpacing: "var(--ls-snug)"
  },
  ctaSub: {
    fontSize: "var(--fs-md)",
    color: "rgba(0,0,0,.7)",
    margin: "8px 0 0"
  },
  foot: {
    background: "var(--nv-black)",
    color: "var(--nv-gray-300)"
  },
  footInner: {
    maxWidth: 1280,
    margin: "0 auto",
    padding: "56px 28px 36px"
  },
  footGrid: {
    display: "grid",
    gridTemplateColumns: "1.4fr 1fr 1fr 1fr 1fr",
    gap: 32,
    paddingBottom: 40,
    borderBottom: "1px solid var(--nv-gray-800)"
  },
  brandCol: {
    display: "flex",
    flexDirection: "column",
    gap: 22
  },
  social: {
    display: "flex",
    gap: 8
  },
  socialLink: {
    width: 36,
    height: 36,
    borderRadius: "var(--radius-sm)",
    border: "1px solid var(--nv-gray-800)",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center"
  },
  colHead: {
    color: "var(--nv-white)",
    fontSize: "var(--fs-sm)",
    fontWeight: "var(--fw-semibold)",
    marginBottom: 16
  },
  colList: {
    listStyle: "none",
    padding: 0,
    margin: 0
  },
  colLink: {
    color: "var(--nv-gray-300)",
    fontSize: "var(--fs-sm)",
    textDecoration: "none"
  },
  legal: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    paddingTop: 24,
    fontSize: "var(--fs-xs)",
    color: "var(--nv-gray-500)",
    flexWrap: "wrap",
    gap: 12
  },
  legalLink: {
    color: "var(--nv-gray-500)",
    textDecoration: "none"
  }
};
window.Footer = Footer;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/web/Footer.jsx", error: String((e && e.message) || e) }); }

// ui_kits/web/Hero.jsx
try { (() => {
// Hero — black full-bleed band with eyebrow, large display title, CTAs, stats.
function Hero() {
  const {
    Button,
    Badge,
    Stat
  } = window.DesignSystem_6d5263;
  const Icon = window.UiIcon;
  return /*#__PURE__*/React.createElement("section", {
    style: heroStyles.wrap
  }, /*#__PURE__*/React.createElement("div", {
    style: heroStyles.glow
  }), /*#__PURE__*/React.createElement("div", {
    style: heroStyles.inner
  }, /*#__PURE__*/React.createElement(Badge, {
    variant: "outline",
    tone: "green",
    style: {
      background: "rgba(118,185,0,.08)"
    }
  }, "New \xB7 Blackwell Architecture"), /*#__PURE__*/React.createElement("h1", {
    style: heroStyles.title
  }, "The Engine of", /*#__PURE__*/React.createElement("br", null), "Accelerated Computing"), /*#__PURE__*/React.createElement("p", {
    style: heroStyles.sub
  }, "One full-stack platform \u2014 from GPU silicon to CUDA libraries and AI frameworks \u2014 powering training and inference at every scale."), /*#__PURE__*/React.createElement("div", {
    style: heroStyles.ctas
  }, /*#__PURE__*/React.createElement(Button, {
    size: "lg",
    iconRight: /*#__PURE__*/React.createElement(Icon, {
      name: "ArrowRight",
      size: 17,
      color: "#000"
    })
  }, "Explore the Platform"), /*#__PURE__*/React.createElement(Button, {
    size: "lg",
    variant: "secondary",
    style: {
      color: "#fff",
      borderColor: "var(--nv-gray-700)"
    },
    iconLeft: /*#__PURE__*/React.createElement(Icon, {
      name: "Play",
      size: 16,
      color: "#fff"
    })
  }, "Watch the Keynote")), /*#__PURE__*/React.createElement("div", {
    style: heroStyles.stats
  }, /*#__PURE__*/React.createElement(Stat, {
    value: "208B",
    label: "Transistors",
    delta: "2.6\xD7"
  }), /*#__PURE__*/React.createElement("span", {
    style: heroStyles.div
  }), /*#__PURE__*/React.createElement(Stat, {
    value: "20 PFLOPS",
    label: "FP4 inference",
    delta: "5\xD7"
  }), /*#__PURE__*/React.createElement("span", {
    style: heroStyles.div
  }), /*#__PURE__*/React.createElement(Stat, {
    value: "192 GB",
    label: "HBM3e memory",
    delta: "8 TB/s"
  }))));
}
const heroStyles = {
  wrap: {
    position: "relative",
    background: "var(--nv-black)",
    color: "var(--nv-white)",
    overflow: "hidden"
  },
  glow: {
    position: "absolute",
    top: -160,
    right: -120,
    width: 620,
    height: 620,
    background: "radial-gradient(circle, rgba(118,185,0,.22), transparent 62%)",
    pointerEvents: "none"
  },
  inner: {
    position: "relative",
    maxWidth: 1280,
    margin: "0 auto",
    padding: "104px 28px 88px"
  },
  title: {
    fontFamily: "var(--font-display)",
    fontWeight: "var(--fw-medium)",
    fontSize: 76,
    lineHeight: 1.03,
    letterSpacing: "var(--ls-tight)",
    margin: "24px 0 0",
    maxWidth: 920,
    color: "#fff"
  },
  sub: {
    fontSize: "var(--fs-lg)",
    lineHeight: 1.5,
    color: "var(--nv-gray-300)",
    maxWidth: 600,
    margin: "22px 0 0"
  },
  ctas: {
    display: "flex",
    gap: 14,
    margin: "40px 0 0",
    flexWrap: "wrap"
  },
  stats: {
    display: "flex",
    alignItems: "center",
    gap: 36,
    marginTop: 72,
    paddingTop: 36,
    borderTop: "1px solid var(--nv-gray-800)",
    "--text-primary": "#ffffff",
    "--text-secondary": "var(--nv-gray-300)",
    "--nv-green-700": "var(--nv-green-300)"
  },
  div: {
    width: 1,
    height: 48,
    background: "var(--nv-gray-800)"
  }
};

// Stat text is dark by default; recolor for the dark hero.
window.Hero = Hero;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/web/Hero.jsx", error: String((e && e.message) || e) }); }

// ui_kits/web/NavBar.jsx
try { (() => {
// Top navigation bar — black surface, NVIDIA logo, primary nav, search + CTA.
function NavBar({
  active,
  onNavigate
}) {
  const {
    Button,
    IconButton
  } = window.DesignSystem_6d5263;
  const Icon = window.UiIcon;
  const links = ["Platform", "Products", "Solutions", "Developers", "Industries"];
  return /*#__PURE__*/React.createElement("header", {
    style: navStyles.bar
  }, /*#__PURE__*/React.createElement("div", {
    style: navStyles.inner
  }, /*#__PURE__*/React.createElement("div", {
    style: navStyles.left
  }, /*#__PURE__*/React.createElement("img", {
    src: "../../assets/logos/nvidia-logo-white.png",
    alt: "NVIDIA",
    style: {
      height: 22
    }
  }), /*#__PURE__*/React.createElement("nav", {
    style: navStyles.links
  }, links.map(l => /*#__PURE__*/React.createElement("a", {
    key: l,
    href: "#",
    onClick: e => {
      e.preventDefault();
      onNavigate && onNavigate(l);
    },
    style: {
      ...navStyles.link,
      color: active === l ? "var(--nv-white)" : "var(--nv-gray-300)"
    }
  }, l, active === l && /*#__PURE__*/React.createElement("span", {
    style: navStyles.activeBar
  }))))), /*#__PURE__*/React.createElement("div", {
    style: navStyles.right
  }, /*#__PURE__*/React.createElement(IconButton, {
    "aria-label": "Search",
    style: {
      color: "var(--nv-gray-200)"
    }
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "Search",
    size: 18
  })), /*#__PURE__*/React.createElement(IconButton, {
    "aria-label": "Account",
    style: {
      color: "var(--nv-gray-200)"
    }
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "User",
    size: 18
  })), /*#__PURE__*/React.createElement(Button, {
    size: "sm",
    iconRight: /*#__PURE__*/React.createElement(Icon, {
      name: "ArrowRight",
      size: 15,
      color: "#000"
    })
  }, "Get Started"))));
}
const navStyles = {
  bar: {
    position: "sticky",
    top: 0,
    zIndex: 100,
    background: "rgba(0,0,0,0.92)",
    backdropFilter: "saturate(120%) blur(8px)",
    borderBottom: "1px solid var(--nv-gray-800)"
  },
  inner: {
    maxWidth: 1280,
    margin: "0 auto",
    height: 64,
    padding: "0 28px",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between"
  },
  left: {
    display: "flex",
    alignItems: "center",
    gap: 36
  },
  links: {
    display: "flex",
    alignItems: "center",
    gap: 26
  },
  link: {
    position: "relative",
    fontSize: "var(--fs-sm)",
    fontWeight: "var(--fw-medium)",
    textDecoration: "none",
    padding: "20px 0",
    transition: "color var(--dur-fast)"
  },
  activeBar: {
    position: "absolute",
    left: 0,
    right: 0,
    bottom: 0,
    height: 2,
    background: "var(--nv-green)"
  },
  right: {
    display: "flex",
    alignItems: "center",
    gap: 6
  }
};
window.NavBar = NavBar;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/web/NavBar.jsx", error: String((e && e.message) || e) }); }

// ui_kits/web/Showcase.jsx
try { (() => {
// Product showcase — filterable grid of product cards built from Card + Badge.
function Showcase() {
  const {
    Card,
    Badge,
    Tabs,
    Button,
    Tag
  } = window.DesignSystem_6d5263;
  const Icon = window.UiIcon;
  const [cat, setCat] = React.useState("All");
  const products = [{
    name: "DGX B200",
    cat: "Data Center",
    tag: "Flagship",
    tone: "green",
    desc: "Unified AI platform for training and inference at enterprise scale.",
    spec: "1.4 TB GPU memory · 72 PFLOPS"
  }, {
    name: "GeForce RTX 50",
    cat: "Gaming",
    tag: "New",
    tone: "info",
    desc: "Neural rendering and DLSS for the next generation of real-time graphics.",
    spec: "Up to 4,000 AI TOPS"
  }, {
    name: "Jetson Orin",
    cat: "Edge",
    tag: "Robotics",
    tone: "neutral",
    desc: "Compact edge AI for autonomous machines and embedded systems.",
    spec: "275 TOPS · 60 W"
  }, {
    name: "RTX PRO 6000",
    cat: "Workstation",
    tag: "Pro",
    tone: "neutral",
    desc: "Professional visualization and simulation for creators and engineers.",
    spec: "96 GB GDDR7"
  }, {
    name: "CUDA Toolkit",
    cat: "Software",
    tag: "Free",
    tone: "success",
    desc: "The parallel computing platform and programming model for GPUs.",
    spec: "12.x · Linux / Windows"
  }, {
    name: "NIM Microservices",
    cat: "Software",
    tag: "Cloud",
    tone: "info",
    desc: "Optimized inference microservices for deploying AI models anywhere.",
    spec: "OpenAI-compatible API"
  }];
  const cats = ["All", "Data Center", "Gaming", "Edge", "Workstation", "Software"];
  const shown = cat === "All" ? products : products.filter(p => p.cat === cat);
  return /*#__PURE__*/React.createElement("section", {
    style: showStyles.wrap
  }, /*#__PURE__*/React.createElement("div", {
    style: showStyles.inner
  }, /*#__PURE__*/React.createElement("div", {
    style: showStyles.head
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    style: showStyles.eyebrow
  }, "Products"), /*#__PURE__*/React.createElement("h2", {
    style: showStyles.title
  }, "Built for Every Workload")), /*#__PURE__*/React.createElement(Button, {
    variant: "secondary",
    iconRight: /*#__PURE__*/React.createElement(Icon, {
      name: "ArrowRight",
      size: 15
    })
  }, "View all products")), /*#__PURE__*/React.createElement("div", {
    style: {
      marginBottom: 26
    }
  }, /*#__PURE__*/React.createElement(Tabs, {
    items: cats,
    value: cat,
    onChange: setCat
  })), /*#__PURE__*/React.createElement("div", {
    style: showStyles.grid
  }, shown.map(p => /*#__PURE__*/React.createElement(Card, {
    key: p.name,
    interactive: true,
    accent: p.tone === "green",
    padding: "22px"
  }, /*#__PURE__*/React.createElement("div", {
    style: showStyles.cardTop
  }, /*#__PURE__*/React.createElement(Badge, {
    tone: p.tone,
    variant: p.tone === "green" ? "solid" : "soft"
  }, p.tag), /*#__PURE__*/React.createElement("span", {
    style: showStyles.cat
  }, p.cat)), /*#__PURE__*/React.createElement("h3", {
    style: showStyles.cardTitle
  }, p.name), /*#__PURE__*/React.createElement("p", {
    style: showStyles.cardDesc
  }, p.desc), /*#__PURE__*/React.createElement("div", {
    style: showStyles.spec
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "Cpu",
    size: 14,
    color: "var(--nv-green-700)"
  }), /*#__PURE__*/React.createElement("span", null, p.spec)), /*#__PURE__*/React.createElement("a", {
    href: "#",
    style: showStyles.link,
    onClick: e => e.preventDefault()
  }, "Learn more ", /*#__PURE__*/React.createElement(Icon, {
    name: "ArrowRight",
    size: 14,
    color: "var(--nv-green-700)"
  })))))));
}
const showStyles = {
  wrap: {
    background: "var(--surface-subtle)",
    borderTop: "1px solid var(--border-subtle)"
  },
  inner: {
    maxWidth: 1280,
    margin: "0 auto",
    padding: "72px 28px 88px"
  },
  head: {
    display: "flex",
    alignItems: "flex-end",
    justifyContent: "space-between",
    marginBottom: 30
  },
  eyebrow: {
    fontSize: "var(--fs-2xs)",
    fontWeight: "var(--fw-semibold)",
    textTransform: "uppercase",
    letterSpacing: "var(--ls-caps)",
    color: "var(--text-accent)"
  },
  title: {
    fontFamily: "var(--font-display)",
    fontWeight: "var(--fw-medium)",
    fontSize: "var(--fs-2xl)",
    letterSpacing: "var(--ls-snug)",
    margin: "8px 0 0"
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(3, 1fr)",
    gap: 20
  },
  cardTop: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 14
  },
  cat: {
    fontSize: "var(--fs-2xs)",
    color: "var(--text-tertiary)",
    fontWeight: "var(--fw-medium)"
  },
  cardTitle: {
    fontSize: "var(--fs-lg)",
    fontWeight: "var(--fw-semibold)",
    margin: "0 0 8px"
  },
  cardDesc: {
    fontSize: "var(--fs-sm)",
    lineHeight: 1.5,
    color: "var(--text-secondary)",
    margin: "0 0 16px",
    minHeight: 42
  },
  spec: {
    display: "flex",
    alignItems: "center",
    gap: 7,
    fontSize: "var(--fs-xs)",
    fontFamily: "var(--font-mono)",
    color: "var(--text-secondary)",
    paddingBottom: 16,
    borderBottom: "1px solid var(--border-subtle)",
    marginBottom: 14
  },
  link: {
    display: "inline-flex",
    alignItems: "center",
    gap: 6,
    fontSize: "var(--fs-sm)",
    fontWeight: "var(--fw-semibold)",
    color: "var(--nv-green-700)",
    textDecoration: "none"
  }
};
window.Showcase = Showcase;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/web/Showcase.jsx", error: String((e && e.message) || e) }); }

// ui_kits/web/UiIcon.jsx
try { (() => {
// Lucide icon helper — NVIDIA uses a thin, consistent line-icon style; Lucide
// (≈1.75px stroke, rounded) is the closest CDN match. Documented in the README.
// Builds an inline SVG from the Lucide icon node (["svg", attrs, children]).
function lucideSvg(name, size, strokeWidth) {
  if (!window.lucide || !lucide.icons || !lucide.icons[name]) return "";
  const [, attrs, children] = lucide.icons[name];
  const merged = {
    ...attrs,
    width: size,
    height: size,
    "stroke-width": strokeWidth
  };
  const attrStr = Object.entries(merged).map(([k, v]) => `${k}="${v}"`).join(" ");
  const inner = (children || []).map(([tag, a]) => `<${tag} ${Object.entries(a).map(([k, v]) => `${k}="${v}"`).join(" ")}/>`).join("");
  return `<svg ${attrStr}>${inner}</svg>`;
}

// Named UiIcon to avoid colliding with the design system's official <Icon>.
function UiIcon({
  name,
  size = 18,
  color = "currentColor",
  strokeWidth = 1.75,
  style
}) {
  const ref = React.useRef(null);
  React.useEffect(() => {
    if (ref.current) ref.current.innerHTML = lucideSvg(name, size, strokeWidth);
  }, [name, size, strokeWidth]);
  return /*#__PURE__*/React.createElement("span", {
    ref: ref,
    style: {
      display: "inline-flex",
      color,
      ...style
    }
  });
}
window.UiIcon = UiIcon;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/web/UiIcon.jsx", error: String((e && e.message) || e) }); }

__ds_ns.Logo = __ds_scope.Logo;

__ds_ns.Button = __ds_scope.Button;

__ds_ns.IconButton = __ds_scope.IconButton;

__ds_ns.Avatar = __ds_scope.Avatar;

__ds_ns.Badge = __ds_scope.Badge;

__ds_ns.Card = __ds_scope.Card;

__ds_ns.Stat = __ds_scope.Stat;

__ds_ns.Tag = __ds_scope.Tag;

__ds_ns.Banner = __ds_scope.Banner;

__ds_ns.Spinner = __ds_scope.Spinner;

__ds_ns.Checkbox = __ds_scope.Checkbox;

__ds_ns.Input = __ds_scope.Input;

__ds_ns.Select = __ds_scope.Select;

__ds_ns.Switch = __ds_scope.Switch;

__ds_ns.Icon = __ds_scope.Icon;

__ds_ns.Breadcrumb = __ds_scope.Breadcrumb;

__ds_ns.Tabs = __ds_scope.Tabs;

__ds_ns.Tooltip = __ds_scope.Tooltip;

})();
