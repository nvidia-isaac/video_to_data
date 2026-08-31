/* @ds-bundle: {"format":3,"namespace":"CohereDesignSystem_b87497","components":[{"name":"Badge","sourcePath":"components/core/Badge.jsx"},{"name":"Button","sourcePath":"components/core/Button.jsx"},{"name":"Card","sourcePath":"components/core/Card.jsx"},{"name":"CardBullet","sourcePath":"components/core/Card.jsx"},{"name":"Chip","sourcePath":"components/core/Chip.jsx"},{"name":"MonoLabel","sourcePath":"components/core/MonoLabel.jsx"},{"name":"Input","sourcePath":"components/forms/Input.jsx"},{"name":"Select","sourcePath":"components/forms/Select.jsx"},{"name":"AnnouncementBar","sourcePath":"components/navigation/AnnouncementBar.jsx"},{"name":"TopNav","sourcePath":"components/navigation/TopNav.jsx"}],"sourceHashes":{"components/core/Badge.jsx":"644e4baa7895","components/core/Button.jsx":"8714999705e1","components/core/Card.jsx":"a35bbb3bda16","components/core/Chip.jsx":"3c9085cb0907","components/core/MonoLabel.jsx":"0fbf00ac0cdd","components/forms/Input.jsx":"6cea775b17ad","components/forms/Select.jsx":"49adcee16b94","components/navigation/AnnouncementBar.jsx":"3dc55e84bc44","components/navigation/TopNav.jsx":"18f885eff875","ui_kits/marketing/Blog.jsx":"130736853942","ui_kits/marketing/Contact.jsx":"206b0b6b823e","ui_kits/marketing/Home.jsx":"87ad976e1897","ui_kits/marketing/Icon.jsx":"03f83bf55946","ui_kits/marketing/Research.jsx":"57c630cc2b26"},"inlinedExternals":[],"unexposedExports":[]} */

(() => {

const __ds_ns = (window.CohereDesignSystem_b87497 = window.CohereDesignSystem_b87497 || {});

const __ds_scope = {};

(__ds_ns.__errors = __ds_ns.__errors || []);

// components/core/Badge.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * Small status chip — used inside agent-console mockups (running / live /
 * connected) and as compact state markers. Quiet by default; tones add a dot.
 */
function Badge({
  children,
  tone = 'neutral',
  dot = false,
  style,
  ...rest
}) {
  const tones = {
    neutral: {
      color: 'var(--text-tertiary)',
      border: 'var(--border-rule)',
      dotColor: 'var(--color-muted)'
    },
    success: {
      color: 'var(--color-green-deep)',
      border: 'color-mix(in srgb, var(--color-green-deep) 30%, white)',
      dotColor: 'var(--color-green-deep)'
    },
    accent: {
      color: 'var(--color-near-black)',
      border: 'var(--accent-editorial)',
      dotColor: 'var(--accent-editorial)'
    },
    info: {
      color: 'var(--color-blue-action)',
      border: 'color-mix(in srgb, var(--color-blue-action) 30%, white)',
      dotColor: 'var(--color-blue-action)'
    },
    onDark: {
      color: 'rgba(255,255,255,0.9)',
      border: 'var(--border-on-dark)',
      dotColor: 'var(--accent-editorial)'
    }
  };
  const t = tones[tone] || tones.neutral;
  return /*#__PURE__*/React.createElement("span", _extends({
    style: {
      display: 'inline-flex',
      alignItems: 'center',
      gap: '6px',
      fontFamily: 'var(--font-mono)',
      fontSize: '12px',
      letterSpacing: '0.28px',
      textTransform: 'uppercase',
      padding: '4px 10px',
      borderRadius: 'var(--radius-full)',
      border: `1px solid ${t.border}`,
      color: t.color,
      background: 'transparent',
      ...style
    }
  }, rest), dot && /*#__PURE__*/React.createElement("span", {
    style: {
      width: '6px',
      height: '6px',
      borderRadius: '9999px',
      background: t.dotColor,
      flex: '0 0 auto'
    }
  }), children);
}
Object.assign(__ds_scope, { Badge });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/core/Badge.jsx", error: String((e && e.message) || e) }); }

// components/core/Button.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * Cohere Button.
 * Primary actions are near-black (or white on dark) pills. Secondary actions
 * are underlined text links. Outline pills are used for taxonomy / filters.
 */
function Button({
  variant = 'primary',
  size = 'md',
  onDark = false,
  href,
  children,
  iconRight,
  iconLeft,
  disabled = false,
  style,
  ...rest
}) {
  const sizes = {
    sm: {
      padding: '8px 18px',
      fontSize: '14px'
    },
    md: {
      padding: '12px 24px',
      fontSize: '14px'
    },
    lg: {
      padding: '14px 28px',
      fontSize: '16px'
    }
  };
  const base = {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '8px',
    fontFamily: 'var(--font-body)',
    fontWeight: 'var(--weight-medium)',
    lineHeight: 1.2,
    border: '1px solid transparent',
    cursor: disabled ? 'not-allowed' : 'pointer',
    textDecoration: 'none',
    transition: 'opacity var(--duration-base) var(--ease-standard), background var(--duration-base) var(--ease-standard), color var(--duration-base) var(--ease-standard)',
    opacity: disabled ? 0.4 : 1,
    whiteSpace: 'nowrap',
    ...sizes[size]
  };
  const variants = {
    primary: {
      background: onDark ? 'var(--action-on-dark-bg)' : 'var(--action-primary-bg)',
      color: onDark ? 'var(--action-on-dark-fg)' : 'var(--action-primary-fg)',
      borderRadius: 'var(--radius-pill)'
    },
    secondary: {
      background: 'transparent',
      color: onDark ? 'var(--text-on-dark)' : 'var(--text-primary)',
      borderRadius: 0,
      padding: '4px 0',
      textDecoration: 'underline',
      textUnderlineOffset: '4px',
      textDecorationThickness: '1px'
    },
    outline: {
      background: 'transparent',
      color: onDark ? 'var(--text-on-dark)' : 'var(--text-primary)',
      border: `1px solid ${onDark ? 'var(--border-on-dark)' : 'var(--text-primary)'}`,
      borderRadius: 'var(--radius-xl)'
    }
  };
  const Tag = href ? 'a' : 'button';
  return /*#__PURE__*/React.createElement(Tag, _extends({
    href: href,
    disabled: href ? undefined : disabled,
    style: {
      ...base,
      ...variants[variant],
      ...style
    },
    onMouseEnter: e => {
      if (!disabled) e.currentTarget.style.opacity = '0.78';
    },
    onMouseLeave: e => {
      if (!disabled) e.currentTarget.style.opacity = '1';
    }
  }, rest), iconLeft, children, iconRight);
}
Object.assign(__ds_scope, { Button });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/core/Button.jsx", error: String((e && e.message) || e) }); }

// components/core/Card.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * Cohere surface card. Flat — depth comes from surface tone, rounded corners
 * and thin borders, never drop shadows.
 *   warm  = stone product/model card
 *   plain = white card with hairline border
 *   dark  = translucent surface inside a dark green/navy band
 */
function Card({
  variant = 'plain',
  radius = 'md',
  children,
  style,
  ...rest
}) {
  const radii = {
    sm: 'var(--radius-sm)',
    md: 'var(--radius-md)',
    lg: 'var(--radius-lg)'
  };
  const variants = {
    warm: {
      background: 'var(--surface-card-warm)',
      border: '1px solid transparent',
      color: 'var(--text-primary)'
    },
    plain: {
      background: 'var(--surface-card)',
      border: '1px solid var(--border-rule)',
      color: 'var(--text-primary)'
    },
    dark: {
      background: 'rgba(255,255,255,0.04)',
      border: '1px solid var(--border-on-dark)',
      color: 'var(--text-on-dark)'
    }
  };
  return /*#__PURE__*/React.createElement("div", _extends({
    style: {
      borderRadius: radii[radius] || radii.md,
      padding: 'var(--space-32)',
      ...variants[variant],
      ...style
    }
  }, rest), children);
}

/** Checkmark bullet row for product/capability cards. */
function CardBullet({
  children,
  onDark = false
}) {
  return /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      gap: '12px',
      alignItems: 'flex-start',
      padding: '8px 0'
    }
  }, /*#__PURE__*/React.createElement("span", {
    "aria-hidden": "true",
    style: {
      flex: '0 0 auto',
      marginTop: '3px',
      width: '16px',
      height: '16px',
      display: 'inline-flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: onDark ? 'var(--accent-editorial)' : 'var(--color-green-deep)',
      fontFamily: 'var(--font-body)',
      fontSize: '15px',
      lineHeight: 1
    }
  }, "\u2713"), /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-body)',
      fontSize: 'var(--type-body-size)',
      lineHeight: 'var(--type-body-lh)',
      color: onDark ? 'rgba(255,255,255,0.82)' : 'var(--text-primary)'
    }
  }, children));
}
Object.assign(__ds_scope, { Card, CardBullet });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/core/Card.jsx", error: String((e && e.message) || e) }); }

// components/core/Chip.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * Cohere editorial chip. The blog index uses oversized coral taxonomy chips:
 * active inverts to coral fill with dark text; inactive uses coral outline on
 * a pale fill. `size="lg"` is the hero-level blog control; `sm` is a label.
 */
function Chip({
  children,
  active = false,
  size = 'md',
  onClick,
  style,
  ...rest
}) {
  const sizes = {
    sm: {
      padding: '4px 12px',
      fontSize: '14px',
      borderRadius: 'var(--radius-sm)'
    },
    md: {
      padding: '8px 18px',
      fontSize: '16px',
      borderRadius: 'var(--radius-xl)'
    },
    lg: {
      padding: '12px 24px',
      fontSize: '18px',
      borderRadius: 'var(--radius-xl)'
    }
  };
  const base = {
    display: 'inline-flex',
    alignItems: 'center',
    fontFamily: 'var(--font-body)',
    fontWeight: 'var(--weight-regular)',
    lineHeight: 1.2,
    cursor: onClick ? 'pointer' : 'default',
    transition: 'background var(--duration-base) var(--ease-standard), color var(--duration-base) var(--ease-standard)',
    border: '1px solid var(--accent-editorial)',
    ...sizes[size]
  };
  const tone = active ? {
    background: 'var(--accent-editorial)',
    color: 'var(--color-near-black)'
  } : {
    background: 'color-mix(in srgb, var(--accent-editorial) 8%, white)',
    color: 'var(--color-near-black)'
  };
  return /*#__PURE__*/React.createElement("button", _extends({
    type: "button",
    onClick: onClick,
    style: {
      ...base,
      ...tone,
      ...style
    }
  }, rest), children);
}
Object.assign(__ds_scope, { Chip });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/core/Chip.jsx", error: String((e && e.message) || e) }); }

// components/core/MonoLabel.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * Uppercase technical label rendered in CohereMono. Used as a category /
 * system marker, especially on product and research pages.
 */
function MonoLabel({
  children,
  color,
  style,
  ...rest
}) {
  return /*#__PURE__*/React.createElement("span", _extends({
    style: {
      fontFamily: 'var(--font-mono)',
      fontSize: 'var(--type-mono-size)',
      lineHeight: 'var(--type-mono-lh)',
      letterSpacing: 'var(--type-mono-ls)',
      textTransform: 'uppercase',
      color: color || 'var(--text-tertiary)',
      ...style
    }
  }, rest), children);
}
Object.assign(__ds_scope, { MonoLabel });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/core/MonoLabel.jsx", error: String((e && e.message) || e) }); }

// components/forms/Input.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * Cohere text input. Rectangular field, thin gray border, compact label.
 * Used inside contact-form cards and the footer newsletter line.
 */
function Input({
  label,
  id,
  type = 'text',
  placeholder,
  value,
  onChange,
  required = false,
  style,
  ...rest
}) {
  const inputId = id || (label ? label.toLowerCase().replace(/\s+/g, '-') : undefined);
  return /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      flexDirection: 'column',
      gap: '8px'
    }
  }, label && /*#__PURE__*/React.createElement("label", {
    htmlFor: inputId,
    style: {
      fontFamily: 'var(--font-body)',
      fontSize: '14px',
      color: 'var(--text-primary)'
    }
  }, label, required && /*#__PURE__*/React.createElement("span", {
    style: {
      color: 'var(--text-muted)'
    }
  }, " *")), /*#__PURE__*/React.createElement("input", _extends({
    id: inputId,
    type: type,
    placeholder: placeholder,
    value: value,
    onChange: onChange,
    required: required,
    style: {
      fontFamily: 'var(--font-body)',
      fontSize: 'var(--type-body-size)',
      color: 'var(--text-primary)',
      background: 'var(--color-white)',
      padding: '12px 16px',
      border: '1px solid var(--border-utility)',
      borderRadius: 'var(--radius-xs)',
      outline: 'none',
      transition: 'border-color var(--duration-base) var(--ease-standard)',
      ...style
    },
    onFocus: e => {
      e.currentTarget.style.borderColor = 'var(--color-focus-violet)';
    },
    onBlur: e => {
      e.currentTarget.style.borderColor = 'var(--border-utility)';
    }
  }, rest)));
}
Object.assign(__ds_scope, { Input });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/Input.jsx", error: String((e && e.message) || e) }); }

// components/forms/Select.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * Cohere select. Same rectangular field language as Input, with a thin
 * chevron. Used in contact forms (region, team size, use case).
 */
function Select({
  label,
  id,
  value,
  onChange,
  children,
  required = false,
  style,
  ...rest
}) {
  const selectId = id || (label ? label.toLowerCase().replace(/\s+/g, '-') : undefined);
  return /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      flexDirection: 'column',
      gap: '8px'
    }
  }, label && /*#__PURE__*/React.createElement("label", {
    htmlFor: selectId,
    style: {
      fontFamily: 'var(--font-body)',
      fontSize: '14px',
      color: 'var(--text-primary)'
    }
  }, label, required && /*#__PURE__*/React.createElement("span", {
    style: {
      color: 'var(--text-muted)'
    }
  }, " *")), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'relative'
    }
  }, /*#__PURE__*/React.createElement("select", _extends({
    id: selectId,
    value: value,
    onChange: onChange,
    required: required,
    style: {
      appearance: 'none',
      width: '100%',
      fontFamily: 'var(--font-body)',
      fontSize: 'var(--type-body-size)',
      color: 'var(--text-primary)',
      background: 'var(--color-white)',
      padding: '12px 40px 12px 16px',
      border: '1px solid var(--border-utility)',
      borderRadius: 'var(--radius-xs)',
      outline: 'none',
      cursor: 'pointer',
      ...style
    },
    onFocus: e => {
      e.currentTarget.style.borderColor = 'var(--color-focus-violet)';
    },
    onBlur: e => {
      e.currentTarget.style.borderColor = 'var(--border-utility)';
    }
  }, rest), children), /*#__PURE__*/React.createElement("span", {
    "aria-hidden": "true",
    style: {
      position: 'absolute',
      right: '16px',
      top: '50%',
      transform: 'translateY(-50%)',
      pointerEvents: 'none',
      color: 'var(--text-tertiary)',
      fontSize: '12px'
    }
  }, "\u25BE")));
}
Object.assign(__ds_scope, { Select });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/Select.jsx", error: String((e && e.message) || e) }); }

// components/navigation/AnnouncementBar.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * Full-width black strip above the nav. 36px tall, centered microcopy with an
 * underlined "Learn more" link and a close control at the far right.
 */
function AnnouncementBar({
  children,
  linkLabel = 'Learn more',
  href = '#',
  onClose,
  style,
  ...rest
}) {
  const [open, setOpen] = React.useState(true);
  if (!open) return null;
  return /*#__PURE__*/React.createElement("div", _extends({
    style: {
      minHeight: '36px',
      background: 'var(--surface-announcement)',
      color: 'var(--color-white)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      position: 'relative',
      padding: '8px 44px',
      fontFamily: 'var(--font-body)',
      fontSize: 'var(--type-caption-size)',
      ...style
    }
  }, rest), /*#__PURE__*/React.createElement("span", {
    style: {
      textAlign: 'center'
    }
  }, children, linkLabel && /*#__PURE__*/React.createElement("a", {
    href: href,
    style: {
      color: 'var(--color-white)',
      textDecoration: 'underline',
      textUnderlineOffset: '3px',
      marginLeft: '8px'
    }
  }, linkLabel)), /*#__PURE__*/React.createElement("button", {
    type: "button",
    "aria-label": "Dismiss",
    onClick: () => {
      setOpen(false);
      onClose && onClose();
    },
    style: {
      position: 'absolute',
      right: '16px',
      top: '50%',
      transform: 'translateY(-50%)',
      background: 'transparent',
      border: 'none',
      color: 'var(--color-white)',
      cursor: 'pointer',
      fontSize: '16px',
      lineHeight: 1,
      padding: '4px',
      opacity: 0.8
    }
  }, "\xD7"));
}
Object.assign(__ds_scope, { AnnouncementBar });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/navigation/AnnouncementBar.jsx", error: String((e && e.message) || e) }); }

// components/navigation/TopNav.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * Cohere global nav. Three-zone layout: wordmark left, centered menu, and
 * sign-in + CTA right. Sits directly under the AnnouncementBar.
 */
function TopNav({
  links = ['Products', 'Solutions', 'Research', 'Resources', 'Company'],
  onDark = false,
  ctaLabel = 'Request a demo',
  style,
  ...rest
}) {
  const fg = onDark ? 'var(--text-on-dark)' : 'var(--text-primary)';
  return /*#__PURE__*/React.createElement("nav", _extends({
    style: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      gap: '24px',
      padding: '16px 24px',
      background: onDark ? 'transparent' : 'var(--surface-page)',
      borderBottom: onDark ? '1px solid var(--border-on-dark)' : '1px solid var(--border-card)',
      fontFamily: 'var(--font-body)',
      ...style
    }
  }, rest), /*#__PURE__*/React.createElement("a", {
    href: "#",
    style: {
      fontFamily: 'var(--font-display)',
      fontSize: '22px',
      letterSpacing: '-0.5px',
      color: fg,
      textDecoration: 'none',
      flex: '0 0 auto'
    }
  }, "cohere"), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      gap: '28px',
      alignItems: 'center',
      flex: '1 1 auto',
      justifyContent: 'center'
    }
  }, links.map(l => /*#__PURE__*/React.createElement("a", {
    key: l,
    href: "#",
    style: {
      fontSize: 'var(--type-body-size)',
      color: fg,
      textDecoration: 'none',
      opacity: 0.92
    }
  }, l))), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      gap: '20px',
      alignItems: 'center',
      flex: '0 0 auto'
    }
  }, /*#__PURE__*/React.createElement("a", {
    href: "#",
    style: {
      fontSize: 'var(--type-body-size)',
      color: fg,
      textDecoration: 'none'
    }
  }, "Sign in"), /*#__PURE__*/React.createElement(__ds_scope.Button, {
    variant: "primary",
    size: "sm",
    onDark: onDark
  }, ctaLabel)));
}
Object.assign(__ds_scope, { TopNav });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/navigation/TopNav.jsx", error: String((e && e.message) || e) }); }

// ui_kits/marketing/Blog.jsx
try { (() => {
/* global React, CohereDesignSystem_b87497, Icon */

/**
 * Blog index — oversized coral taxonomy chips as a hero-level control,
 * a search field, and a grid of rounded article cards.
 */
function Blog() {
  const {
    Chip,
    MonoLabel
  } = window.CohereDesignSystem_b87497;
  const cats = ['All', 'Product', 'Research', 'Customers', 'Company'];
  const [active, setActive] = React.useState('All');
  const posts = [{
    cat: 'Product',
    title: 'Introducing Command A: our most capable enterprise model',
    read: '6 min read',
    tone: 'var(--color-stone)'
  }, {
    cat: 'Research',
    title: 'Grounded generation: reducing hallucination with verifiable citations',
    read: '9 min read',
    tone: 'var(--color-wash-blue)'
  }, {
    cat: 'Customers',
    title: 'How a global bank deployed private AI agents in 90 days',
    read: '5 min read',
    tone: 'var(--color-wash-green)'
  }, {
    cat: 'Company',
    title: 'Security at the foundation: our approach to enterprise trust',
    read: '4 min read',
    tone: 'var(--color-stone)'
  }, {
    cat: 'Research',
    title: 'Multilingual retrieval at scale across 100+ languages',
    read: '8 min read',
    tone: 'var(--color-wash-blue)'
  }, {
    cat: 'Product',
    title: 'North: the secure workspace for AI agents',
    read: '7 min read',
    tone: 'var(--color-wash-green)'
  }];
  const shown = active === 'All' ? posts : posts.filter(p => p.cat === active);
  return /*#__PURE__*/React.createElement("div", {
    style: {
      maxWidth: 1180,
      margin: '0 auto',
      padding: '64px 24px 110px'
    }
  }, /*#__PURE__*/React.createElement(MonoLabel, {
    color: "var(--accent-editorial)"
  }, "The Cohere Blog"), /*#__PURE__*/React.createElement("h1", {
    style: {
      fontFamily: 'var(--font-display)',
      fontWeight: 400,
      fontSize: 'clamp(40px,5vw,64px)',
      lineHeight: 1.0,
      letterSpacing: '-1.2px',
      margin: '12px 0 36px'
    }
  }, "Ideas, research & product"), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'relative',
      maxWidth: 360,
      marginBottom: 28
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      position: 'absolute',
      left: 14,
      top: '50%',
      transform: 'translateY(-50%)'
    }
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "search",
    size: 16,
    color: "var(--text-tertiary)"
  })), /*#__PURE__*/React.createElement("input", {
    placeholder: "Search articles",
    style: {
      width: '100%',
      boxSizing: 'border-box',
      fontFamily: 'var(--font-body)',
      fontSize: 16,
      padding: '12px 16px 12px 40px',
      border: '1px solid var(--border-utility)',
      borderRadius: 'var(--radius-xs)',
      outline: 'none'
    }
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      gap: 12,
      flexWrap: 'wrap',
      marginBottom: 48
    }
  }, cats.map(c => /*#__PURE__*/React.createElement(Chip, {
    key: c,
    size: "lg",
    active: active === c,
    onClick: () => setActive(c)
  }, c))), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'grid',
      gridTemplateColumns: 'repeat(3, 1fr)',
      gap: 28
    }
  }, shown.map(p => /*#__PURE__*/React.createElement("article", {
    key: p.title,
    style: {
      cursor: 'pointer'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      background: p.tone,
      borderRadius: 'var(--radius-sm)',
      height: 180,
      marginBottom: 16
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'inline-flex',
      border: '1px solid var(--accent-editorial)',
      color: 'var(--color-near-black)',
      background: 'color-mix(in srgb, var(--accent-editorial) 8%, white)',
      borderRadius: 'var(--radius-sm)',
      padding: '2px 10px',
      fontSize: 13,
      marginBottom: 10
    }
  }, p.cat), /*#__PURE__*/React.createElement("h3", {
    style: {
      fontFamily: 'var(--font-body)',
      fontWeight: 400,
      fontSize: 22,
      lineHeight: 1.25,
      letterSpacing: '-0.2px',
      margin: '0 0 10px'
    }
  }, p.title), /*#__PURE__*/React.createElement("p", {
    style: {
      fontSize: 14,
      color: 'var(--text-muted)',
      margin: 0
    }
  }, p.read)))));
}
Object.assign(window, {
  Blog
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/marketing/Blog.jsx", error: String((e && e.message) || e) }); }

// ui_kits/marketing/Contact.jsx
try { (() => {
/* global React, CohereDesignSystem_b87497, Icon */

/**
 * Contact — a rounded white form card on a deep-green band, followed by the
 * dark footer newsletter block.
 */
function Contact() {
  const {
    Input,
    Select,
    Button,
    MonoLabel
  } = window.CohereDesignSystem_b87497;
  return /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("section", {
    style: {
      background: 'var(--color-green-deep)',
      padding: '96px 24px'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      maxWidth: 920,
      margin: '0 auto',
      display: 'grid',
      gridTemplateColumns: '1fr 1.1fr',
      gap: 56,
      alignItems: 'start'
    }
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement(MonoLabel, {
    color: "var(--accent-editorial)"
  }, "Get in touch"), /*#__PURE__*/React.createElement("h1", {
    style: {
      fontFamily: 'var(--font-body)',
      fontWeight: 400,
      fontSize: 'clamp(32px,4vw,48px)',
      lineHeight: 1.05,
      letterSpacing: '-0.48px',
      color: '#fff',
      margin: '12px 0 16px'
    }
  }, "Talk to our enterprise team"), /*#__PURE__*/React.createElement("p", {
    style: {
      fontSize: 18,
      lineHeight: 1.4,
      color: 'rgba(255,255,255,0.78)',
      margin: 0,
      maxWidth: 320
    }
  }, "See how Cohere deploys secure, private AI inside your environment.")), /*#__PURE__*/React.createElement("div", {
    style: {
      background: '#fff',
      borderRadius: 'var(--radius-md)',
      padding: 32
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gap: 20
    }
  }, /*#__PURE__*/React.createElement(Input, {
    label: "First name",
    placeholder: "Ada"
  }), /*#__PURE__*/React.createElement(Input, {
    label: "Last name",
    placeholder: "Lovelace"
  }), /*#__PURE__*/React.createElement(Input, {
    label: "Work email",
    type: "email",
    placeholder: "you@company.com",
    required: true
  }), /*#__PURE__*/React.createElement(Input, {
    label: "Company",
    placeholder: "Acme Inc."
  }), /*#__PURE__*/React.createElement(Select, {
    label: "Team size",
    required: true
  }, /*#__PURE__*/React.createElement("option", null, "1\u201350"), /*#__PURE__*/React.createElement("option", null, "51\u2013500"), /*#__PURE__*/React.createElement("option", null, "500+")), /*#__PURE__*/React.createElement(Select, {
    label: "Use case"
  }, /*#__PURE__*/React.createElement("option", null, "Agents"), /*#__PURE__*/React.createElement("option", null, "Search & retrieval"), /*#__PURE__*/React.createElement("option", null, "Summarization"))), /*#__PURE__*/React.createElement("div", {
    style: {
      marginTop: 24
    }
  }, /*#__PURE__*/React.createElement(Button, {
    variant: "primary"
  }, "Request a demo"))))), /*#__PURE__*/React.createElement(Footer, null));
}

/** Dark footer with coral newsletter block + muted link columns. */
function Footer() {
  const {
    MonoLabel
  } = window.CohereDesignSystem_b87497;
  const cols = {
    Products: ['Command', 'Embed', 'Rerank', 'North'],
    Solutions: ['Financial services', 'Customer support', 'Search', 'Security'],
    Company: ['About', 'Careers', 'Research', 'Blog']
  };
  return /*#__PURE__*/React.createElement("footer", {
    style: {
      background: 'var(--color-near-black)',
      color: '#fff',
      padding: '72px 24px 48px'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      maxWidth: 1180,
      margin: '0 auto'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      justifyContent: 'space-between',
      gap: 40,
      flexWrap: 'wrap',
      borderBottom: '1px solid var(--border-on-dark)',
      paddingBottom: 48,
      marginBottom: 48
    }
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement(MonoLabel, {
    color: "var(--accent-editorial)"
  }, "AI moves fast"), /*#__PURE__*/React.createElement("h2", {
    style: {
      fontFamily: 'var(--font-body)',
      fontWeight: 400,
      fontSize: 32,
      lineHeight: 1.1,
      letterSpacing: '-0.32px',
      margin: '12px 0 0',
      maxWidth: 360
    }
  }, "Stay ahead with the Cohere newsletter")), /*#__PURE__*/React.createElement("div", {
    style: {
      minWidth: 320
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      alignItems: 'center',
      borderBottom: '1px solid rgba(255,255,255,0.4)',
      paddingBottom: 10
    }
  }, /*#__PURE__*/React.createElement("input", {
    placeholder: "Enter your email",
    style: {
      flex: 1,
      background: 'transparent',
      border: 'none',
      outline: 'none',
      color: '#fff',
      fontFamily: 'var(--font-body)',
      fontSize: 16
    }
  }), /*#__PURE__*/React.createElement(Icon, {
    name: "arrow-right",
    size: 18,
    color: "#fff"
  })), /*#__PURE__*/React.createElement("p", {
    style: {
      fontSize: 12,
      color: 'var(--text-muted)',
      margin: '12px 0 0',
      lineHeight: 1.4
    }
  }, "By subscribing you agree to our Privacy Policy."))), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'grid',
      gridTemplateColumns: 'repeat(4, 1fr)',
      gap: 32
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontFamily: 'var(--font-display)',
      fontSize: 26,
      letterSpacing: '-0.5px'
    }
  }, "cohere"), Object.entries(cols).map(([head, items]) => /*#__PURE__*/React.createElement("div", {
    key: head
  }, /*#__PURE__*/React.createElement("p", {
    style: {
      fontSize: 14,
      color: '#fff',
      margin: '0 0 16px'
    }
  }, head), items.map(i => /*#__PURE__*/React.createElement("a", {
    key: i,
    href: "#",
    style: {
      display: 'block',
      fontSize: 14,
      color: 'var(--text-muted)',
      textDecoration: 'none',
      marginBottom: 10
    }
  }, i)))))));
}
Object.assign(window, {
  Contact,
  Footer
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/marketing/Contact.jsx", error: String((e && e.message) || e) }); }

// ui_kits/marketing/Home.jsx
try { (() => {
/* global React, CohereDesignSystem_b87497, Icon */

/**
 * Home — centered hero declaration over a two-card media composition,
 * a quiet trust-logo strip, and a dark agent-console feature band.
 */
function Home() {
  const {
    Button,
    MonoLabel,
    Badge
  } = window.CohereDesignSystem_b87497;
  return /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("section", {
    style: {
      padding: '88px 24px 56px',
      maxWidth: 1100,
      margin: '0 auto',
      textAlign: 'center'
    }
  }, /*#__PURE__*/React.createElement("h1", {
    style: {
      fontFamily: 'var(--font-display)',
      fontWeight: 400,
      fontSize: 'clamp(48px, 7vw, 84px)',
      lineHeight: 1.0,
      letterSpacing: '-1.6px',
      margin: '0 auto 24px',
      maxWidth: 900,
      color: 'var(--text-primary)'
    }
  }, "The AI platform built for enterprise"), /*#__PURE__*/React.createElement("p", {
    style: {
      fontSize: 18,
      lineHeight: 1.4,
      color: 'var(--text-primary)',
      maxWidth: 600,
      margin: '0 auto 32px'
    }
  }, "Secure, private, and adaptable models and agents \u2014 deployed wherever your data lives."), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      gap: 20,
      justifyContent: 'center',
      alignItems: 'center',
      flexWrap: 'wrap'
    }
  }, /*#__PURE__*/React.createElement(Button, {
    variant: "primary",
    size: "lg"
  }, "Request a demo"), /*#__PURE__*/React.createElement(Button, {
    variant: "secondary",
    iconRight: /*#__PURE__*/React.createElement(Icon, {
      name: "arrow-right",
      size: 16
    })
  }, "Explore products"))), /*#__PURE__*/React.createElement("section", {
    style: {
      maxWidth: 1180,
      margin: '0 auto 96px',
      padding: '0 24px',
      display: 'grid',
      gridTemplateColumns: '1.5fr 1fr',
      gap: 20
    }
  }, /*#__PURE__*/React.createElement(AgentConsole, null), /*#__PURE__*/React.createElement("div", {
    style: {
      borderRadius: 'var(--radius-lg)',
      background: 'var(--color-stone)',
      minHeight: 360,
      display: 'flex',
      alignItems: 'flex-end',
      padding: 24
    }
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement(MonoLabel, null, "Enterprise"), /*#__PURE__*/React.createElement("p", {
    style: {
      fontFamily: 'var(--font-body)',
      fontSize: 24,
      lineHeight: 1.3,
      margin: '8px 0 0',
      maxWidth: 240
    }
  }, "Photography placeholder")))), /*#__PURE__*/React.createElement("section", {
    style: {
      padding: '0 24px 110px',
      textAlign: 'center'
    }
  }, /*#__PURE__*/React.createElement("p", {
    style: {
      fontSize: 16,
      color: 'var(--text-tertiary)',
      marginBottom: 40
    }
  }, "Trusted by the world's leading enterprises"), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      justifyContent: 'center',
      gap: 64,
      flexWrap: 'wrap',
      alignItems: 'center',
      opacity: 0.55
    }
  }, ['Notion', 'Oracle', 'Fujitsu', 'Bell', 'Dell', 'LG'].map(n => /*#__PURE__*/React.createElement("span", {
    key: n,
    style: {
      fontFamily: 'var(--font-display)',
      fontSize: 22,
      letterSpacing: '-0.5px',
      color: 'var(--color-near-black)'
    }
  }, n)))), /*#__PURE__*/React.createElement("section", {
    style: {
      background: 'var(--color-green-deep)',
      padding: '96px 24px'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      maxWidth: 1180,
      margin: '0 auto'
    }
  }, /*#__PURE__*/React.createElement(MonoLabel, {
    color: "var(--accent-editorial)"
  }, "Command"), /*#__PURE__*/React.createElement("h2", {
    style: {
      fontFamily: 'var(--font-body)',
      fontWeight: 400,
      fontSize: 'clamp(32px,4vw,48px)',
      lineHeight: 1.1,
      letterSpacing: '-0.48px',
      color: '#fff',
      margin: '12px 0 40px',
      maxWidth: 620
    }
  }, "Build agents that act securely across your enterprise"), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'grid',
      gridTemplateColumns: 'repeat(3, 1fr)',
      gap: 20
    }
  }, [{
    icon: 'shield-check',
    t: 'Private deployment',
    b: 'Run in your VPC, on-prem, or air-gapped — your data never leaves.'
  }, {
    icon: 'plug',
    t: 'Connect any source',
    b: 'Native retrieval across the tools and documents your teams already use.'
  }, {
    icon: 'workflow',
    t: 'Agentic workflows',
    b: 'Multi-step reasoning and tool use, grounded in verifiable citations.'
  }].map(c => /*#__PURE__*/React.createElement("div", {
    key: c.t,
    style: {
      borderTop: '1px solid var(--border-on-dark)',
      paddingTop: 20
    }
  }, /*#__PURE__*/React.createElement(Icon, {
    name: c.icon,
    size: 28,
    color: "rgba(255,255,255,0.9)",
    strokeWidth: 1.25
  }), /*#__PURE__*/React.createElement("h3", {
    style: {
      fontFamily: 'var(--font-body)',
      fontWeight: 400,
      fontSize: 24,
      lineHeight: 1.3,
      color: '#fff',
      margin: '20px 0 10px'
    }
  }, c.t), /*#__PURE__*/React.createElement("p", {
    style: {
      fontSize: 16,
      lineHeight: 1.5,
      color: 'rgba(255,255,255,0.72)',
      margin: 0
    }
  }, c.b)))))));
}

/** Dark agent-console mockup card. */
function AgentConsole() {
  const {
    Badge,
    MonoLabel
  } = window.CohereDesignSystem_b87497;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      borderRadius: 'var(--radius-lg)',
      background: 'var(--color-near-black)',
      color: '#fff',
      padding: 24,
      minHeight: 360,
      display: 'flex',
      flexDirection: 'column',
      gap: 16
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      alignItems: 'center',
      gap: 10
    }
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "bot",
    size: 18,
    color: "var(--accent-editorial)"
  }), /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-body)',
      fontSize: 15
    }
  }, "Research Agent")), /*#__PURE__*/React.createElement(Badge, {
    tone: "onDark",
    dot: true
  }, "Running")), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      gap: 8,
      flexWrap: 'wrap'
    }
  }, /*#__PURE__*/React.createElement(Badge, {
    tone: "onDark"
  }, "Salesforce"), /*#__PURE__*/React.createElement(Badge, {
    tone: "onDark"
  }, "Confluence"), /*#__PURE__*/React.createElement(Badge, {
    tone: "onDark"
  }, "Snowflake")), /*#__PURE__*/React.createElement("div", {
    style: {
      background: 'rgba(255,255,255,0.05)',
      border: '1px solid var(--border-on-dark)',
      borderRadius: 'var(--radius-sm)',
      padding: 14,
      fontSize: 14,
      color: 'rgba(255,255,255,0.85)',
      lineHeight: 1.5
    }
  }, "Summarize Q3 pipeline risk across all enterprise accounts and cite sources."), /*#__PURE__*/React.createElement("div", {
    style: {
      background: 'rgba(255,255,255,0.03)',
      border: '1px solid var(--border-on-dark)',
      borderRadius: 'var(--radius-sm)',
      padding: 14,
      fontSize: 14,
      color: 'rgba(255,255,255,0.72)',
      lineHeight: 1.5,
      flex: 1
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      alignItems: 'center',
      gap: 8,
      marginBottom: 8
    }
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "sparkles",
    size: 14,
    color: "var(--accent-editorial)"
  }), /*#__PURE__*/React.createElement(MonoLabel, {
    color: "rgba(255,255,255,0.5)"
  }, "Response")), "Three accounts show elevated renewal risk driven by stalled deployments. Pipeline coverage is 2.4\xD7, down from 3.1\xD7 last quarter\u2026"));
}
Object.assign(window, {
  Home
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/marketing/Home.jsx", error: String((e && e.message) || e) }); }

// ui_kits/marketing/Icon.jsx
try { (() => {
/* global React, lucide */

/**
 * Thin-line geometric icon (Lucide) — matches Cohere's research/capability
 * illustration style. Renders a placeholder <i> that Lucide upgrades to SVG.
 */
function Icon({
  name,
  size = 20,
  color = 'currentColor',
  strokeWidth = 1.5,
  style
}) {
  const ref = React.useRef(null);
  React.useEffect(() => {
    if (window.lucide && ref.current) {
      window.lucide.createIcons({
        attrs: {
          'stroke-width': strokeWidth
        },
        nameAttr: 'data-lucide'
      });
    }
  });
  return /*#__PURE__*/React.createElement("i", {
    ref: ref,
    "data-lucide": name,
    style: {
      display: 'inline-flex',
      width: size,
      height: size,
      color,
      ...style
    }
  });
}
Object.assign(window, {
  Icon
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/marketing/Icon.jsx", error: String((e && e.message) || e) }); }

// ui_kits/marketing/Research.jsx
try { (() => {
/* global React, CohereDesignSystem_b87497, Icon */

/**
 * Research index — compact outlined filter pills above a rule-separated
 * publication list: title left, topic pills center, date right.
 */
function Research() {
  const {
    Button,
    MonoLabel
  } = window.CohereDesignSystem_b87497;
  const filters = ['All', 'LLMs', 'Retrieval', 'Safety', 'Multilingual', 'Efficiency', 'Agents'];
  const [active, setActive] = React.useState('All');
  const papers = [{
    title: 'Verifiable citations for grounded enterprise generation',
    topics: ['Retrieval', 'Safety'],
    date: 'Jun 2026'
  }, {
    title: 'Command A: scaling instruction-following for enterprise tasks',
    topics: ['LLMs'],
    date: 'May 2026'
  }, {
    title: 'Cross-lingual embeddings for 100+ language retrieval',
    topics: ['Multilingual', 'Retrieval'],
    date: 'Apr 2026'
  }, {
    title: 'Tool-use planning in long-horizon agent workflows',
    topics: ['Agents'],
    date: 'Mar 2026'
  }, {
    title: 'Quantization-aware training for private deployment',
    topics: ['Efficiency'],
    date: 'Feb 2026'
  }];
  const shown = active === 'All' ? papers : papers.filter(p => p.topics.includes(active));
  return /*#__PURE__*/React.createElement("div", {
    style: {
      maxWidth: 1080,
      margin: '0 auto',
      padding: '64px 24px 110px'
    }
  }, /*#__PURE__*/React.createElement(MonoLabel, null, "Research"), /*#__PURE__*/React.createElement("h1", {
    style: {
      fontFamily: 'var(--font-display)',
      fontWeight: 400,
      fontSize: 'clamp(40px,5vw,64px)',
      lineHeight: 1.0,
      letterSpacing: '-1.2px',
      margin: '12px 0 36px'
    }
  }, "Publications"), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      gap: 10,
      flexWrap: 'wrap',
      marginBottom: 16
    }
  }, filters.map(f => /*#__PURE__*/React.createElement("button", {
    key: f,
    type: "button",
    onClick: () => setActive(f),
    style: {
      fontFamily: 'var(--font-body)',
      fontSize: 14,
      padding: '6px 16px',
      borderRadius: 'var(--radius-xl)',
      cursor: 'pointer',
      border: '1px solid ' + (active === f ? 'var(--color-near-black)' : 'var(--border-rule)'),
      background: active === f ? 'var(--color-near-black)' : 'transparent',
      color: active === f ? '#fff' : 'var(--text-primary)'
    }
  }, f))), /*#__PURE__*/React.createElement("div", {
    style: {
      borderTop: '1px solid var(--border-rule)'
    }
  }, shown.map(p => /*#__PURE__*/React.createElement("div", {
    key: p.title,
    style: {
      display: 'grid',
      gridTemplateColumns: '1fr auto auto',
      gap: 24,
      alignItems: 'center',
      padding: '28px 0',
      borderBottom: '1px solid var(--border-rule)'
    }
  }, /*#__PURE__*/React.createElement("a", {
    href: "#",
    style: {
      fontFamily: 'var(--font-body)',
      fontSize: 20,
      lineHeight: 1.3,
      letterSpacing: '-0.2px',
      color: 'var(--text-primary)',
      textDecoration: 'none'
    }
  }, p.title), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      gap: 8
    }
  }, p.topics.map(t => /*#__PURE__*/React.createElement("span", {
    key: t,
    style: {
      fontFamily: 'var(--font-body)',
      fontSize: 13,
      padding: '4px 12px',
      borderRadius: 'var(--radius-xl)',
      border: '1px solid var(--border-rule)',
      color: 'var(--text-tertiary)'
    }
  }, t))), /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-mono)',
      fontSize: 13,
      letterSpacing: '0.28px',
      color: 'var(--text-muted)',
      whiteSpace: 'nowrap'
    }
  }, p.date)))));
}
Object.assign(window, {
  Research
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/marketing/Research.jsx", error: String((e && e.message) || e) }); }

__ds_ns.Badge = __ds_scope.Badge;

__ds_ns.Button = __ds_scope.Button;

__ds_ns.Card = __ds_scope.Card;

__ds_ns.CardBullet = __ds_scope.CardBullet;

__ds_ns.Chip = __ds_scope.Chip;

__ds_ns.MonoLabel = __ds_scope.MonoLabel;

__ds_ns.Input = __ds_scope.Input;

__ds_ns.Select = __ds_scope.Select;

__ds_ns.AnnouncementBar = __ds_scope.AnnouncementBar;

__ds_ns.TopNav = __ds_scope.TopNav;

})();
