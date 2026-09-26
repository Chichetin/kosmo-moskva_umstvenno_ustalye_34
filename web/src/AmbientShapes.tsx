import { useId } from "react";

/** Quiet geometric texture around the map; all motion stays in CSS. */
export function AmbientShapes({ analysis = false }: { analysis?: boolean }) {
  const prefix = useId().replaceAll(":", "");
  const pattern = (name: string) => `url(#${prefix}-${name})`;

  return (
    <svg
      className={`ambient-shapes${analysis ? " ambient-shapes-analysis" : ""}`}
      viewBox="0 0 1440 900"
      preserveAspectRatio="none"
      aria-hidden="true"
      focusable="false"
    >
      <defs>
        <pattern id={`${prefix}-dots`} width="12" height="12" patternUnits="userSpaceOnUse">
          <circle cx="6" cy="6" r="1.7" fill="#a7b5c4" />
        </pattern>
        <pattern id={`${prefix}-squares`} width="12" height="12" patternUnits="userSpaceOnUse">
          <rect x="4.2" y="4.2" width="3.6" height="3.6" fill="#a3afbd" />
        </pattern>
        <pattern id={`${prefix}-triangles`} width="12" height="12" patternUnits="userSpaceOnUse">
          <path d="M6 3.7 9.1 8.8H2.9Z" fill="#96a6b7" />
        </pattern>
        <radialGradient id={`${prefix}-left-top`} gradientUnits="userSpaceOnUse" cx="175" cy="190" r="520">
          <stop stopColor="white" stopOpacity="0.68" />
          <stop offset="0.58" stopColor="white" stopOpacity="0.35" />
          <stop offset="1" stopColor="white" stopOpacity="0" />
        </radialGradient>
        <radialGradient id={`${prefix}-left-bottom`} gradientUnits="userSpaceOnUse" cx="135" cy="770" r="455">
          <stop stopColor="white" stopOpacity="0.57" />
          <stop offset="0.55" stopColor="white" stopOpacity="0.27" />
          <stop offset="1" stopColor="white" stopOpacity="0" />
        </radialGradient>
        <radialGradient id={`${prefix}-right`} gradientUnits="userSpaceOnUse" cx="1280" cy="470" r="620">
          <stop stopColor="white" stopOpacity="0.65" />
          <stop offset="0.63" stopColor="white" stopOpacity="0.29" />
          <stop offset="1" stopColor="white" stopOpacity="0" />
        </radialGradient>
        <mask id={`${prefix}-mask-left-top`}>
          <rect width="1440" height="900" fill={pattern("left-top")} />
        </mask>
        <mask id={`${prefix}-mask-left-bottom`}>
          <rect width="1440" height="900" fill={pattern("left-bottom")} />
        </mask>
        <mask id={`${prefix}-mask-right`}>
          <rect width="1440" height="900" fill={pattern("right")} />
        </mask>
      </defs>
      <g className="ambient-shape-field ambient-shape-field-a">
        <rect width="1440" height="900" fill={pattern("squares")} mask={pattern("mask-left-top")} />
        <rect width="1440" height="900" fill={pattern("triangles")} mask={pattern("mask-left-bottom")} />
      </g>
      <g className="ambient-shape-field ambient-shape-field-b">
        <rect width="1440" height="900" fill={pattern("dots")} mask={pattern("mask-right")} />
      </g>
    </svg>
  );
}
