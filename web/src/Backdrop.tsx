import { useEffect, useRef } from "react";
import "./backdrop.css";

/**
 * Background of the globe screen (the analysis screen stays flat): graphite light pools drifting
 * slowly, a slight cursor parallax and a barely visible water caustic. Everything is one opaque
 * WebGL canvas at ⅓ resolution and at most 20 fps, so the compositor has a single layer to update.
 * Reduced motion or a CPU rasteriser gives a still frame; a hidden tab or `paused` (the map covers
 * the screen, or the analysis screen is open) renders nothing; no WebGL leaves the static CSS gradient.
 */
export default function Backdrop({ paused = false }: { paused?: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const pausedRef = useRef(paused);
  const sync = useRef<() => void>(() => {});

  useEffect(() => {
    const scene = startScene(canvas.current!, () => pausedRef.current);
    if (!scene) return;
    sync.current = scene.sync;
    return () => {
      sync.current = () => {};
      scene.stop();
    };
  }, []);

  useEffect(() => {
    pausedRef.current = paused;
    sync.current();
  }, [paused]);

  return (
    <div className="backdrop" aria-hidden="true">
      <canvas className="backdrop-canvas" ref={canvas} />
      <div className="backdrop-grain" />
    </div>
  );
}

const VERTEX = "attribute vec2 p;void main(){gl_Position=vec4(p,0.0,1.0);}";

// Pools: the former static radials (68% 38%, 12% 92%) plus a faint upper-left one, each drifting
// on 58–88 s cycles. Caustic: soft borders of two layers of drifting Worley cells bent by a slow
// two-octave warp, brighter toward the top; its terms repeat in 24–33 s. uShift is the parallax.
const FRAGMENT = `
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform vec2 uRes;
uniform float uTime;
uniform vec2 uShift;
vec2 hash2(vec2 p) {
  p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
  return fract(sin(p) * 43758.5453);
}
float border(vec2 p, float t) {
  vec2 cell = floor(p);
  vec2 f = fract(p);
  float d1 = 8.0;
  float d2 = 8.0;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      vec2 g = vec2(float(x), float(y));
      vec2 o = 0.5 + 0.4 * sin(t + 6.2831 * hash2(cell + g));
      float d = length(g + o - f);
      if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) { d2 = d; }
    }
  }
  return d2 - d1;
}
// Linear falloff like a CSS radial-gradient(closest-side, color, transparent); r in vmax.
float pool(vec2 px, vec2 center, vec2 r) {
  float vmax = max(uRes.x, uRes.y);
  return clamp(1.0 - length((px - center * uRes) / (r * vmax)), 0.0, 1.0);
}
void main() {
  vec2 px = gl_FragCoord.xy;
  vec2 uv = px / uRes;
  float t = uTime;
  vec3 col = mix(vec3(0.020, 0.024, 0.027), vec3(0.043, 0.051, 0.063), uv.y);
  vec2 b = vec2(0.12, 0.08) + vec2(0.07 * sin(t * 0.108), 0.05 * sin(t * 0.083 + 1.0)) + uShift * 0.5;
  col = mix(col, vec3(0.090, 0.106, 0.122), pool(px, b, vec2(0.39, 0.30) * (1.0 + 0.05 * sin(t * 0.07))));
  vec2 a = vec2(0.68, 0.62) + vec2(0.06 * sin(t * 0.085), 0.04 * sin(t * 0.071 + 2.0)) + uShift;
  col = mix(col, vec3(0.149, 0.169, 0.188), pool(px, a, vec2(0.52, 0.39) * (1.0 + 0.04 * sin(t * 0.06 + 1.0))));
  vec2 c = vec2(0.26, 0.82) + vec2(0.10 * sin(t * 0.071 + 3.0), 0.06 * sin(t * 0.058)) + uShift * 1.6;
  col = mix(col, vec3(0.118, 0.137, 0.153), 0.72 * pool(px, c, vec2(0.30, 0.22)));

  vec2 p = (px / uRes.y + uShift * 1.6) * 4.6;
  p += 0.55 * vec2(sin(p.y * 1.9 + t * 0.21), cos(p.x * 1.7 - t * 0.19));
  p += 0.15 * vec2(sin(p.y * 4.1 - t * 0.23), cos(p.x * 3.7 + t * 0.2));
  float ca = exp(-border(p, t * 0.26) * 16.0);
  float cb = exp(-border(p * 1.35 + 5.0, t * 0.22 + 2.0) * 20.0);
  float caustic = pow(ca * 0.9 + cb * 0.55, 1.5);
  vec2 d = (uv - vec2(0.62, 1.05)) * vec2(uRes.x / uRes.y, 1.0);
  caustic *= 0.35 + 0.65 * smoothstep(1.35, 0.1, length(d));
  col += vec3(0.075 * caustic);
  gl_FragColor = vec4(col, 1.0);
}`;

const FRAME_MS = 50; // ≤ 20 fps: every motion cycle is 24–88 s, and 30 fps cost ~40 % more CPU
const SLOW_MS = 20; // a GPU slower than this per frame gets a still image
const LATE_MS = 100; // frames arriving this late (< 10 fps) mean the page and the map are starved
const START_MS = 2500; // let the globe load its first tiles before the background starts moving
const PARALLAX = 0.012; // share of the screen a layer shifts at the pointer's extreme
// CPU rasterisers (VMs, no GPU): rendering competes with the map's tile workers, so a still frame.
const SOFTWARE = /swiftshader|llvmpipe|softpipe|software|basic render/i;

function startScene(canvas: HTMLCanvasElement, isPaused: () => boolean) {
  const gl = canvas.getContext("webgl", {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    powerPreference: "low-power",
  });
  const program = gl && link(gl);
  if (!gl || !program) {
    canvas.hidden = true;
    return null;
  }
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  gl.useProgram(program);
  const position = gl.getAttribLocation(program, "p");
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  const uRes = gl.getUniformLocation(program, "uRes");
  const uTime = gl.getUniformLocation(program, "uTime");
  const uShift = gl.getUniformLocation(program, "uShift");

  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const finePointer = matchMedia("(pointer: fine)");
  const info = gl.getExtension("WEBGL_debug_renderer_info");
  const renderer = String(gl.getParameter(info ? info.UNMASKED_RENDERER_WEBGL : gl.RENDERER));
  const start = performance.now();
  let raf = 0;
  let last = 0;
  let frozen = SOFTWARE.test(renderer);
  let ready = false;
  let measured = 0;
  let slow = 0;
  let late = 0;
  // parallax: pointer target and eased value, both in −1…1
  let tx = 0;
  let ty = 0;
  let sx = 0;
  let sy = 0;

  const draw = (now: number) => {
    const w = Math.max(1, Math.ceil(innerWidth / 3));
    const h = Math.max(1, Math.ceil(innerHeight / 3));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      gl.viewport(0, 0, w, h);
    }
    gl.uniform2f(uRes, w, h);
    gl.uniform1f(uTime, ((now - start) / 1000) % 3600);
    gl.uniform2f(uShift, -sx * PARALLAX, sy * PARALLAX);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  };
  const running = () => ready && !frozen && !reduced.matches && !document.hidden && !isPaused();
  const frame = (now: number) => {
    raf = 0;
    if (!running()) return;
    raf = requestAnimationFrame(frame);
    if (last && now - last < FRAME_MS) return;
    late = last && now - last > LATE_MS ? late + 1 : 0;
    last = now;
    sx += (tx - sx) * 0.08;
    sy += (ty - sy) * 0.08;
    const t0 = performance.now();
    draw(now);
    if (measured < 30) {
      gl.finish();
      measured += 1;
      if (performance.now() - t0 > SLOW_MS) slow += 1;
    }
    if (slow >= 5 || late >= 45) {
      frozen = true;
      canvas.dataset.state = "still";
    }
  };
  const sync = () => {
    if (running()) {
      if (!raf) {
        last = 0;
        late = 0;
        canvas.dataset.state = "live";
        raf = requestAnimationFrame(frame);
      }
    } else {
      cancelAnimationFrame(raf);
      raf = 0;
      canvas.dataset.state = frozen || reduced.matches ? "still" : ready ? "paused" : "starting";
    }
  };
  const move = (event: PointerEvent) => {
    if (!finePointer.matches) return;
    tx = (event.clientX / innerWidth) * 2 - 1;
    ty = (event.clientY / innerHeight) * 2 - 1;
  };
  const redraw = () => {
    if (!raf && !canvas.hidden) draw(performance.now());
  };
  const lost = (event: Event) => {
    event.preventDefault();
    frozen = true;
    canvas.hidden = true;
  };
  canvas.dataset.state = "still";
  draw(start);
  const delay = setTimeout(() => {
    ready = true;
    sync();
  }, START_MS);
  reduced.addEventListener("change", sync);
  document.addEventListener("visibilitychange", sync);
  addEventListener("pointermove", move, { passive: true });
  addEventListener("resize", redraw);
  canvas.addEventListener("webglcontextlost", lost);
  return {
    sync,
    stop() {
      clearTimeout(delay);
      cancelAnimationFrame(raf);
      raf = 0;
      reduced.removeEventListener("change", sync);
      document.removeEventListener("visibilitychange", sync);
      removeEventListener("pointermove", move);
      removeEventListener("resize", redraw);
      canvas.removeEventListener("webglcontextlost", lost);
      gl.deleteBuffer(buffer);
      gl.deleteProgram(program);
    },
  };
}

function link(gl: WebGLRenderingContext): WebGLProgram | null {
  const program = gl.createProgram();
  for (const [type, source] of [
    [gl.VERTEX_SHADER, VERTEX],
    [gl.FRAGMENT_SHADER, FRAGMENT],
  ] as const) {
    const shader = gl.createShader(type);
    if (!shader || !program) return null;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) return null;
    gl.attachShader(program, shader);
    gl.deleteShader(shader);
  }
  gl.linkProgram(program!);
  return gl.getProgramParameter(program!, gl.LINK_STATUS) ? program : null;
}
