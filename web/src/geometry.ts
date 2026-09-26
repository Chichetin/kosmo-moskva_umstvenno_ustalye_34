import type { Feature } from "./types";

export const pointInZone = (asset: Feature, zone: Feature) => {
  if (asset.geometry.type !== "Point" || zone.geometry.type !== "Polygon")
    return false;
  const [x, y] = asset.geometry.coordinates as number[];
  const rings = zone.geometry.coordinates as number[][][];
  const inside = (ring: number[][]) => {
    let hit = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i],
        [xj, yj] = ring[j];
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi)
        hit = !hit;
    }
    return hit;
  };
  return !!rings[0] && inside(rings[0]) && !rings.slice(1).some(inside);
};
