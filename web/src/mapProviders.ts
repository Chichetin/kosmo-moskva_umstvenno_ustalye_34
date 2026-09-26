/** Public maps provide geographic context only; they never enter model calculations. */
export const mapProviders = {
  map: {
    style: "https://tiles.openfreemap.org/styles/dark",
    label: "OpenFreeMap · OpenStreetMap",
    reliefTiles: ["https://tiles.openfreemap.org/natural_earth/ne2sr/{z}/{x}/{y}.png"],
  },
  optical: {
    tiles: [
      "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2019_3857/default/g/{z}/{y}/{x}.jpg",
    ],
    label: "EOX Sentinel-2 cloudless · мозаика 2019",
    attribution:
      '<a href="https://s2maps.eu" target="_blank" rel="noopener noreferrer">EOxCloudless by EOX</a> (Contains modified Copernicus Sentinel data 2019), CC BY-NC-SA 4.0',
    maxzoom: 15,
  },
} as const;
