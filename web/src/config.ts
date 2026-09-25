/** Runtime configuration for the viewer. Everything here is keyless by default. */
export const CONFIG = {
  /** Route bundles live in public/data/<route_id>/ (written by `tcs run`). */
  dataRoot: 'data',        // relative: works at a server root and inside a subfolder
  defaultRoute: 'ecml_kgx_edb',
  basemap: {
    // CARTO basemap styles (free with attribution) - swap for MapTiler/Protomaps/OpenFreeMap as needed.
    light: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
    dark: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
  },
  terrain: {
    // AWS Terrain Tiles (Terrarium encoding) - open data, no key. Attribution: Mapzen / AWS Open Data.
    tiles: ['https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'],
    encoding: 'terrarium' as const,
    tileSize: 256,
    maxzoom: 15,
    exaggeration: 1.4,
  },
  ribbons: {
    laneOffsetM: 60,        // combined WAN lane sits this far LEFT of the track, cellular lanes to the right       // lateral spacing between cellular lanes (right-hand side of travel)
    centreWidthM: 30,
    laneWidthM: 18,
    skyLiftM: 120,          // satcom lane floats above the route
    groundLiftM: 4,        // keeps ground ribbons above the terrain mesh
  },
  camera: { chaseZoom: 16.85, chasePitch: 60, chaseYaw: -32, chaseAheadM: -55, obliqueZoom: 13.4, obliquePitch: 55, obliqueYaw: -38 },
  playback: { defaultSpeed: 20, startFraction: 0.63 },
}
