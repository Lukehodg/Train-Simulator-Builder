/** Escapes text for HTML markup and quoted attributes. Route data (OSM tunnel and station names, OpenCelliD cell
 *  fields, route config) and uploaded train designs are written into innerHTML, so every such value goes through this. */
export const esc = (s: unknown): string => String(s ?? '').replace(/[&<>"']/g, c => `&#${c.charCodeAt(0)};`)
