/**
 * Procedural rolling-stock geometry for deck.gl SimpleMeshLayer (no external assets).
 * Local frame: +x = direction of travel, +y = left, +z = up; units are metres. Flat-shaded, per-vertex colours.
 */
type Attr = { size: number; value: Float32Array }
/** deck.gl SimpleMeshLayer mesh: attribute objects with size + value. */
export interface MeshData { positions: Attr; normals: Attr; colors: Attr }
type V3 = [number, number, number]
type RGB = [number, number, number]

const hex = (h: string): RGB => { const n = parseInt(h.replace('#', ''), 16); return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255] }

class Builder {
  pos: number[] = []; nrm: number[] = []; col: number[] = []
  tri(a: V3, b: V3, c: V3, color: RGB) {
    const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2], vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2]
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx
    const l = Math.hypot(nx, ny, nz) || 1; nx /= l; ny /= l; nz /= l
    for (const p of [a, b, c]) { this.pos.push(p[0], p[1], p[2]); this.nrm.push(nx, ny, nz); this.col.push(color[0], color[1], color[2]) }
  }
  /** Quad a-b-c-d (counter-clockwise seen from outside). */
  quad(a: V3, b: V3, c: V3, d: V3, color: RGB) { this.tri(a, b, c, color); this.tri(a, c, d, color) }
  box(cx: number, cy: number, cz: number, l: number, w: number, h: number, color: RGB, top?: RGB) {
    const x0 = cx - l / 2, x1 = cx + l / 2, y0 = cy - w / 2, y1 = cy + w / 2, z0 = cz - h / 2, z1 = cz + h / 2
    this.quad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], top ?? color)   // top
    this.quad([x0, y1, z0], [x1, y1, z0], [x1, y0, z0], [x0, y0, z0], color)          // bottom
    this.quad([x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1], color)          // front (+x)
    this.quad([x0, y1, z0], [x0, y0, z0], [x0, y0, z1], [x0, y1, z1], color)          // back
    this.quad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], color)          // right (−y)
    this.quad([x1, y1, z0], [x0, y1, z0], [x0, y1, z1], [x1, y1, z1], color)          // left (+y)
  }
  /** Loft rectangular cross-sections along x. Each section: x, half-width, z bottom, z top. */
  loft(sections: { x: number; w: number; z0: number; z1: number; side: RGB; top: RGB; bottom?: RGB }[], capStart = true, capEnd = true) {
    for (let i = 0; i < sections.length - 1; i++) {
      const a = sections[i], b = sections[i + 1]
      const A = { fl: [a.x, a.w, a.z1] as V3, fr: [a.x, -a.w, a.z1] as V3, bl: [a.x, a.w, a.z0] as V3, br: [a.x, -a.w, a.z0] as V3 }
      const B = { fl: [b.x, b.w, b.z1] as V3, fr: [b.x, -b.w, b.z1] as V3, bl: [b.x, b.w, b.z0] as V3, br: [b.x, -b.w, b.z0] as V3 }
      this.quad(A.fr, B.fr, B.fl, A.fl, b.top)                       // top
      this.quad(A.bl, B.bl, B.br, A.br, b.bottom ?? b.side)          // bottom
      this.quad(A.br, B.br, B.fr, A.fr, b.side)                      // right side (−y)
      this.quad(A.fl, B.fl, B.bl, A.bl, b.side)                      // left side (+y)
    }
    const s0 = sections[0], sN = sections[sections.length - 1]
    if (capStart) this.quad([s0.x, s0.w, s0.z0], [s0.x, -s0.w, s0.z0], [s0.x, -s0.w, s0.z1], [s0.x, s0.w, s0.z1], s0.side)
    if (capEnd) this.quad([sN.x, -sN.w, sN.z0], [sN.x, sN.w, sN.z0], [sN.x, sN.w, sN.z1], [sN.x, -sN.w, sN.z1], sN.side)
  }
  build(): MeshData { return { positions: { size: 3, value: new Float32Array(this.pos) }, normals: { size: 3, value: new Float32Array(this.nrm) }, colors: { size: 3, value: new Float32Array(this.col) } } }
}

export interface Livery { body: string; band: string; stripe: string; roof: string; frame: string; glass: string; door: string; nose: string }
/** LNER Azuma (Class 800/801): white body, charcoal window band, red doors and lower stripe, red cab front. */
export const LNER_AZUMA: Livery = { body: '#eef0f2', band: '#1c2228', stripe: '#c8102e', roof: '#9aa0a6', frame: '#2a2f35', glass: '#161d24', door: '#c8102e', nose: '#c8102e' }
export const NEUTRAL_LIVERY: Livery = { body: '#d7dbe0', band: '#141b23', stripe: '#c8102e', roof: '#8d949b', frame: '#2a2f35', glass: '#1a2530', door: '#141b23', nose: '#141b23' }
export const DEFAULT_LIVERY: Livery = LNER_AZUMA

/** One carriage. kind: cab nose at +x ('cabF'), at −x ('cabR'), or none ('mid'). Length in metres. */
export function carriageMesh(kind: 'cabF' | 'cabR' | 'mid', L = 26, livery: Livery = DEFAULT_LIVERY): MeshData {
  const b = new Builder()
  const body = hex(livery.body), band = hex(livery.band), stripe = hex(livery.stripe), roof = hex(livery.roof), frame = hex(livery.frame), glass = hex(livery.glass), door = hex(livery.door), noseC = hex(livery.nose)
  const W = 1.36, zF = 1.15, zB = 3.55, zR = 3.95          // half width, floor, body top, roof top
  const noseLen = 4.2
  const xf = kind === 'cabF' ? L / 2 - noseLen : L / 2, xb = kind === 'cabR' ? -L / 2 + noseLen : -L / 2
  // body shell + chamfered roof
  b.loft([{ x: xb, w: W, z0: zF, z1: zB, side: body, top: body }, { x: xf, w: W, z0: zF, z1: zB, side: body, top: body }], kind !== 'cabR', kind !== 'cabF')
  b.loft([{ x: xb, w: W, z0: zB, z1: zB, side: roof, top: roof }, { x: xb, w: W - 0.32, z0: zB, z1: zR, side: roof, top: roof }, { x: xf, w: W - 0.32, z0: zB, z1: zR, side: roof, top: roof }, { x: xf, w: W, z0: zB, z1: zB, side: roof, top: roof }], false, false)
  // window band and stripe (thin boxes proud of the sides so they read at distance)
  const bodyLen = xf - xb, cx = (xf + xb) / 2
  for (const sy of [W + 0.02, -W - 0.02]) {
    b.box(cx, sy, 2.75, bodyLen - 1.6, 0.04, 0.9, glass)
    b.box(cx, sy, 1.45, bodyLen - 0.6, 0.04, 0.22, stripe)
  }
  // doors (darker recesses near each end)
  for (const dx of [xb + 1.6, xf - 1.6]) for (const sy of [W + 0.015, -W - 0.015]) b.box(dx, sy, 2.3, 1.3, 0.03, 2.2, door)
  // nose(s): streamlined loft ending in a rounded stub; dark windscreen on the sloping top
  const nose = (dir: 1 | -1) => {
    const x0 = dir === 1 ? xf : xb
    const s = (d: number, w: number, z0: number, z1: number, top: RGB, side: RGB = body) => ({ x: x0 + dir * d, w, z0, z1, side, top })
    const secs = [s(0, W, zF, zR - 0.1, roof), s(1.2, W - 0.03, zF - 0.05, zB - 0.05, glass), s(2.6, W - 0.25, zF - 0.1, 2.75, glass, noseC), s(3.6, W - 0.7, 1.05, 2.05, band, noseC), s(noseLen, W - 1.05, 1.15, 1.55, noseC, noseC)]
    if (dir === 1) b.loft(secs, false, true); else { secs.reverse(); b.loft(secs, true, false) }   // lofts must run in +x for outward normals
  }
  if (kind === 'cabF') nose(1)
  if (kind === 'cabR') nose(-1)
  // underframe, bogies, wheels
  b.box((xf + xb) / 2, 0, 0.98, xf - xb - 0.4, 2.4, 0.34, frame)
  for (const bx of [xb + 4.2, xf - 4.2]) {
    b.box(bx, 0, 0.62, 3.0, 2.3, 0.4, frame)
    for (const wx of [bx - 1.05, bx + 1.05]) for (const wy of [0.78, -0.78]) b.box(wx, wy, 0.46, 0.92, 0.14, 0.92, band)
  }
  return b.build()
}

/** Roof-mounted EDGE Rail active antenna (schematic, ~3x real size so it reads from the chase camera). */
export function edgeRailMesh(): MeshData {
  const b = new Builder()
  b.box(0, 0, 0.16, 1.32, 1.14, 0.32, hex('#3b4148'), hex('#4a5159'))
  b.box(0, 0, 0.36, 0.9, 0.8, 0.08, hex('#ff731e'))
  return b.build()
}
/** Roof-mounted satcom terminal (flat panel). */
export function satcomMesh(): MeshData {
  const b = new Builder()
  b.box(0, 0, 0.12, 1.8, 1.1, 0.24, hex('#e8ecef'), hex('#f4f6f8'))
  b.box(0, 0, 0.3, 0.5, 0.5, 0.12, hex('#9aa3ab'))
  return b.build()
}
/** Simple tree: trunk + two stacked canopy cones. */
export function treeMesh(): MeshData {
  const b = new Builder()
  const trunk = hex('#4a3826'), leaf = hex('#2f6b35'), leaf2 = hex('#3a7f40')
  b.box(0, 0, 1.2, 0.5, 0.5, 2.4, trunk)
  const cone = (z0: number, h: number, r: number, c: RGB) => {
    const n = 7; const apex: V3 = [0, 0, z0 + h]
    for (let k = 0; k < n; k++) { const a0 = (k / n) * Math.PI * 2, a1 = ((k + 1) / n) * Math.PI * 2; b.tri([Math.cos(a0) * r, Math.sin(a0) * r, z0], [Math.cos(a1) * r, Math.sin(a1) * r, z0], apex, c) }
  }
  cone(2.0, 4.5, 2.6, leaf); cone(4.2, 4.2, 2.0, leaf2)
  return b.build()
}
/** Catenary mast: post + cantilever arm. */
export function mastMesh(): MeshData {
  const b = new Builder()
  const steel = hex('#6b7278')
  b.box(0, 0, 3.3, 0.22, 0.22, 6.6, steel)
  b.box(0, -1.3, 5.9, 0.12, 2.8, 0.12, steel)
  return b.build()
}
