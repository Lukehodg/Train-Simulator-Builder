export type ProviderType = 'cellular' | 'satcom'

export interface ProviderMeta {
  id: string
  name: string
  type: ProviderType
  capacity_prior_mbps: Record<string, number> | number
  terminal?: string
  terminal_default?: string
  enabled?: boolean
  enabled_default?: boolean
  service_area?: { countries: string[] }
  capacity_priors?: Record<string, number>
  min_elevation_deg?: Record<string, number>
  latency_prior_ms?: { base: number; obstruction_penalty: number }
  availability?: {
    sky_threshold: number
    speed_penalty_per_100kph: number
    weather: Record<string, { sky_penalty: number; capacity_factor: number }>
  }
}

export interface StationMeta {
  crs: string; name: string; lat: number; lon: number; distance_m: number; sample_id: number; stop: boolean; trainshed: boolean; scheduled: string | null
}

export interface TunnelMeta { name: string; from_m: number; to_m: number; das: boolean }

export interface Accuracy { points: number; bias_db?: number; mae_db?: number; rmse_db?: number; within_6db?: number; correlation?: number | null; usable_agreement?: number }
/** What the cellular predictions were calibrated against (tcs calibrate-national / tcs calibrate) and how well they held up. */
export interface CalibrationMeta {
  scope: 'national' | 'route'
  source?: string
  fit_period?: { from: string; to: string }
  test_period?: { from: string; to: string }
  routes?: number
  bias?: Record<string, number>
  overall?: Accuracy
  overall_before?: Accuracy
  route?: { calibrated: Accuracy; uncalibrated: Accuracy; held_out: Accuracy | null; measurements?: Record<string, number> } | null
  /** the model against later measurements (tcs check-national), level-matched */
  current_check?: { source?: string; fit_period?: { from: string; to: string }; test_period?: { from: string; to: string }; level_offset_db?: number; overall?: Accuracy; route?: Accuracy | null } | null
  /** share of measured route points with usable 5G, per network (name -> 0..1) */
  five_g?: { source?: string; period?: { from: string; to: string }; bands?: string[]; networks?: Record<string, number>; route?: { points: number; networks: Record<string, number> } | null } | null
}

export interface Meta {
  route: { id: string; name: string; country: string; operator?: string; service_id?: string; direction?: string; origin_crs: string; destination_crs: string; sample_spacing_m: number }
  geometry_source: string
  geometry_straight_legs?: string[]         // OSM legs with no rail path, drawn straight ('BWK-DUN')
  terrain_source: string
  lidar_share?: number                      // share of samples with open 2 m LiDAR near the track (GB bundles from Oct 2026 on)
  lidar_sources?: Record<string, number>    // survey -> share of the LiDAR samples ('lidar_ea', 'lidar_wales', 'lidar_scotland')
  length_m: number
  duration_s: number
  departure: string
  n_samples: number
  providers: ProviderMeta[]
  stations: StationMeta[]
  tunnels: TunnelMeta[]
  line: [number, number][]
  sim: any
  sim_defaults?: any
  provenance: any
  coverage_sources: string[]
  coverage_share?: Record<string, number>   // share of sample-operator pairs per coverage source (bundles from late 2026 on)
  cell_source: string
  calibration?: CalibrationMeta | null   // bundles from Oct 2026 on
  model_version: string
  warnings: string[]
}

/** Column-oriented route bundle (one entry per 50 m sample). */
export interface RouteData {
  n: number
  meta: Meta
  distance: Float64Array
  lat: Float64Array
  lon: Float64Array
  elev: Float32Array
  terrain: Float32Array
  bearing: Float32Array
  inTunnel: Uint8Array
  tunnelName: (string | null)[]
  cutting: Float32Array
  canopy: Float32Array
  urban: Float32Array
  sky: Float32Array
  /** 1 where open LiDAR covered the sample: cutting depth and the near skyline come from it (0 in older bundles) */
  lidar: Uint8Array
  /** share of the antenna's sky roofed by bridges over the line, from LiDAR (0 where none) */
  overhead: Float32Array
  speed: Float32Array
  t: Float64Array
  nextStation: (string | null)[]
  ttn: Float32Array
  stationNear: (string | null)[]
  /** Per provider base inputs exported by the pipeline. */
  base: Record<string, ProviderBase>
  /** Python-computed outputs for the default scenario (used for parity checks / provenance). */
  py: { bonded: Float32Array; wifi: Float32Array; cls: (string | null)[]; active: (string | null)[] }
  cells: CellRecord[]
  cellIndex: Map<string, CellRecord>
}

export interface ProviderBase {
  rsrpSlope?: Float32Array
  rsrpIntercept?: Float32Array
  qb: Float32Array         // cellular: quality before vehicle loss / tunnel override; satcom: raw sky visibility
  hp: Float32Array         // handover penalty 0..1 (satcom: temporary beam handover flag)
  conf: Float32Array
  tech: (string | null)[]
  cell: (string | null)[]
  celld: Float32Array
  src: (string | null)[]
  rsrp: Float32Array       // Python's signal_primary for reference
}

export interface CellRecord { key: string; provider: string; radio: string; lat: number; lon: number; samples: number; range: number; source: string }

export type Policy = 'FAILOVER' | 'WEIGHTED_LOAD_BALANCING' | 'PACKET_BONDING' | 'CELLULAR_PRIMARY_STARLINK_BACKUP' | 'STARLINK_PRIMARY_CELLULAR_BACKUP' | 'POLICY_BASED'
export type Vehicle = 'EXTERNAL_ROOFTOP_ANTENNA' | 'PASSENGER_HANDSET_INSIDE_CARRIAGE' | 'EDGE_RAIL_ACTIVE_ANTENNA'
export type Metric = 'quality' | 'capacity' | 'latency' | 'loss' | 'avail' | 'wifi' | 'conf'
export type CameraMode = 'chase' | 'oblique' | 'free' | 'route'

export interface Scenario { policy: Policy; vehicle: Vehicle; weather: string }

/** Per-provider simulated outputs for the current scenario. */
export interface LinkResult {
  q: Float32Array; cap: Float32Array; lat: Float32Array; loss: Float32Array; avail: Uint8Array; rsrp: Float32Array; reason: string[]; score: Float32Array
}

export interface SimResult {
  links: Record<string, LinkResult>
  active: Uint32Array          // bitmask over meta.providers order
  bonded: Float32Array
  lat: Float32Array
  loss: Float32Array
  conf: Float32Array
  perUser: Float32Array
  activeUsers: Float32Array
  wifi: Float32Array
  wifiClass: Uint8Array        // 0 EXCELLENT .. 4 OUTAGE
}
