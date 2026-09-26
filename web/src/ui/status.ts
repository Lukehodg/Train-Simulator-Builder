/** Simulation state chip in the dock: always visible, and always word + icon + colour (never colour alone). */
export type SimState = 'loading' | 'running' | 'paused' | 'finished' | 'error'

const LABEL: Record<SimState, string> = { loading: 'Loading', running: 'Running', paused: 'Paused', finished: 'Finished', error: 'Error' }

export function setSimState(state: SimState, detail = '') {
  const el = document.getElementById('simState')
  if (!el || (el.dataset.state === state && el.dataset.detail === detail)) return   // unchanged: no DOM write, no re-announcement
  el.dataset.state = state
  el.dataset.detail = detail
  document.getElementById('simStateText')!.textContent = LABEL[state]
  document.getElementById('simStateRate')!.textContent = detail
}
