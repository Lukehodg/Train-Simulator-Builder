/** Keyboard helpers shared by the tab strips, the scenario popover and the help dialog. UI only. */

/** Arrow-key navigation for a group of buttons (roving focus). `activate` runs for tab strips that follow focus. */
export function arrowNav(buttons: HTMLElement[], orientation: 'horizontal' | 'vertical', activate?: (b: HTMLElement) => void) {
  const prev = orientation === 'horizontal' ? 'ArrowLeft' : 'ArrowUp', next = orientation === 'horizontal' ? 'ArrowRight' : 'ArrowDown'
  buttons.forEach((b, k) => b.addEventListener('keydown', e => {
    let j = -1
    if (e.key === prev) j = (k - 1 + buttons.length) % buttons.length
    else if (e.key === next) j = (k + 1) % buttons.length
    else if (e.key === 'Home') j = 0
    else if (e.key === 'End') j = buttons.length - 1
    if (j < 0) return
    e.preventDefault(); e.stopPropagation()               // keep the app's own arrow/Home/End shortcuts out of it
    buttons[j].focus()
    activate?.(buttons[j])
  }))
}

const FOCUSABLE = 'a[href], button:not([disabled]), select:not([disabled]), input:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'

/** Keep Tab / Shift+Tab inside `container` while it is open. Returns a handler to attach to keydown. */
export function trapTab(container: HTMLElement) {
  return (e: KeyboardEvent) => {
    if (e.key !== 'Tab') return
    const items = [...container.querySelectorAll<HTMLElement>(FOCUSABLE)].filter(el => el.offsetParent !== null)
    if (!items.length) return
    const first = items[0], last = items[items.length - 1]
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus() }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus() }
  }
}
