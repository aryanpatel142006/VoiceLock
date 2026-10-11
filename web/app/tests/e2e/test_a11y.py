"""T-404: Listen's live regions and the Diagnostics sheet's focus handling, in a real browser."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]


def checker():
    spec = importlib.util.spec_from_file_location('app_check', ROOT / 'scripts/app_check.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LIVE = """() => {
  const stage = document.querySelector('main#stage');
  const listen = stage.querySelector('.listen');
  const status = listen.querySelectorAll('.sr-only[role=status]');
  return {
    stageLive: stage.getAttribute('aria-live'),
    status: status.length,
    statusText: status[0]?.textContent || '',
    captionAnnouncer: listen.querySelectorAll('.sr-only[aria-live=polite]').length,
    linesLive: listen.querySelector('.ls-cap-lines').getAttribute('aria-live'),
    nestedLive: [...stage.querySelectorAll('[aria-live]')].filter(n => n.parentElement.closest('[aria-live]')).length,
  };
}"""

FOCUS = """() => ({
  sheetHidden: document.querySelector('#sheet').hidden,
  focused: document.activeElement?.className || document.activeElement?.id || document.activeElement?.tagName,
  stageInert: document.querySelector('#stage').inert,
  chromeInert: document.querySelector('#chrome').inert,
})"""


@pytest.mark.slow
def test_listen_live_regions_and_sheet_focus(tmp_path):
    from playwright.sync_api import sync_playwright

    module = checker()
    with module.server(tmp_path) as base, sync_playwright() as pw:
        browser = pw.chromium.launch(args=['--ignore-certificate-errors', '--mute-audio'])
        try:
            for size, covers in (('390', True), ('1440', False)):
                ctx = browser.new_context(**module.VIEWPORTS[size], ignore_https_errors=True)
                page = ctx.new_page()
                page.goto(f'{base}/app/#listen', wait_until='networkidle')
                page.wait_for_function("window.hearingApp && hearingApp.ready && document.querySelector('.listen')")

                live = page.evaluate(LIVE)
                assert live['stageLive'] is None, 'the whole stage must not be one live region'
                assert live['status'] == 1, 'Listen has exactly one status region for the state word'
                assert live['statusText'].strip(), 'the status region names the state'
                assert live['captionAnnouncer'] == 1, 'one hidden region reads each final caption'
                assert live['linesLive'] is None, 'the visible caption list is re-rendered on partials, not live'
                assert live['nestedLive'] == 0

                page.locator('#stage').focus()
                page.keyboard.press('d')
                page.wait_for_function("!document.querySelector('#sheet').hidden")
                page.wait_for_function("document.activeElement?.classList.contains('sheet-close')")
                opened = page.evaluate(FOCUS)
                # below 900 px the sheet covers the screen and the page under it leaves the focus order
                assert opened['stageInert'] is covers and opened['chromeInert'] is covers, (size, opened)

                page.keyboard.press('Escape')
                page.wait_for_function("document.querySelector('#sheet').hidden")
                closed = page.evaluate(FOCUS)
                assert closed['focused'] == 'stage', (size, closed)
                assert closed['stageInert'] is False and closed['chromeInert'] is False, (size, closed)
                ctx.close()
        finally:
            browser.close()
