"""Setup for Pytest."""

# Standard library imports
from pathlib import Path
import shutil
from sys import platform
import warnings
import webbrowser

# Third party imports
import pytest
from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright

# ---- Constants

OPEN_BROWSER_OPTION = '--open-browser'
COMPARE_SCREENSHOTS_OPTION = "--compare-screenshots"
UPDATE_REFERENCE_SCREENSHOTS_OPTION = "--update-reference-screenshots"


# ---- Pytest hooks

def pytest_addoption(parser):
    """Add command line options."""
    parser.addoption(
        OPEN_BROWSER_OPTION,
        action='store_true',
        default=False,
        help='For tests that generate HTML output, open it in a web browser',
    )
    parser.addoption(
        COMPARE_SCREENSHOTS_OPTION,
        action='store_true',
        default=False,
        help='For tests that generate HTML output, run visual regression tests on them',
    )
    parser.addoption(
        UPDATE_REFERENCE_SCREENSHOTS_OPTION,
        action='store_true',
        default=False,
        help='For tests that generate HTML output, update reference screenshots for the visual regression tests',
    )


# ---- Fixtures

@pytest.fixture
def open_browser(request):
    """Show the passed URL in the user's web browser if passed."""
    def _open_browser(url):
        if request.config.getoption(OPEN_BROWSER_OPTION):
            warnings.filterwarnings(
                'ignore', category=ResourceWarning, module='subprocess.*')

            webbrowser.open_new_tab(url)
    return _open_browser


@pytest.fixture
def compare_screenshots(request):
    """Do visual regression tests on the output."""
    def _compare_screenshots(test_id, url):
        if (request.config.getoption(COMPARE_SCREENSHOTS_OPTION) or
            request.config.getoption(UPDATE_REFERENCE_SCREENSHOTS_OPTION)):
            # Filtering warnings generated by playwright
            warnings.filterwarnings(
                'ignore', category=DeprecationWarning, module='pyee.*')
            warnings.filterwarnings(
                'ignore', category=ResourceWarning, module='subprocess')
            warnings.filterwarnings(
                'ignore', category=ResourceWarning, module='subprocess.*')
            warnings.filterwarnings(
                'ignore', category=ResourceWarning, module='asyncio.*')

            test_dir = Path(__file__).parent / 'docrepr' / 'tests'
            image = f'test-{test_id}-{platform}.png'
            reference = (test_dir / 'references' / image).resolve()
            screenshot = (test_dir / 'screenshots' / image).resolve()
            diff = (test_dir / 'diffs' / image).resolve()

            # Create diff directory
            (test_dir / 'diffs').mkdir(parents=True, exist_ok=True)

            # Take a screenshot of the generated HTML
            with sync_playwright() as p:
                browser = p.firefox.launch()
                page = browser.new_page()
                page.goto(f'file://{url}')

                # Wait for mathjax to finish rendering
                page.wait_for_selector('#MathJax_Message', state='hidden')

                page.screenshot(path=screenshot)
                browser.close()

            if request.config.getoption(UPDATE_REFERENCE_SCREENSHOTS_OPTION):
                shutil.copyfile(screenshot, reference)
                # Do not run the visual regression test
                return

            # Compare the screenshot with the reference
            reference_im = Image.open(reference).convert('RGB')
            screenshot_im = Image.open(screenshot).convert('RGB')
            diff_im = ImageChops.difference(screenshot_im, reference_im)

            bbox = diff_im.getbbox()
            if bbox is not None:
                diff_im.save(diff)

            assert bbox is None, \
                f'{test_id} screenshot and reference do not match'
    return _compare_screenshots
