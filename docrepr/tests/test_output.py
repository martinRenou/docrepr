"""Simple tests of docrepr's output."""

# Standard library imports
import copy
import subprocess
import sys
from time import sleep
from pathlib import Path
import warnings

# Third party imports
import numpy as np
from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright
import pytest
from IPython.core.oinspect import Inspector, object_info

# Local imports
import docrepr
import docrepr.sphinxify as sphinxify


# ---- Test data

# A sample function to test
def get_random_ingredients(kind=None):
    """
    Return a list of random ingredients as strings.

    :param kind: Optional "kind" of ingredients.
    :type kind: list[str] or None
    :raise ValueError: If the kind is invalid.
    :return: The ingredients list.
    :rtype: list[str]

    """
    if 'spam' in kind:
        return ['spam', 'spam', 'eggs', 'spam']
    return ['eggs', 'bacon', 'spam']


# A sample class to test
class SpamCans:
    """
    Cans of spam.

    :param n_cans: Number of cans of spam.
    :type n_cans: int
    :raise ValueError: If spam is negative.

    """

    def __init__(self, n_cans=1):
        """Spam init."""
        if n_cans < 0:
            raise ValueError('Spam must be non-negative!')
        self.n_cans = n_cans

    def eat_one(self):
        """
        Eat one can of spam.

        :raise ValueError: If we're all out of spam.
        :return: The number of cans of spam left.
        :rtype: int

        """
        if self.n_cans <= 0:
            raise ValueError('All out of spam!')
        self.n_cans -= 1
        return self.n_cans


PLOT_DOCSTRING = """
.. plot::

   >>> import matplotlib.pyplot as plt
   >>> plt.plot([1,2,3], [4,5,6])
"""

# Test cases
TEST_CASES = {
    'empty_oinfo': {
        'obj': None,
        'oinfo': {},
        'options': {},
        },
    'basic': {
        'obj': None,
        'oinfo': {
            'name': 'Foo',
            'argspec': {},
            'docstring': 'A test',
            'type_name': 'Function',
            },
        'options': {},
        },
    'function_sphinx': {
        'obj': get_random_ingredients,
        'oinfo': {'name': 'get_random_ingredients'},
        'options': {},
        },
    'class_sphinx': {
        'obj': SpamCans,
        'oinfo': {'name': 'SpamCans'},
        'options': {},
        },
    'method_sphinx': {
        'obj': SpamCans().eat_one,
        'oinfo': {'name': 'SpamCans.eat_one'},
        'options': {},
        },
    'render_math': {
        'obj': None,
        'oinfo': {
            'name': 'Foo',
            'docstring': 'This is some math :math:`a^2 = b^2 + c^2`',
            },
        'options': {},
        },
    'no_render_math': {
        'obj': None,
        'oinfo': {
            'name': 'Foo',
            'docstring': 'This is a rational number :math:`\\frac{x}{y}`',
            },
        'options': {'render_math': False},
        },
    'numpy_module': {
        'obj': np,
        'oinfo': {'name': 'NumPy'},
        'options': {},
        },
    'numpy_sin': {
        'obj': np.sin,
        'oinfo': {'name': 'sin'},
        'options': {},
        },
    'collapse': {
        'obj': np.sin,
        'oinfo': {'name': 'sin'},
        'options': {'collapse_sections': True},
        },
    'outline': {
        'obj': np.sin,
        'oinfo': {'name': 'sin'},
        'options': {'outline': True},
        },
    'plot': {
        'obj': None,
        'oinfo': {
            'name': 'Foo',
            'docstring': PLOT_DOCSTRING
            },
        'options': {},
        },
    'python_docs': {
        'obj': subprocess.run,
        'oinfo': {'name': 'run'},
        'options': {},
        },
    'no_docstring': {
        'obj': None,
        'oinfo': {'docstring': '<no docstring>'},
        'options': {},
        },
    }


# ---- Helper functions

def _test_cases_to_params(test_cases):
    return [
        [test_id, *test_case.values()]
        for test_id, test_case in test_cases.items()
    ]


# ---- Fixtures

@pytest.fixture(name='build_oinfo')
def fixture_build_oinfo():
    """Generate object information for tests."""
    def _build_oinfo(obj=None, **oinfo_data):
        if obj is not None:
            oinfo = Inspector().info(obj)
        else:
            oinfo = object_info()
        oinfo = {**oinfo, **oinfo_data}
        return oinfo
    return _build_oinfo


@pytest.fixture(name='set_docrepr_options')
def fixture_set_docrepr_options():
    """Set docrepr's rendering options and restore them after."""
    default_options = copy.deepcopy(docrepr.options)

    def _set_docrepr_options(**docrepr_options):
        docrepr.options.update(docrepr_options)

    yield _set_docrepr_options
    docrepr.options.clear()
    docrepr.options.update(default_options)


# ---- Tests

@pytest.mark.parametrize(
    ('test_id', 'obj', 'oinfo_data', 'docrepr_options'),
    _test_cases_to_params(TEST_CASES),
    ids=list(TEST_CASES.keys()),
    )
def test_sphinxify(
        build_oinfo, set_docrepr_options, open_browser,
        test_id, obj, oinfo_data, docrepr_options,
        ):
    if (oinfo_data.get("docstring", None) == PLOT_DOCSTRING
            and sys.version_info.major == 3
            and sys.version_info.minor == 6
            and sys.platform.startswith("win")):
        pytest.skip(
            "Plot fails on Py3.6 on Windows; older version of Matplotlib?")

    # Filtering warnings generated by playwright
    warnings.filterwarnings(
        'ignore', category=DeprecationWarning, module='pyee.*')
    warnings.filterwarnings(
        'ignore', category=ResourceWarning, module='subprocess.*')
    warnings.filterwarnings(
        'ignore', category=ResourceWarning, module='asyncio.*')

    oinfo = build_oinfo(obj, **oinfo_data)
    set_docrepr_options(**docrepr_options)

    url = sphinxify.rich_repr(oinfo)

    output_file = Path(url)
    assert output_file.is_file()
    assert output_file.suffix == '.html'
    assert output_file.stat().st_size > 512
    file_text = output_file.read_text(encoding='utf-8', errors='strict')
    assert len(file_text) > 512

    test_dir = Path(__file__).parent
    image = f'test-{test_id}.png'
    reference = (test_dir / 'references' / image).resolve()
    screenshot = (test_dir / 'screenshots' / image).resolve()
    diff = (test_dir / 'diffs' / image).resolve()

    # Create diff directory
    (test_dir / 'diffs').mkdir(parents=True, exist_ok=True)

    # Take a screenshot of the generated HTML
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f'file://{url}')

        # Wait for mathjax to finish rendering
        page.wait_for_selector('#MathJax_Message', state='hidden')

        page.screenshot(path=screenshot)
        browser.close()

    # Compare the screenshot with the reference
    reference_im = Image.open(reference).convert('RGB')
    screenshot_im = Image.open(screenshot).convert('RGB')
    diff_im = ImageChops.difference(screenshot_im, reference_im)

    bbox = diff_im.getbbox()
    if bbox is not None:
        diff_im.save(diff)

    assert bbox is None, \
        f'{test_id} screenshot and reference do not match'
