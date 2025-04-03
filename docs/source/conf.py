import os
import sys
import jinja2

# Add the project directory to the sys.path
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath("sphinxext"))



# Configuration file for the Sphinx documentation builder
# .
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mlquantify'
author = 'Luiz Fernando'
release = '0.1.1'

html_baseurl = "https://luizfernandolj.github.io/mlquantify/"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.linkcode",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx-prompt",
    "sphinx_copybutton",
    "sphinxext.opengraph",
    "matplotlib.sphinxext.plot_directive",
    "sphinxcontrib.sass",
    "sphinx_remove_toctrees",
    "sphinx_design",
    # See sphinxext/
    "override_pst_pagetoc",
]

templates_path = ['_templates']
exclude_patterns = [
    "_build",
    "_templates",
]

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False


source_suffix = ".rst"

# The encoding of source files.
source_encoding = "utf-8"

# The main toctree document.
root_doc = "index"

default_role = "literal"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']


html_additional_pages = {"index": "index.html"}


# If false, no module index is generated.
html_domain_indices = False

# If false, no index is generated.
html_use_index = False

html_css_files = ["custom.css"]

html_sidebars = {
    "install": [],
    "getting_started": [],
}




html_theme_options = {
    # -- General configuration ------------------------------------------------
    "sidebar_includehidden": True,
    "use_edit_page_button": True,
    "external_links": [],
    "icon_links_label": "Icon Links",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/luizfernandolj/mlquantify/tree/master",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
    "analytics": {
        "plausible_analytics_domain": "scikit-learn.org",
        "plausible_analytics_url": "https://views.scientific-python.org/js/script.js",
    },
    # If "prev-next" is included in article_footer_items, then setting show_prev_next
    # to True would repeat prev and next links. See
    # https://github.com/pydata/pydata-sphinx-theme/blob/b731dc230bc26a3d1d1bb039c56c977a9b3d25d8/src/pydata_sphinx_theme/theme/pydata_sphinx_theme/layout.html#L118-L129
    "show_prev_next": False,
    "search_bar_text": "Search the docs ...",
    "navigation_with_keys": False,
    "collapse_navigation": False,
    "navigation_depth": 2,
    "show_nav_level": 1,
    "show_toc_level": 1,
    "navbar_align": "left",
    "header_links_before_dropdown": 5,
    "header_dropdown_text": "More",
    # The switcher requires a JSON file with the list of documentation versions, which
    # is generated by the script `build_tools/circle/list_versions.py` and placed under
    # the `js/` static directory; it will then be copied to the `_static` directory in
    # the built documentation
    # check_switcher may be set to False if docbuild pipeline fails. See
    # https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/version-dropdown.html#configure-switcher-json-url
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "surface_warnings": True,
    # navbar_persistent is persistent right (even when on mobiles)
    "navbar_persistent": ["search-button"],
    "article_header_start": ["breadcrumbs"],
    "article_header_end": [],
    "article_footer_items": ["prev-next"],
    "content_footer_items": [],
    # Use html_sidebars that map page patterns to list of sidebar templates
    "primary_sidebar_end": [],
    "footer_start": ["copyright"],
    "footer_center": [],
    "footer_end": [],
    # When specified as a dictionary, the keys should follow glob-style patterns, as in
    # https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-exclude_patterns
    # In particular, "**" specifies the default for all pages
    # Use :html_theme.sidebar_secondary.remove: for file-wide removal
    "secondary_sidebar_items": {
        "**": [
            "page-toc",
            "sourcelink",
        ],
    },
    "show_version_warning_banner": True,
    "announcement": None,
}






def linkcode_resolve(domain, info):
    if domain != 'py' or not info['module']:
        return None

    filename = info['module'].replace('.', '/')
    # Adjust branch and repository as needed
    return f"https://github.com/luizfernandolj/mlquantify/tree/master/{filename}.py"




from api_reference import API_REFERENCE

rst_templates = [
    ("index", "index", {}),
    (
        "api/index",
        "api/index",
        {
            "API_REFERENCE": sorted(API_REFERENCE.items(), key=lambda x: x[0]),
        },
    )
]

for module in API_REFERENCE:
    rst_templates.append(
        (
            "api/module",
            f"api/{module}",
            {"module": module, "module_info": API_REFERENCE[module]},
        )
    )

for rst_template_name, rst_target_name, rst_context in rst_templates:
    
    rst_template = f"{rst_template_name}.rst.template"
    rst_target = f"{rst_target_name}.rst"

    with open(rst_template, "r", encoding="utf-8") as f:
        t = jinja2.Template(f.read())
    # Write the modified content to the target file
    with open(rst_target, "w", encoding="utf-8") as f:
        f.write(t.render(**rst_context))
