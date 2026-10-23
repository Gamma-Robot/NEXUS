"""Public NEXUS documentation; no training imports are needed to build it."""

project = "NEXUS"
author = "Xiangyu Miao"
language = "en"
root_doc = "index"
extensions = ["myst_parser"]
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build", "README.md", "internal/**"]
myst_enable_extensions = ["dollarmath"]
myst_heading_anchors = 3
html_theme = "sphinx_book_theme"
html_title = "NEXUS Documentation"
html_show_sourcelink = False
html_theme_options = {
    "home_page_in_toc": True,
    "show_navbar_depth": 1,
    "max_navbar_depth": 1,
    "collapse_navbar": True,
    "show_toc_level": 2,
    "toc_title": "On this page",
    "use_download_button": False,
    "use_fullscreen_button": False,
    "primary_sidebar_end": [],
    "article_header_start": ["breadcrumbs.html"],
    "article_header_end": ["nexus-header.html"],
    "repository_url": "https://github.com/Gamma-Robot/NEXUS",
    "use_repository_button": True,
    "use_issues_button": False,
}

html_static_path = ["_static"]
html_css_files = ["css/documentation.css"]

templates_path = ["_templates"]
html_context = {"author_url": "https://xiangyumiao.pages.dev/"}


def _set_home_navigation_title(app, pagename, templatename, context, doctree):
    """Label the sidebar home entry independently of the page heading."""
    context["root_title"] = "Overview"


def setup(app):
    app.connect("html-page-context", _set_home_navigation_title, priority=800)
