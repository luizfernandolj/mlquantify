import os
import sys
from datetime import datetime

# Adicione o diretório do projeto ao sys.path
sys.path.insert(0, os.path.abspath('../../'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mlquantify'
copyright = f"{datetime.now().year}, Luiz Fernando"
author = 'Luiz Fernando'
version = '0.0.11.8'
release = '0.0.11.8'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Suporte ao estilo Google/Numpy docstring
    "sphinx.ext.viewcode",  # Adiciona links para o código fonte
    "sphinx.ext.githubpages",  # Para hospedar no GitHub Pages
    "sphinx.ext.todo",  # Suporte para TODOs no código
    "sphinx.ext.mathjax",  # Renderização de fórmulas matemáticas
    "sphinx_copybutton",  # Botão para copiar snippets de código
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Configuração para autodoc
autodoc_typehints = "description"

# Caminho para templates customizados
templates_path = ["_templates"]

# Arquivos e diretórios a ignorar
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Configuração para renderizar matemáticas
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

# -- Configurações para HTML -------------------------------------------------
html_theme = "pydata_sphinx_theme"

# Opções do tema
html_theme_options = {
    "navigation_depth": 2,
    "collapse_navigation": False,
    "show_prev_next": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/luizfernandolj/mlquantify",
            "icon": "fa-brands fa-github",
        },
    ],
}

# Caminho para arquivos estáticos
html_static_path = ["_static"]

# Configurações para o botão copiar código
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# -- Extensões adicionais ---------------------------------------------------
# Suporte a docstrings no estilo Google/Numpy
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Configuração para tarefas TODO
todo_include_todos = True

# -- Versões de documentação ------------------------------------------------
# Se sua biblioteca tiver versões, configure o menu de troca aqui:
html_theme_options["switcher"] = {
    "json_url": "https://seu-site/_static/versions.json",
    "version_match": version,
}
