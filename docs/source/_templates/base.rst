{{ objname | escape | underline(line="=") }}
{%- if objtype == "module" %}
.. automodule:: {{ fullname }}
{%- elif objtype == "function" %}
.. currentmodule:: {{ module }}
.. autofunction:: {{ objname }}
{%- elif objtype == "class" %}
.. currentmodule:: {{ module }}
.. autoclass:: {{ objname }}
    :members:
    :inherited-members: Module
    :special-members: __call__
{%- else -%}
.. currentmodule:: {{ module }}
{%- endif -%}