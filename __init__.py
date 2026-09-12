"""Directory-install shim.

`hermes plugins install bachellerieloic/hermes-nari` clones this repository into
~/.hermes/plugins/nari and imports this file as a package; the real code lives in the
hermes_nari package. The absolute fallback covers tools that import the file on its own.
"""

try:
    from .hermes_nari.plugin import register  # noqa: F401
except ImportError:  # imported without a parent package
    from hermes_nari.plugin import register  # noqa: F401
