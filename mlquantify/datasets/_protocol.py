"""mlquantify evaluation-protocol bridge for :mod:`datasets`.

Wraps mlquantify.model_selection (APP/NPP/UPP/PPP) and mlquantify.utils.get_prev_from_labels so
the fetchers' ``protocol`` argument produces sample bags + true prevalences.
"""


def make_protocol(protocol, sample_size, n_samples, random_state=None):
    """Build an mlquantify protocol from a name or return a passed-in instance.

    'app' -> APP, 'npp' -> NPP, 'upp' -> UPP, 'ppp' -> PPP (needs an explicit instance).
    sample_size -> protocol ``batch_size``; n_samples -> number of prevalence points.
    """
    if isinstance(protocol, str):
        from mlquantify.model_selection import APP, NPP, UPP, PPP  # noqa: F401
        name = protocol.lower()
        if name == "app":
            return APP(batch_size=sample_size, n_prevalences=n_samples, random_state=random_state)
        if name == "upp":
            return UPP(batch_size=sample_size, n_prevalences=n_samples, random_state=random_state)
        if name == "npp":
            return NPP(batch_size=sample_size, n_samples=n_samples, random_state=random_state)
        if name == "ppp":
            raise ValueError("PPP needs explicit target prevalences; pass a configured "
                             "mlquantify.model_selection.PPP(...) instance as protocol=")
        raise ValueError("protocol must be 'app'|'npp'|'upp'|'ppp' or an mlquantify protocol instance (got %r)" % (protocol,))
    if hasattr(protocol, "split"):
        return protocol
    raise ValueError("protocol must be 'app'|'npp'|'upp'|'ppp' or an mlquantify protocol instance (got %r)" % (protocol,))


def run_protocol(protocol, y, sample_size, n_samples, random_state):
    """Return (protocol_obj, samples, prevalences) using mlquantify's protocol + get_prev_from_labels."""
    import numpy as np
    from mlquantify.utils import get_prev_from_labels
    proto = make_protocol(protocol, sample_size, n_samples, random_state)
    y = np.asarray(y)
    classes, codes = np.unique(y, return_inverse=True)
    Xph = np.arange(len(y)).reshape(-1, 1)
    samples = [np.asarray(list(b), dtype=int) for b in proto.split(Xph, codes)]
    prev = np.array([get_prev_from_labels(y[b], format="array", classes=list(classes)) for b in samples])
    return proto, samples, prev


def protocol_name(protocol):
    return type(protocol).__name__ if (not isinstance(protocol, str) and hasattr(protocol, "split")) else str(protocol).upper()
