"""Download/cache (scikit-learn style) + Bunch container + return assemblers for :mod:`datasets`."""
import os, time, ssl, urllib.request

from ._protocol import run_protocol, protocol_name


class Bunch(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)
    def __setattr__(self, k, v):
        self[k] = v


def get_data_home(data_home=None):
    if data_home is None:
        data_home = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")
    os.makedirs(data_home, exist_ok=True)
    return data_home


_progress_hook = None


def set_progress_hook(hook):
    """Register a global callback to report download progress (or ``None`` to disable).

    The library never renders a progress bar itself and pulls in **no** progress
    dependency (no ``tqdm``). Instead, register any callable here and every dataset
    fetcher will report its downloads to it, so you decide how (or whether) to display
    them. The callable is invoked repeatedly while a file is downloading, as::

        hook(downloaded, total, url)

    where ``downloaded`` is the number of bytes received so far, ``total`` is the total
    size in bytes (or ``None`` when the server sends no ``Content-Length`` header), and
    ``url`` is the file being fetched. Returns the previously registered hook, so you can
    restore it later.

    Examples
    --------
    Plain text, no extra dependencies::

        from mlquantify.datasets import set_progress_hook

        def show(downloaded, total, url):
            pct = f"{downloaded / total:6.1%}" if total else f"{downloaded} B"
            print(f"\r{url.split('/')[-1]}: {pct}", end="", flush=True)

        set_progress_hook(show)

    Driving a tqdm bar (tqdm stays *your* dependency, not the library's)::

        from tqdm import tqdm

        bars = {}
        def show(downloaded, total, url):
            bar = bars.setdefault(url, tqdm(total=total, unit="B", unit_scale=True))
            bar.update(downloaded - bar.n)

        set_progress_hook(show)
    """
    global _progress_hook
    old = _progress_hook
    _progress_hook = hook
    return old


def get_progress_hook():
    """Return the currently registered global download-progress hook (or ``None``)."""
    return _progress_hook


def _write(resp, dest, progress=None, url=None):
    try:
        cl = resp.headers.get("Content-Length")
        total = int(cl) if cl else None
    except Exception:
        total = None
    done = 0
    tmp = dest + ".part"
    with resp, open(tmp, "wb") as f:
        while True:
            b = resp.read(1 << 20)
            if not b:
                break
            f.write(b)
            done += len(b)
            if progress is not None:
                progress(done, total, url)
    os.replace(tmp, dest)
    return dest


def fetch_remote(url, dest, download_if_missing=True, n_retries=3, delay=1.0, progress=None):
    """urllib download with local cache + retries (like sklearn). Retries once unverified on TLS errors.

    Pass ``progress`` (a ``callable(downloaded, total, url)``) to report byte-level
    progress for this download; when ``None`` the global hook set via
    :func:`set_progress_hook` is used, if any. See :func:`set_progress_hook`.
    """
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    if not download_if_missing:
        raise IOError("%s is missing and download_if_missing=False" % dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    hook = progress if progress is not None else _progress_hook
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (quant-datasets)"})
    last = None
    tried_unverified = False
    for attempt in range(1, int(n_retries) + 1):
        print("  downloading (try %d): %s" % (attempt, url))
        try:
            return _write(urllib.request.urlopen(req, timeout=300, context=ssl.create_default_context()), dest, hook, url)
        except Exception as e:
            last = e
            msg = str(getattr(e, "reason", e)) + " " + str(e)
            if (not tried_unverified) and ("CERTIFICATE_VERIFY" in msg or "SSL" in msg):
                tried_unverified = True
                print("  TLS verification failed; retrying without verification")
                try:
                    return _write(urllib.request.urlopen(req, timeout=300, context=ssl._create_unverified_context()), dest, hook, url)
                except Exception as e2:
                    last = e2
        if attempt < int(n_retries):
            time.sleep(delay)
    raise last


def finish_tabular(X, y, df, as_frame, return_X_y, protocol, n_samples, sample_size, random_state, name, source):
    """scikit-learn-style return for a tabular dataset; protocol -> mlquantify sampling."""
    import pandas as pd
    tn = sorted(map(str, pd.unique(y)))
    if protocol is not None and protocol is not False:
        proto, samples, prev = run_protocol(protocol, y, sample_size, n_samples, random_state)
        data = X if as_frame else X.to_numpy()
        target = y.rename("target") if as_frame else y.to_numpy()
        return Bunch(data=data, target=target, samples=samples, prevalences=prev, protocol=proto,
                     feature_names=list(X.columns), target_names=tn,
                     DESCR="%s (%s). mlquantify %s: .samples index into .data; .prevalences[i]=bag i; .protocol = the mlquantify protocol." % (name, source, protocol_name(proto)))
    if return_X_y:
        return (X, y.rename("target")) if as_frame else (X.to_numpy(), y.to_numpy())
    if as_frame:
        target = y.rename("target")
        frame = X.copy()
        frame["target"] = target.to_numpy()
        return Bunch(data=X, target=target, frame=frame, feature_names=list(X.columns),
                     target_names=tn, DESCR="%s (%s)" % (name, source))
    return Bunch(data=X.to_numpy(), target=y.to_numpy(), frame=None, feature_names=list(X.columns),
                 target_names=tn, DESCR="%s (%s)" % (name, source))


def finish_xy(data, target, return_X_y, protocol, n_samples, sample_size, random_state, name, source, **extra):
    """scikit-learn-style return for vector/text/image/graph datasets; protocol -> mlquantify sampling."""
    import numpy as np
    y = np.asarray(target)
    if protocol is not None and protocol is not False:
        proto, samples, prev = run_protocol(protocol, y, sample_size, n_samples, random_state)
        return Bunch(data=data, target=y, samples=samples, prevalences=prev, protocol=proto,
                     DESCR="%s (%s). mlquantify %s: .samples index into .data" % (name, source, protocol_name(proto)), **extra)
    if return_X_y:
        return data, y
    return Bunch(data=data, target=y, DESCR="%s (%s)" % (name, source), **extra)
