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


def _write(resp, dest):
    tmp = dest + ".part"
    with resp, open(tmp, "wb") as f:
        while True:
            b = resp.read(1 << 20)
            if not b:
                break
            f.write(b)
    os.replace(tmp, dest)
    return dest


def fetch_remote(url, dest, download_if_missing=True, n_retries=3, delay=1.0):
    """urllib download with local cache + retries (like sklearn). Retries once unverified on TLS errors."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    if not download_if_missing:
        raise IOError("%s is missing and download_if_missing=False" % dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (quant-datasets)"})
    last = None
    tried_unverified = False
    for attempt in range(1, int(n_retries) + 1):
        print("  downloading (try %d): %s" % (attempt, url))
        try:
            return _write(urllib.request.urlopen(req, timeout=300, context=ssl.create_default_context()), dest)
        except Exception as e:
            last = e
            msg = str(getattr(e, "reason", e)) + " " + str(e)
            if (not tried_unverified) and ("CERTIFICATE_VERIFY" in msg or "SSL" in msg):
                tried_unverified = True
                print("  TLS verification failed; retrying without verification")
                try:
                    return _write(urllib.request.urlopen(req, timeout=300, context=ssl._create_unverified_context()), dest)
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
