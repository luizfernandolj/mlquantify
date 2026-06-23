"""Synthetic and helper datasets for quantification.

Mirrors the spirit of :mod:`sklearn.datasets`, but the generators produce
*bags* (samples with controlled class prevalence) suitable for evaluating
quantifiers under distribution shift.
"""

from ._samples_generator import make_quantification

from ._tabular import (
    fetch_mushroom,
    fetch_banknote_authentication,
    fetch_haberman_survival,
    fetch_miniboone,
    fetch_digits_optical_penbased,
    fetch_dry_bean,
    fetch_covertype,
    fetch_yeast,
    fetch_sensorless_drive,
    fetch_statlog_shuttle,
    fetch_wine_quality,
    fetch_online_news_popularity,
    fetch_pima_diabetes,
    fetch_electricity_elec2,
    fetch_airlines,
)
from ._text import (
    fetch_newsgroups20,
    fetch_imdb,
    fetch_multidomain_sentiment,
    fetch_sentiment140,
    fetch_rcv1_v2,
)
from ._image import (
    fetch_mnist_usps,
    fetch_cifar10,
)
from ._graph import fetch_planetoid_cora_citeseer_pubmed
from ._synthetic import fetch_sea_concepts
from ._lequa import fetch_lequa2024
from ._base import Bunch, get_data_home, fetch_remote, set_progress_hook, get_progress_hook
from ._protocol import make_protocol

__all__ = [
    "fetch_mushroom", "fetch_banknote_authentication", "fetch_haberman_survival", "fetch_miniboone",
    "fetch_digits_optical_penbased", "fetch_dry_bean", "fetch_covertype", "fetch_yeast",
    "fetch_sensorless_drive", "fetch_statlog_shuttle", "fetch_wine_quality", "fetch_online_news_popularity",
    "fetch_pima_diabetes", "fetch_electricity_elec2", "fetch_airlines",
    "fetch_newsgroups20", "fetch_imdb", "fetch_multidomain_sentiment", "fetch_sentiment140", "fetch_rcv1_v2",
    "fetch_mnist_usps", "fetch_cifar10",
    "fetch_planetoid_cora_citeseer_pubmed",
    "fetch_sea_concepts",
    "fetch_lequa2024",
    "make_protocol", "Bunch", "get_data_home", "fetch_remote",
    "set_progress_hook", "get_progress_hook",
    "make_quantification",
]
