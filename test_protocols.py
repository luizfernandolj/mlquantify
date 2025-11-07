from mlquantify.model_selection import (
    BaseProtocol,
    APP,
    NPP,
    UPP,
    PPP
)



app = APP(batch_size=10,
          n_prevalences=10,
          repeats=1,
          random_state=42)
npp = NPP(batch_size=10,
          n_samples=10,
          random_state=42)
upp = UPP(batch_size=10,
          n_prevalences=20,
          repeats=1,
          random_state=42)
ppp = PPP(batch_size=10,
          prevalences=[[0.2, 0.5, 0.3], [0.5, 0.3, 0.2], [0.3, 0.4, 0.3]],
          repeats=1,
          random_state=42)
