# Considerando um problema de 3 classes

app1 = APP(
    n_prev=20,
    tr_sample_size=[100, 200],
    ts_sample_size=[50, 100],
    train_prev=None, # sem necessidade de usar
    test_prev=None # sem necessidade de usar
)


app1 = APP(
    n_prev=None, # sem necessidade de usar
    tr_sample_size=[100, 200],
    ts_sample_size=[50, 100],
    train_prev=[
        [0.2, 0.3, 0.5],   # configuração 1 treino
        [0.1, 0.4, 0.5]    # configuração 2 treino
    ],
    test_prev=[
        [0.3, 0.3, 0.4],   # configuração 1 teste
        [0.2, 0.4, 0.4]    # configuração 2 teste
    ]
)


app2 = APP(
    n_prev = 20,
    batch_size = ([100, 200], [50, 100])
)

app2 = APP(
    n_prev = (
        [
            [0.2, 0.3, 0.5],   # configuração 1 treino
            [0.1, 0.4, 0.5]    # configuração 2 treino
        ],
        [
            [0.3, 0.3, 0.4],   # configuração 1 teste
            [0.2, 0.4, 0.4]    # configuração 2 teste
        ]
    ),
    batch_size = ([100, 200], [50, 100]),
)

