import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from mlquantify.adjust_counting import *

sns.set_theme(style="whitegrid")

def gerar_dataset():
    X, y = make_classification(
        n_samples=2000,
        n_features=6,
        n_informative=4,
        n_classes=2,
        weights=[0.8, 0.2],
        random_state=32
    )
    return X, y

def treinar_modelo(X_train, y_train):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    return clf

def obter_melhor_threshold(thresholds, tprs, fprs, ajustador):
    return ajustador.get_best_threshold(thresholds, tprs, fprs)

def plotar_fpr_1menos_tpr(thresholds, tprs, fprs, melhores_pontos, metodo_nomes):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, fprs, label='FPR', linewidth=2, color='darkred')
    ax.plot(thresholds, 1 - tprs, label='1 - TPR', linewidth=2, color='black')

    # Para cada método, desenhar a linha vertical e texto
    for (best_threshold, best_tpr, best_fpr), metodo_nome in zip(melhores_pontos, metodo_nomes):
        max_y = max(best_fpr, 1 - best_tpr)
        # Linha vertical
        ax.plot([best_threshold, best_threshold], [0, max_y], linestyle='--', linewidth=2, color='gray')
        # Linha horizontal ligando ao eixo y
        ax.plot([0, best_threshold], [max_y, max_y], linestyle='--', linewidth=2, color='gray')
        ax.text(best_threshold, max_y, metodo_nome, fontsize=14, rotation=90, va='bottom', ha='center')

    ax.set_xlabel('Threshold', fontsize=18)
    ax.set_ylabel('Rate', fontsize=18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
    ax.set_title('Threshold Selection Policies', fontsize=16)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0)
    plt.tight_layout()
    plt.savefig('threshold-selection-policies.png')

def main():
    X, y = gerar_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

    clf = treinar_modelo(X_train, y_train)
    thresholds, tprs, fprs = evaluate_thresholds(y=y_test, probabilities=clf.predict_proba(X_test)[:, 1])

    # Métodos que deseja adicionar as linhas
    metodos = [MAX(), X_method(), ACC(), T50()]
    nomes_metodos = ['MAX', 'X', 'ACC', 'T50']

    melhores_pontos = [obter_melhor_threshold(thresholds, tprs, fprs, metodo) for metodo in metodos]

    plotar_fpr_1menos_tpr(thresholds, tprs, fprs, melhores_pontos, nomes_metodos)

if __name__ == "__main__":
    main()
