import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D


def MoSS_multivariate_normal(n=1000, n_classes=3, alpha=None, m=0.5):
    """
    Gera scores sintéticos multiclasse usando Gaussian Mixture Models.
    
    Parâmetros
    ----------
    n : int
        Número total de amostras.
    n_classes : int
        Número de classes.
    alpha : list ou np.ndarray
        Proporção de amostras por classe.
    m : float
        Controla a nitidez/confiança (0.1 = classes bem separadas, 1.0 = difusas).
    n_components : int
        Número de gaussianas por classe (mistura interna).
    """
    if alpha is None:
        alpha = np.ones(n_classes) / n_classes

    alpha = np.array(alpha)
    
    # garante número inteiro de amostras por classe
    n_per_class = np.floor(n * alpha).astype(int)
    n_per_class[-1] = n - n_per_class[:-1].sum()
    
    
    # listas de saída
    X, y = [], []
    
    # controla a sobreposição (variância proporcional a m)
    variance = 0.05 + 0.5 * m  # m pequeno → variância baixa
    
    # gera centróides das classes no simplex (para manter somatório = 1)
    centers = np.eye(n_classes) * (1 - variance)
    centers += (variance / n_classes)
    
    for c in range(n_classes):
        mean = centers[c, :]
        cov = np.diag(np.full(n_classes, variance))

        # amostras sintéticas ao redor do centro da classe
        X_class = np.random.multivariate_normal(mean, cov, size=n_per_class[c])
        X_class = np.abs(X_class)
        X_class /= X_class.sum(axis=1, keepdims=True)
        
        X.append(X_class)
        y.append(np.full(n_per_class[c], c))
    
    X = np.vstack(X)
    y = np.concatenate(y)
    return X, y


def MoSS_Dirichlet_multiclass(n=1000, n_classes=3, alpha=None, m=0.5):
    """
    Gera scores sintéticos multiclasse usando distribuição Dirichlet.
    
    Parâmetros
    ----------
    n : int
        Número total de amostras.
    n_classes : int
        Número de classes.
    alpha : list ou np.ndarray
        Proporção de amostras por classe.
    m : float
        Controla a dispersão (concentração) da Dirichlet.
        m pequeno → classes mais concentradas em torno do centróide;
        m grande → classes mais dispersas.
    """
    if alpha is None:
        alpha = np.ones(n_classes) / n_classes
    
    n_per_class = np.floor(n * alpha).astype(int)
    n_per_class[-1] = n - n_per_class[:-1].sum()
    
    X, y = [], []
    
    # Definindo centróides no simplex (similares à versão original)
    variance = 0.05 + 0.5 * m
    centers = np.eye(n_classes) * (1 - variance)
    centers += (variance / n_classes)
    
    # Parâmetros para controle explícito da concentração
    max_concentration = 20
    min_concentration = 2
    
    for c in range(n_classes):
        mean = centers[c, :]
        
        # Concentração variável conforme m, para controle efetivo da dispersão
        concentration_factor = max_concentration * (1 - m) + min_concentration * m
        concentration = mean * concentration_factor
        
        X_class = np.random.dirichlet(concentration, size=n_per_class[c])
        
        X.append(X_class)
        y.append(np.full(n_per_class[c], c))
    
    X = np.vstack(X)
    y = np.concatenate(y)
    return X, y



def MoSS(n=1000, alpha=0.5, m=0.5):
    """
    Gera amostras sintéticas binárias com controle de dispersão via potência m.
    
    Parâmetros
    ----------
    n : int
        Número total de amostras.
    alpha : float
        Proporção da classe positiva (classe 1).
    m : float
        Controle da concentração/dispersão das amostras.
        m pequeno → amostras mais próximas a 0 ou 1;
        m grande → amostras mais dispersas.
    
    Retorna
    -------
    X : np.ndarray, shape (n, 2)
        Amostras bidimensionais geradas.
    y : np.ndarray, shape (n,)
        Labels correspondentes (0 ou 1).
    """
    n_pos = int(n * alpha)
    n_neg = n - n_pos
    
    # Scores positivos
    p_score = np.random.uniform(size=n_pos) ** m
    # Scores negativos
    n_score = 1 - (np.random.uniform(size=n_neg) ** m)
    
    # Construção dos arrays de features (duas colunas iguais)
    X_pos = np.column_stack((p_score, p_score))
    X_neg = np.column_stack((n_score, n_score))
    
    # Labels correspondentes
    y_pos = np.ones(n_pos, dtype=int)
    y_neg = np.zeros(n_neg, dtype=int)
    
    # Concatenar dados positivos e negativos
    X = np.vstack((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg))
    
    return X, y





# =========================================================
# 2. Plot 3D
# =========================================================
def plot_3d(X, y, title):
    n_classes = len(np.unique(y))
    
    if n_classes == 2:
        # Histograma para 2 classes
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for c in np.unique(y):
            subset = X[y == c]
            # Plota histograma da primeira coluna (score da classe 0)
            ax.hist(subset[:, 0], 
                    bins=30,
                    density=True,
                    alpha=0.6, 
                    label=f"Classe {c}")
        
        ax.set_xlabel('Score Classe 0')
        ax.set_ylabel('Frequência')
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.show()
    
    elif n_classes == 3:
        # Scatter 3D para 3 classes
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        for c in np.unique(y):
            subset = X[y == c]
            ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2], 
                      label=f"Classe {c}", alpha=0.6)
        
        ax.set_xlabel('Score Classe 0')
        ax.set_ylabel('Score Classe 1')
        ax.set_zlabel('Score Classe 2')
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.show()
    
    else:
        # Scatter 2D para outros casos
        fig, ax = plt.subplots(figsize=(6, 5))
        
        for c in np.unique(y):
            subset = X[y == c]
            ax.scatter(subset[:, 0], subset[:, 1], label=f"Classe {c}", alpha=0.6)
        
        ax.set_xlabel('Score 1')
        ax.set_ylabel('Score 2')
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.show()


# =========================================================
# 3. Exemplo de uso
# =========================================================
if __name__ == "__main__":
    for m in [0.1, 0.5, 0.9]:
        X_gmm, y_gmm = MoSS_multivariate_normal(n=1000, n_classes=3, alpha=[0.5, 0.25, 0.25],  m=m)
        print(X_gmm)
        plot_3d(X_gmm, y_gmm, f"multivariate normal (m={m})")
        
        #X_dir, y_dir = MoSS_Dirichlet_multiclass(n=1000, n_classes=2, m=m)
        #plot_3d(X_dir, y_dir, f"Dirichlet (m={m})")
        
        #X_moss, y_moss = MoSS(n=1000, alpha=0.5, m=m)
        #plot_3d(X_moss, y_moss, f"MoSS Binário (m={m})")
