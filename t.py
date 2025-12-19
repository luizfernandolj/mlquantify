import quapy as qp
from quapy.method.meta import QuaNet
from quapy.classification.neural import NeuralClassifierTrainer, CNNnet
import mlquantify as mq
from mlquantify.metrics import MAE
import torch

qp.environ['SAMPLE_SIZE'] = 10

# load the Kindle dataset as text, and convert words to numerical indexes
dataset = qp.datasets.fetch_reviews('kindle', pickle=True)
qp.data.preprocessing.index(dataset, min_df=5, inplace=True)

# device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# the text classifier is a CNN trained by NeuralClassifierTrainer
cnn = CNNnet(dataset.vocabulary_size, dataset.n_classes)
classifier = NeuralClassifierTrainer(cnn, device=device, epochs=100)

# train QuaNet (QuaNet is an alias to QuaNetTrainer)
model = QuaNet(classifier, 100, device=device)
modelq = mq.neural.QuaNet(classifier, 100, device=device)
model.fit(*dataset.training.Xy)
Xq, yq = dataset.training.Xy
modelq.fit(Xq, yq)

# estimation and evaluation
estim_prevalence = model.predict(dataset.test.instances)
estim_prevalenceq = modelq.predict(dataset.test.instances)
true_prevalence = dataset.test.prevalence()

print(f"True prevalence: {true_prevalence}")
print(f"Estimated prevalence Quapy: {estim_prevalence}")
print(f"Estimated prevalence mlquantify: {estim_prevalenceq}")
print(f"MAE Quapy:", MAE(true_prevalence, estim_prevalence))
print(f"MAE mlquantify:", MAE(true_prevalence, estim_prevalenceq))
