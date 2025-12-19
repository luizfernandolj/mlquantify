import quapy as qp
from quapy.method.meta import QuaNet
from quapy.classification.neural import NeuralClassifierTrainer, CNNnet

# use samples of 100 elements
qp.environ['SAMPLE_SIZE'] = 100

# load the Kindle dataset as text, and convert words to numerical indexes
dataset = qp.datasets.fetch_reviews('kindle', pickle=True)
qp.data.preprocessing.index(dataset, min_df=5, inplace=True)

# the text classifier is a CNN trained by NeuralClassifierTrainer
cnn = CNNnet(dataset.vocabulary_size, dataset.n_classes)
classifier = NeuralClassifierTrainer(cnn, device='cuda')

# train QuaNet (QuaNet is an alias to QuaNetTrainer)
model = QuaNet(classifier, qp.environ['SAMPLE_SIZE'], device='cuda')
model.fit(*dataset.training.Xy)
estim_prevalence = model.predict(dataset.test.instances)

print(estim_prevalence)
print(dataset.test.prevalence())

