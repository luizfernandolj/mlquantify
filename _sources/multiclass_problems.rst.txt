.. _multiclass_problems:

Multiclass Problems
-------------------

In multiclass problems, most methods that only work with binary problems can still be used, applying them to each class separately, using the one-vs-all approach. This means that for each class, the method is applied as if it were a binary problem, treating the selected class as the positive class and all other classes as the negative class. The results are normalized to obtain the final class distribution.