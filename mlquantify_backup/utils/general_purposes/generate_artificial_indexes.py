import numpy as np

def generate_artificial_indexes(y, prevalence: list, sample_size:int, classes:list):        
    # Ensure the sum of prevalences is 1
    assert np.isclose(sum(prevalence), 1), "The sum of prevalences must be 1"
    # Ensure the number of prevalences matches the number of classes

    sampled_indexes = []
    total_sampled = 0

    for i, class_ in enumerate(classes):

        if i == len(classes) - 1:
            num_samples = sample_size - total_sampled
        else:
            num_samples = int(sample_size * prevalence[i])
        
        # Get the indexes of the current class
        class_indexes = np.where(y == class_)[0]

        # Sample the indexes for the current class
        sampled_class_indexes = np.random.choice(class_indexes, size=num_samples, replace=True)
        
        sampled_indexes.extend(sampled_class_indexes)
        total_sampled += num_samples

    np.random.shuffle(sampled_indexes)  # Shuffle after collecting all indexes
        
    return sampled_indexes