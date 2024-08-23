class MLDataSet:
    """
    Class representing protein data for ML.

    Attributes
    -----------
    """

    def __init__(self, protein_data_set):
        self.protein_data_set = protein_data_set
        self.training_indices = None
        self.validation_indices = None
        self.test_indices = None
        self.x_training = None
        self.y_training = None
        self.x_validation = None
        self.y_validation = None
        self.x_test = None
        self.y_test = None
