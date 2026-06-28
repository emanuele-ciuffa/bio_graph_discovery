class Dataset:
    def __init__(self, path, columns):
        """
        Associate dataset path metadata
        :param path: dataset path (str)
        :param columns: dataset columns (str array)
        """
        self.path = path  # Attribute
        self.columns = columns  # Attribute