import os

class MyDirectories:
    """
    This class provides methods to retrieve directories for TAQ trade and quote data.
    Modify the base directory according to your dataset's location.
    """
    
    # Define the base directory where TAQ data is stored
    BASE_DIR = "/Users/jk/Documents/NYU/Academics/Sem 4/ATQS/Impact_Model/"  
    
    @classmethod
    def getTradesDir(cls):
        """
        Returns the directory containing TAQ trade data.
        """
        return os.path.join(cls.BASE_DIR, "trades")

    @classmethod
    def getQuotesDir(cls):
        """
        Returns the directory containing TAQ quote data.
        """
        return os.path.join(cls.BASE_DIR, "quotes")

    @classmethod
    def getDataDir(cls):
        """
        Returns the directory containing TAQ quote data.
        """
        return os.path.join(cls.BASE_DIR, "Processed_Data")
    