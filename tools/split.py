from sklearn.model_selection import train_test_split
import numpy as np

class split:

    def __init__(self, split, data_obj):
        """
            Initialization of split object

            :param split: Flag to enable or disable splitting
            :param data_obj: Object that contains the data for the run
        """  
        self.train_data = None
        self.test_data = None
        self.val_data = None

        self.train_groups = None
        self.test_groups = None
        self.val_groups = None

        if split:
            self.split_in_train_test(data_obj)
        else:
            self.no_split(data_obj)

        self.fix_groups()
        self.check_data_set()

    def split_in_train_test(self, data_obj):
        """
            Split the data into 60% train, 20% validation and 20% test. And sets them as object variables

            :param data_obj: Object that contains the data for the run
        """  
        train_data, self.test_data = train_test_split(data_obj.data, test_size=0.2, random_state=42)

        self.train_data, self.val_data = train_test_split(train_data,  test_size=0.25, random_state=42)


    def no_split(self, data_obj):
        """
            Performs no splitting on the data

            :param data_obj: Object that contains the data for the run
        """  
        self.train_data = data_obj.data
        self.test_data = data_obj.data
        self.val_data = data_obj.data
    
    def check_data_set(self):
        """
            Performs a check if all variables are set on the split object. 
        """  
        check = [self.__dict__.values()]
        if any(x is None for x in check):
            exit("Failed to split the data!")
    
    def fix_groups(self):
        """
            Fixing of groups is needed to use groups as numpy arrays in further process steps.
        """  
        self.train_groups = np.array([np.array(xi) for xi in self.train_data["groups"].values])
        self.test_groups = np.array([np.array(xi) for xi in self.test_data["groups"].values])
        self.val_groups = np.array([np.array(xi) for xi in self.val_data["groups"].values])

        self.train_data.drop(columns=["groups"])
        self.test_data.drop(columns=["groups"])
        self.val_data.drop(columns=["groups"])