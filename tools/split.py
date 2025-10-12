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

        self.run = data_obj.run
        self.save_dir = data_obj.save_dir

        if split:
            self.split_in_train_test(data_obj)
        else:
            self.no_split(data_obj)

        self.fix_groups()
        self.check_data_set()

    def split_in_train_test(self, data_obj, fractions = [0.5, 0.25, 0.25]):
        """
            Split the data into 60% train, 20% validation and 20% test. And sets them as object variables

            :param data_obj: Object that contains the data for the run
        """  
        df = data_obj.data
        # clean split along problems
        group_names = ["train", "val", "test"]
        rng = np.random.default_rng(42)
        unique_names = df['names'].unique()
        rng.shuffle(unique_names)
        
        n = len(unique_names)
        sizes = (np.array(fractions) * n).astype(int)
        sizes[-1] = n - sizes[:-1].sum()  # fix rounding
        splits = np.split(unique_names, np.cumsum(sizes)[:-1])


        name_to_split = {name: group for group, names in zip(group_names, splits) for name in names}
        df["split"] = df["names"].map(name_to_split)

        # --- 4. get split dataframes ---
        self.train_data = df[df["split"] == "train"]
        self.val_data = df[df["split"] == "val"]
        self.test_data = df[df["split"] == "test"]
        
        
        #train_data, self.test_data = train_test_split(data_obj.data, test_size=0.2, random_state=42)

        #self.train_data, self.val_data = train_test_split(train_data,  test_size=0.25, random_state=42)


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
        
    #######################################################
    
    def get_train_probs(self):
        return self.train_data["probs"]
    
    def get_train_is_correct(self):
        return self.train_data["is_correct"]
    
    def get_train_groups(self):
        return self.train_groups
    
    #######################################################

    def get_test_probs(self):
        return self.test_data["probs"]
    
    def set_test_probs(self, new_probs):
        self.test_data["probs"] = new_probs

    def get_test_is_correct(self):
        return self.test_data["is_correct"]

    def get_test_groups(self):
        return self.test_groups

    def get_test_languages(self):
        return self.test_data["languages"]

    def get_test_names(self):
        return self.test_data["names"]

    def get_test_programs(self):
        return self.test_data["programs"]

    def get_test_prompts(self):
        return self.test_data["prompts"]

    def get_test_token_logprobs(self):
        return self.test_data["token_logprobs"]
    
    #######################################################
    
    def get_val_probs(self):
        return self.val_data["probs"]
    
    def get_val_is_correct(self):
        return self.val_data["is_correct"]
    
    def get_val_groups(self):
        return self.val_groups