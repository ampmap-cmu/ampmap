
class Budget(object):
    def __init__(self, initial_val):
        self.budget = initial_val
    def increase(self, x): 
        self.budget = self.budget + x 
