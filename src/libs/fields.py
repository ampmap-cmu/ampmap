



class Field: #Stores metadata about dns fields.
    def __init__(self, name, size, accepted_range, is_int, value):
        
        if size == -1 :
            assert( accepted_range !=  None  ) 


        self.name = name
        self.size = size
        self.accepted_range = accepted_range
        self.is_int = is_int

        #If accepted range is not provided, then assume it 
        #takes contiguous
        if self.accepted_range == None: 
            self.accepted_range = range(0,self.size)
        else:
            self.size = len(self.accepted_range)
        
        self.value = value

    def printField(self):
        print ("\t\tField Name : %s" % self.name)
        print ("\t\tField size : %d" % self.size)
        print ("\t\tAccepted range : " , self.accepted_range)
        print ("\t\tValue : %s" % self.value)
        print("\t\tIs Integer: %s" % self.is_int)


def field(name, size, accepted_range, is_int, value = None ):
    return Field(name, size, accepted_range, is_int, value)

