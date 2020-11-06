class QueryNodeTest(object):
    def __init__(self, id, field_values,  amplification_value, server_id):
        self.id = id 
        self.field_values = field_values 
        self.AF = amplification_value
        self.server_id = server_id

    def print_compact(self): 
        print("\t\t\t", self.AF , " : ", self.field_values)
    def print(self): 
        print("\t\tPrinting queryNode id " , self.id)
        print("\t\t\tfield values ", self.field_values)
        print("\t\t\tAF ", self.AF)
        print("\t\t\tserver_id ", self.server_id)


class QueryNode(object):
    def __init__(self, field_values, depth, amplification_value, server_id, cluster_id):
        self.field_values = field_values 
        self.depth = depth 
        self.AF = amplification_value
        self.server_id = server_id
        self.cluster_id = cluster_id


    def update_cluster_id(self,cluster_id): 
        self.cluster_id = cluster_id 

    def print(self): 
        print("\t\tPrinting queryNode")
        print("\t\t\tfield values ", self.field_values)
        print("\t\t\tcluster id ", self.cluster_id)
        print("\t\t\tAF ", self.AF)
        print("\t\t\tdepth ", self.depth , " server_id ", self.server_id)



def get_query_node(id, field_values , AF, server_id ):
    return QueryNodeTest(id, field_values, AF, server_id)



