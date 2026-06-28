class Graph_log:
    def __init__(self, nodes_file, edges_file, organism, status, import_time):
        """
        Represent a neo4j log row.
        :param nodes_file: imported nodes file.
        :param edges_file: imported edges file.
        :param organism: imported organism.
        :param status: import status.
        :param import_time: import time.
        :return:
        """
        self.nodes_file = nodes_file
        self.edges_file = edges_file

        self.organism = organism
        self.status = status

        self.import_time = import_time
