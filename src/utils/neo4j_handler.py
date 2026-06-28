from neo4j import GraphDatabase

import pandas as pd

from src.utils import file_utils
from src.graph_imp.graph_log import Graph_log


class Neo4j_handler:
    def __init__(self, uri, user, password, logger):
        """
        Initialize Neo4j_handler.

        :param uri: neo4j URI.
        :param user: user to authenticate.
        :param password: password to authenticate.
        :param logger: configured logger.
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.logger = logger

        # Set driver based on provided uri, user and password
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def clear_database(self):
        """
        This function allows to clear neo4j database.
        This deletes everything! Use with caution.
        """
        with self.driver.session() as session:
            #session.run("MATCH (n) DETACH DELETE n")
            session.run("""
                            CALL apoc.periodic.iterate(
                              "MATCH (n) RETURN n",
                              "DETACH DELETE n",
                              {batchSize: 5000, parallel: false}
                            );
                        """)
            self.logger.info("Graph has been cleared!")
        self.driver.close()

    def get_recent_import_logs(self, limit=5):
        """
        Connects to Neo4j and retrieves the most recent ImportLog entries.

        :param limit: number of records to return.
        :return: object (nodes, edges, organism, status, [importTime] as imported)
        """
        driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

        query = """
                MATCH (log:ImportLog)
                RETURN log.nodesFile AS Nodes, 
                       log.edgesFile AS Edges, 
                       log.organism AS Organism, 
                       log.status AS Status, 
                       log.importTime AS Imported
                ORDER BY log.importTime DESC
                LIMIT $limit
                """

        logs = []

        try:
            with driver.session() as session:
                response = session.run(query, limit=limit)

                for record in response:
                    # Create the GraphLog object directly from the record
                    log_obj = Graph_log(
                        nodes_file=record["Nodes"],
                        edges_file=record["Edges"],
                        organism=record["Organism"],
                        status=record["Status"],
                        import_time=record["Imported"]
                    )
                    logs.append(log_obj)
        finally:
            driver.close()

        return logs


    def test_connection(self):
        """
        This function allows to check the connection to neo4j database.

        :return: neo4j connection status
        """

        query = "RETURN 'Connection succeceeded!' AS message, datetime() AS serve_time"

        try:
            self.logger.info("Attempt to ping Neo4j...")

            with self.driver.session() as session:
                result = session.run(query)
                data = [record.data() for record in result]

                df_result = pd.DataFrame(data)

                if df_result is not None:
                    self.logger.info("Neo4j connection has been established!")
                    self.logger.info("\n--- Result from DB ---")
                    self.logger.info(df_result)
                    return True
                else:
                    self.logger.warning("\nCan't connect to Neo4j.")
                    return False

        except Exception as e:
            self.logger.error(f"Error during the attempt to ping DB: {e}")
            return None
        finally:
            self.driver.close()


    def _remove_temporary_labels(self, logger, driver):
        """
        This function removes temporary label from the graphs.
        Temporary labels were created only to improve loading performance.
        :param logger: configured logger
        :param driver: neo4j driver
        """
        # --- CLEANUP (Remove temporary label in batches) ---
        logger.info("Cleaning up: Removing temporary 'Node' label from all nodes...")

        # We remove the label in batches of 10,000 to prevent memory crashes
        cleanup_query = """
                    MATCH (n:Node)
                    CALL (n) {
                        REMOVE n:Node
                    } IN TRANSACTIONS OF 10000 ROWS
                    RETURN count(*) as count
                """

        try:
            with driver.session() as session:
                session.run(cleanup_query).consume()

            logger.info("  - Successfully cleared all :Node labels.")

            # Drop the constraint
            logger.info("Dropping temporary Node constraint...")
            driver.execute_query("DROP CONSTRAINT import_node_id IF EXISTS")
        except Exception as e:
            logger.error(f"Failed to remove temporary labels/constraints: {e}")


    def create_db(self, logger, nodes_file_name, edges_file_name):
        """
        Invoking this function graph DB will be created importing data from nodes and edges files.

        Creates the graph using a generic :Node label for high-performance indexing,
        then cleans up the schema after the import is finished.

        Warning: this function will import data to the neo4j default DB.

        :param logger: configured logger
        :param nodes_file_name: file name containing nodes data.
        :param edges_file_name: file name containing edges data.
        """
        if not (file_utils.is_file_name(nodes_file_name) & file_utils.is_file_name(edges_file_name)):
            raise ValueError(f"Expected only a filename, but received a path: \n{nodes_file_name}\n{edges_file_name}")

        with self.driver.session() as session:
            # CREATE INDEXES & CONSTRAINTS ---
            logger.info("Creating indexes and named constraints...")
            # Specific constraints for your biological entities
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:protein) REQUIRE n.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:chemical) REQUIRE n.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:reaction) REQUIRE n.id IS UNIQUE")

            # Named constraint for the generic 'Node' label (used for edge matching)
            # We name it 'import_node_id' so we can drop it by name later.
            session.run("""
                CREATE CONSTRAINT import_node_id IF NOT EXISTS 
                FOR (n:Node) REQUIRE n.id IS UNIQUE
            """)

            # Wait for Neo4j to finish building these indexes before proceeding
            logger.info("Waiting for indexes to go ONLINE...")
            session.run("CALL db.awaitIndexes()")

            # LOAD NODES (Adding the secondary label 'Node') ---
            node_labels = ["protein", "chemical", "reaction"]
            for label in node_labels:
                logger.info(f"  - Importing {label} nodes...")

                node_query = f"""
                    LOAD CSV WITH HEADERS FROM 'file:///{nodes_file_name}' AS row
                    WITH row WHERE row.type = '{label}'
                    CALL (row) {{
                        CREATE (n:{label}:Node {{ 
                            id: row.id, 
                            name: row.name, 
                            chem_ID: row.chem_ID, 
                            gene_ID: row.gene_ID,
                            action: row.action 
                        }})
                    }} IN TRANSACTIONS OF 5000 ROWS
                """

                session.run(node_query)

            # LOAD EDGES (Using the 'Node' label index) ---
            logger.info("Loading edges...")

            edge_query = f"""
                LOAD CSV WITH HEADERS FROM 'file:///{edges_file_name}' AS row
                CALL (row) {{
                    MATCH (a:Node {{id: row.source_id}})
                    MATCH (b:Node {{id: row.target_id}})

                    // Create relationship with properties
                    CALL apoc.create.relationship(
                        a,
                        row.relationship,
                        {{
                            type: row.type,
                            relationship: row.relationship,
                            reaction_id: row.reaction_id
                        }},
                        b
                    )
                    YIELD rel

                    // Dummy update for Unit Subquery rule
                    SET rel._temp = 1
                    REMOVE rel._temp
                }} IN TRANSACTIONS OF 10000 ROWS
            """

            session.run(edge_query)

        # Remove temporary label used to perform loading process
        self._remove_temporary_labels(logger=logger, driver=self.driver)

        # Keep only nodes belonging to a triangle
        #self._triangle_filtering(logger=logger, driver=self.driver)

        logger.info("--- Import Complete! ---")
        logger.info(f"Nodes loaded from: {nodes_file_name}")
        logger.info(f"Edges loaded from: {edges_file_name}")
        logger.info("Temporary labels and indexes have been removed.")


    def count_nodes(self):
        """
        This function counts the number of nodes in the database.
        :return: number of nodes in the database.
        """
        with self.driver.session() as session:
            check_query = "MATCH (n) RETURN count(n) AS c"
            result = session.run(check_query)
            count = result.single()["c"]

            return count


    def fetch_edges(self, query):
        """
        Connects to Neo4j and returns edges as a Pandas DataFrame.

        :param query: neo4j query (ex: "MATCH (s)-[r]->(t) RETURN s.id AS source, t.id AS target").
        :return: pandas dataframe.
        """

        #driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

        with self.driver.session() as session:
            result = session.run(query)
            df = pd.DataFrame([dict(record) for record in result])
        #driver.close()
        return df

    def get_neo4j_last_log(self, logger):
        """
        Get the last log from Neo4j.

        :param logger: logger.
        :return: a Graph_log object containing the last neo4j log.
        """
        # Instatiate Neo4j_handler to interact with graph DB
        neo4j_handler = Neo4j_handler(self.uri, self.user, self.password, logger)

        neo4j_log = neo4j_handler.get_recent_import_logs(limit=1)

        return neo4j_log[0]

    def get_reaction_ids_from_agent(self, destination_id):
        """
        Get reaction_ids of the 'reaction' node having a given agent.

        :param: destination_id: node id to be filtered which is identified as the agent of the reaction node to be returned.
        :return: reaction_ids list of the reaction nodes.
        """
        #driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

        query = """
            MATCH (a:chemical|protein|reaction {id: $destination_id})-[r {relationship: "AGENT_OF"}]->(re:reaction)
    RETURN re.id AS reaction_id
            """

        with self.driver.session() as session:
            result = session.run(query, destination_id=destination_id)
            # Extract the IDs into a flat Python list
            reaction_ids = [record["reaction_id"] for record in result]
        self.driver.close()

        return reaction_ids

    def get_reaction_ids_from_target(self, destination_id):
        """
        Get reaction_ids of the 'reaction' node having a given target.

        :param: destination_id: node id to be filtered which is identified as the agent of the reaction node to be returned.
        :return: reaction_ids list of the reaction nodes.
        """
        #driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

        query = """
            MATCH (a:chemical|protein|reaction {id: $destination_id})-[r {relationship: "TARGET_OF"}]->(re:reaction)
            RETURN re.id AS reaction_id
            """

        with self.driver.session() as session:
            result = session.run(query, destination_id=destination_id)
            # Extract the IDs into a flat Python list
            reaction_ids = [record["reaction_id"] for record in result]
        self.driver.close()

        return reaction_ids

    def get_node_id_by_name(self, node_name):
        """
        Retrieves the internal Neo4j element_id (or a custom id property)
        by matching the 'name' property case-insensitively.

        :param: node_name: node name to be filtered.
        :return: node id.
        """
        # Cypher query using toLower for case-insensitive matching
        query = """
        MATCH (n)
        WHERE toLower(n.name) = toLower($node_name)
        RETURN n.id AS custom_id, elementId(n) AS neo4j_id
        LIMIT 1
        """

        with self.driver.session() as session:
            result = session.run(query, node_name=node_name)
            record = result.single()
        self.driver.close()

        return record["custom_id"]

    def get_organism(self, logger):
        """
            Read the organism from the Neo4j database.

            :param: logger: configured logger.
            :return: organism.
        """
        # Get last neo4j log
        last_neo4j_log = self.get_neo4j_last_log(logger=logger)

        # Read organism from last log
        return last_neo4j_log.organism
