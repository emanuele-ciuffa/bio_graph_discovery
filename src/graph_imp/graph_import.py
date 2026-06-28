from src.utils.config_handler import Config_handler
from src.utils.logging_handler import Logging_handler
from src.utils.neo4j_handler import Neo4j_handler

import os
from src.utils import file_utils
import traceback

def _get_default_neo4j_browser_uri(neo4j_server_uri):
    """
    This function is used to get the default browser URI to query neo4j graph.
    Warning: if default port has changed the returend URI will not accessible from browser.
    :param neo4j_server_uri: neo4j server URI (ex: 'bolt://localhost:7687')
    :return: default neo4j browser URI.
    """
    uri_ = neo4j_server_uri.split("://")[1]

    return "http://" + uri_.split(":")[0] + ":7474/browser/"


def _neo4j_log_creation(neo4j_handler, nodes_file_name, edges_file_name, organism):
    """
    This function is used to create the log for neo4j graph database.
    WARNING: a new node representing log will be created. This node should be discarded when analysing the graph.
    :param neo4j_handler: neo4j_handler.
    :param nodes_file_name: node file name.
    :param edges_file_name: edge file name.
    :param organism: organism.
    """
    # --- INITIALIZE METADATA LOG ---
    import_metadata = {
        "nodes_file": nodes_file_name,
        "edges_file": edges_file_name,
        "organism": organism,
        "status": "IMPORTED",
        "import_time": "datetime()"  # Cypher function
    }

    log_query = """
        CREATE (log:ImportLog {
            id: randomUUID(),
            nodesFile: $nodes_file,
            edgesFile: $edges_file,
            organism: $organism,
            status: 'COMPLETED',
            importTime: datetime()
        }) RETURN log.id as log_id
        """

    # Use the driver's native execute_query method
    records, summary, keys = neo4j_handler.driver.execute_query(
        log_query,
        parameters_=import_metadata
    )

    '''
    ******** READ CREATED LOG WITH CYPHER 5 ****************
    MATCH (log:ImportLog)
    RETURN
        log.importTime AS Date,
        log.organism AS Organism,
        log.nodesFile AS Nodes_Source,
        log.edgesFile AS Edges_Source,
        log.status AS Status,
        log.id AS Session_ID
    ORDER BY log.importTime DESC
    ***********************************************************
    '''

    return records[0].data() if records else None


def execute(common_config_handler, nodes_path, edges_path, organism):
    """
    This function creates the Neo4j graph DB and import data (nodes and edges).
    :param nodes_path: nodes dataset (input path).
    :param edges_path: edges dataset (input path).
    :param organism: selected organism.
    :param common_config_handler: common config handler.
    """

    try:
        config_handler = Config_handler("config-neo4j.yml")

        logger = Logging_handler(common_config_handler).get_logger(module_name="graph_import")

        # Reading references to connect to neo4j DB
        uri = config_handler.read_property("neo4j.uri")
        user = config_handler.read_property("neo4j.user")
        password = config_handler.read_property("neo4j.password")

        logger.info(f"Neo4j URI: {uri}")
        logger.info(f"Neo4j User: {user}")

        # Instatiate Neo4j_handler to interact with graph DB
        neo4j_handler = Neo4j_handler(uri, user, password, logger)

        # Reading config parameter (overwrite db can be True or Fals)
        has_to_overwrite_db = config_handler.read_property("database.overwrite")

        logger.info(f"Overwriting database: {has_to_overwrite_db}")

        # Clear neo4j database only if the config parameter ('database.overwrite') is set to True
        if has_to_overwrite_db:
            logger.warning("Database will be overwritten.")
            neo4j_handler.clear_database()

        # Get file name from nodes path
        nodes_file_name = file_utils.get_name_with_organism(
            file_path=os.path.basename(nodes_path),
            organism=organism
        )

        logger.debug(f"nodes_file_name: {nodes_file_name}")

        # Get file name from edges path
        edges_file_name = file_utils.get_name_with_organism(
            file_path=os.path.basename(edges_path),
            organism=organism
        )

        logger.debug(f"edges_file_name: {edges_file_name}")

        # We check if nodes exist
        is_connected = neo4j_handler.test_connection()

        if is_connected:
            # We check if nodes exist
            count = neo4j_handler.count_nodes()

            # check if DB is empty
            if count == 0:
                logger.info("Empty DB. A new graph will be created importing available data")

                logger.info(f"Importing data related to the organism '{organism}'")

                logger.info(f"Importing nodes file: {nodes_file_name}")
                logger.info(f"Importing edges file: {edges_file_name}")

                # Create DB and import data
                neo4j_handler.create_db(logger=logger,
                                        nodes_file_name=nodes_file_name,
                                        edges_file_name=edges_file_name)

                # Create log for neo4j (log node)
                _neo4j_log_creation(neo4j_handler=neo4j_handler,
                                    nodes_file_name=nodes_file_name,
                                    edges_file_name=edges_file_name,
                                    organism=organism)

                logger.info("Data has been successfully imported.")
                logger.info(f"You can access to the graph by browser: {_get_default_neo4j_browser_uri(uri)}")
            else:
                logger.warning(f"Skip graph creation: found {count} existing nodes.")
        else:
            logger.warning(f"Skip graph creation: db connection not established.")

    except Exception as e:
        logger.error(f"{e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.warning("Check the presence of files in import folder (sub-directory of neo4j, ex: /neo4j-community-5.26.24/import/[file_name].csv)")
        raise Exception("Error while importing data to the graph.")






