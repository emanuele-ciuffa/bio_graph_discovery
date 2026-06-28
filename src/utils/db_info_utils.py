from src.utils.config_handler import Config_handler
from src.utils.neo4j_handler import Neo4j_handler


def get_db_organism():
    """
    Establish connection to the Neo4j database to retrieve the stored organism.
    return: the organism from the Neo4j database.
    """
    try:
        # Instatiate config handler for retrieving neo4j parameters
        config_handler_neo4j = Config_handler("config-neo4j.yml")

        # Reading references to connect to neo4j DB
        uri = config_handler_neo4j.read_property("neo4j.uri")
        user = config_handler_neo4j.read_property("neo4j.user")
        password = config_handler_neo4j.read_property("neo4j.password")

        # Instantiate neo4j_handler
        neo4j_handler = Neo4j_handler(uri, user, password, None)

        # Get last neo4j log
        organism = neo4j_handler.get_organism(None)

        return organism

    except Exception as e:
        raise ValueError(
            f"It's impossibile to retrieve the organism from Neo4j: {e}"
        )