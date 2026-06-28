import os
import yaml

class Config_handler:
    def __init__(self, config_filename):
        self.config_filename = config_filename

    def read_property(self, property_path):
        """
        Reads a value from config-data_prep.yml.
        Supports nested keys using dot notation (e.g., 'database.port').
        :param property_path: property name.
        """
        try:
            # Setup paths (Mirroring your logic)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            config_path = os.path.join(project_root, "config", self.config_filename)

            # Load the YAML file
            with open(config_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)

            # Traverse the dictionary for nested keys
            keys = property_path.split('.')
            value = config_data
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    print(f"ERROR - {property_path} not found in YAML")
                    return None

            return value

        except Exception as e:
            print(f"ERROR - Cannot read YAML file: {e}")
            return None
