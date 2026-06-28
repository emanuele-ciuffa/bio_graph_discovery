import os
from datetime import datetime


def get_name_with_organism(file_path, organism):
    """
    This function adds to the file name a suffix composed by the organism:
    [name]_[organism].[extension]
    :param file_path: file path to update adding the suffix with temporal info.
    :param organism: selected organism (species)
    :return: new file path with a suffix.
    """
    # Extract directory
    directory = os.path.dirname(file_path)

    # Extract file name
    file_name = os.path.basename(file_path)

    # Separate name and extension
    name, extension = os.path.splitext(file_name)

    # Get the current time
    now = datetime.now()

    # Format date and hour: 'YYYYMMDD_hhmm'
    timestamp = now.strftime("%Y%m%d_%H%M")

    # Normalize organism by removing spaces and using underscore, finally apply lower case
    normalized_organism = organism.replace(" ", "_").lower()

    #new_file_name = f"{name}_{normalized_organism}_{timestamp}{extension}"

    new_file_name = f"{name}_{normalized_organism}{extension}"

    return os.path.join(directory, new_file_name)


def is_file_name(file_name):
    """
    This function checks if the file name is a valid file name.
    :param file_name: file name to check.
    :return: True if it is a valid file name, False otherwise.
    """
    # Check if it's just a filename
    is_only_filename = file_name == os.path.basename(file_name)

    if is_only_filename:
        return True
    else:
        return False
