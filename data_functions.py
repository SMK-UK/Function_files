'''
Sean Keenan, PhD Physics
Quantum Memories Group, Heriot-Watt University, Edinburgh
2023

Functions designed to load in and parse data form .csv or .txt
'''

# TODO: 
# update this to operate as class that can be inherited into other files for data handling needs.

from natsort import natsorted
import numpy as np
import csv, json, os, re
import pandas as pd

from typing import Any, Dict, List, Tuple, Union

def check_digits(
    input_string: str
    ) -> bool:
    """
    Checks if a string contains digits and validates if all characters 
    are allowed.

    Parameters
    ----------
    input_string : str
        The string to check.

    Returns
    -------
    bool
        True if the string contains digits and all characters are 
        allowed; False otherwise.
    """
    allowed = "0123456789\n\t\r eE-+,.;"

    # Check if the string contains any digit
    contains_digit = any(
        char.isdigit() for char in input_string
        )

    # Check if all characters in the string are among allowed 
    # characters
    characters_allowed = all(
        char in allowed for char in input_string
        )

    return contains_digit and characters_allowed

def check_dir(
        directory:str,
        ):
    """
    Check if a directory exists already and if not, create it

    Parameters
    ----------

    directory: str
        directory path to check
    
    """
    if os.path.exists(directory):
        pass
    else:
        make_dir(directory) 

def make_filename_copy(self,
                       format: str
                       ) -> str:
        """
        Check if a filename already exists and if so generate a new one
        
        Parameters
        ----------
        format: str
            Format of the file to be saved (e.g., 'csv', 'txt')
        
        Returns
        -------
        file_name: str
            New filename or existing filename if it does not exist

        """
        i = 0
        # check for directory and make if not
        if self.folder:
            self.check_dir(f'{self.path}{self.folder}')
            base_name = f'{self.path}{self.folder}{self.fname}'
        else:
            self.check_dir(f'{self.path}')
            base_name = f'{self.path}{self.fname}'
        # check for existing file and make a copy if yes
        file_name = f'{base_name}.{format}'
        while os.path.isfile(file_name):
            i += 1
            file_name = f'{base_name} ({i}).{format}'

        if self.verbose:
            print(f"Saving file as {file_name}")

        return file_name

def check_str(
    subset_string: str,
    main_string: str
    ) -> bool:
    # TODO 
    # add case insensitivity switch
    """
    Checks if all characters in `subset_string` are present in 
    `main_string`.

    Parameters
    ----------
    subset_string : str
        The string whose characters are to be checked.
    main_string : str
        The string against which the subset_string is validated.

    Returns
    -------
    bool
        True if all characters in subset_string are found in 
        main_string, False otherwise.
    """
    # Create sets for subset and main strings
    char_allow = set(subset_string)
    validation = set(main_string)
   
    # Check if all characters in char_allow are present in validation
    return char_allow.issubset(validation)

def dir_interrogate(
    path: str,
    extensions: List[str] = [],
    exceptions: List[str] = [],
    folders: List[str] = []
    ) -> Tuple[List[str], List[str]]:
    # TODO
    # add case insensitivty to the folder search function
    """
    Interrogate a directory to extract folders and files based on 
    optional filters.

    This function extracts folders and files that are nested once 
    in the given directory.
    It does not handle files in subfolders or the main directory 
    itself.

    Parameters
    ----------
    path : str
        The path to the main directory to interrogate.
    extensions : List[str], optional
        A list of file extensions to include. If empty, no filtering 
        by extension is applied.
    exceptions : List[str], optional
        A list of file extensions or substrings to exclude from the 
        results. If empty, no exclusions are applied.
    folders : List[str], optional
        A list of folder names to include. If empty, all folders are 
        considered.

    Returns
    -------
    Tuple[List[str], List[str]]
        A tuple where:
        - The first element is a list of folder names.
        - The second element is a list of file names.

    Notes
    -----
    - Only files nested one level below the specified directories are 
    included.
    - The function uses natural sorting for directory and file names.
    """
    folder_list = []
    file_list = []
    for root, dirs, files in natsorted(os.walk(path)):

        # Process directories
        if dirs:
            dirs = natsorted(dirs)
            if not folders:
                folder_list = dirs
            else:
                folder_list = [
                    folder for folder in dirs if folder in folders
                    ]
            if exceptions:
                folder_list = [
                    folder for folder in folder_list
                               if not any(
                                exc in folder for exc in exceptions
                                )]
        # Process files
        if not dirs:
            temp_files = []
            if not folders:
                temp_files = files
            elif any(
                folder in os.path.split(root)[-1] for folder in folders
                ):
                temp_files = files
            if exceptions:
                temp_files = [file for file in temp_files
                              if not any(
                                exc in file for exc in exceptions
                                )]
            if extensions:
                temp_files = [
                    file for file in temp_files
                              if file.endswith(
                                tuple(extensions)
                                )]
            if temp_files:
                file_list.append(natsorted(temp_files))
    # Flatten the list if there is only one sublist
    if len(file_list) == 1:
        file_list = [
            file_name for sublist in file_list for file_name in sublist
            ]

    return folder_list, file_list

def extract_dirs(
    path: str,
    folder_1:str,
    folder_2:str
    ):
    """
    Extract the folders and subfolders in a directory using keywords 
    found in the folder names.

    Parameters
    ----------
    path : str
        The path to the main directory to interrogate.
    folder_1 : str
        Keyword(s) in folders to extract from the first layer
    folder_2 : str
        Keyword(s) in folders to extract from the second layer

    Returns
    -------
    Tuple[List[str], List[str]]
        A tuple where:
        - The first element is a list of folder names.
        - The second element is a list of sub folder names.

    """
    folders = []
    sub_folders = []
    count = 0
    for _, dirs, _ in natsorted(os.walk(path)):
        for dir in dirs:
            if folder_1 in dir:
                folders.append(dir)
                sub_folders.append([])
            if folder_2 in dir:
                sub_folders[count].append(dir)
        
        if any(folder_2 in x for x in dirs):
            count += 1

    return folders, sub_folders

def find_numbers(
    string:str,
    pattern:str='-?\ *\d+\.?\d*(?:[Ee]\ *-?\ *\d+)?'
    ) -> list[str]:
    """
    Checks string for whole numbers (does not split numbers i.e. 180 
    is returned as '180' and not ['1','8','0']) and returns a list 
    of all numbers found in the string
    
    Parameters
    ----------
    string : string
        String to check for numbers

    Returns
    -------
    numbers : list
        Number found

    """
    # compile the m atch register
    match_number = re.compile(pattern)
    # create list of numbers found
    numbers = re.findall(match_number, string)

    # handle no numbers found
    if not numbers:
        return None

    return numbers

def make_dir(
        directory:str,
        verbose: bool= True
        ):
    """
    Check if a directory exists and if not create it at the desired
    location.
    
    Parameters
    ----------
    keys : list
        Data values to create a dictionary from

    Returns
    -------
    dictionary :
        Key = value pairs for the data in keys
    """
    try:
        os.makedirs(directory)
        print(f'{directory} does not exist. \n \
                Creating directory... \n')
    except FileExistsError:
        print(f'{directory} already exists \n')
        pass    

def make_index_dict(
    keys: list
    ):
    """
    Make a simple dictionary containing values with increasing index 
    numbers (starting at 0).
    
    Parameters
    ----------
    keys : list
        Data values to create a dictionary from

    Returns
    -------
    dictionary :
        Key = value pairs for the data in keys

    """
    return {x: index for index, x in enumerate(keys)} 

import pandas as pd
import numpy as np
from typing import List

def open_csv(path: str, separators: str = ",", header=None) -> np.ndarray:
    """
    Open a CSV file and convert its contents to a NumPy array.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    separators : str, optional
        Separator(s) used in splitting the columns. Default is ','.
    
    Returns
    -------
    np.ndarray
        A NumPy array containing the data from the CSV file.
        Each row in the array corresponds to a row in the CSV file.

    Raises
    ------
    FileNotFoundError
        If the file specified by `path` is not found.
    Exception
        If any unexpected error occurs while reading the file.
    """
    try:
        # Read the CSV file into a DataFrame
        temp_df = pd.read_csv(path, sep=separators, engine='python', header=header)
        # Convert the entire DataFrame to a NumPy array
        csv_data = temp_df.to_numpy()
        return csv_data
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        return np.array([])
    except Exception as e:
        print(f"Error: An unexpected error occurred while reading the file '{path}': {str(e)}")
        return np.array([])
   
def open_text(
    path: str
    ) -> List[List[Union[str, List[str]]]]:
    """
    Open a text file and read its columns into lists. Handles columns 
    of varying lengths.

    Parameters
    ----------
    path : str
        Path to the text file.

    Returns
    -------
    List[List[Union[str, List[str]]]]
        A list where each sublist represents a column of data. If 
        there is only one column, the outer list will be flattened 
        to a single list of strings.
    """
    data_list = []
    try:
        with open(path, 'r', newline='') as raw_file:
            for row in raw_file:
                data_temp = [i for i in re.split(r"[\t|,|;]", row) if i.strip()]
                if not data_list:
                    data_list = [[] for _ in range(len(data_temp))]
                for index, data in enumerate(data_temp):
                    if len(data_list) < index + 1:
                        data_list.append([])
                    data_list[index].append(data)
        # flatten the list if neccesary
        if len(data_list) == 1:
            data_list = [data for sublist in data_list for data in sublist]
    # handle error exceptions        
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        return []
    except Exception as e:
        print(f"Error: An unexpected error occurred \
               while reading the file '{path}': {str(e)}")
        return []

    return data_list

def read_file(
    path: str
    ) -> Tuple[List[List[Any]], List[List[float]]]:
    """
    Open a given file and read the first two columns to lists. Handles 
    varying column lengths.

    Parameters
    ----------
    path : str
        Path of the file to open.

    Returns
    -------
    Tuple[List[List[Any]], List[List[float]]]
        A tuple containing:
        - A list of metadata lists (each inner list contains metadata 
        entries).
        - A list of data lists (each inner list contains numerical 
        data).
    """
    data_list = []
    metadata_list = []

    try:
        with open(path, 'r', newline='') as raw_file:
            for row in raw_file:
                columns = [
                    i.strip() for i in re.split(r'[,\t;]', row) 
                    if i.strip()
                    ]
                if check_digits(row):
                    # Initialize data_list if empty
                    if not data_list:
                        data_list = [[] for _ in range(len(columns))]
                    for index, value in enumerate(columns):
                        # Ensure data_list has enough nested lists
                        while len(data_list) <= index:
                            data_list.append([])
                        data_list[index].append(float(value))
                else:
                    # Initialize metadata_list if empty
                    if not metadata_list:
                        metadata_list = [
                            [] for _ in range(len(columns))
                            ]
                    for index, value in enumerate(columns):
                        # Ensure metadata_list has enough nested lists
                        while len(metadata_list) <= index:
                            metadata_list.append([])
                        metadata_list[index].append(value)

    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        return [], []
    except Exception as e:
        print(f"Error: An unexpected error occurred while reading \
            the file '{path}': {e}")
        return [], []

    return metadata_list, data_list

def read_json(
    file_name: str
    ) -> dict:
    """
    Read from a JSON file.

    Parameters
    ----------
    file_name : str
        Name of the file to read.
    
    Returns
    -------
    dict
        The contents of the JSON file as a dictionary.
    """
    with open(file_name, 'r') as f:
        return json.load(f)

def search_paths(
    folders: List[str],
    files: List[List[str]],
    include: List[str] = [],
    exclude: List[str] = []
    ) -> List[str]:
    """
    TODO handle empty folders 
    
    Search a list of paths for include and exclude, joining
    files to folders if the conditions are met.
    
    Parameters
    ----------
    folders : List[str]
        List of folder names.
    files : List[List[str]]
        List of file names, each sublist corresponding to the respective folder.
    include : List[str], optional
        Keywords to include in the search (default is []).
    exclude : List[str], optional
        Keywords to exclude from the search (default is []).
    
    Returns
    -------
    List[str]
        List of desired paths.
        
    """
    paths = []
    for index, folder in enumerate(folders):
        desired = []
        for file in files[index]:
            path = os.path.join(folder, file)
            if include:
                if any([x in path for x in include]):
                    desired.append(path)
            else:
                desired.append(path)
            if exclude:
                desired = [x for x in desired
                               if not any([y in path for y in exclude])]
        if desired:
            paths.append(desired)

    if len(paths) == 1:
        paths = [data for sublist in paths for data in sublist]

    return paths

def separate_lists(
    list_to_split: List[List[Tuple]]
    ) -> Tuple[List[List], List[List]]:
    """
    Separates a list of lists of tuples into two lists of lists.

    Parameters
    ----------
    list_to_split : List[List[Tuple]]
        A list of lists, where each sublist contains tuples of two elements.
    
    Returns
    -------
    Tuple[List[List], List[List]]
        Two lists of lists, where the first contains all the first elements 
        of the tuples and the second contains all the second elements of the tuples.
    """
    return (
        [[x for x, y in sublist] for sublist in list_to_split],
        [[y for x, y in sublist] for sublist in list_to_split]
    )

def spectrum_extract(
    paths: List[str],
    keys: List[str] = None,
    tail: int = 1,
    include: bool = True
    ) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    """
    Extract data from files based on the presence of keys in their paths.

    Parameters
    ----------
    paths : List[str]
        List of file paths to process.
    keys : List[str], optional
        List of key values to search for in the paths. If None or empty, all paths are processed (default is None).
    tail : int, optional
        Determines which part of the path to search: 0 for head, 1 for tail (default is 1).
    include : bool, optional
        If True, include files containing the key. If False, include files not containing the key (default is True).

    Returns
    -------
    Tuple[List[Dict[str, Any]], List[np.ndarray]]
        A tuple containing two lists:
        - A list of metadata dictionaries read from the files.
        - A list of NumPy arrays of data read from the files.
    """
    extracted_metadata = []
    extracted_data = []
    
    if not paths:
        return extracted_metadata, extracted_data

    if not keys:
        for path in paths:
            metadata, data = read_file(path)
            extracted_metadata.append(metadata)
            extracted_data.append(np.array(data))
    else:
        for key in keys:
            metadata_children = []
            data_children = []
            for path in paths:
                path_segment = os.path.split(path)[tail]
                if (include and key in path_segment) or (not include and key not in path_segment):
                    metadata, data = read_file(path)
                    metadata_children.append(metadata)
                    data_children.append(np.array(data))
            extracted_metadata.append(metadata_children)
            extracted_data.append(data_children)
    
    return extracted_metadata, extracted_data

def write_csv(file_name, data, delimiter=';', headers=None, numpy=False):
    # TODO add alternate csv writer

    if numpy:
        np.savetxt(file_name, data, delimiter=delimiter, header=headers)
    else:
        with open(file_name, 'w') as file:
            writer = csv.writer(file, delimiter=delimiter)

        return False
    
def write_file(file_name:str, save_data, format:str='txt', **kwargs):
    '''
    Write data to a file
    
    Supoorted types are .txt, .json, .csv
    '''
    fname = f"{file_name}.{format}"

    i = kwargs.get('delimiter') if kwargs else ';'
    if format == 'csv':
        j = kwargs.get('headers') if kwargs else False
        write_csv(file_name=fname, data=save_data, delimiter=i, headers=j)
    elif format == 'json':
        i = kwargs.get('indent') if kwargs else 5
        write_json(file_name=fname, data=save_data, indent=i)
    else:
        write_text(file_name=fname, data=save_data, delimiter=i)

def write_json(file_name:str, data, indent:int=5):
    '''
    Write a Json file

    file_name : str
        Name of file to save as
    data : 
        Json eligible data to save to file  
    indent : int (default 5)
        Amount of indenting to have in the file
    '''
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=indent)

def write_text(file_name, data, delimiter=','):

    with open(file_name, 'wb') as file:

        writer = csv.writer(file, delimiter=delimiter)
        if isinstance(data, zip):
            writer.writerows(data)
        else:
            writer.writerow(data)
