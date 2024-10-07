'''
Sean Keenan, PhD Physics
Quantum Memories Group, Heriot-Watt University, Edinburgh
2023

Functions designed to load in and parse data form .csv or .txt
'''

from natsort import natsorted
import numpy as np
import os, re
import pandas as pd
import json

def check_str(subset_string: str,
              main_string: str) -> bool:
    """
    Checks a string against another string and tests if the first is a subset of the second

    Parameters
    ----------
    subset_string : string
        Validation string to check against
    main_string : string
        String to check 

    Returns
    -------
    Boolean
        True or False depending on the condition of check_str
    
    """
    # string to check against - should be a subset of char_allow
    char_allow = set(subset_string)
    # should contain the string you want to check
    validation = set(main_string)
   
    return char_allow.issubset(validation)

def check_digits(input_string: str) -> bool:
    """
    Checks if a string contains only digits and flags if True
    
    Parameters
    ----------
    input_string : string
        String to check the contents of

    Returns
    -------
    logical : Boolean
        True or False depending on the condition of check_str

    """
    # check input string for digits and flag True
    if any(char.isdigit() for char in input_string) == True:

        logical = check_str((input_string), "0123456789\n\t\r eE-+,.;")
    else:
        logical = False

    return logical

def dir_interrogate(path: str, 
                   extensions: list[str] = [], 
                   exceptions: list[str] = [], 
                   folders: list[str] = []) -> tuple[list[str], list[str]]:
    """
    Interogate directory and extract all folders and files. Optional: 'extensions', 
    'exceptions' and 'folders' enables selective read of files and folders.

    Only capable of extracting files that are nested once. Will not work for 
    subfolders or extract files from the main directory.

    Parameters
    ----------
    path : string - main folder / directory to interrogate
    extensions : tuple - file extensions to check for in directory
    exceptions : tuple - file extensions / strings to exclude
    folders : tuple - select folders to extract from

    Returns
    -------
    folder_list : list of folder names
    file_list : list of file names

    """       
    folder_list = []
    file_list = []
    for root, dirs, files in natsorted(os.walk(path)):

        if dirs:
            dirs = natsorted(dirs)
            if not folders:
                folder_list = dirs
            else:
                folder_list = [folder for folder in dirs 
                               if folder in folders]
            if exceptions:
                folder_list = [folder for folder in folder_list
                               if not any([x in folder for x in exceptions])]

        if not dirs:
            temp_files = []
            if not folders:
                temp_files = files
            elif any([x in os.path.split(root) for x in folders]):
                temp_files = files
            if exceptions:
                temp_files = [file for file in temp_files
                              if not any([x in file for x in exceptions])]
            else:
                temp_files = [file for file in temp_files]
            if extensions:
                    temp_files = [file for file in temp_files
                              if file.endswith(tuple(extensions))]
            if temp_files:
                file_list.append(natsorted(temp_files))

    if len(file_list) == 1:
        file_list = [file_name for sublist in file_list
                     for file_name in sublist]
    
    return folder_list, file_list

def find_numbers(string:str, 
                 pattern:str='-?\ *\d+\.?\d*(?:[Ee]\ *-?\ *\d+)?') -> list[str]:
    """
    Checks string for whole numbers (does not split numbers i.e. 180 is 
    returned as '180' and not ['1','8','0']) and returns a list of all 
    numbers found in the string
    
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

def make_index_dict(keys: list):
    """
    Make a simple dictionary containing values with increasing index numbers (starting at 0)
    
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

def open_csv(path: str, 
             separators: str=",") -> list:
    """
    Open a given excel / csv file and convert each column 
    to an np.array and place into a list
    
    Parameters
    ----------
    path : string
        path to extract the excel data from
    separators : string
        Seperators to use in splitting the columns. Default is comma ','
    
    Returns
    -------
    excel_data : list
        Column data from pandas data frame as list of np.arrays

    """
    # attempt to open the file
    try:
        temp_df = pd.read_csv(path, sep=separators, engine='python')
        csv_data = temp_df.to_numpy()
        return csv_data
    # flag error if unable to open
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        return []
    except Exception as e:
        print(f"Error: An unexpected error occurred while \
              reading the file '{path}': {str(e)}")
        return []
    
def open_text(path: str):
    """
    Open a text file and read the first two columns to a list. Works with
    columns of different length

    Parameters
    ----------
    path : file path
    
    Returns
    -------
    data_list : list of data read from path
    metadata_list : list of metadata read from path
    
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

def read_file(path: str) -> tuple:
    """
    Open a given file and read the first two columns to a list. Works with
    different lengths of columns 

    Parameters
    ----------
    path : str
        Path of file to open
    
    Returns
    -------
    metadata_list : list
        List of infromation / metadata read from the file
    data_list : list
        List of number data read from file
    
    """
    data_list = []
    metadata_list = []
    try:
        with open(path, 'r', newline='') as raw_file:
            for row in raw_file:
                if check_digits(row):
                    # generate list to populate with column data
                    data_temp = [i for i in re.split(r"[\t|,|;]", row) if i.strip()]
                    # create the empty list with approriate number of nested lists
                    if not data_list:
                        data_list = [[] for _ in range(len(data_temp))]
                    for index, data in enumerate(data_temp):
                        if len(data_list) < index + 1:
                            data_list.append([])
                        data_list[index].append(float(data))
                else:
                    metadata_temp = [i for i in re.split(r"[\t|,|;]", row) if i.strip()]
                    if not metadata_list:
                        metadata_list = [[] for _ in range(len(metadata_temp))]
                    for index, metadata in enumerate(metadata_temp):
                        if len(metadata_list) < index + 1:
                            metadata_list.append([])
                        metadata_list[index].append(metadata)
    # handle error exceptions
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        return [], []
    except Exception as e:
        print(f"Error: An unexpected error occurred \
              while reading the file '{path}': {str(e)}")
        return [], []

    return metadata_list, data_list

def read_json(file_name):
    '''
    Read from a Json file

    file_name : str
        Name of file to read
    
    '''
    with open(file_name, 'r') as f:
        return json.load(f)

def search_paths(folders: list[str], files: list[str], include: list[str] = [], exclude: list[str] = []) -> list[str]:
    """
    Search a list of paths for include and exclude, joining
    files to folders if the conditions are met
        
    Parameters
    ----------
    folders : list
        List of folder names to 
    files : list
        list of file names
    keys : list
        keywords to search folders and files for
    
    Returns
    -------
    paths : list of desired paths

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

def seperate_lists(list_to_split):

    return [[x for x, y in sublist] for sublist in list_to_split], [[y for x, y in sublist] for sublist in list_to_split]

def spectrum_extract(paths: list[str],
                     keys: list[str]=[], 
                     tail: int=1,
                     include: bool=True) -> tuple:
    """
    search a list of paths for strings and extract the data
    from selected files depending on the discriminator (keys)

    Parameters
    ----------
    paths : file paths
    keys : list of key values to search for in path if required
    tail : int value 1 or 0 to search head or tail of path
    include : True or False to include the data with key or exclude
    
    Returns
    -------
    extracted_metadata : list of metadata read from path
    extracted_data : list of data read from path
    
    """
    extracted_metadata = []
    extracted_data = []
    if not keys:
        for path in paths:
            extracted_metadata.append(read_file(path)[0])
            extracted_data.append(np.array(read_file(path)[1]))
    else:
        for key in keys:
            extracted_data_children = []
            extracted_metadata_children = []
            for path in paths:
                if include == True:
                    # extract data from path if it contains the key
                    if key in os.path.split(path)[tail]:
                        extracted_metadata_children.append(read_file(path)[0])
                        extracted_data_children.append(np.array(read_file(path)[1])) 
                    else:
                        continue
                else:
                    # extract data from path if it does not contain the key
                    if key in os.path.split(path)[tail]:
                        continue
                    else:                
                        # extract data from path if it doesn't contain the key
                        extracted_metadata_children.append(read_file(path)[0])
                        extracted_data_children.append(np.array(read_file(path)[1]))
            extracted_metadata.append(extracted_metadata_children)
            extracted_data.append(extracted_data_children)

    return extracted_metadata, extracted_data

def write_csv(file_name, data, delimiter=';', headers=False):

    if isinstance(data, np.ndarray):
        print(file_name)
        np.savetxt(file_name, data, delimiter=delimiter)
    
def write_file(file_name:str, save_data, format:str='txt', **kwargs):
    '''
    Write data to a file
    
    Supoorted typesd are .txt, .json, .csv
    '''
    fname = f"{file_name}.{format}"
    if format == 'csv':
        i = kwargs.get('delimiter') if kwargs else ';'
        j = kwargs.get('headers') if kwargs else False
        write_csv(file_name=fname, data=save_data, delimiter=i, headers=j)
    elif format == 'json':
        i = kwargs.get('indent') if kwargs else 5
        write_json(file_name=fname, data=save_data, indent=i)
    else:
        write_text(file_name=fname, data=save_data)

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

def write_text(file_name:str, data):
    '''
    Write a text file
    
    '''
    if type(data) != str:
        print('Data type should be str!')
    else:
        with open(file_name, 'w') as writer:
            writer.write(data)

def numpy_extract(file_name:str, delimiter=';'):
    data = np.loadtxt(file_name=file_name, delimiter=delimiter)
    
    return data[0], data[1]
