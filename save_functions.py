'''
Sean Keenan, PhD Physics
Quantum Memories Group, Heriot-Watt University, Edinburgh
2023

Functions for saving / loading of data.

'''

def check_dir(
        directory:str,
        verbose: bool=True
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
    if os.path.exists(directory):
        if verbose:
            print(f'{directory} already exists \n')
    else:
        try:
            os.makedirs(directory)
            if verbose:
                print(f'{directory} does not exist. \n \
                        Creating directory... \n')
        except FileExistsError:
            if verbose:
                print(f'{directory} already exists \n')
            pass 

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