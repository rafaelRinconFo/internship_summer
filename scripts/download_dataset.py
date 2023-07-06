import os
import zipfile
import argparse
import urllib.request
from tqdm import tqdm


import os
from urllib.request import urlretrieve
from tqdm import tqdm

def my_hook(t):
    """Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    >>> with tqdm(...) as t:
    ...     reporthook = my_hook(t)
    ...     urllib.urlretrieve(..., reporthook=reporthook)
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to

def download(url, save_dir):
    filename = url.split('/')[-1]
    with tqdm(unit = 'B', unit_scale = True, unit_divisor = 1024, miniters = 1, desc = filename) as t:
        urllib.request.urlretrieve(url, filename = os.path.join(save_dir, filename), reporthook = my_hook(t), data = None)
class DatasetDownloader:
    def __init__(self, years: list, des_dir: str, download_all: bool = False) -> None:
        self.available_years = {
            2015: "98240.zip",
            2016: "98289.zip",
            2018: "98314.zip",
            2020: "98356.zip"
        }
        self.dataset_url = "https://www.seanoe.org/data/00810/92226/data/"
        if download_all:
            self.dataset_years = list(self.available_years.keys())
        else:
            self.dataset_years = [int(''.join(year)) for year in years]
        print(f"Downloading dataset for years {self.dataset_years}")
        self.download_dataset(des_dir)
        print(f'Decompressing dataset for years {self.dataset_years}')
        self.decompress_dataset(des_dir)
        print(f"Deleting non-necessary .zip files {self.dataset_years}")
        self.delete_zip_files(des_dir)

    def delete_zip_files(self, dir: str) -> None:
        for year in self.dataset_years:
            if year not in self.available_years:
                raise ValueError(f"Year {year} is not available. Available years are {self.available_years.keys()}")
            else:
                print(f"Deleting .zip file for year {year}")
                os.system(f"rm {os.path.join(dir,self.available_years[year])}")


    def decompress_dataset(self, dir: str) -> None:
        for year in self.dataset_years:
            if year not in self.available_years:
                raise ValueError(f"Year {year} is not available. Available years are {self.available_years.keys()}")
            elif os.path.exists(os.path.join(dir,str(year))):
                print(f"Dataset for year {year} already decompressed")
            else:
                print(f"Decompressing dataset for year {year}")
                with zipfile.ZipFile(os.path.join(dir,self.available_years[year]), 'r') as zip_ref:
                    zip_ref.extractall(dir)

    
    def download_dataset(self, dir: str) -> None:
        for year in self.dataset_years:
            if year not in self.available_years:
                raise ValueError(f"Year {year} is not available. Available years are {self.available_years.keys()}")
            elif os.path.exists(os.path.join(dir,self.available_years[year])):
                print(f"Dataset for year {year} already downloaded")                
            else:
                print(f"Downloading dataset for year {year}")
                download(self.dataset_url + self.available_years[year], dir)
                
        

def main():
    parser = argparse.ArgumentParser(description="Download dataset from https://www.seanoe.org/data/00810/92226/")
    parser.add_argument("--years", type=list, help="List of years to download", default=[],nargs='+')
    parser.add_argument("--des_dir", type=str, help="Destination directory to save the dataset", default="datasets")
    parser.add_argument("--download_all", type=bool, help="Download all available years", default=False)
    args = parser.parse_args()
    years = args.years
    des_dir = args.des_dir
    download_all = args.download_all
    dataset_downloader = DatasetDownloader(years, des_dir, download_all)

if __name__ == "__main__":    
    main()

