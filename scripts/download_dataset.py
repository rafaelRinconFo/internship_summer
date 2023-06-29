import os
import zipfile
import argparse
import urllib.request
import progressbar


class MyProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar=progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()

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
                urllib.request.urlretrieve(self.dataset_url + self.available_years[year], os.path.join(dir,self.available_years[year]), MyProgressBar())

        

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

