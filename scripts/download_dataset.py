import os
import zipfile
import argparse
import urllib.request



class DatasetDownloader:
    def __init__(self, years: list, des_dir: str) -> None:
        self.available_years = {
            2015: "98240.zip",
            2016: "98289.zip",
            2018: "98314.zip",
            2020: "98356.zip"
        }
        self.dataset_url = "https://www.seanoe.org/data/00810/92226/data/"

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
                print(f"Deleted .zip file for year {year}")

    def decompress_dataset(self, dir: str) -> None:
        for year in self.dataset_years:
            if year not in self.available_years:
                raise ValueError(f"Year {year} is not available. Available years are {self.available_years.keys()}")
            else:
                print(f"Decompressing dataset for year {year}")
                with zipfile.ZipFile(os.path.join(dir,self.available_years[year]), 'r') as zip_ref:
                    zip_ref.extractall(dir)
                print(f"Decompressed dataset for year {year}")
    
    def download_dataset(self, dir: str) -> None:
        for year in self.dataset_years:
            if year not in self.available_years:
                raise ValueError(f"Year {year} is not available. Available years are {self.available_years.keys()}")
            else:
                print(f"Downloading dataset for year {year}")
                urllib.request.urlretrieve(self.dataset_url + self.available_years[year], os.path.join(dir,self.available_years[year]))
                print(f"Downloaded dataset for year {year}")
        

def main():
    parser = argparse.ArgumentParser(description="Download dataset from https://www.seanoe.org/data/00810/92226/")
    parser.add_argument("--years", type=list, help="List of years to download", default=[],nargs='+')
    parser.add_argument("--des_dir", type=str, help="Destination directory to save the dataset", default="datasets")
    args = parser.parse_args()
    years = args.years
    des_dir = args.des_dir
    dataset_downloader = DatasetDownloader(years, des_dir)

if __name__ == "__main__":    
    main()

