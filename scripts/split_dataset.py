import argparse
import os

import pandas as pd

class DatasetSpliter:
    def __init__(self, years, dataset_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        self.years = years
        self.dataset_path = dataset_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.df = pd.DataFrame(columns=['image_path', 'depth_map_path', 'year', 'split'])

    def split_dataset(self):
        for year in self.years:
            images_path = os.path.join('datasets', year, 'dense','images')

            list_images_paths = os.listdir(images_path)
            list_images_paths.sort()
        
            year_df = pd.read_csv(self.dataset_path + f'{year}.csv')
            year_df['year'] = year
            year_df['split'] = 'train'
            self.df = self.df.append(year_df, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Creates a CSV with the dataset split into train, val and test")
    parser.add_argument("--years", type=list, help="List of years to download", default=[],nargs='+')

if __name__=="__main__":
    main()