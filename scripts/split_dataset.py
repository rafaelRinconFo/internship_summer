import argparse
import os
import random

import pandas as pd


class DatasetSpliter:
    def __init__(
        self,
        years: list,
        dataset_path: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        split_type: str = "shuffled",
        segment_size: int = 1000,
    ):
        """
        Args:
            
            years (list): List of years to be used in the split
            dataset_path (str): Path to the dataset
            train_ratio (float, optional): Ratio of the train set. Defaults to 0.8.
            val_ratio (float, optional): Ratio of the validation set. Defaults to 0.1.          
            test_ratio (float, optional): Ratio of the test set. Defaults to 0.1.
            split_type (str, optional): Type of split. Defaults to 'shuffled'.
            segment_size (int, optional): Size of the segment to be used in the sequential split. Defaults to 1000.
        """
        self.years = years
        self.years = [int("".join(year)) for year in years]
        self.dataset_path = dataset_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_type = split_type
        self.segment_size = segment_size
        self.df = pd.DataFrame(columns=["image_name", "year", "split"])
        print("Initializing DatasetSpliter")

    def split_dataset(self):
        year_dict = {}
        print(f"Reading images for years {self.years}")
        for year in self.years:
            images_path = os.path.join("datasets", str(year), "dense", "images")
            list_images_paths = os.listdir(images_path)
            list_images_paths.sort()
            year_dict[year] = list_images_paths
        print(f"Splitting {self.split_type} dataset")
        if self.split_type == "shuffled":
            self.shuffled_split(year_dict)
        elif self.split_type == "sequential":
            self.sequential_split(year_dict)

        print("Writing CSV")
        self.df.to_csv(
            os.path.join(self.dataset_path, f"{self.split_type}_images_split.csv"),
            index=False,
        )

    def shuffled_split(self, year_dict) -> None:
        ### shuffle the list of images
        for year, image_list in year_dict.items():
            list_to_shuffle = image_list
            random.shuffle(list_to_shuffle)
            self.split_and_write(list_to_shuffle, year, sort_data=True)

    def sequential_split(self, year_dict) -> None:
        for year, image_list in year_dict.items():
            if len(image_list) < self.segment_size:
                print(
                    f"Year {year} has less than {self.segment_size} images. Skipping..."
                )
                continue
            # split the list of images in segments of size segment_size
            segments = [
                image_list[i : i + self.segment_size]
                for i in range(0, len(image_list), self.segment_size)
            ]
            # split the segments in train, val and test
            for segment in segments:
                self.split_and_write(segment, year)

    def split_and_write(
        self, image_list: list, year: str, sort_data: bool = False
    ) -> None:
        train_size = int(len(image_list) * self.train_ratio)
        val_size = int(len(image_list) * self.val_ratio)
        test_size = int(len(image_list) * self.test_ratio)
        train_list = image_list[:train_size]
        val_list = image_list[train_size : train_size + val_size]
        test_list = image_list[
            train_size + val_size : train_size + val_size + test_size
        ]
        if sort_data:
            train_list.sort()
            val_list.sort()
            test_list.sort()
        ### add the images to the dataframe
        for image in train_list:
            df_new_row = pd.DataFrame(
                {"image_name": [image], "year": [year], "split": ["train"]}
            )
            self.df = pd.concat([self.df, df_new_row], ignore_index=True)
        for image in val_list:
            df_new_row = pd.DataFrame(
                {"image_name": [image], "year": [year], "split": ["val"]}
            )
            self.df = pd.concat([self.df, df_new_row], ignore_index=True)
        for image in test_list:
            df_new_row = pd.DataFrame(
                {"image_name": [image], "year": [year], "split": ["test"]}
            )
            self.df = pd.concat([self.df, df_new_row], ignore_index=True)
        if sort_data:
            self.df.sort_values(by=["image_name"], inplace=True)


def main():
    parser = argparse.ArgumentParser(
        description="Creates a CSV with the dataset split into train, val and test"
    )
    parser.add_argument(
        "--years", type=list, help="List of years to download", default=[], nargs="+"
    )
    parser.add_argument(
        "--dataset_path", type=str, help="Path to the dataset", default="datasets"
    )
    parser.add_argument(
        "--train_ratio", type=float, help="Ratio of the train set", default=0.8
    )
    parser.add_argument(
        "--val_ratio", type=float, help="Ratio of the validation set", default=0.1
    )
    parser.add_argument(
        "--test_ratio", type=float, help="Ratio of the test set", default=0.1
    )
    parser.add_argument(
        "--split_type", type=str, help="Type of split", default="shuffled"
    )
    parser.add_argument(
        "--segment_size",
        type=int,
        help="Size of the segment to be used in the sequential split",
        default=1000,
    )
    args = parser.parse_args()
    dataset_spliter = DatasetSpliter(
        args.years,
        args.dataset_path,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.split_type,
        args.segment_size,
    )
    dataset_spliter.split_dataset()


if __name__ == "__main__":
    main()
