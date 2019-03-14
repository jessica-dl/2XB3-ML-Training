import argparse
from trainer.dataset import Dataset
from trainer.models.aging_model import AgingModel


def main(bucket, job_dir, dataset_url, **args):
    data = []  # TODO get data from API
    dataset = Dataset(dataset_url, data)

    aging_model = AgingModel()

    aging_model.train(dataset, bucket + job_dir + "/logs/tensorboard")

    aging_model.save(bucket + job_dir + "/model.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument("--bucket",
                        help="GCS cloud bucket",
                        required=True)

    parser.add_argument("--job-dir",
                        help="GCS relative location within bucket to write checkpoints and export models",
                        required=True)

    parser.add_argument("--dataset_url",
                        help="GCS path to dataset",
                        required=True)

    arguments = parser.parse_args()
    arguments = arguments.__dict__
    print("ARGUMENTS", arguments)

    main(**arguments)
