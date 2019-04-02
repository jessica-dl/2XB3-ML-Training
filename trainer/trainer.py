import argparse
from trainer.dataset import Dataset
from trainer.models.cgan_model import CGANModel


def main(job_dir, dataset_url, **args):
    data = []  # TODO get data from API
    dataset = Dataset(dataset_url, data)

    aging_model = CGANModel(job_dir + "/")

    aging_model.train_gpu(dataset, job_dir + "/logs/tensorboard")

    aging_model.generator_samples()

    aging_model.save()


if __name__ == "__main__":
    # main("", "")
    # exit()

    parser = argparse.ArgumentParser()

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
