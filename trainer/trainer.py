import argparse
from trainer.dataset import Dataset
from trainer.models.cgan_model import CGANModel


def main(job_dir, dataset_path=None, local=False, generator_weights=None, discriminator_weights=None, encoder_weights=None):
    dataset = Dataset(dataset_path, local)

    aging_model = CGANModel(job_dir + "/", local, generator_weights, discriminator_weights, encoder_weights)

    aging_model.train_gpu(dataset, job_dir + "/logs/tensorboard")

    aging_model.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path",
                        help="Path to dataset")

    parser.add_argument("--job-dir",
                        help="GCS relative location within bucket to write checkpoints and export models",
                        default="out")

    parser.add_argument("--local",
                        help="True if training should be run locally",
                        type=bool)

    parser.add_argument("--generator_weights",
                        help="Path to generator weights")

    parser.add_argument("--discriminator_weights",
                        help="Path to discriminator weights")

    parser.add_argument("--encoder_weights",
                        help="Path to encoder weights")

    arguments = parser.parse_args()
    arguments = arguments.__dict__

    main(**arguments)

