import click
import os, sys
import logging
from lib.ImageStacker import align_and_stack_images
import cv2

@click.command()
@click.option('-f', '--folder', type=str)
@click.option('-o', '--output', type=str)
@click.option('--bound', type=bool, default=True)
def cli(folder, output, bound):
    path = os.path.abspath(folder)
    files = [entry.path for entry in os.scandir(path) if entry.is_file()]
    logging.debug(f'{len(files)} Files Targeted: {files}')

    align_and_stack_images(files, output, should_crop=bound)

if __name__ == "__main__":
    fmt = '[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=fmt)
    cli()