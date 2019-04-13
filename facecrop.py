#!/usr/bin/env python

from os import listdir, makedirs
from os.path import isfile, isdir, join, dirname, exists
import time
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from glob import glob
import cv2
import click
import dlib
from face_cropper import FaceCropper

@click.command()
@click.option('--output', '-o', default='dst', help='Destination folder')
@click.option('--input', '-i', default='.', help='Input path (file or folder)')
@click.option('--mask', '-m', default='*.jpg',
              help='Mask (regex) to match files in the input path \
              (ignored if input path is a file)')
@click.option('--recursive', '-r', is_flag=True,
              help='Apply given mask recursively (ignored if input \
              path is a file)')
@click.option('--processes', '-p', default=1,
              help='Number of processes to be spawned')
@click.option('--height', '-H', default=0, help='Height of result image')
@click.option('--width', '-W', default=0, help='Width of result image')
@click.option('--ratio', '-R', default=0,
              help='Aspect ratio of result image (ignored if both height and \
              width are set)')
@click.option('--color_treshold', '-c', default=240,
              help='Grayscale color treshold for blank pixels')
@click.option('--angle_treshold', '-a', default=5,
              help='Angle treshold for tilted faces')
@click.option('--predictor_path', '-P',prompt='Path to predictor file',
              help='Path to predictor file (.dat)')
def main(input, output, processes, mask, recursive, height, width, ratio,
         color_treshold, angle_treshold, predictor_path):
    if not isfile(input) and not isdir(input):
        print("Given file or folder doesn't exist")
        return
    if not isdir(output):
        print("Output directory doesn't exist")
        return
    filenames, input = get_filenames(input, mask, recursive)
    predictor =  dlib.shape_predictor(predictor_path)
    mapper = partial(facecrop, input=input, output=output,
                     cropper=FaceCropper(predictor, height, width, ratio,
                     color_treshold, angle_treshold))
    with Pool(processes=processes) as pool:
        with tqdm(total=len(filenames)) as progress_bar:
            for i, _ in tqdm(enumerate(pool.imap(mapper, filenames))):
                progress_bar.update()

def get_filenames(filepath, mask, recursive):
    if isfile(filepath):
        filenames = [filepath]
        filepath = dirname(filepath)
        filepath = filepath if filepath else '.'
    elif isdir(filepath):
        filenames = glob(join(filepath, mask), recursive=recursive)
    return filenames, filepath

def facecrop(filepath, input, output, cropper):
    image = cv2.imread(filepath)
    face = cropper.crop(image)
    save_image(face, filepath, input, output)
    del face

def save_image(image, filepath, input, output):
    output_path = filepath.replace(input, output)
    directory = dirname(output_path)
    if not exists(directory):
        makedirs(directory)
    cv2.imwrite(output_path, image)

if __name__ == "__main__":
    main()
