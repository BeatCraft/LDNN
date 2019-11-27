# coding: utf-8

import os
import os.path
import numpy
from matplotlib import pyplot

IMAGE_CSV = "original_data/image.csv"
LABEL_CSV = "original_data/label.csv"

MAX_OUTPUT_IMAGE = 10

def save_csv_image(file_name, data_csv):
    data = numpy.asfarray(data_csv.split(',')).reshape((28, 28))
    pyplot.imshow(data, cmap="Greys", interpolation='None')
    pyplot.savefig(file_name)

def main():
    classified_data = [[] for _ in range(10)]

    # read original data
    with open(IMAGE_CSV) as fp:
        images = fp.readlines()
    with open(LABEL_CSV) as fp:
        labels = fp.readlines()

    data_len = min(len(images), len(labels))
    print("data length {}".format(data_len))

    # classify
    for i in range(data_len):
        index = int(labels[i])
        classified_data[index].append(images[i])

    # write image.csv and png files
    for _cls in range(len(classified_data)):
        data_dir = os.path.join("classified_data", "class_{}".format(_cls))
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        data_file = os.path.join(data_dir, "image.csv")
        with open(data_file, "w") as fp:
            fp.writelines(classified_data[_cls])

        image_dir = os.path.join(data_dir, "image")
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        for index, data_str in enumerate(classified_data[_cls]):
            if index >= MAX_OUTPUT_IMAGE:
                break
            image_file = os.path.join(image_dir, "class_{}_{:0>4}.png".format(_cls, index))
            save_csv_image(image_file, data_str)


if __name__ == "__main__":
    main()

