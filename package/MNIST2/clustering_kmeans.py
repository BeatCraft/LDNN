# coding: utf-8

from sklearn.cluster import KMeans
import numpy
from matplotlib import pyplot

import os.path

CLUSTERS = 10
DATA_MAX = 0
OUTPUT_IMAGE_MAX = 5

def save_csv_image(file_name, data_csv):
    data = numpy.asfarray(data_csv.split(',')).reshape((28, 28))
    pyplot.imshow(data, cmap="Greys", interpolation='None')
    pyplot.savefig(file_name)

def load_data(_cls):
    data_file = os.path.join("classified_data", "class_{}".format(_cls), "image.csv")
    if not os.path.exists(data_file):
        print("Data not found {}".format(_cls))
        return []

    with open(data_file) as fp:
        data_list = fp.readlines()
    if DATA_MAX > 0:
        data_list = data_list[:DATA_MAX]

    return data_list

def normalize_data(data_list):
    norm_data_list = []
    for data in data_list:
        norm_data = [int(i)/256 for i in data.split(",")]
        norm_data_list.append(norm_data)
    return norm_data_list

def save_clustered_data(_cls, data_list, labels, distances):
    # make clustered_data
    clustered_data_distance = [[] for _ in range(CLUSTERS)]
    for i, cluster in enumerate(labels):
        clustered_data_distance[cluster].append([
            distances[i][cluster],
            data_list[i],
        ])

    # save data
    for cluster in range(CLUSTERS):
        sorted_data_distance = sorted(clustered_data_distance[cluster])

        # print distances
        # print([d[0] for d in sorted_data_distance])

        save_file = os.path.join(
            "clustered_data",
            "class_{}".format(_cls),
            "cluster_{}".format(cluster),
            "image.csv"
        )
        save_dir = os.path.dirname(save_file)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        csv_data = [d[1] for d in sorted_data_distance]
        with open(save_file, "w") as fp:
            fp.writelines(csv_data)

        for i in range(len(csv_data)):
            if i >= OUTPUT_IMAGE_MAX:
                break
            image_dir = os.path.join(save_dir, "image")
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            image_file = os.path.join(image_dir, "class_{}_cluster_{}_{:0>4}.png".format(_cls, cluster, i))
            save_csv_image(image_file, csv_data[i])

    return

def main():
    for _cls in range(10):
        print("Class {}".format(_cls))
        data_list = load_data(_cls)
        norm_data_list = normalize_data(data_list)

        km = KMeans(n_clusters=CLUSTERS)
        labels = km.fit_predict(norm_data_list)
        distances = km.transform(norm_data_list)
        save_clustered_data(_cls, data_list, labels, distances)

if __name__ == "__main__":
    main()

