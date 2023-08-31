# coding: utf-8
import os.path

import numpy as np
from torchvision import datasets, transforms


def build_iid_data(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        user_items = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - user_items)
        dict_users[i] = list(user_items)
    return dict_users


def build_noniid_data(dataset, num_users):
    num_shards, num_imgs = 2 * num_users, len(dataset.targets) // (2 * num_users)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def build_dir_data(dataset, num_users, alpha):
    train_labels = np.array(dataset.targets, dtype='int64')
    n_classes = np.max(train_labels) + 1
    # Dirichlet distribution
    label_distribution = np.random.dirichlet([alpha] * num_users, n_classes)
    # (K, N) category label distribution matrix X, recording how much each client occupies each category

    class_idxs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]
    # Record the sample subscript corresponding to each K category

    client_idxs = [[] for _ in range(num_users)]
    # Record the indexes of N clients corresponding to the sample collection
    for c, fracs in zip(class_idxs, label_distribution):
        # np.split divides the samples of category k into N subsets according to the proportion
        # for i, idxs is the index of traversing the sample set corresponding to the i-th client
        for i, idxs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idxs[i] += [idxs]

    client_idxs = [np.concatenate(idxs) for idxs in client_idxs]

    dict_users = {}
    for i in range(len(client_idxs)):
        dict_users[i] = client_idxs[i]
    return dict_users


def draw_data_distribution(dict_users, dataset, num_class, fig_path="./"):
    import matplotlib.pyplot as plt
    targets = dataset.targets
    # plt.figure(figsize=(20, 3))

    plt.hist([np.array(targets)[list(idc)] for idc in dict_users.values()], stacked=True,
             bins=np.arange(min(targets) - 0.5, max(targets) + 1.5, 1),
             label=["C{}".format(i) for i in range(len(dict_users))], rwidth=0.5)
    plt.xticks(np.arange(num_class), rotation=70)
    plt.legend(loc=(0.95, -0.1))
    plt.savefig(os.path.join(fig_path, "data_distribution.jpg"))
    plt.show()


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 10
    d = build_iid_data(dataset_train, num)
    draw_data_distribution(d, dataset_train, 10)

    dataset_train = datasets.CIFAR10('../data/cifar/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 10
    d = build_iid_data(dataset_train, num)
    draw_data_distribution(d, dataset_train, 10)
