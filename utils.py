# %%
import torch
import numpy as np

# %%
def provide_lables_instances_number(number_of_labels):
    """Provide an array of data-set's lables and number of instances of each labels

    Parameters
    ----------
    number_of_labels: int
        The number of existing labels (classes) in data-set

    Returns
    -------
    lables_instances: 2D-array
        A 2D-numpy array of existing lables and number of each instances of each lable(0)
    """
    each_instances_numnber = np.zeros((number_of_labels), dtype=int)
    labels = np.arange(number_of_labels)
    lables_instances = np.array(
        [labels, each_instances_numnber]).transpose()

    return lables_instances


def provide_few_shot_dataset(train_dataset, lables_instances_array, intances_each_lable_number=10):
    """Provide an array of data-set's lables and number of instances of each labels

    Parameters
    ----------
    train_dataset: torch train dataset
        Downloaded dataset from torch.dataset

    lables_instances_array:  2D-array
        A 2D-numpy array of existing lables and number of each instances of each lable(0)

    intances_each_lable_number: int, optional (default is 10)
        A number of instances for each label (class) [few-shot]
    Returns
    -------
    train_dataset: torch train dataset
        A torch train dataset with few instances of each label
    """
    image_matrix = [np.array(train_dataset[0][0].numpy())]
    labels = [np.array([train_dataset[0][1]])]

    lables_instances_array[train_dataset[0][1]
                           ][1] = lables_instances_array[train_dataset[0][1]][1] + 1
    for i, (el) in enumerate(train_dataset):

        if lables_instances_array[train_dataset[i][1]][1] < intances_each_lable_number:
            image_matrix = np.append(
                image_matrix, [np.array(train_dataset[i][0].numpy())], axis=0)
            labels = np.append(
                labels, [np.array([train_dataset[i][1]])], axis=0)
            lables_instances_array[train_dataset[i][1]
                                   ][1] = lables_instances_array[train_dataset[i][1]][1] + 1

        if len(np.unique(lables_instances_array[:, 1])) == 1:
            if np.unique(lables_instances_array[:, 1]) == [intances_each_lable_number]:
                break

    image_matrices_tensor = torch.stack(
        [torch.Tensor(i) for i in image_matrix])

    # transform to torch tensors
    labels_tensor = torch.stack([torch.Tensor(i) for i in labels]).long()

    train_dataset = torch.utils.data.TensorDataset(
        image_matrices_tensor, labels_tensor)

    return train_dataset


def provide_dic_label(train_loader, number_of_each_classes_instance):
    """Provide a dictionary of labels with number of each label instances

    Parameters
    ----------
    train_loader: torch data loader
        The torch data loader

    Returns
    -------
    number_of_each_classes_instance: dictionary
        A dictionary of lables and numnber of instances of each label
    """
    for i, (images, labels) in enumerate(train_loader):
        for i, (label) in enumerate(labels):
            number_of_each_classes_instance[str(
                label.item())] = number_of_each_classes_instance[str(label.item())] + 1
    return number_of_each_classes_instance
