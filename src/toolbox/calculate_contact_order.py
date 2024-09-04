import torch
def calculate_contact_matrix(coords, threshold=6.0):
    """
    Calculate the contact matrix.
    :param coords: Atomic coordinates of the protein, shape (..., N, 3)
    :param threshold: Distance threshold for contacts, in Ångströms
    :return: Contact matrix, shape (..., N, N)
    """
    dist_matrix = torch.cdist(coords, coords)  # Calculate distance matrix
    contact_matrix = (dist_matrix < threshold).float()  # Generate contact matrix
    return contact_matrix

def calculate_contact_order(coords, contact_matrix):
    """
    Calculate the contact order.
    :param coords: Atomic coordinates of the protein, shape (..., N, 3)
    :param contact_matrix: Contact matrix, shape (..., N, N)
    :return: Contact order
    """
    L = coords.shape[-2]
    indices = torch.arange(L, device=coords.device)
    i, j = torch.meshgrid(indices, indices, indexing='ij')
    seq_distances = torch.abs(i - j).float()

    contact_order = (contact_matrix * seq_distances).sum(dim=(-2, -1)) / contact_matrix.sum(dim=(-2, -1)) / L
    return contact_order