import numpy as np
from scipy.ndimage import find_objects

import numba
from numba import njit, prange


@njit("(float64[:], int32[:], int32[:], int32, int32, int32, int32)", nogil=True)
def _extend_centers(T, y, x, ymed, xmed, Lx, niter):
    """Run diffusion from the center of the mask on the mask pixels.

    Args:
        T (numpy.ndarray): Array of shape (Ly * Lx) where diffusion is run.
        y (numpy.ndarray): Array of y-coordinates of pixels inside the mask.
        x (numpy.ndarray): Array of x-coordinates of pixels inside the mask.
        ymed (int): Center of the mask in the y-coordinate.
        xmed (int): Center of the mask in the x-coordinate.
        Lx (int): Size of the x-dimension of the masks.
        niter (int): Number of iterations to run diffusion.

    Returns:
        numpy.ndarray: Array of shape (Ly * Lx) representing the amount of diffused particles at each pixel.
    """
    for t in range(niter):
        T[ymed * Lx + xmed] += 1
        T[y * Lx +
          x] = 1 / 9. * (T[y * Lx + x] + T[(y - 1) * Lx + x] + T[(y + 1) * Lx + x] +
                         T[y * Lx + x - 1] + T[y * Lx + x + 1] +
                         T[(y - 1) * Lx + x - 1] + T[(y - 1) * Lx + x + 1] +
                         T[(y + 1) * Lx + x - 1] + T[(y + 1) * Lx + x + 1])
    return T


def _slices_to_ndarrays(slices: list[(slice, slice)]) -> (np.ndarray, np.ndarray):
    """
    Takes a list of (slice, slice) objects representing bounding boxes
    on a 2D grid. Parses the start/end coordinates into int32 numpy arrays
    for compatibility with downstream numba functions.

    Objects are also validated here, with those with null extents being omitted.

    Arguments:
    - slices (tuple(slice, slice)): row and column slices passed from find_objects

    Returns:
    - labels (int32[:]): unique integer label for each valid object
    - bounds (int32[:, 4]): array of each valid object's bounding box, ordered as (row_start, row_end, col_start, col_end)
    """
    labels = []
    row_starts = []
    row_ends = []
    col_starts = []
    col_ends = []
    for i, si in enumerate(slices):
        if si is None:
            continue
        labels.append(i)
        rs, re = si[0].start, si[0].stop
        cs, ce = si[1].start, si[1].stop
        row_starts.append(rs)
        row_ends.append(re)
        col_starts.append(cs)
        col_ends.append(ce)
    labels = np.array(labels, dtype=np.int32)

    row_starts = np.array(row_starts, dtype=np.int32)
    row_ends = np.array(row_ends, dtype=np.int32)
    col_starts = np.array(col_starts, dtype=np.int32)
    col_ends = np.array(col_ends, dtype=np.int32)
    bounds = np.stack([row_starts, row_ends, col_starts, col_ends], axis=1)
    return labels, bounds


@njit(inline='always')
def _calculate_object_centroid(masks, label, row_start, row_end, col_start, col_end):
    """
    Calculates the centroid pixel of an object defined as where `masks == label`, and
    bounded by (row_start, row_end, col_start, col_end).

    Arguments:
    - masks (uint32[:, :]): big array of unique object labels for each pixel
    - label (int32): label of the object being processed
    - row_start, row_end, col_start, col_end (int32): bounding box for the object on masks

    Returns:
    - (int32, int32) coordinates of the object centroid in its cropped coordinate frame
    """

    # Mean coordinate within the object
    count = 0
    sum_y = 0.0
    sum_x = 0.0
    for r in range(row_start, row_end):
        for c in range(col_start, col_end):
            if masks[r, c] == label + 1:
                sum_y += (r - row_start + 1)
                sum_x += (c - col_start + 1)
                count += 1
    if count == 0:  # should be unreachable as long as the objects are pre-validated
        return -1, -1
    mean_y = sum_y / count
    mean_x = sum_x / count

    # Argmin squared distance from the mean coordinate
    min_dist = count
    min_y = 0
    min_x = 0
    for r in range(row_start, row_end):
        for c in range(col_start, col_end):
            if masks[r, c] == label + 1:
                obj_y = r - row_start + 1
                obj_x = c - col_start + 1
                dy = obj_y - mean_y
                dx = obj_x - mean_x
                dist = dy * dy + dx * dx
                if dist < min_dist:
                    min_dist = dist
                    min_y = obj_y
                    min_x = obj_x

    return min_y - 1, min_x - 1


@njit(parallel=True)
def _calculate_all_centroids(masks, labels, bounds):
    """
    Parallel calculation of the centroids of all objects in masks.

    Arguments:
    - masks (uint32[:, :]): big array of unique object labels for each pixel
    - labels (int32[:]): labels of all valid objects
    - bounds (int32[:, 4]): bounding boxes for each object on masks

    Returns:
    - int32[:, 2] coordinates of each object's centroid in its cropped coordinate frame
    """
    n = labels.shape[0]
    centroids = np.zeros((n, 2), dtype=np.int32)
    for i in prange(n):
        cy, cx = _calculate_object_centroid(masks,
                                            label=labels[i],
                                            row_start=bounds[i, 0],
                                            row_end=bounds[i, 1],
                                            col_start=bounds[i, 2],
                                            col_end=bounds[i, 3]
                                            )
        centroids[i, 0] = cy
        centroids[i, 1] = cx
    return centroids


@njit(inline='always')
def _get_object_coordinates(masks, label, row_start, row_end, col_start, col_end):
    """
    Finds all coordinates within a bounding box where `masks == label+1`
    and returns them as 1D arrays. Coordinates are defined relative to the full
    masks frame.

    Arguments:
    - masks (uint32[:, :]): big array of unique object labels for each pixel
    - label (int32): label of the object being processed
    - row_start, row_end, col_start, col_end (int32): bounding box for the object on masks

    Returns:
    - (int32[:], int32[:]) flattened lists of the object's coordinates on masks
    """
    row_list = []
    col_list = []
    for i in range(row_start, row_end):
        for j in range(col_start, col_end):
            if masks[i, j] == label + 1:
                row_list.append(i)
                col_list.append(j)
    row_list = np.array(row_list, dtype=np.int32)
    col_list = np.array(col_list, dtype=np.int32)
    return row_list, col_list


@njit(inline='always')
def _compute_flow(T, row_list_rel, col_list_rel, num_cols):
    """
    Given the flattened, diffused array T, calculate the gradients (dy, dx)
    at the given mask coordinates. All coordinates are defined relative to
    the object's cropped bounding box.

    Arguments:
    - T (float64[:]): flattened diffused field, from _extend_centers()
    - row_list_rel (int32[:]): array of y-coordinates for the object under consideration
    - col_list_rel (int32[:]): array of x-coordinates for the object under consideration
    - num_cols (int32): width of the diffused field, for indexing into the flattened field

    Returns:
    - (float64[:], float64[:]) gradients
    """
    n = row_list_rel.shape[0]
    dy = np.empty(n, dtype=T.dtype)
    dx = np.empty(n, dtype=T.dtype)
    for i in range(n):
        r = row_list_rel[i]
        c = col_list_rel[i]
        dy[i] = T[(r + 1) * num_cols + c] - T[(r - 1) * num_cols + c]
        dx[i] = T[r * num_cols + (c + 1)] - T[r * num_cols + (c - 1)]
    return dy, dx


@njit
def _update_gradient_field(mu, row_list, col_list, dy, dx):
    """
    Updates `mu` in-place, writing (dy, dx) to the coordinates
    specified by row_list and col_list.

    Arguments:
    - mu (float64[2, :, :]): gradient field to be modified
    - row_list (int32[:]): array of y-coordinates for the object under consideration
    - col_list (int32[:]): array of x-coordinates for the object under consideration
    - dx, dy (float64[:]): flattened gradient values at the object's coordinates
    """
    n = dy.shape[0]
    for i in range(n):
        r = row_list[i]
        c = col_list[i]
        mu[0, r, c] = dy[i]
        mu[1, r, c] = dx[i]


@njit(parallel=True)
def _get_all_gradients(masks, centroids, labels, bounds, niter):
    """
    Given a set of objects on masks, diffuse on each object in parallel
    to calculate the corresponding gradient field.

    Arguments:
    - masks (uint32[:, :]): big array of unique object labels for each pixel
    - centroids: int32[:, 2] coordinates of each object's centroid in its cropped coordinate frame
    - labels (int32[:]): labels of all valid objects
    - bounds (int32[:, 4]): bounding boxes for each object on masks
    - niter (int32): number of iterations to expand (if None, calculated based on object size)

    Returns:
    - mu (float64[2, :, :]): gradient field for all objects
    """

    Ly, Lx = masks.shape
    mu = np.zeros((2, Ly, Lx), np.float64)

    n_objects = labels.shape[0]
    for i in prange(n_objects):

        if centroids[i][0] < 0 or centroids[i][1] < 0:
            # _calculate_object_centroid returned (-1, -1) due to the object having a null span
            continue

        row_list, col_list = _get_object_coordinates(masks,
                                                     label=labels[i],
                                                     row_start=bounds[i, 0],
                                                     row_end=bounds[i, 1],
                                                     col_start=bounds[i, 2],
                                                     col_end=bounds[i, 3])

        row_list_rel = row_list - bounds[i, 0] + np.int32(1)
        col_list_rel = col_list - bounds[i, 2] + np.int32(1)

        ly = (bounds[i, 1] - bounds[i, 0]) + np.int32(2)
        lx = (bounds[i, 3] - bounds[i, 2]) + np.int32(2)

        n_iter = 2 * np.int32(ly + lx) if niter is None else niter
        T = np.zeros(ly * lx, dtype=np.float64)
        T = _extend_centers(T, row_list_rel, col_list_rel, centroids[i][0], centroids[i][1], lx, n_iter)
        dy, dx = _compute_flow(T, row_list_rel, col_list_rel, lx)
        _update_gradient_field(mu, row_list, col_list, dy, dx)

    return mu


def _normalize_gradient_field(mu):
    """
    Normalizes the gradient field in-place.

    Arguments:
    - mu (float64[2, :, :]
    """
    epsilon = 1e-60
    mu /= (epsilon + (mu ** 2).sum(axis=0) ** 0.5)


def masks_to_flows_cpu_parallel(masks, device=None, niter=None):
    """
    Same as dynamics.masks_to_flows_cpu(), but using numba to parallelize across objects.

    Convert masks to flows using diffusion from center pixel.

    Center of masks where diffusion starts is defined to be the closest pixel to the mean of all pixels that is inside the mask.
    Result of diffusion is converted into flows by computing the gradients of the diffusion density map.

    Args:
        masks (int, 2D or 3D array): Labelled masks 0=NO masks; 1,2,...=mask labels
        device: not used in this method

    Returns:
        tuple containing
            - mu (float, 3D or 4D array): Flows in Y = mu[-2], flows in X = mu[-1].
                If masks are 3D, flows in Z = mu[0].
            - meds (float, 2D or 3D array): cell centers
    """
    slices = find_objects(masks)
    labels, bounds = _slices_to_ndarrays(slices)
    centroids = _calculate_all_centroids(masks, labels, bounds)
    mu = _get_all_gradients(masks, centroids + 1, labels, bounds, niter)  # +1 to account for padding
    _normalize_gradient_field(mu)
    return mu, centroids
