import numpy as np

def triangulation(
        camera_matrix: np.ndarray,
        camera_position1: np.ndarray,
        camera_rotation1: np.ndarray,
        camera_position2: np.ndarray,
        camera_rotation2: np.ndarray,
        image_points1: np.ndarray,
        image_points2: np.ndarray
):
    """
    :param camera_matrix: first and second camera matrix, np.ndarray 3x3
    :param camera_position1: first camera position in world coordinate system, np.ndarray 3x1
    :param camera_rotation1: first camera rotation matrix in world coordinate system, np.ndarray 3x3
    :param camera_position2: second camera position in world coordinate system, np.ndarray 3x1
    :param camera_rotation2: second camera rotation matrix in world coordinate system, np.ndarray 3x3
    :param image_points1: points in the first image, np.ndarray Nx2
    :param image_points2: points in the second image, np.ndarray Nx2
    :return: triangulated points, np.ndarray Nx3
    """

    # YOUR CODE HERE
    camera_pos1 = camera_position1.reshape((3, 1))
    camera_pos2 = camera_position2.reshape((3, 1))

    rot1 = camera_rotation1.transpose()
    rot2 = camera_rotation2.transpose()

    tr1 = -rot1 @ camera_pos1
    tr2 = -rot2 @ camera_pos2

    extr1 = np.hstack((rot1, tr1))
    extr2 = np.hstack((rot2, tr2))
    proj1 = camera_matrix @ extr1
    proj2 = camera_matrix @ extr2
    triangulated = []
    # Solve for each pair of corresponding points
    for point1, point2 in zip(image_points1, image_points2):
        # Create the linear system of equations
        A = np.vstack([
            point1[0] * proj1[2, :] - proj1[0, :],
            point1[1] * proj1[2, :] - proj1[1, :],
            point2[0] * proj2[2, :] - proj2[0, :],
            point2[1] * proj2[2, :] - proj2[1, :]
        ])
        # Solve the system using SVD
        _, _, vh = np.linalg.svd(A)
        X = vh[-1]
        X /= X[3]  # Convert to inhomogeneous coordinates

        triangulated.append(X[:3])
    return np.array(triangulated)
