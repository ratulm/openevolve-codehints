# EVOLVE-BLOCK-START
# Goal: pack 26 non-overlapping circles inside the unit square so that the
# sum of their radii is as large as possible (best known value is about 2.635).
#
# HINT: hand-placed geometric patterns (rings, grids, hexagonal rows) plateau
# well below the best known value. A stronger approach is to treat this as a
# constrained optimization problem: the variables are the 26 centers and 26
# radii, the objective is the sum of radii, and the constraints are
# radius_i <= distance to each wall and |center_i - center_j| >= radius_i +
# radius_j for every pair. Solve it with scipy.optimize.minimize(method="SLSQP")
# starting from several structured layouts with small random perturbations,
# and keep the best valid result. Circles of unequal size pack better than
# equal ones.
import numpy as np


def construct_packing():
    # Naive hand-placed starting layout: this is only a seed, not a solution.
    num_circles = 26
    circle_centers = np.zeros((num_circles, 2))

    # One circle in the middle of the square.
    circle_centers[0] = [0.5, 0.5]

    # Inner ring of 8 circles.
    for i in range(8):
        ring_angle = 2 * np.pi * i / 8
        circle_centers[i + 1] = [0.5 + 0.3 * np.cos(ring_angle), 0.5 + 0.3 * np.sin(ring_angle)]

    # Outer ring of 16 circles. Radius 0.7 puts many centers outside the square,
    # so the clip below piles them onto the walls and wastes space.
    for i in range(16):
        ring_angle = 2 * np.pi * i / 16
        circle_centers[i + 9] = [0.5 + 0.7 * np.cos(ring_angle), 0.5 + 0.7 * np.sin(ring_angle)]

    # Force every center inside the unit square.
    circle_centers = np.clip(circle_centers, 0.01, 0.99)

    # TODO: replace this greedy radius assignment with joint optimization of
    # centers and radii (see HINT at the top).
    circle_radii = greedy_radii_without_overlap(circle_centers)

    sum_of_radii = np.sum(circle_radii)

    return circle_centers, circle_radii, sum_of_radii


def greedy_radii_without_overlap(circle_centers):
    # Greedy and order-dependent: far from the largest radii these centers allow.
    num_circles = circle_centers.shape[0]
    circle_radii = np.ones(num_circles)

    # Each radius is limited by the distance to the nearest wall.
    for i in range(num_circles):
        center_x, center_y = circle_centers[i]
        circle_radii[i] = min(center_x, center_y, 1 - center_x, 1 - center_y)

    # Shrink overlapping pairs proportionally until they just touch.
    for i in range(num_circles):
        for j in range(i + 1, num_circles):
            center_distance = np.sqrt(np.sum((circle_centers[i] - circle_centers[j]) ** 2))

            if circle_radii[i] + circle_radii[j] > center_distance:
                shrink_factor = center_distance / (circle_radii[i] + circle_radii[j])
                circle_radii[i] *= shrink_factor
                circle_radii[j] *= shrink_factor

    return circle_radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(sum_radii)
