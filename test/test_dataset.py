import context

from dataset.colmap_solution import ColmapSolution


def test_net_gen():

    test = ColmapSolution("/database/colmap_test", 0)

    test.unit_rescale()

    rays = test.calculate_rays()

    print(rays)