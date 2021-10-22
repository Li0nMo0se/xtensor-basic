#include <cassert>
#include <numeric>
#include <pybind11/pybind11.h>
#include <xtensor.hpp>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>

double sum_of_sines(xt::pyarray<double>& m)
{
    return xt::sum(xt::sin(m), {0})(0);
}

// Not that useful but test tbb efficiency
xt::pyarray<int> tbb_sum(xt::pyarray<int>& m, unsigned int ksize)
{
    assert(m.dimension() == 3);

    xt::xarray<int>::shape_type shape = {m.shape(0), m.shape(1)};
    xt::xarray<int> res(shape);

    tbb::parallel_for(
        tbb::blocked_range2d<int>(0, m.shape(0) / ksize, 0, m.shape(1) / ksize),
        [&res, &m, ksize](const tbb::blocked_range2d<int>& r) {
            for (int y = std::begin(r.rows()); y != std::end(r.rows()); y++)
            {
                for (int x = std::begin(r.cols()); x != std::end(r.cols()); x++)
                {
                    size_t real_y = y * ksize;
                    size_t real_x = x * ksize;
                    auto m_view = xt::view(m,
                                           xt::range(real_y, real_y + ksize),
                                           xt::range(real_x, real_x + ksize),
                                           xt::all());
                    xt::view(res,
                             xt::range(real_y, real_y + ksize),
                             xt::range(real_x, real_x + ksize)) =
                        xt::sum(m_view, {2});
                }
            }
        });
    return res;
}

xt::pyarray<int> sum(xt::pyarray<int>& m, unsigned int ksize)
{
    assert(m.dimension() == 3);
    xt::xarray<int>::shape_type shape = {m.shape(0), m.shape(1)};
    xt::xarray<int> res(shape);

    // auto range_y = xt::range(0, m.shape(0), ksize);
    // auto range_x = xt::range(0, m.shape(1), ksize);

    for (size_t y = 0; y < m.shape(0); y += ksize)
    {
        for (size_t x = 0; x < m.shape(1); x += ksize)
        {
            auto m_view = xt::view(m,
                                   xt::range(y, y + ksize),
                                   xt::range(x, x + ksize),
                                   xt::all());
            // std::cout << xt::adapt(m_view.shape()) << "\n";
            xt::view(res, xt::range(y, y + ksize), xt::range(x, x + ksize)) =
                xt::sum(m_view, {2});
        }
    }
    return res;
}

PYBIND11_MODULE(mymodule, m)
{
    xt::import_numpy();
    m.doc() = "Test module for xtensor python bindings";
    m.def("sum_of_sines",
          sum_of_sines,
          "Sum the sines of the input values using smid operations");
    m.def("sum", sum, "Sum (axis=2) processed without tbb");
    m.def("tbb_sum", tbb_sum, "Sum (axis=2) processed with tbb");
}
