/*
 * Bowyer-Watson algorithm
 * C++ implementation of http://paulbourke.net/papers/triangulate .
 **/

#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>

#include "common.hpp"

using namespace sycl;

using PipelinedLSU = ext::intel::lsu<>;

template<typename T>
double delaunay_triang_kernel(queue &q, const std::vector<Point<T>> &points, 
                                        const Delaunay<T> d) {
  std::cout << "Static HLS\n";

  /* Init Delaunay triadgulation. */
  buffer points_buf(points);
  buffer d_triangles_buf(d.triangles);
  buffer d_edges_buf(d.edges);

  sycl::event event = q.submit([&](handler &hnd) {
    accessor points(points_buf, hnd, read_only);
    accessor d_triangles(d_triangles_buf, hnd, read_write);
    accessor d_edges(d_edges_buf, hnd, read_write);

    hnd.single_task<class StaticDelaunayTriangKernel>([=]() [[intel::kernel_args_restrict]] {
      using Node = Point<T>;

      auto xmin = points[0].x;
      auto xmax = xmin;
      auto ymin = points[0].y;
      auto ymax = ymin;
      for (uint i_point = 0; i_point < NUM_POINTS; ++i_point) {
        const auto pt = points[i_point];

        xmin = min(xmin, pt.x);
        xmax = max(xmax, pt.x);
        ymin = min(ymin, pt.y);
        ymax = max(ymax, pt.y);
      }

      const auto dx = xmax - xmin;
      const auto dy = ymax - ymin;
      const auto dmax = std::max(dx, dy);
      const auto midx = (xmin + xmax) / static_cast<T>(2.);
      const auto midy = (ymin + ymax) / static_cast<T>(2.);

      const auto p0 = Node{midx - 20 * dmax, midy - dmax};
      const auto p1 = Node{midx, midy + 20 * dmax};
      const auto p2 = Node{midx + 20 * dmax, midy - dmax};
      d_triangles[0] = Triangle<T>{p0, p1, p2};

      for (uint i_point = 0; i_point < NUM_POINTS; ++i_point) {
        const auto pt = points[i_point];

        Edge<T> edges[NUM_POINTS*9];
        uint i_edges = 0;
        Triangle<T> tmps[NUM_POINTS*9];
        uint i_tmps = 0;

        for (uint i_tri = 0; i_tri < NUM_POINTS; ++i_tri) {
          const auto tri = d_triangles[i_tri];

          /* Check if the point is inside the triangle circumcircle. */
          const auto dist = (tri.circle.x - pt.x) * (tri.circle.x - pt.x) +
                            (tri.circle.y - pt.y) * (tri.circle.y - pt.y);
          if ((dist - tri.circle.radius) <= eps) {
            edges[i_edges++] = Edge(tri.e0);
            edges[i_edges++] = Edge(tri.e1);
            edges[i_edges++] = Edge(tri.e2);
          } else {
            tmps[i_tmps++] = Triangle(tri);
          }
        }

        /* Delete duplicate edges. */
        bool remove[NUM_POINTS*3];
        for (uint it1 = 0; it1 < NUM_POINTS*3; ++it1) {
          remove[it1] = false;
          for (uint it2 = it1; it2 < NUM_POINTS*3; ++it2) {
            if (it1 == it2) {
              continue;
            }
            if (edges[it1] == edges[it2]) {
              remove[it1] = true;
              remove[it2] = true;
            }
          }
        }

        /* Update triangulation. */
        for (uint i = 0; i < NUM_POINTS*3; ++i) {
          if (!remove[i]) {
            tmps[i_tmps++] = {edges[i].p0, edges[i].p1, {pt.x, pt.y}};
          }
        }
        for (uint i = 0; i < i_tmps; ++i) {
          d_triangles[i] = tmps[i];
        }
      }

      /* Add edges. */
      uint i_edges = 0;
      for (int i = 0; i < NUM_POINTS; ++i) {
        /* Remove original super triangle. */
        if ((d_triangles[i].p0 == p0 || d_triangles[i].p1 == p0 || d_triangles[i].p2 == p0) ||
            (d_triangles[i].p0 == p1 || d_triangles[i].p1 == p1 || d_triangles[i].p2 == p1) ||
            (d_triangles[i].p0 == p2 || d_triangles[i].p1 == p2 || d_triangles[i].p2 == p2)) {
          continue;
        }
        else {
          d_edges[i_edges++] = d_triangles[i].e0;
          d_edges[i_edges++] = d_triangles[i].e1;
          d_edges[i_edges++] = d_triangles[i].e2;
        }
      }
    });
  });

  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
