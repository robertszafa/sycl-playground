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

  const uint num_points = points.size();

  /* Init Delaunay triadgulation. */
  buffer points_buf(points);
  buffer d_triangles_buf(d.triangles);
  buffer d_triangles2_buf(d.triangles);
  buffer d_edges_buf(d.edges);

  buffer<Edge<T>> edges_buf(range<1>{MAX_POINTS*3});
  buffer<Triangle<T>> tmps_buf(range<1>{MAX_POINTS*3});
  buffer<bool> remove_buf(range<1>{MAX_POINTS*3});

  uint num_final_triangles;
  buffer num_final_triangles_buf(&num_final_triangles, range<1>(1));

  sycl::event event = q.submit([&](handler &hnd) {
    accessor points(points_buf, hnd, read_only);
    accessor d_triangles(d_triangles_buf, hnd, read_write);
    accessor d_triangles2(d_triangles_buf, hnd, read_write);
    accessor d_edges(d_edges_buf, hnd, read_write);

    accessor tmps(tmps_buf, hnd, read_write, no_init);
    accessor edges(edges_buf, hnd, read_write, no_init);
    accessor remove(remove_buf, hnd, read_write, no_init);

    accessor num_final_triangles(num_final_triangles_buf, hnd, write_only, no_init);

    hnd.single_task<class StaticDelaunayTriangKernel>([=]() [[intel::kernel_args_restrict]] {
      using Node = Point<T>;

      T xmin = points[0].x;
      T xmax = xmin;
      T ymin = points[0].y;
      T ymax = ymin;
      for (uint i_point = 0; i_point < num_points; ++i_point) {
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

      // add super-triangle to triangulation 
      const auto p0 = Node{midx - T(20.0) * dmax, midy - dmax};
      const auto p1 = Node{midx, midy + T(20.0) * dmax};
      const auto p2 = Node{midx + T(20.0) * dmax, midy - dmax};
      d_triangles[0] = Triangle<T>{p0, p1, p2};

      uint num_triangles = 1;

      for (uint i_point = 0; i_point < num_points; ++i_point) {
        const auto pt = points[i_point];
        uint i_edges = 0;

        // first find all the triangles that are no longer valid due to the insertion
        uint num_good_triangles = 0;
        for (uint i_tri = 0; i_tri < num_triangles; ++i_tri) {
          const auto tri = d_triangles[i_tri];

          // Check if the point is inside the triangle circumcircle. 
          const auto dist = (tri.circle.x - pt.x) * (tri.circle.x - pt.x) +
                            (tri.circle.y - pt.y) * (tri.circle.y - pt.y);
          if ((dist - tri.circle.radius) <= eps) {
            // add edges to pool
            edges[i_edges++] = tri.e0;
            edges[i_edges++] = tri.e1;
            edges[i_edges++] = tri.e2;
          }
          else {
            // Keep good triangles in seperate buffer.
            d_triangles2[num_good_triangles++] = d_triangles[i_tri];
          }
        }

        // Delete duplicate edges.
        for (uint it1 = 0; it1 < i_edges; ++it1) {
          remove[it1] = false;
        }
        for (uint it1 = 0; it1 < i_edges; ++it1) {
          for (uint it2 = it1+1; it2 < i_edges; ++it2) {
            if (almost_equal(edges[it1], edges[it2])) {
              remove[it1] = true;
              remove[it2] = true;
            }
          }
        }
        uint num_good_edges = 0;
        for (uint i = 0; i < i_edges; ++i) {
          if (!remove[i]) {
            edges[num_good_edges++] = edges[i];
          }
        }

        // Update triangulation.
        for (uint i = 0; i < num_good_triangles; ++i) {
          d_triangles[i] = d_triangles2[i];
        }
        num_triangles = num_good_triangles;
        for (uint i = 0; i < num_good_edges; ++i) {
          d_triangles[num_triangles++] = {edges[i].p0, edges[i].p1, pt};
        }
      } // end top loop

      // Remove original super triangle.
      uint num_good_triangles = 0;
      for (int i = 0; i < num_triangles; ++i) {
        if ((d_triangles[i].p0 == p0 || d_triangles[i].p1 == p0 || d_triangles[i].p2 == p0) ||
            (d_triangles[i].p0 == p1 || d_triangles[i].p1 == p1 || d_triangles[i].p2 == p1) ||
            (d_triangles[i].p0 == p2 || d_triangles[i].p1 == p2 || d_triangles[i].p2 == p2)) {
          continue;
        }
        else {
          d_triangles[num_good_triangles++] = d_triangles[i];
        }
      }

      num_final_triangles[0] = num_good_triangles;
    });
  });

  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;


  std::cout << "Number of final triangles: " << num_final_triangles << "\n";
  host_accessor d_triangles_host(d_triangles_buf);

  if (num_final_triangles < 10) {
    for (int i = 0; i < num_final_triangles; ++i) {
      std::cout << d_triangles_host[i] << "\n";
    }
  }

  return time_in_ms;
}
