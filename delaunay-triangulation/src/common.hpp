#pragma once

#include <algorithm>
#include <iostream>
#include <vector>


constexpr double eps = 1e-4;

constexpr uint NUM_POINTS = N_POINTS;

template <typename T>
struct Point {
  T x, y;

  Point() : x{0}, y{0} {}
  Point(T _x, T _y) : x{_x}, y{_y} {}

  template <typename U>
  Point(U _x, U _y) : x{static_cast<T>(_x)}, y{static_cast<T>(_y)}
  {
  }

  friend std::ostream& operator<<(std::ostream& os, const Point<T>& p)
  {
    os << "x=" << p.x << "  y=" << p.y;
    return os;
  }

  bool operator==(const Point<T>& other) const
  {
    return (other.x == x && other.y == y);
  }

  bool operator!=(const Point<T>& other) const { return !operator==(other); }
};

template <typename T>
struct Edge {
  using Node = Point<T>;
  Node p0, p1;

  Edge() : p0{}, p1{} {}

  Edge(const Edge &e) : p0{e.p0}, p1{e.p1} {}

  Edge(Node const& _p0, Node const& _p1) : p0{_p0}, p1{_p1} {}

  friend std::ostream& operator<<(std::ostream& os, const Edge& e)
  {
    os << "p0: [" << e.p0 << " ] p1: [" << e.p1 << "]";
    return os;
  }

  bool operator==(const Edge& other) const
  {
    return ((other.p0 == p0 && other.p1 == p1) ||
            (other.p0 == p1 && other.p1 == p0));
  }
};

template <typename T>
struct Circle {
  T x, y, radius;
  Circle() = default;
};

template <typename T>
struct Triangle {
  using Node = Point<T>;
  Node p0, p1, p2;
  Edge<T> e0, e1, e2;
  Circle<T> circle;

  Triangle() 
      : p0{},
        p1{},
        p2{},
        e0{},
        e1{},
        e2{},
        circle{}
  {}

  Triangle(const Node& _p0, const Node& _p1, const Node& _p2)
      : p0{_p0},
        p1{_p1},
        p2{_p2},
        e0{_p0, _p1},
        e1{_p1, _p2},
        e2{_p0, _p2},
        circle{}
  {
    const auto ax = p1.x - p0.x;
    const auto ay = p1.y - p0.y;
    const auto bx = p2.x - p0.x;
    const auto by = p2.y - p0.y;

    const auto m = p1.x * p1.x - p0.x * p0.x + p1.y * p1.y - p0.y * p0.y;
    const auto u = p2.x * p2.x - p0.x * p0.x + p2.y * p2.y - p0.y * p0.y;
    const auto s = 1. / (2. * (ax * by - ay * bx));

    circle.x = ((p2.y - p0.y) * m + (p0.y - p1.y) * u) * s;
    circle.y = ((p0.x - p2.x) * m + (p1.x - p0.x) * u) * s;

    const auto dx = p0.x - circle.x;
    const auto dy = p0.y - circle.y;
    circle.radius = dx * dx + dy * dy;
  }
};

template <typename T>
struct Delaunay {
  std::vector<Triangle<T>> triangles;
  std::vector<Edge<T>> edges;

  Delaunay(const uint n_points)
    : triangles(std::vector<Triangle<T>>(n_points*9)),
      edges(std::vector<Edge<T>>(n_points*9))
  {}
  
};
