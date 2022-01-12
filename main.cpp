#include <boost/math/quadrature/gauss.hpp>  // for Gauss-Legendre integration

#include <charconv>
#include <cmath>
#include <concepts>
#include <fstream>
#include <iostream>
#include <numbers>
#include <optional>
#include <ranges>
#include <vector>

template<typename T>
struct Range {
  T begin;
  T end;
};

namespace math_utilities {
  template<std::floating_point T>
  double integrate(const auto fn, Range<T> range) {
    static constexpr auto points_count = 5;

    return boost::math::quadrature::gauss<T, points_count>::integrate(fn, range.begin, range.end);
  }

  auto gaussJordan(std::vector<std::vector<double>> A, std::vector<double> b) {
    const auto swap_row = [&A, &b](int i, int j) {
      for (const auto k : std::views::iota(0u, A.size())) std::swap(A[i][k], A[j][k]);
      std::swap(b[i], b[j]);
    };

    for (const auto k : std::views::iota(0u, A.size())) {
      auto i_max = k;
      auto v_max = A[i_max][k];

      for (const auto i : std::views::iota(k + 1, A.size())) {
        if (std::abs(A[i][k]) > std::abs(v_max)) {
          v_max = A[i][k];
          i_max = i;
        }
      }

      if (i_max != k)
        swap_row(k, i_max);

      for (const auto i : std::views::iota(k + 1, A.size())) {
        const auto f = A[i][k] / A[k][k];

        for (const auto j : std::views::iota(k + 1, A.size() + 1)) {
          if (j == A.size())
            b[i] -= b[k] * f;
          else
            A[i][j] -= A[k][j] * f;
        }

        A[i][k] = 0;
      }
    }

    std::vector<double> solution(A.size(), 0.0);
    for (const auto i : std::views::iota(0, static_cast<int>(A.size()))
                            | std::views::transform([&A](const auto i) { return A.size() - i - 1; })) {
      solution[i] = b[i];

      for (const auto j : std::views::iota(i + 1u, A.size())) solution[i] -= A[i][j] * solution[j];

      solution[i] = solution[i] / A[i][i];
    }

    return solution;
  }
}  // namespace math_utilities

namespace sulution_constants {
  inline constexpr double G = 6.6743e-11;
  inline constexpr auto solution_range = Range{ 0.0, 3.0 };
  inline constexpr auto p = [](const auto x) {
    return (x > 1 && x <= 2) ? 1 : 0;
  };
  inline constexpr auto shifted = [](const auto x) {
    return 5.0 - (x / 3.0);
  };
  inline constexpr auto dshifted = [](const auto _) {
    return -1.0 / 3.0;
  };

}  // namespace sulution_constants

auto get_base_functions(const double width) {
  const auto getRanges = [width](const auto i) {
    return std::array{ width * (i - 1), width * i, width * (i + 1) };
  };

  const auto e = [width, &getRanges](const auto i) {
    return [width, i, ranges = getRanges(i)](const auto x) {
      if (x > ranges[0] && x <= ranges[1])
        return (ranges[0] - x) / width;
      else if (x > ranges[1] && x < ranges[2])
        return (x - ranges[2]) / width;
      else
        return 0.0;
    };
  };

  const auto de = [width, &getRanges](const auto i) {
    return [width, i, ranges = getRanges(i)](const auto x) {
      if (x > ranges[0] && x <= ranges[1])
        return 1.0 / width;
      else if (x > ranges[1] && x < ranges[2])
        return -1.0 / width;
      else
        return 0.0;
    };
  };

  return std::pair{ e, de };
}

auto getMatrixA(const int elements_count, const auto de, const double width) {
  std::vector<std::vector<double>> A(elements_count + 1, std::vector<double>(elements_count + 1, 0.0f));

  A[0][0] = A[elements_count][elements_count] = 1.0;
  for (const auto i : std::views::iota(1, elements_count)) {
    for (const auto j : std::views::iota(i - 1, i + 2)) {  // abs(i - j) <= 1
      if (j == 0 || j == elements_count)
        continue;

      A[i][j] = A[j][i]
          = -math_utilities::integrate([&de, i, j](const auto x) { return de(i)(x) * de(j)(x); },
                                       (i == j) ? Range{ (i - 1) * width, (i + 1) * width }
                                                : Range{ std::min(i, j) * width, std::max(i, j) * width });
    }
  }

  return A;
}

auto getMatrixL(const int elements_count, const auto de, const auto e, const double width) {
  std::vector<double> L(elements_count + 1, 0.0);

  for (const auto i : std::views::iota(1, elements_count)) {
    const auto range = Range{ (i - 1) * width, (i + 1) * width };
    const auto l1 = [&e, i](const auto x) {
      return 4 * std::numbers::pi * sulution_constants::G * sulution_constants::p(x) * e(i)(x);
    };
    const auto l2 = [&de, i](const auto x) {
      return sulution_constants::dshifted(x) * de(i)(x);
    };

    L[i] = math_utilities::integrate(l1, Range{ range.begin, range.end - width })
           + math_utilities::integrate(l1, Range{ range.begin + width, range.end })
           + math_utilities::integrate(l2, Range{ range.begin, range.end - width })
           + math_utilities::integrate(l2, Range{ range.begin + width, range.end });
  }

  return L;
}

std::vector<double> MES(const int elements_count) {
  const auto width
      = (sulution_constants::solution_range.end - sulution_constants::solution_range.begin) / elements_count;
  const auto [e, de] = get_base_functions(width);

  return math_utilities::gaussJordan(getMatrixA(elements_count, de, width), getMatrixL(elements_count, de, e, width));
}

template<typename T>
std::optional<T> get_num_input(std::string_view input) {
  T result{};
  const auto [ptr, ec] = std::from_chars(input.data(), input.data() + input.size(), result);

  if (ec == std::errc())
    return result;
  else
    return std::nullopt;
}

void draw_chart(std::string_view file_output_name, const int points_count, const std::vector<double>& solutionMatrix) {
  const auto width = (sulution_constants::solution_range.end - sulution_constants::solution_range.begin) / points_count;

  std::ofstream file(file_output_name.data());

  for (const auto [i, x] : std::views::iota(0, points_count)
                               | std::views::transform([width](const auto i) { return std::pair(i, width * i); })) {
    file << x << ' ' << solutionMatrix[i] + sulution_constants::shifted(x) << '\n';
  }

  file.close();
}

int main(int argc, char** argv) {
  if (argc == 2) {
    if (const auto parsed_input = get_num_input<int>(argv[1]); parsed_input.has_value()) {
      draw_chart("output.dat", parsed_input.value(), MES(parsed_input.value()));
      return 0;
    }
  }

  std::cout << "To use: [./app N] where N is elements count \n";
  return 1;
}
