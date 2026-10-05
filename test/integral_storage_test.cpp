#include <green/symmetry/integral_pair_map.h>
#include <catch2/catch_test_macros.hpp>
#include <limits>

TEST_CASE("SG representative preload size is checked", "[integral-storage]") {
  using green::symmetry::integral_storage_bytes;
  REQUIRE(integral_storage_bytes(24,10,8,16,245760)==245760);
  REQUIRE_THROWS_AS(integral_storage_bytes(24,10,8,16,245759),std::runtime_error);
  REQUIRE_THROWS_AS(integral_storage_bytes(std::numeric_limits<std::size_t>::max(),10,8,16,
                                         std::numeric_limits<std::size_t>::max()),std::overflow_error);
}
