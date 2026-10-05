project(symmetry_test)

add_executable(integral_archive_probe integral_archive_probe.cpp)
target_link_libraries(integral_archive_probe PRIVATE symmetry)

Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.3 # or a later release
)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

add_executable(symmetry_test symmetry_test.cpp integral_storage_test.cpp)
target_compile_definitions(symmetry_test PRIVATE TEST_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(symmetry_test
        PRIVATE
        Catch2::Catch2WithMain
        GREEN::SYMMETRY)

include(CTest)
include(Catch)
catch_discover_tests(symmetry_test)

set(GREEN_SG_VALIDATION_ROOT "" CACHE PATH "Optional externally generated physical SG fixtures")
if(GREEN_SG_VALIDATION_ROOT)
    foreach(kind hf correlation)
        if(kind STREQUAL "hf")
            set(fixture carbon-p-nk333-pw31-normalized)
        else()
            set(fixture carbon-p-nk333-pw31-ewald)
        endif()
        foreach(budget 0 4096 268435456)
            add_test(NAME SGPhysical_${kind}_${budget} COMMAND integral_archive_probe
                "${GREEN_SG_VALIDATION_ROOT}/${fixture}/sg"
                "${GREEN_SG_VALIDATION_ROOT}/${fixture}/native-reference.h5"
                "${GREEN_SG_VALIDATION_ROOT}/${fixture}/native-input.h5" ${budget} ${kind})
        endforeach()
    endforeach()
endif()
