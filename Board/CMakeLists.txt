cmake_minimum_required(VERSION 3.20.0)

list(APPEND ZEPHYR_EXTRA_MODULES
    /Users/artaosmani/sdk-edge-ai/edge-ai
)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(polyp_classifier)

include(${ZEPHYR_BASE}/samples/subsys/usb/common/common.cmake)

target_sources(app PRIVATE
    src/main.c
    src/tjpgd.c
)

target_include_directories(app PRIVATE
    src
)


