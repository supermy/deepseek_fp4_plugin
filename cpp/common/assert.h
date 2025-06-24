#pragma once

#include <stdexcept>
#include <string>

#define TLLM_CHECK_WITH_INFO(condition, message) \
    do { \
        if (!(condition)) { \
            throw std::runtime_error(message); \
        } \
    } while (0)

#define TLLM_CHECK(condition) TLLM_CHECK_WITH_INFO(condition, "Assertion failed: " #condition) 