#pragma once
#include <span>

template <typename T, typename U>
int sequence_search_template(std::span<T> data, U target) {
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] == target) return static_cast<int>(i);
    }
    return -1;
}