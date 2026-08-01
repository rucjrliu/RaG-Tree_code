#pragma once

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>

#include <iostream>
#include <utility>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <filesystem>
#include <ctime>

#include <NumCpp.hpp>
#include <cnpy.h>

using namespace std;

// bool create_folder_if_not_exists(const std::string& path)

// {

//     namespace fs = std::filesystem;

//     try

//     {

//         // 已存在

//         if (fs::exists(path))

//         {

//             return false;

//         }

//         // 创建文件夹

//         return fs::create_directories(path);

//     }

//     catch (const fs::filesystem_error& e)

//     {

//         cerr << "filesystem error: " << e.what() << endl;

//         return false;

//     }

// }

uint64_t folder_size(const char *c_path)

{
    ostringstream ossp;
    ossp << c_path;
    string path = ossp.str();
    namespace fs = std::filesystem;

    uint64_t size = 0;

    for (const auto& e : fs::recursive_directory_iterator(path))

        if (e.is_regular_file())

            size += e.file_size();

    return size;

}