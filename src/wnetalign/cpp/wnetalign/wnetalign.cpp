#include <iostream>

#include <nanobind/nanobind.h>


NB_MODULE(wnetalign_cpp, m) {
    m.doc() = "WNetAlign C++ imlementation module";
    m.def("wnetalign_cpp_hello", []() {
        std::cout << "Hello from WNetAlign (C++)!" << std::endl;
    }, "A simple hello world function for the WNetAlign (C++) extension");
}