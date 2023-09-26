#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "ebsynth.h"
#include <algorithm>
#include <cmath>
#define NOMINMAX

#ifdef _WIN32
#include <windows.h>
#endif

namespace py = pybind11;

class Ebsynth {
/**
 * @brief Wrapper class for the ebsynth library
 * 
 * @details This class wraps the ebsynth library and provides a Python interface
 * 
 * @param style The style image
 * @param guides A list of tuples of the form (source, target, weight)
 * @param weight The weight of the style image
 * @param uniformity The uniformity parameter
 * @param patchsize The patch size (must be odd)
 * @param pyramidlevels The number of pyramid levels
 * @param searchvoteiters The number of search vote iterations
 * @param patchmatchiters The number of patch match iterations
 * @param extrapass3x3 Whether to use an extra pass with 3x3 patches
 * @param backend_str The backend to use (cpu, cuda, auto)
 * 
 * @return A tuple containing the output image and the NNF (if output_nnf is true)
 * 
*/
public:
    enum class EbsynthBackend {
        CPU = EBSYNTH_BACKEND_CPU,
        CUDA = EBSYNTH_BACKEND_CUDA,
        AUTO = EBSYNTH_BACKEND_AUTO
    };

    EbsynthBackend backend; // Define the member variable


Ebsynth(py::object style, py::list guides_py, py::object weight = py::none(),
        float uniformity = 3500.0f, int patchsize = 5, int pyramidlevels = 6,
        int searchvoteiters = 12, int patchmatchiters = 6, bool extrapass3x3 = false,
        std::string backend_str = "cuda")
    : uniformity(uniformity), patchsize(patchsize), pyramidlevels(pyramidlevels),
      searchvoteiters(searchvoteiters), patchmatchiters(patchmatchiters),
      extrapass3x3(extrapass3x3), backend(EbsynthBackend::CUDA) // Default backend
{
        if (patchsize < 3) {
        throw std::runtime_error("patch_size is too small");
        }
        if (patchsize % 2 == 0) {
            throw std::runtime_error("patch_size must be an odd number");
        }
        // Handle style image
       if (py::isinstance<py::str>(style)) {
            std::string style_path = style.cast<std::string>();
            this->style = cv::imread(style_path, cv::IMREAD_UNCHANGED);
            if (this->style.empty()) {
                throw std::runtime_error("Failed to read style image from path: " + style_path);
            }
        } else if (py::isinstance<py::array>(style)) {
        py::array_t<unsigned char> buffer = style.cast<py::array_t<unsigned char>>();
        this->style = cv::Mat(buffer.shape(0), buffer.shape(1), CV_8UC3).clone(); // Assuming it's a 3-channel image
        std::memcpy(this->style.data, buffer.data(), buffer.size() * sizeof(unsigned char));
        } else {
            throw std::invalid_argument("style should be either a file path or a numpy array.");
        }
}

void clear_guides() {
    /**
     * @brief Clear the guides
     * 
     * @details This function clears the guides
     * 
    */
    guides.clear();
}

void add_guide(py::object source, py::object target, float weight = 1.0f) {
    if (!(py::isinstance<py::str>(source) || py::isinstance<py::array>(source))) {
        throw std::invalid_argument("source should be either a file path or a numpy array.");
    }

    if (!(py::isinstance<py::str>(target) || py::isinstance<py::array>(target))) {
        throw std::invalid_argument("target should be either a file path or a numpy array.");
    }

    cv::Mat source_mat;
    cv::Mat target_mat;

    // Handle source
    if (py::isinstance<py::str>(source)) {
        source_mat = cv::imread(source.cast<std::string>(), cv::IMREAD_UNCHANGED);
    } else if (py::isinstance<py::array>(source)) {
        py::array_t<unsigned char> buffer = source.cast<py::array_t<unsigned char>>();
        source_mat = cv::Mat(buffer.shape(0), buffer.shape(1), CV_8UC3).clone();
        std::memcpy(source_mat.data, buffer.data(), buffer.size() * sizeof(unsigned char));
    }

    // Handle target
    if (py::isinstance<py::str>(target)) {
        target_mat = cv::imread(target.cast<std::string>(), cv::IMREAD_UNCHANGED);
    } else if (py::isinstance<py::array>(target)) {
        py::array_t<unsigned char> buffer = target.cast<py::array_t<unsigned char>>();
        target_mat = cv::Mat(buffer.shape(0), buffer.shape(1), CV_8UC3).clone();
        std::memcpy(target_mat.data, buffer.data(), buffer.size() * sizeof(unsigned char));
    }

    // Add to the guides container (assuming it's suitable for storing these objects)
    guides.emplace_back(source_mat, target_mat, weight);
}

    py::object run(bool output_nnf = true) {
    /**
     * @brief Run the ebsynth algorithm
     * 
     * @details This function runs the ebsynth algorithm and returns the output image and the NNF (if output_nnf is true)
     * 
     * @param output_nnf Whether to output the NNF
     * 
     * @return A tuple containing the output image and the NNF (if output_nnf is true)
    */
    if (guides.empty()) {
        throw std::runtime_error("at least one guide must be specified");
    }

    // Normalize style image
    cv::Mat img_style = _normalize_img_shape(style);
    int sh = img_style.rows, sw = img_style.cols, sc = img_style.channels();

    // Validate style channels
    if (sc > EBSYNTH_MAX_STYLE_CHANNELS) {
        throw std::runtime_error("error: too many style channels");
    }

    std::vector<cv::Mat> guides_source_channels, guides_target_channels;
    std::vector<float> guides_weights;
    std::vector<float> style_weights(sc, 1.0f / sc);

    int t_h = 0, t_w = 0, t_c = 0;

    for (const auto& guide_tuple : guides) {
        cv::Mat source_guide = _normalize_img_shape(std::get<0>(guide_tuple));
        cv::Mat target_guide = _normalize_img_shape(std::get<1>(guide_tuple));
        float guide_weight = std::get<2>(guide_tuple);

        int s_c = source_guide.channels();
        float guide_weight_scaled = guide_weight / s_c;
        guides_weights.insert(guides_weights.end(), s_c, guide_weight_scaled);

        if (t_h == 0 && t_w == 0 && t_c == 0) {
            t_h = target_guide.rows;
            t_w = target_guide.cols;
            t_c = target_guide.channels();
        } else if (t_h != target_guide.rows || t_w != target_guide.cols) {
            std::stringstream ss;
            ss << "guides target resolutions must be equal. Current target_guide dimensions: ("
            << target_guide.rows << ", " << target_guide.cols << "), Expected dimensions: ("
            << t_h << ", " << t_w << ")";
            throw std::runtime_error(ss.str());
        }

        std::vector<cv::Mat> source_channels, target_channels;
        cv::split(source_guide, source_channels);
        cv::split(target_guide, target_channels);

        guides_source_channels.insert(guides_source_channels.end(), source_channels.begin(), source_channels.end());
        guides_target_channels.insert(guides_target_channels.end(), target_channels.begin(), target_channels.end());
    }

    cv::Mat concatenated_guides_source, concatenated_guides_target;
    cv::merge(guides_source_channels, concatenated_guides_source);
    cv::merge(guides_target_channels, concatenated_guides_target);

    // Compute max pyramid levels
    int maxPyramidLevels = computeMaxPyramidLevels(sh, sw, t_h, t_w, patchsize);

    // Set pyramid levels
    int num_pyramid_levels = pyramidlevels == -1 ? maxPyramidLevels : std::min(pyramidlevels, maxPyramidLevels);

    // Initialize parameters for pyramid levels and iterations
    std::vector<int> num_search_vote_iters_per_level(num_pyramid_levels, searchvoteiters);
    std::vector<int> num_patch_match_iters_per_level(num_pyramid_levels, patchmatchiters);
    std::vector<int> stop_threshold_per_level(num_pyramid_levels, 0.0f); // Assuming default value

    std::vector<float> output_error; // Empty vector
    std::vector<int32_t> nnf_buffer; // Ensure int32_t type
    std::vector<uint8_t> output_buffer(t_h * t_w * sc);
    if(output_nnf) {
        nnf_buffer.resize(t_h * t_w * 2); // Size needed for the NNF buffer (same size as Python)
        output_error.resize(t_h * t_w); // Size needed for the output error buffer (same size as Python)
    }
    // Call ebsynthRun
    ebsynthRun(static_cast<int>(backend), sc, concatenated_guides_source.channels(), sw, sh,
            img_style.data, concatenated_guides_source.data, t_w, t_h,
            concatenated_guides_target.data, nullptr, style_weights.data(),
            guides_weights.data(), uniformity, patchsize, EBSYNTH_VOTEMODE_WEIGHTED,
            num_pyramid_levels, num_search_vote_iters_per_level.data(),
            num_patch_match_iters_per_level.data(), stop_threshold_per_level.data(),
            extrapass3x3 ? 1 : 0, nnf_buffer.empty() ? nullptr : nnf_buffer.data(), output_buffer.data(), output_error.empty() ? nullptr : output_error.data());

    py::array_t<uint8_t> result_image({t_h, t_w, sc}, output_buffer.data());

    if(!nnf_buffer.empty()) {
        py::array_t<int32_t> output_nnf({t_h, t_w, 2}, nnf_buffer.data());
        py::array_t<float> output_error({t_h, t_w}, output_error.data());
        return py::make_tuple(result_image, output_error);
    }

    return result_image;
}



private:
    cv::Mat style;
    std::vector<std::tuple<cv::Mat, cv::Mat, float>> guides;
    float uniformity;
    int patchsize;
    int pyramidlevels;
    int searchvoteiters;
    int patchmatchiters;
    bool extrapass3x3;

    std::map<std::vector<int>, std::unique_ptr<char[]>> buffer_pool;

    cv::Mat _normalize_img_shape(const cv::Mat& img) {
        int sc = img.channels();

        cv::Mat result;
        // Handle the case where the image has only one channel (grayscale)
        if (sc == 1) {
            cv::cvtColor(img, result, cv::COLOR_GRAY2RGB); // Convert to 3-channel BGR image
        } else {
            result = img;
        }
        
        return result;
    }

    char* _get_buffer(const std::vector<int>& shape) {
        auto it = buffer_pool.find(shape);
        if (it != buffer_pool.end()) {
            return it->second.get();
        }
        int size = shape[0] * shape[1] * shape[2];
        buffer_pool[shape] = std::make_unique<char[]>(size);
        return buffer_pool[shape].get();
    }

    int computeMaxPyramidLevels(int sh, int sw, int t_h, int t_w, int patchsize) {
        int maxPyramidLevels = 0;
        for (int level = 32; level >= 0; level--) {
            if (std::min(std::min(sh, t_h) * std::pow(2.0, -level),
                        std::min(sw, t_w) * std::pow(2.0, -level)) >= (2 * patchsize + 1)) {
                maxPyramidLevels = level + 1;
                break;
            }
        }
        return maxPyramidLevels;
    }
    
};

PYBIND11_MODULE(ebsynth, m) {
    py::class_<Ebsynth>(m, "Ebsynth", R"pbdoc(
        A wrapper for the ebsynth library.

        Parameters:
        -----------
        style : object
            A File path or Numpy Array.
        guides : list
            A Tuple of [Source, target, weight].
        weight : object, optional
            Additional weight parameter, default is None.
        uniformity : float, optional
            Uniformity parameter, default is 3500.0.
        patchsize : int, optional
            Patch size, default is 5.
        pyramidlevels : int, optional
            Pyramid levels, default is 6.
        searchvoteiters : int, optional
            Search vote iterations, default is 12.
        patchmatchiters : int, optional
            Patch match iterations, default is 6.
        extrapass3x3 : bool, optional
            Extra pass 3x3, default is false.
        backend : str, optional
            Backend to use, default is "cuda".
    )pbdoc")
    .def(py::init<py::object, py::list, py::object, float,
                  int, int, int, int, bool, std::string>(),
         py::arg("style"), py::arg("guides"), py::arg("weight") = py::none(),
         py::arg("uniformity") = 3500.0f, py::arg("patchsize") = 5,
         py::arg("pyramidlevels") = 6, py::arg("searchvoteiters") = 12,
         py::arg("patchmatchiters") = 6, py::arg("extrapass3x3") = false,
         py::arg("backend") = "cuda")
    .def("clear_guides", &Ebsynth::clear_guides)
    .def("add_guide", &Ebsynth::add_guide,
         py::arg("source"), py::arg("target"), py::arg("weight"))
    .def("run", &Ebsynth::run, py::arg("output_nnf") = false);
}
