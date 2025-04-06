#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <algorithm>
#include <chrono>
#include <onnxruntime_cxx_api.h>

// Define these macros if they're not available in your ONNX Runtime version
#ifndef ORT_API_VERSION
#define ORT_API_VERSION 13
#endif

// Helper function to print ONNX tensor info
template<typename T>
void printTensorInfo(const T& info) {
    try {
        std::cout << "  Dimensions: ";
        auto shape = info.GetShape();
        for (const auto& dim : shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        
        std::cout << "  Element type: " << info.GetElementType() << std::endl;
        
        // Safely calculate element count
        size_t elementCount = 1;
        bool hasDynamicDim = false;
        for (const auto& dim : shape) {
            if (dim < 0) {
                hasDynamicDim = true;
                continue;
            }
            
            // Check for overflow before multiplying
            if (dim > 0 && elementCount > std::numeric_limits<size_t>::max() / dim) {
                std::cout << "  Element count: Too large to calculate (would overflow)" << std::endl;
                return;
            }
            elementCount *= dim > 0 ? dim : 1; // Treat negative dimensions as 1 for count
        }
        
        if (hasDynamicDim) {
            std::cout << "  Element count: Dynamic (depends on input dimensions)" << std::endl;
        } else {
            std::cout << "  Element count: " << elementCount << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "  Error retrieving tensor info: " << e.what() << std::endl;
    }
}

// Load raw state data from binary file
std::vector<float> loadStateData(const std::string& filePath, int inputChannels, int windowSize) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filePath);
    }
    
    // Calculate expected size (each element is a float)
    size_t expectedSize = inputChannels * windowSize * sizeof(float);
    
    // Read the file content
    std::vector<char> buffer(expectedSize);
    file.read(buffer.data(), expectedSize);
    
    if (file.gcount() != expectedSize) {
        throw std::runtime_error("Failed to read expected data size from file");
    }
    
    // Convert to float vector
    std::vector<float> data(inputChannels * windowSize);
    std::memcpy(data.data(), buffer.data(), expectedSize);
    
    return data;
}

class RiskPredictor {
private:
    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::vector<const char*> inputNames_;
    std::vector<const char*> outputNames_;
    int inputChannels_;
    int windowSize_;

public:
    RiskPredictor(const std::string& modelPath, int inputChannels, int windowSize, bool useCUDA = false)
        : env_(ORT_LOGGING_LEVEL_WARNING, "RiskPredictor"),
          inputChannels_(inputChannels),
          windowSize_(windowSize) {
        
        // Create session options
        if (useCUDA) {
            try {
                // Try to enable CUDA execution provider
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions_, 0));
                std::cout << "CUDA execution provider enabled." << std::endl;
            } catch (const Ort::Exception& e) {
                std::cout << "Warning: Failed to enable CUDA execution provider. Falling back to CPU." << std::endl;
                std::cout << "Error details: " << e.what() << std::endl;
                useCUDA = false;
            }
        }
        
        // Set graph optimization level
        sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Create session
        session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions_);
        
        // Get input and output names
        size_t numInputs = session_->GetInputCount();
        for (size_t i = 0; i < numInputs; i++) {
            // Modern ONNX Runtime API uses GetInputNameAllocated
            Ort::AllocatedStringPtr inputNamePtr = session_->GetInputNameAllocated(i, allocator_);
            inputNames_.push_back(inputNamePtr.get());
        }
        
        size_t numOutputs = session_->GetOutputCount();
        for (size_t i = 0; i < numOutputs; i++) {
            // Modern ONNX Runtime API uses GetOutputNameAllocated
            Ort::AllocatedStringPtr outputNamePtr = session_->GetOutputNameAllocated(i, allocator_);
            outputNames_.push_back(outputNamePtr.get());
        }
        
        // Print model info
        std::cout << "Model loaded successfully from: " << modelPath << std::endl;
        std::cout << "Number of inputs: " << numInputs << std::endl;
        std::cout << "Number of outputs: " << numOutputs << std::endl;
        
        // Print input info
        for (size_t i = 0; i < numInputs; i++) {
            try {
                Ort::TypeInfo typeInfo = session_->GetInputTypeInfo(i);
                auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
                
                // Print name safely
                const char* name = inputNames_[i];
                if (name && std::isprint(name[0])) {
                    std::cout << "Input " << i << " name: " << name << std::endl;
                } else {
                    std::cout << "Input " << i << " name: <unprintable>" << std::endl;
                }
                
                printTensorInfo(tensorInfo);
            } catch (const std::exception& e) {
                std::cout << "Error getting input info for input " << i << ": " << e.what() << std::endl;
            }
        }
        
        // Print output info
        for (size_t i = 0; i < numOutputs; i++) {
            try {
                Ort::TypeInfo typeInfo = session_->GetOutputTypeInfo(i);
                auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
                
                // Print name safely
                const char* name = outputNames_[i];
                if (name && std::isprint(name[0])) {
                    std::cout << "Output " << i << " name: " << name << std::endl;
                } else {
                    std::cout << "Output " << i << " name: <unprintable>" << std::endl;
                }
                
                printTensorInfo(tensorInfo);
            } catch (const std::exception& e) {
                std::cout << "Error getting output info for output " << i << ": " << e.what() << std::endl;
            }
        }
    }
    
    // With the newer ONNX Runtime API using AllocatedStringPtr,
    // we don't need to manually free the memory in the destructor
    ~RiskPredictor() {
        // Memory is automatically managed by AllocatedStringPtr
    }
    
    float predict(const std::vector<float>& statesData) {
        try {
            // Check input size
            if (statesData.size() != inputChannels_ * windowSize_) {
                throw std::runtime_error("Input data size doesn't match expected size: " + 
                                         std::to_string(statesData.size()) + " vs " + 
                                         std::to_string(inputChannels_ * windowSize_));
            }
            
            // Create input tensor
            std::vector<int64_t> inputShape = {1, static_cast<int64_t>(inputChannels_), static_cast<int64_t>(windowSize_)};
            auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            
            std::cout << "Creating input tensor with shape: [";
            for (size_t i = 0; i < inputShape.size(); i++) {
                std::cout << inputShape[i] << (i < inputShape.size() - 1 ? ", " : "");
            }
            std::cout << "]" << std::endl;
            
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo, const_cast<float*>(statesData.data()), statesData.size(), 
                inputShape.data(), inputShape.size()
            );
            
            // Print first few values for debugging
            std::cout << "Input tensor values (first 5): ";
            for (size_t i = 0; i < std::min(size_t(5), statesData.size()); i++) {
                std::cout << statesData[i] << " ";
            }
            std::cout << std::endl;
            
            // Run inference with hardcoded names
            auto startTime = std::chrono::high_resolution_clock::now();
            
            // Use hardcoded input and output names
            const char* inputName = "input";
            const char* outputName = "output";
            
            std::cout << "Running inference with fixed input name: '" << inputName 
                      << "' and output name: '" << outputName << "'" << std::endl;
            
            auto outputTensors = session_->Run(
                Ort::RunOptions{nullptr}, 
                &inputName,      // Use hardcoded input name 
                &inputTensor, 
                1, 
                &outputName,     // Use hardcoded output name
                1
            );
            
            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
            
            // Get output value
            float* outputData = outputTensors[0].GetTensorMutableData<float>();
            float riskValue = outputData[0];
            
            std::cout << "Inference time: " << duration.count() / 1000.0 << " ms" << std::endl;
            std::cout << "Predicted risk value: " << riskValue << std::endl;
            
            return riskValue;
        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime error during prediction: " << e.what() << std::endl;
            return -1.0f; // Indicate error
        } catch (const std::exception& e) {
            std::cerr << "Error during prediction: " << e.what() << std::endl;
            return -1.0f; // Indicate error
        }
    }
    
    // Predicts risk from a trajectory file (numpy .npy format)
    float predictFromNumpyFile(const std::string& filePath) {
        std::cout << "Attempting to read NumPy file: " << filePath << std::endl;
        
        // Simple implementation for NumPy '.npy' files
        // This assumes the file structure matches what your Python code generates
        std::ifstream file(filePath, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file: " + filePath);
        }
        
        // Read the entire file into memory
        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> buffer(fileSize);
        if (!file.read(buffer.data(), fileSize)) {
            throw std::runtime_error("Failed to read file");
        }
        
        std::cout << "File size: " << fileSize << " bytes" << std::endl;
        
        // For debugging purposes, print the first few bytes of the file
        std::cout << "File header bytes (hex): ";
        for (int i = 0; i < std::min(size_t(16), fileSize); i++) {
            printf("%02x ", static_cast<unsigned char>(buffer[i]));
        }
        std::cout << std::endl;
        
        // This is a simple implementation for demonstration
        // In a production system, you'd want to use a proper NumPy file parser
        // that can handle different data types, endianness, etc.
        
        // Extract the last window from the trajectory
        // For proper implementation, consider using cnpy library
        
        std::cout << "Warning: Using simplified NumPy parsing logic." << std::endl;
        std::cout << "This implementation assumes specific file format." << std::endl;
        
        // Create synthetic data as a fallback
        std::vector<float> stateData(inputChannels_ * windowSize_);
        for (size_t i = 0; i < stateData.size(); ++i) {
            stateData[i] = 0.5f;  // Default values
        }
        
        try {
            // Try to extract some meaningful data
            // This is very simplified and assumes a specific structure
            // Look for a pattern that might indicate float data
            for (size_t i = 128; i < fileSize - stateData.size() * sizeof(float); i++) {
                // Try using the data at this position
                std::memcpy(stateData.data(), buffer.data() + i, stateData.size() * sizeof(float));
                
                // Check if the data seems reasonable
                bool reasonable = true;
                for (float val : stateData) {
                    if (std::isnan(val) || std::isinf(val) || std::abs(val) > 1000.0f) {
                        reasonable = false;
                        break;
                    }
                }
                
                if (reasonable) {
                    std::cout << "Found potentially valid data at offset: " << i << std::endl;
                    std::cout << "Sample values: [" << stateData[0] << ", " << stateData[1] << ", ...]" << std::endl;
                    break;
                }
            }
        } catch (const std::exception& e) {
            std::cout << "Error extracting data: " << e.what() << std::endl;
            std::cout << "Using default values instead." << std::endl;
        }
        
        return predict(stateData);
    }
};

int main(int argc, char* argv[]) {
    try {
        // Default values
        std::string modelPath = "model.onnx";
        int inputChannels = 19;
        int windowSize = 1;
        bool useCUDA = false;
        std::string inputFile;
        bool isNumpyFile = false;
        
        // Parse command line arguments
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            
            if (arg == "--model" && i + 1 < argc) {
                modelPath = argv[++i];
            } else if (arg == "--channels" && i + 1 < argc) {
                inputChannels = std::stoi(argv[++i]);
            } else if (arg == "--window_size" && i + 1 < argc) {
                windowSize = std::stoi(argv[++i]);
            } else if (arg == "--cuda") {
                useCUDA = true;
            } else if (arg == "--input" && i + 1 < argc) {
                inputFile = argv[++i];
            } else if (arg == "--numpy") {
                isNumpyFile = true;
            } else if (arg == "--help") {
                std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  --model <path>       Path to ONNX model file (default: model.onnx)" << std::endl;
                std::cout << "  --channels <int>     Number of input channels (default: 19)" << std::endl;
                std::cout << "  --window_size <int>  Size of the input window (default: 1)" << std::endl;
                std::cout << "  --cuda               Use CUDA for acceleration (default: false)" << std::endl;
                std::cout << "  --input <path>       Path to input data file" << std::endl;
                std::cout << "  --numpy              Input file is a numpy .npy file (default: false)" << std::endl;
                std::cout << "  --help               Show this help message" << std::endl;
                return 0;
            }
        }
        
        std::cout << "Initializing RiskPredictor..." << std::endl;
        RiskPredictor predictor(modelPath, inputChannels, windowSize, useCUDA);
        
        if (!inputFile.empty()) {
            // Run prediction on input file
            float risk;
            
            if (isNumpyFile) {
                risk = predictor.predictFromNumpyFile(inputFile);
            } else {
                // Load raw state data
                std::vector<float> stateData = loadStateData(inputFile, inputChannels, windowSize);
                risk = predictor.predict(stateData);
            }
            
            std::cout << "Predicted risk: " << risk << std::endl;
        } else {
            // Create synthetic data for demonstration
            std::cout << "No input file provided. Running with synthetic data..." << std::endl;
            std::vector<float> syntheticData(inputChannels * windowSize);
            
            // Fill with random values
            for (size_t i = 0; i < syntheticData.size(); ++i) {
                syntheticData[i] = static_cast<float>(rand()) / RAND_MAX;
            }
            
            float risk = predictor.predict(syntheticData);
            std::cout << "Predicted risk with synthetic data: " << risk << std::endl;
        }
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}