# NVIDIA DeepStream SDK Source Code Repository

This repository contains source code extracted from official NVIDIA DeepStream SDK Docker images (`nvcr.io/nvidia/deepstream:6.x-triton` , `nvcr.io/nvidia/deepstream:6.x-triton-multiarch` and `nvcr.io/nvidia/deepstream:7.x-triton-multiarch`). It serves as a reference and development resource for DeepStream applications.

## Repository Structure

```
.
├── deepstream-6.0      # DeepStream 6.0 source code
├── deepstream-6.0.1    # DeepStream 6.0.1 source code
├── deepstream-6.1      # DeepStream 6.1 source code
├── deepstream-6.1.1    # DeepStream 6.1.1 source code
├── deepstream-6.2      # DeepStream 6.2 source code
├── deepstream-6.3      # DeepStream 6.3 source code
├── deepstream-6.4      # DeepStream 6.4 source code
├── deepstream-7.0      # DeepStream 7.0 source code
├── deepstream-7.1      # DeepStream 7.1 source code
└── README.md           
```

## Version Details

### DeepStream 6.x Series
- **6.0/6.0.1**: Initial release of the 6.x series with Triton Inference Server integration
- **6.1/6.1.1**: Enhanced features and bug fixes from 6.0
- **6.2**: Additional optimizations and feature updates
- **6.3**: Further improvements and stability enhancements
- **6.4**: Latest release in the 6.x series with advanced features

### DeepStream 7.x Series
- **7.0**: Major version upgrade with new features and improvements
- **7.1**: Latest version with enhanced capabilities

## Source Origin

All source code in this repository is extracted from official NVIDIA Docker images:
```
#version 6.0, 6.0.1, 6.1, 6.1.1, 6.2
nvcr.io/nvidia/deepstream:6.x-triton
#version 6.3, 6.4
nvcr.io/nvidia/deepstream:6.x-triton-multiarch
#version 7.0, 7.1
nvcr.io/nvidia/deepstream:7.x-triton-multiarch
```

## Prerequisites

To work with these source files, you need:
- NVIDIA GPU with compatible drivers
- Ubuntu 20.04 or later
- CUDA Toolkit
- TensorRT
- Triton Inference Server

## Usage

Each directory contains the complete source code for its respective DeepStream version. To use:

1. Navigate to the desired version directory
2. Run the bash script `download_folder_gitignore.sh` to download `lib`, `samples` and `bin` for the SDK that I have stored separately on Google Drive
3. Follow the version-specific README and documentation
4. Build and run according to NVIDIA's DeepStream SDK guidelines

## Documentation

For detailed information about each version, please refer to:
- [DeepStream Developer Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/)

## License

This repository contains source code from NVIDIA DeepStream SDK, which is subject to NVIDIA's software license agreement. Please refer to NVIDIA's licensing terms for usage restrictions and conditions.

## Contributing

For issues or improvements related to the source code organization in this repository:
1. Open an issue describing the problem or enhancement
2. Submit pull requests with clear descriptions of changes
3. Follow existing code structure and documentation patterns

## Support

For DeepStream SDK related questions and support:
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/accelerated-computing/intelligent-video-analytics/deepstream-sdk/)
- [DeepStream SDK Support](https://developer.nvidia.com/deepstream-sdk)