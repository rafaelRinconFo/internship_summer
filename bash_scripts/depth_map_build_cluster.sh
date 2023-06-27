ml load cmake
ml load boost
ml load opencv
ml load libcudnn
cd ../depth_map_2_mesh_ray_tracer
chmod +x build_thirdparty.sh
./build_thirdparty.sh
mkdir build
cd build
cmake ..
make -j4