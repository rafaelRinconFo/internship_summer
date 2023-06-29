cd ../depth_map_2_mesh_ray_tracer
ls
chmod +x build_thirdparty.sh
./build_thirdparty.sh
mkdir build
cd build
cmake ..
make -j4