#include <iostream>
#include <fstream>
#include <float.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"


__global__ void create_world(hitable **d_esferas, hitable **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_esferas)   = new sphere(vec3(0,0,-1), 0.5);
        *(d_esferas+1) = new sphere(vec3(0,-100.5,-1), 100);
        *d_world    = new hitable_list(d_esferas,2);
    }
}

__device__ vec3 color(const ray& r, hitable **world) {
    hit_record rec;
    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
        return 0.5f*vec3(rec.normal.x()+1.0f, rec.normal.y()+1.0f, rec.normal.z()+1.0f);
    }
    else {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5f*(unit_direction.y() + 1.0f);
        return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
    }
}
__global__ void render(vec3 *fb, int max_x, int max_y,
                       vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin,
                       hitable **world) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    ray r(origin, lower_left_corner + u*horizontal + v*vertical);
    fb[pixel_index] = color(r, world);
}
int main(){

  int nx = 1200;
  int ny = 600;

  //arquivo de saida
  std::ofstream myfile;
  myfile.open ("image.ppm");
  myfile << "P3\n" << nx << " " << ny << "\n255\n";

  cudaError_t error;

  int resolution = nx*ny;
  int color_channel = 3;
  int img_buffer_size = resolution*color_channel*sizeof(float);

  // allocate img_buffer
  vec3 *img_buffer;
  error = cudaMallocManaged((void **)&img_buffer, img_buffer_size);
  if(error!=cudaSuccess) {
        printf("Memory Allocation CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));
        exit(EXIT_FAILURE);
  }


  //alocando espaço para as esferas
  hitable **d_esferas;
  error = cudaMalloc((void **)&d_esferas, 2*sizeof(hitable *));
  if(error!=cudaSuccess) {
      printf("Memory Allocation CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));
      exit(EXIT_FAILURE);
  }

  //alocando espaço para o mundo
  hitable **d_world;
  error = cudaMalloc((void **)&d_world, sizeof(hitable *));
  if(error!=cudaSuccess) {
      printf("Memory Allocation CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));
      exit(EXIT_FAILURE);
  }

  //criando o mundo 
  create_world<<<1,1>>>(d_esferas,d_world);

  cudaDeviceSynchronize();



  float block_size = 16.0;
  // Dimensoes para organizar na GPU
  dim3 dimGrid(ceil(ny/block_size), ceil(nx/block_size), 1);
  dim3 dimBlock((int) block_size, int (block_size), 1);


  render<<<dimGrid, dimBlock>>>(img_buffer, nx, ny,
                            vec3(-2.0, -1.0, -1.0),
                            vec3(4.0, 0.0, 0.0),
                            vec3(0.0, 2.0, 0.0),
                            vec3(0.0, 0.0, 0.0),
                            d_world);

  cudaDeviceSynchronize();

  //jogando os pixels calculado para o arquivo de saida formador da imagem
  for (int j = ny-1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
        size_t pixel_index = j*nx + i;
        int ir = int(255.99*img_buffer[pixel_index].r());
        int ig = int(255.99*img_buffer[pixel_index].g());
        int ib = int(255.99*img_buffer[pixel_index].b());
        myfile << ir << " " << ig << " " << ib << "\n";
    }
  }
  
  cudaDeviceReset();
}

