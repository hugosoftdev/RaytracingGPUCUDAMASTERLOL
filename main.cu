#include <iostream>
#include <fstream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"


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

__global__ void render(vec3 *img_buffer, int max_x, int max_y, int ns, camera **cam, hitable **world) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j*max_x + i;
    curandState *randomState;
    curand_init(2000, pixel_index, 0, &randomState);

    vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&randomState)) / float(max_x);
        float v = float(j + curand_uniform(&randomState)) / float(max_y);
        ray r = (*cam)->get_ray(u,v);
        col += color(r, world);
    }
    img_buffer[pixel_index] = col/float(ns);
}



int main(){

  int nx = 1200;
  int ny = 600;
  int ns = 10;

  //arquivo de saida
  std::ofstream myfile;
  myfile.open ("image.ppm");
  myfile << "P3\n" << nx << " " << ny << "\n255\n";

  cudaError_t error;

  int resolution = nx*ny;
  int color_channel = 3;
  int img_buffer_size = resolution*color_channel*sizeof(vec3);

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

  camera **d_camera;
  error = cudaMalloc((void **)&d_camera, sizeof(camera *));
  if(error!=cudaSuccess) {
      printf("Memory Allocation CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(error));
      exit(EXIT_FAILURE);
  }

  //criando o mundo 
  create_world<<<1,1>>>(d_esferas,d_world);

  cudaDeviceSynchronize();



  float n_threads= 8.0;
  // Dimensoes para organizar na GPU
  dim3 blocks(nx/n_threads+1,ny/n_threads+1);
  dim3 threads(n_threads,n_threads);
  render<<<blocks, threads>>>(img_buffer, nx, ny, ns, d_camera, d_world);

  cudaDeviceSynchronize();

  //jogando os pixels calculado para o arquivo de saida formador da imagem
  for (int j = ny-1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
        int pixel_index = j*nx + i;
        int ir = int(255.99*img_buffer[pixel_index].r());
        int ig = int(255.99*img_buffer[pixel_index].g());
        int ib = int(255.99*img_buffer[pixel_index].b());
        myfile << ir << " " << ig << " " << ib << "\n";
    }
  }
  
  cudaDeviceReset();
}

