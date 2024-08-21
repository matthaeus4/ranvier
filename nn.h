#ifndef NN_H
#define NN_H
#include <mnist.h>
struct vector{
    int n;
    float *a;
};

struct vector *new_vector(int n);

int print_vector(struct vector *v);

int print_sample(struct vector *v);

void ch();

struct matrix{
    int m,n;
    float *a;
};
struct layer{
    int in, out;
    struct matrix *weights;
    struct vector *biases;
    float (*activation)(float x);
    float (*derivative)(float x);
};
struct gradient{
    struct matrix **nabla_w;
    struct vector **nabla_b;
};
struct network{
    int *shape;
    int n_layers;

    float eta;
    int batch_size;

    struct dataset *train;
    struct dataset *test;

    struct vector *input;
    struct vector *output;
    struct vector *expected;

    struct vector **delt;
    struct matrix **product;

    struct vector **z;
    struct vector **a;
    
    struct layer **layers;
};
struct image{
    uint8_t *data;
};
struct label{
    uint8_t *data;
};

struct dataset{
    struct mnist_image_hdr image_hdr;
    struct mnist_label_hdr label_hdr;
    struct label *labels;
    struct image *images;
};
int load_data(struct network *net);
int free_data(struct dataset *d);
#endif
