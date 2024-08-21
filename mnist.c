#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <nn.h>

int load_images(struct dataset *set, FILE *fp){
    if (fp == NULL || set == NULL) return -1;
    set->images = malloc(sizeof(struct image));
    set->images->data = malloc(set->image_hdr.n_items * set->image_hdr.col_cnt * set->image_hdr.row_cnt);
    fread(set->images->data, set->image_hdr.n_items * set->image_hdr.col_cnt * set->image_hdr.row_cnt, 1, fp);
    return 0;
}

int load_labels(struct dataset *set, FILE *fp){
    if (fp == NULL || set == NULL) return -1;
    set->labels = malloc(sizeof(struct label));
    set->labels->data = malloc(set->label_hdr.n_items);
    fread(set->labels->data, set->label_hdr.n_items, 1, fp);
    return 0;
}
int load_data(struct network *net){
    FILE *image_fp;
    FILE *label_fp;
    uint32_t *buf_i = malloc(sizeof(struct mnist_image_hdr));
    uint32_t *buf_l = malloc(sizeof(struct mnist_label_hdr));
    int ret;

    net->train = malloc(sizeof(struct dataset));
    net->test = malloc(sizeof(struct dataset));
    
    image_fp = fopen("data/train-images-idx3-ubyte", "rb");
    if (image_fp == NULL) return -1;

    label_fp = fopen("data/train-labels-idx1-ubyte", "rb");
    if (label_fp == NULL) return -1;
    
    ret = fread(buf_i, sizeof(struct mnist_image_hdr), 1, image_fp);
    if (ret == -1) return -1;

    net->train->image_hdr = (struct mnist_image_hdr) {
        .magic_number = __builtin_bswap32(buf_i[0]),
        .n_items = __builtin_bswap32(buf_i[1]),
        .row_cnt = __builtin_bswap32(buf_i[2]),
        .col_cnt = __builtin_bswap32(buf_i[3])
    };
    
    ret = fread(buf_l, sizeof(struct mnist_label_hdr), 1, label_fp);
    if (ret == -1) return -1;

    net->train->label_hdr = (struct mnist_label_hdr) {
        .magic_number = __builtin_bswap32(buf_l[0]),
        .n_items = __builtin_bswap32(buf_l[1])
    };

    ret = load_images(net->train, image_fp);
    if (ret == -1){
        fprintf(stderr, "Failed to load training images");
        return -1;
    }
    ret = load_labels(net->train, label_fp);
    if (ret == -1){
        fprintf(stderr, "Failed to load training labels");
        return -1;
    }

    fclose(image_fp);
    fclose(label_fp);
    
    image_fp = fopen("data/t10k-images-idx3-ubyte", "rb");
    label_fp = fopen("data/t10k-labels-idx1-ubyte", "rb");

    ret = fread(buf_i, sizeof(struct mnist_image_hdr), 1, image_fp);
    if (ret == -1) return -1;

    net->test->image_hdr = (struct mnist_image_hdr) {
        .magic_number = __builtin_bswap32(buf_i[0]),
        .n_items = __builtin_bswap32(buf_i[1]),
        .row_cnt = __builtin_bswap32(buf_i[2]),
        .col_cnt = __builtin_bswap32(buf_i[3])
    };
    
    ret = fread(buf_l, sizeof(struct mnist_label_hdr), 1, label_fp);
    if (ret == -1) return -1;

    net->test->label_hdr = (struct mnist_label_hdr) {
        .magic_number = __builtin_bswap32(buf_l[0]),
        .n_items = __builtin_bswap32(buf_l[1])
    };

    ret = load_images(net->test, image_fp);
    if (ret == -1) return -1;
    ret = load_labels(net->test, label_fp);
    if (ret == -1) return -1;

    fclose(image_fp);
    fclose(label_fp);
    free(buf_i);
    free(buf_l);
    return 0;
}

int free_data(struct dataset *d){
    if (d == NULL) return -1;
    free(d->labels->data);
    free(d->labels);
    free(d->images->data);
    free(d->images);
    free(d);
    return 0;
}
