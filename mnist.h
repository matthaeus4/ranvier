#ifndef MNIST_H
#define MNIST_H

#define IMAGE_SIG 0x803
#define LABEL_SIG 0x801

struct mnist_image_hdr{
    uint32_t magic_number;
    uint32_t n_items;
    uint32_t row_cnt;
    uint32_t col_cnt;
}__attribute__((packed));

struct mnist_label_hdr{
    uint32_t magic_number;
    uint32_t n_items;
}__attribute__((packed));

#endif
