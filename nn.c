#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <nn.h>

void ch(){
    static int p = 0;
    printf("checkpoint %d\n", p);
    p++;
}
#define ELEM(mat, row, col)  \
    mat->a[(row)*mat->n + (col)]

#define T_ELEM(mat, row, col) \
    mat->a[(col)*mat->m + (row)]

void init_rand(){srand((unsigned)(time(0)));}
//produce pseudorandom floats in the interval [a,b]
float brandy(float a, float b){
    return a + (b - a)*((float)rand()/(float)RAND_MAX);
}
float box_muller(){
    float u1 = ((float) rand())/(RAND_MAX);
    float u2 = ((float) rand())/(RAND_MAX);
    return sqrt(-2*log(u2)) * cos(2*M_PI*u1);
}
int act(struct vector *v, float (*activation)(float)){
    if (v->a == NULL || activation == NULL) return -1;
    for (int i = 0; i < v->n; i++){
        v->a[i] = (*activation)(v->a[i]);
    }
    return 0;
}

float sigmoid(float x){
    return 1.0/(1.0+exp(-x));
}
float sigmoid_prime(float x){
    return sigmoid(x)*(1-sigmoid(x));
}
float relu(float x){
    if (x <= 0.0) return 0.0;
    return x;
}
float relu_prime(float x){
    if (x < 0.0) return 0.0;
    return 1.0;
}
float leaky_relu(float x){
    if (x <= 0.0) return 0.01*x;
    return x;
}
float leaky_relu_prime(float x){
    if (x < 0.0) return 0.01;
    return 1.0;
}

struct vector *new_vector(int n){
    if (n <= 0) return NULL;
    struct vector *v = malloc(sizeof(*v));
    v->n = n;
    v->a = (float *) calloc(n,sizeof(float));
    return v;
}
int delete_vector(struct vector *v){
    if (v == NULL || v->a == NULL) return -1;
    free(v->a);
    free(v);
    return 0;
}
//copies contents of w into v, if v exists.
int copy_vector(struct vector *v, struct vector *w){
    if (v->n <= 0 || v->a == NULL){
        fprintf(stderr, "Vector copy error: invalid vector dimensions or uninitialized vector\n");
        return -1;
    }
    if (v->n != w->n) {
        fprintf(stderr, "Vector copy error: vector dimensions (%d and %d) do not match\n", v->n, w->n);
        return -1;
    }
    for (int i = 0; i < v->n; i++){
        v->a[i] = w->a[i];
    }
    return 0;
}
int rand_vector(struct vector *v){
    if (v->n <= 0 || v->a == NULL) return -1;
    for(int i = 0; i < v->n; i++){
        v->a[i] = box_muller();
    }
    return 0;
}
struct matrix *new_matrix(int m, int n){
    if (m <= 0 || n <= 0) return NULL;
    struct matrix *mat = malloc(sizeof(*mat));
    mat->m = m;
    mat->n = n;
    mat->a = (float *) calloc(m*n,sizeof(float));
    return mat;
}
int delete_matrix(struct matrix *m){
    if (m == NULL || m->a == NULL) return -1;
    free(m->a);
    free(m);
    return 0;
}
int rand_matrix(struct matrix *m){
    if (m->m <= 0 || m->n <= 0) {
        fprintf(stderr, "rand_matrix failed: invalid matrix dimensions (%d, %d)", m->m, m->n);
        return -1;
    }
    if (m->a == NULL) return -1;
    for(int i = 0; i < m->m; i++){
        for(int j = 0; j < m->n; j++){
            ELEM(m, i, j) = box_muller()/sqrt(m->m);
        }
    }
    return 0;
}

int print_matrix(struct matrix *m){
    if (m->m <= 0 || m->n <= 0) return -1;
        for(int i = 0; i < m->n; i++){
            for (int j = 0; j < m->m; j++){
                printf("%f ", ELEM(m,i,j));
            }
            printf("\n");
        }
    printf("\n");
    return 0;
}
struct layer *new_layer(int in, int out){
    if (in <= 0 || out <= 0) return NULL;
    struct layer *l = malloc(sizeof(*l));
    l->in = in;
    l->out = out;
    l->weights = new_matrix(out, in);
    l->biases = new_vector(out);
    return l;
}
int delete_layer(struct layer *l){
    if (l == NULL) return -1;
    delete_matrix(l->weights);
    delete_vector(l->biases);
    free(l);
    return 0;
}
// (m, n) = (out, in) when not transposing
int map(struct vector *v, struct matrix *m, int transpose, struct vector *mapped){
    //struct vector *w;
    //static int count = 0;
    //printf("map #%d\n",count);
    //count++;
    if (m->n == v->n && transpose == 0){
        if (m->m != mapped->n) return -1;
        for (int i = 0; i < m->m; i++){
            for (int j = 0; j < m->n; j++){
                mapped->a[i] += ELEM(m, i, j)*v->a[j];
            }
        }
        return 0;
    }
    else if (m->m == v->n && transpose == 1){
        //w = new_vector(m->n);
        if (m->n != mapped->n) return -1;
        for (int i = 0; i < m->m; i++){
            for (int j = 0; j < m->n; j++){
                mapped->a[j] += T_ELEM(m, i, j)*v->a[i];
            }
        }
        return 0;
    }
    else{
        //free(w);
        fprintf(stderr, "\nMapping failed: vector of dimension %d cannot be mapped by matrix of shape %dx%d\n", v->n, m->m, m->n);
        return -1;
    }

}

float dot(struct vector *v, struct vector *w){
    if (v->n <= 0 || w->n <= 0){
        fprintf(stderr, "Dot product error: invalid vector dimension");
        exit(1);
    }
    float dt = 0.0;
    if (v->n == w->n){
        for (int i = 0; i < v->n; i++){
            dt += v->a[i] * w->a[i];
        }
        return dt;
    }
    else{
        fprintf(stderr, "Dot product error: dimensions do not match (v->n: %d, w->n: %d)", v->n, w->n);
        exit(1);
    }
}

struct vector *hadamard(struct vector *v, struct vector *w, struct vector *h){
    if (v->n <= 0 || w->n <= 0) return NULL;
    if (v->n == w->n){
        for (int i = 0; i < v->n; i++){
            h->a[i] = v->a[i] * w->a[i];
        }
        return h;
    }
    return NULL;
}

int add(struct vector *v, struct vector *w, int sub){
    if (v->n != w->n) return -1;
        for (int i = 0; i < v->n; i++){
            if (sub == 1){v->a[i] -= w->a[i];}
            else {v->a[i] += w->a[i];}
        }
    return 0;
}

int scl_vec(float scl, struct vector *v){
    if (v->n <= 0) return -1;
    for (int i = 0; i < v->n; i++){
        v->a[i] *= scl;
    }
    return 0;
}

int add_m(struct matrix *m, struct matrix *t, int sub){
    if (m->m != t->m || m->n != t->n) return -1;
        for (int i = 0; i < m->m; i++){
            for (int j = 0; j < m->n; j++){
                if (sub == 1){
                    ELEM(m, i, j) -= ELEM(t, i, j);
                }
                else{
                    ELEM(m, i, j) += ELEM(t, i, j);
                }
            }
        }
    return 0;
}

int scl_mat(float scl, struct matrix *m){
    if (m->m <= 0 || m->n <= 0) return -1;
    for (int i = 0; i < m->m; i++){
        for (int j = 0; j < m->n; j++){
            ELEM(m, i, j) *= scl; 
        }
    }
    return 0;
}

int outer(struct vector *out, struct vector *in, struct matrix *o){
    if (out->n <= 0 || in->n <= 0) return -1;
    if (o->m <= 0 || o->n <= 0) return -1;
    if (o->m != out->n || o->n != in->n) return -1;
    for (int i = 0; i < out->n; i++){
        for (int j = 0; j < in->n; j++){
            ELEM(o, i, j) = out->a[i] * in->a[j];
        }
    }
    return 0;
}


int print_vector(struct vector *v){
    if (v->n <= 0 || v->a == NULL) return -1;
    //static int count = 0;
    //printf("Vector %d\n", count);
    for (int i = 0; i < v->n; i++){
        printf("%f ", v->a[i]);
    }
    printf("\n");
    //count++;
    return 0;
}

float cost(struct vector *target, struct vector *output){
    if (target->a == NULL || output->a == NULL) return NAN;
    struct vector *temp = new_vector(10);
    copy_vector(temp, output);
    add(temp, target, 1);
    float c = 0.5*dot(temp,temp);
    free(temp);
    return c;
}
//output and target were swapped before
struct vector *cost_derivative(struct vector *output, struct vector *target, struct vector *dcda){
    if (target->a == NULL || output->a == NULL) return NULL;
    if (dcda->a == NULL) return NULL;
    if (output->n != target->n) return NULL;
    copy_vector(dcda, output);
    add(dcda, target, 1);
    return dcda;
}
struct vector *softmax(struct vector *v){
    float sum;
    for (int i = 0; i < v->n; i++){
        sum += exp(v->a[i]);
    }
    for (int i = 0; i < v->n; i++){
        v->a[i] = exp(v->a[i])/sum;
    }
    return v;
}
int forward(struct network *net){
    if (net->train == NULL || net->test == NULL) return -1;

    map(net->input, net->layers[0]->weights, 0, net->z[0]); // Mv
    add(net->z[0], net->layers[0]->biases, 0); // z = Mv + b
    copy_vector(net->a[0], net->z[0]);
    act(net->a[0], net->layers[0]->activation); 
    //copy_vector(net->a[0], t1);

    map(net->a[0], net->layers[1]->weights, 0, net->z[1]);
    add(net->z[1], net->layers[1]->biases, 0);
    copy_vector(net->output, net->z[1]);
    //print_vector(net->z[1]);
    act(net->output, net->layers[1]->activation);
    //net->a[1] = copy_vector(t2);
    //copy_vector(net->output, t2);
    return 0;
}
int evaluate(struct network *net){
    struct vector *t1 = new_vector(net->shape[1]);
    struct vector *t2 = new_vector(net->shape[2]);
    map(net->input, net->layers[0]->weights, 0, t1);
    add(t1, net->layers[0]->biases, 0);
    act(t1, net->layers[0]->activation);
    map(t1, net->layers[1]->weights, 0, t2);
    add(t2, net->layers[1]->biases, 0);
    act(t2, net->layers[1]->activation);
    copy_vector(net->output, t2);
    delete_vector(t1);
    delete_vector(t2);
    return 0;
}


//Takes error from top layer and propagates it back, computing the error at each layer
int backward(struct network *net, struct vector *delta, struct gradient *grad){
    if (net == NULL || grad == NULL) return -1;
    if (delta->a == NULL) return -1;
    //for the last layer we compute the hadamard product of nabla_c and sigma_prime(z_L) DONE IN train()

    //for each previous layer we compute the hadamard product of (w_(l+1)T * delta_(l+1) and sigma_prime(z_l)
    copy_vector(net->delt[1], delta);
    map(net->delt[1], net->layers[1]->weights, 1, net->delt[0]);

    act(net->z[0], net->layers[0]->derivative);
    hadamard(net->delt[0], net->z[0], net->delt[0]);

    outer(net->delt[1], net->a[0], net->product[1]);
    outer(net->delt[0], net->input, net->product[0]);
    
    add_m(grad->nabla_w[1], net->product[1], 0);
    add_m(grad->nabla_w[0], net->product[0], 0);
    add(grad->nabla_b[1], net->delt[1], 0);
    add(grad->nabla_b[0], net->delt[0], 0);
    //from the vectors delta_l we get dcdw_ljk = a_(l-1)_k * delta_l_j and dcdb_lj = delta_l_j
    //keep in mind that delta_l is a vector and delta_l_j is an element in that vector  
    return 0;
}

int init_net(struct network *net){
    if (net == NULL) return -1;
    printf("Initializing Network...");
    net->input = new_vector(net->shape[0]);
    net->output = new_vector(net->shape[net->n_layers]);
    net->expected = new_vector(net->shape[net->n_layers]);

    net->z = malloc(net->n_layers*sizeof(struct vector *));
    net->a = malloc(sizeof(struct vector *));

    net->delt = malloc(net->n_layers*sizeof(struct vector *));
    net->product = malloc(net->n_layers*sizeof(struct matrix *));
    net->layers = malloc(net->n_layers*sizeof(struct layers *));

    net->a[0] = new_vector(net->shape[1]);
    for (int i = 0; i < net->n_layers; i++){
        net->layers[i] = new_layer(net->shape[i], net->shape[i+1]);
        net->layers[i]->activation = sigmoid;
        net->layers[i]->derivative = sigmoid_prime;
        rand_matrix(net->layers[i]->weights);
        rand_vector(net->layers[i]->biases);
        net->z[i] = new_vector(net->shape[i+1]);
        net->delt[i] = new_vector(net->shape[i+1]);
        net->product[i] = new_matrix(net->shape[i+1], net->shape[i]);
    }
    if (load_data(net) == -1){
        fprintf(stderr, "Failed to load training and/or test data\n");
        return -1;
    }
    printf("done\n");
    return 0;
}
int set_expected(struct vector *v, uint8_t *data, int label){
    if (v == NULL || data == NULL) return -1;
    for(int i = 0; i < v->n; i++){
        v->a[i] = 0.0;
    }
    v->a[data[label]] = 1.0;
    return 0;
}
int image_to_vector(struct vector *v, uint8_t *data, int image){
    if (v == NULL || data == NULL) return -1;
    if (image < 0) return -1;
    for (int i = 0; i < v->n; i++){
        v->a[i] = ((float) data[i+image])/255.0;
    }
    return 0;
}

int print_image(struct vector *image){
    if (image->n <= 0 || image->a == NULL) return -1;
    char b,f;
    for (int i = 0; i < image->n; i++){
        if (i % 28 == 0) printf("\n");
        b = ((image->a[i]*255.0) < 100.0) ? 32 : 0;
        f = ((image->a[i]*255.0) < 10.0) ? 32 : 0;
        printf("%c%d%c", b,(int) (image->a[i]*255.0), f);
    }
    return 0;
}

int eval_sample(struct network *net, int is_train){
    //collect test example, preferably a new one each time
    int ret;
    int s = (is_train == 1) ? rand() % 60000 : rand() % 10000;
    struct dataset *d = (is_train == 1) ? net->train : net->test;
    ret = image_to_vector(net->input, d->images->data, s*784);
    if (ret == -1) return -1;

    ret = set_expected(net->expected, d->labels->data, s);
    if (ret == -1) return -1;
    
    ret = evaluate(net);
    if (ret == -1) return -1;
    return 0;
}

int store_model(struct network *net){
    FILE *fp = fopen("model", "wb");
    fwrite(net->layers[0]->weights->a, 784*net->shape[1]*sizeof(float), 1, fp);
    fwrite(net->layers[0]->biases->a, net->shape[1]*sizeof(float), 1, fp);
    fwrite(net->layers[1]->weights->a, net->shape[1]*10*sizeof(float), 1, fp);
    fwrite(net->layers[1]->biases->a, 10*sizeof(float), 1, fp);
    fclose(fp);
    return 0;
}

int delete_net(struct network *net){
    free_data(net->train);
    free_data(net->test);
    delete_vector(net->input);
    delete_vector(net->output);
    delete_vector(net->expected);
    for (int i = 0; i < net->n_layers; i++){
        delete_vector(net->delt[i]);
        delete_matrix(net->product[i]);
        delete_vector(net->z[i]);
        delete_layer(net->layers[i]);
    }
    free(net->layers);
    free(net->delt);
    free(net->product);
    free(net->z);
    delete_vector(net->a[0]);
    free(net->a);
    free(net);
    return 0;
}

int max_index(struct vector *v){
    if (v == NULL || v->a == NULL) return -1;
    int index = 0;
    for (int i = 1; i < v->n; i++){
        if (v->a[index] > v->a[i]){
            continue;
        }
        else{
            index = i;
        }
    }
    return index;
}

int train(struct network *net, int epochs){
    if (net == NULL || epochs <= 0) return -1;
    printf("Training...\n");
    int ret;
    int n_batches = net->train->image_hdr.n_items/net->batch_size;
    struct gradient *grad = malloc(sizeof(*grad));
    grad->nabla_w = malloc(2*sizeof(struct matrix *));
    grad->nabla_b = malloc(2*sizeof(struct vector *));
    
    struct vector *delta = new_vector(10);
    struct vector *dcda = new_vector(10);
    
    for (int i = 0; i < 2; i++){
        grad->nabla_w[i] = new_matrix(net->layers[i]->weights->m, net->layers[i]->weights->n);
        grad->nabla_b[i] = new_vector(net->layers[i]->biases->n);
    }
    
    for (int i = 0; i < epochs; i++){
        //if (i % 10 == 0 && !(i==0)) {net->eta *= 0.3;}
        for (int j = 0; j < n_batches; j++){
            for (int k = 0; k < net->batch_size; k++){
                //forward pass, collect all intermediate values
                //printf("net->input->n: %d, data pointer address: %p\n", net->input->n, net->train->images->data);
                ret = image_to_vector(net->input, net->train->images->data, 784*(net->batch_size*j + k));
                //print_image(net->input);
                if (ret == -1) return -1;
                ret = forward(net);
                if (ret == -1) return -1;
                //compute cost and error for last layer
                //printf("l : %d", dat->batches[j]->l[k]);
                //printf("net->train->labels->data: %p\n", net->train->labels->data);

                ret = set_expected(net->expected, net->train->labels->data, net->batch_size*j + k);
                //printf("outside of set_expected\n");
                //print_vector(net->expected);
                if (ret == -1) return -1;
                //print_vector(target);


                act(net->z[1], net->layers[1]->derivative);
                
                delta = hadamard(cost_derivative(net->output, net->expected, dcda), net->z[1], delta);
                ret = backward(net, delta, grad);
                if (ret == -1) return -1;
                //c += cost(net->expected, net->output);
            }
            for (int q = 0; q < 2; q++){
                scl_mat((net->eta/((float) net->batch_size)), grad->nabla_w[q]);
                add_m(net->layers[q]->weights, grad->nabla_w[q], 1);
                scl_vec((net->eta/((float) net->batch_size)), grad->nabla_b[q]);
                add(net->layers[q]->biases, grad->nabla_b[q], 1);
            }
        }
        int total = 0;
        //subtract eta-scaled gradient from respective weights and biases
        for (int i = 0; i < 10000; i++){
            ret = image_to_vector(net->input, net->test->images->data, 784*i);
            if (ret == -1) return -1;
            ret = set_expected(net->expected, net->test->labels->data, i);
            if (ret == -1) return -1;
            ret = evaluate(net);
            if (max_index(net->output) == net->test->labels->data[i]){
                total++;
            }
            else{
                continue;
            }
        }
        printf("Epoch %d: %d / 10000\n", i, total);
        eval_sample(net, 0);
        printf("Expected: ");
        print_image(net->input);
        printf("\nOutput: ");
        print_vector(net->output);
        printf("\n");
    }
    delete_vector(dcda);
    delete_vector(delta);
    for (int i = 0; i < 2; i++){
        delete_matrix(grad->nabla_w[i]);
        delete_vector(grad->nabla_b[i]);
    }
    free(grad->nabla_w);
    free(grad->nabla_b);
    free(grad);
    
    return 0;
}


// TODO:
// Cleanup code, possible future refactor
// Add a loading bar that looks like this:
// [##        ] 20%
// [#####     ] 50%
// [######### ] 90%
int main(){
    init_rand();
    int ret;
    struct network *net = malloc(sizeof(*net));
    int sh[] = {784, 30, 10};
    
    net->eta = 3.0;
    net->batch_size = 10;
    net->n_layers = 2;
    net->shape = sh;
    
    ret = init_net(net);
    if (ret == -1) return -1;
    
    ret = train(net, 10);
    if (ret == -1) return -1;
    store_model(net);
    delete_net(net);
    return 0;
}
