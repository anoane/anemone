/* A template argument is a shape without a name, also inside a constant table or a #define. */
template <int row_count> static void stamp(float *rows) { rows[0] = (float)row_count; }
typedef void (*stamped_kernel)(float *);
static const stamped_kernel stamped_kernels[] = {
    stamp<8>, stamp<12>,
    stamp <4096>,
};
#define WIDEST_KERNEL stamp<65536>
static void launch(float *rows) { stamp<16>(rows); }
