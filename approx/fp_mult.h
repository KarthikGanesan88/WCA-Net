#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace std;

#define SGN_BIT_MASK 0x1
#define SGN_BIT_SHIFT 31
#define EXP_BIT_MASK 0x7F800000
#define EXP_BIT_SHIFT 23
#define MNT_BIT_MASK 0x7FFFFF
#define MNT_IMPLIED_ONE (1<<23)
#define MNT_PRODUCT_MASK 0x1
#define MNT_PRODUCT_SHIFT 47
#define MNT_FINAL_MASK 23

union val{
    float floating;
    int integer;
};

enum APPX_MODE {
  PRECISE = 0,
  DA,       // Uses actual approx. from DA.
  BITLEVEL, // Uses the precise bit-level calculation.
};

// Function just for seeing the bits when debugging.
void printBits(size_t const size, void const * const ptr)
{
    unsigned char *b = (unsigned char*) ptr;
    unsigned char byte;
    int i, j;

    for (i = size-1; i >= 0; i--) {
        for (j = 7; j >= 0; j--) {
            byte = (b[i] >> j) & 1;
            printf("%u", byte);
        }
        printf("_");
    }
    printf("\n");
}

template <typename dtype>
#ifdef __NVCC__
__host__ __device__
#endif
dtype int_appx_mul(dtype A, dtype B, int bit_width, int appx_mode){

    // Store each row of partial products as a separate array element
    dtype ab_vec[32]; // This needs to be a fixed value for NVCC.
    dtype b_bit = 0 ; //Holds the value of the current b
    for (int i=0; i<bit_width; i++){
      b_bit = (B>>i)&1;
      ab_vec[i] = b_bit?A:0;
      ab_vec[i] = ab_vec[i]<<i;
      //printBits(sizeof(ab_vec[0]),&ab_vec[i]);
    }

    dtype sum = ab_vec[0]; //initialize sum to first PP row
    dtype sum_bit = 0, ab_bit = 0, new_sum_bit = 0, carry = 0;

    for (int row = 1; row<bit_width; row++){
      for (int col = 0; col<(bit_width<<1); col++){ //This time col must go up to propogate the carry
        if (col>(row-1)){

          sum_bit = (sum >> col)&1;
          ab_bit = (ab_vec[row]>>col)&1;

          switch (appx_mode){
            case DA:
              new_sum_bit = ab_bit;
              carry = sum_bit;
              break;
            case BITLEVEL:
              new_sum_bit = sum_bit^ab_bit^carry;
              carry = (sum_bit&ab_bit) | (ab_bit&carry) | (sum_bit&carry);
              break;
          }

          sum &= ~(1ULL<<col);           // Need to 0 the bit first.
          sum |= (new_sum_bit<<col);
        }
      }
      carry = 0;
    }
    return sum;
}

#ifdef __NVCC__
__host__ __device__
#endif
float fp_appx_mul(float input_A, float input_B, int appx_mode) {

    union val A, B, result;
    A.floating = input_A;
    B.floating = input_B;
    unsigned int sgn_a = 0, sgn_b = 0, sgn_ab = 0;
    unsigned int exp_a = 0, exp_b = 0, exp_ab = 0;
    unsigned long int mnt_a = 0, mnt_b = 0;
    unsigned long int mnt_ab = 0;
    unsigned int mnt_overflow = 0;
    result.floating = 0.0f;

    if (fabs(A.floating)<1e-36 || fabs(B.floating)<1e-36 ||
            A.floating == 0.0f || B.floating == 0.0f){
        return 0.0f;
    }
    else {
         sgn_a = (A.integer>>SGN_BIT_SHIFT)&SGN_BIT_MASK;
         sgn_b = (B.integer>>SGN_BIT_SHIFT)&SGN_BIT_MASK;
         sgn_ab = sgn_a^sgn_b;

         exp_a = (A.integer&EXP_BIT_MASK)>>EXP_BIT_SHIFT;
         exp_b = (B.integer&EXP_BIT_MASK)>>EXP_BIT_SHIFT;

         exp_ab = (exp_a + exp_b - 127);
         exp_ab = (exp_ab>255)?255:exp_ab;

         mnt_a = (A.integer&MNT_BIT_MASK)|MNT_IMPLIED_ONE;
         mnt_b = (B.integer&MNT_BIT_MASK)|MNT_IMPLIED_ONE;

         mnt_ab = int_appx_mul(mnt_a, mnt_b, 24, appx_mode);

         mnt_overflow = (mnt_ab>>MNT_PRODUCT_SHIFT)&MNT_PRODUCT_MASK;
         if (mnt_overflow == 0x1 ){
            mnt_ab = (mnt_ab>>(MNT_FINAL_MASK+1))&MNT_BIT_MASK;
            exp_ab += 1;
         }
         else {
            mnt_ab = ((mnt_ab >>MNT_FINAL_MASK)&MNT_BIT_MASK);
         }

         result.integer = (sgn_ab<<SGN_BIT_SHIFT)|(exp_ab<<EXP_BIT_SHIFT)|mnt_ab;

         return result.floating;
   }
}
