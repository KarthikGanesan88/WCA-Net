#include <cmath>

#define SGN_BIT_MASK 0x1
#define SGN_BIT_SHIFT 31
#define EXP_BIT_MASK 0x7F800000
#define EXP_BIT_SHIFT 23
#define MNT_BIT_MASK 0x7FFFFF
#define MNT_IMPLIED_ONE (1<<23)
#define MNT_PRODUCT_MASK 0x1
#define MNT_PRODUCT_SHIFT 47
#define MNT_FINAL_MASK 23

#define N 24

union val{
    float floating;
    int integer;
};

__device__ unsigned long int int_mult_alt(unsigned long int A, unsigned long int B){

    // Store each row of partial products as a separate array element

    //cout<<"AB_VEC:" << endl;
    unsigned long int ab_vec[N];
    unsigned long int b_bit = 0 ; //Holds the value of the current b
    for (int i=0; i<N; i++){
      b_bit = (B>>i)&1;
      ab_vec[i] = b_bit?A:0;
      ab_vec[i] = ab_vec[i]<<i;
      //printBits(sizeof(ab_vec[0]),&ab_vec[i]);
    }

    unsigned long int sum = ab_vec[0]; //initialize sum to first PP row
    unsigned long int sum_bit = 0, ab_bit = 0, new_sum_bit = 0, carry = 0;

    for (int row = 1; row<N; row++){
      for (int col = 0; col<(N<<1); col++){ //This time col must go up to propogate the carry

        if (col>(row-1)){
          sum_bit = (sum >> col)&1;
          ab_bit = (ab_vec[row]>>col)&1;
          //cout << "(" << sum_bit << "^" << ab_bit << "^" << carry << ")";
          new_sum_bit = sum_bit^ab_bit^carry;
          carry = (sum_bit&ab_bit) | (ab_bit&carry) | (sum_bit&carry);

          sum &= ~(1ULL<<col);           // Need to 0 the bit first.
          sum |= (new_sum_bit<<col);
        }
        //else cout << "("<< sum_bit <<"|||)";
      }
      //cout << endl;
      carry = 0;
    }

    //printBits(sizeof(sum),&sum);
    return sum;

}

__device__ float FP_appx_mul(float input_A, float input_B) {

    union val A, B, result;
    A.floating = input_A;
    B.floating = input_B;
    unsigned int sgn_a = 0, sgn_b = 0, sgn_ab = 0;
    unsigned int exp_a = 0, exp_b = 0, exp_ab = 0;
    unsigned long int mnt_a = 0, mnt_b = 0;
    unsigned long int mnt_ab = 0;
    unsigned int mnt_overflow = 0;
    result.floating = 0.0f;

    if ((!isnormal(A.floating)) || (!isnormal(A.floating)) || A.floating == 0.0f || B.floating == 0.0f){
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
         //printBits(sizeof(mnt_a),&mnt_a);

         mnt_b = (B.integer&MNT_BIT_MASK)|MNT_IMPLIED_ONE;
         //printBits(sizeof(mnt_b),&mnt_b);

         //mnt_ab = mnt_a * mnt_b;
         //printBits(sizeof(mnt_ab),&mnt_ab);
         mnt_ab = int_mult_alt(mnt_a, mnt_b);
         //printBits(sizeof(mnt_ab),&mnt_ab);
         //printBits(sizeof(mnt_ab),&mnt_ab);

         mnt_overflow = (mnt_ab>>MNT_PRODUCT_SHIFT)&MNT_PRODUCT_MASK;
         if (mnt_overflow == 0x1 ){
            mnt_ab = (mnt_ab>>(MNT_FINAL_MASK+1))&MNT_BIT_MASK;
            exp_ab += 1;
         }
         else {
            mnt_ab = ((mnt_ab >>MNT_FINAL_MASK)&MNT_BIT_MASK);
         }
         //printBits(sizeof(mnt_ab),&mnt_ab);

         result.integer = (sgn_ab<<SGN_BIT_SHIFT)|(exp_ab<<EXP_BIT_SHIFT)|mnt_ab;
         //printBits(sizeof(result),&result);

         return result.floating;
   }
}
