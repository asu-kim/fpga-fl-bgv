#ifndef CONSTANT_HPP
#define CONSTANT_HPP

#include <array>

// Integer constants
// const int POLYNOMIAL_DEGREE = 128; // N
// const int PLAINTEXT_MODULUS = 32; // P
// const int CIPHERTEXT_MODULUS = 16974593; // Q
// const int PRIMITIVE_N_TH_ROOT_OF_UNITY = 908870; // W
// const int SQUARE_ROOT_OF_W = 3259673; // E
// const int COEFFICIENT_WIDTH = 27; // ap_int<COEFFICIENT_WIDTH>

// const int POLYNOMIAL_DEGREE = 4;         // N
// const int INVERSE_POLYNOMIAL_DEGREE = 13; // N^-1 (4*13 mod 17 = 52 mod 17 = 1)
// const int CIPHERTEXT_MODULUS = 17;       // Q
// const int PRIMITIVE_N_TH_ROOT_OF_UNITY = 4; // W (4^4 mod 17 = 256 mod 17 = 1)
// const int INVERSE_PRIMITIVE_N_TH_ROOT_OF_UNITY = 13; // W^-1 (4*13 mod 17 = 52 mod 17 = 1)
// const int SQUARE_ROOT_OF_W = 2;          // E (2^2 mod 17 = 4 mod 17 = W)
// const int INVERSE_SQUARE_ROOT_OF_W = 9;  // E^-1 (2*9 mod 17 = 18 mod 17 = 1)

const int POLYNOMIAL_DEGREE = 8; // N
const int INVERSE_POLYNOMIAL_DEGREE = 15; // N^-1 where N * N^-1 mod Q = 1
const int PLAINTEXT_MODULUS = 4; // P
const int CIPHERTEXT_MODULUS = 17; // Q
const int PRIMITIVE_N_TH_ROOT_OF_UNITY = 2; // W
const int INVERSE_PRIMITIVE_N_TH_ROOT_OF_UNITY = 9; // W^-1 where w * w^-1 mod Q = 1
const int SQUARE_ROOT_OF_W = 6; // E
const int INVERSE_SQUARE_ROOT_OF_W = 3; // E^-1 where E * E^-1 mod Q = 1

// const int COEFFICIENT_WIDTH = 5; // ap_int<COEFFICIENT_WIDTH>

// Compile-time LOG2 calculation
constexpr int log2_constexpr(int n) {
    return (n <= 1) ? 0 : 1 + log2_constexpr(n/2);
}

constexpr int LOG2_N = log2_constexpr(POLYNOMIAL_DEGREE);

// Bit-reversal function for compile-time computation
constexpr int bit_reverse(int index, int bits) {
    int result = 0;
    for (int i = 0; i < bits; i++) {
        result = (result << 1) | ((index >> i) & 1);
    }
    return result;
}

// Generate the bit-reversal LUT
template<int N, int BITS>
struct BitReversalTable {
    static constexpr int compute(int i) {
        return bit_reverse(i, BITS);
    }
    
    template<int... Is>
    static constexpr std::array<int, sizeof...(Is)> generate(std::integer_sequence<int, Is...>) {
        return { compute(Is)... };
    }
    
    static constexpr auto table = generate(std::make_integer_sequence<int, N>{});
};

// Define the BIT_REVERSE_LUT using the template
using BitRevTable = BitReversalTable<POLYNOMIAL_DEGREE, LOG2_N>;
constexpr auto BIT_REVERSE_LUT = BitRevTable::table;

// Constexpr modulo function
constexpr int mod_constexpr(int a, int m) {
    int result = a % m;
    return result >= 0 ? result : result + m;
}

// Constexpr modular power function
constexpr int pow_mod_constexpr(int base, int exp, int mod) {
    int result = 1;
    base = mod_constexpr(base, mod);
    while (exp > 0) {
        if (exp & 1)
            result = mod_constexpr(result * base, mod);
        base = mod_constexpr(base * base, mod);
        exp >>= 1;
    }
    return result;
}

// Generate all twiddle factors needed for NTT-based polynomial multiplication
template<int N, int W, int W_INV, int E, int E_INV, int MOD>
struct TwiddleFactorTable {
    // Compute W^i mod MOD
    static constexpr int compute_w_power(int i) {
        return pow_mod_constexpr(W, i, MOD);
    }
    
    // Compute W_INV^i mod MOD
    static constexpr int compute_w_inv_power(int i) {
        return pow_mod_constexpr(W_INV, i, MOD);
    }
    
    // Compute E^i mod MOD (E is square root of W)
    static constexpr int compute_e_power(int i) {
        return pow_mod_constexpr(E, i, MOD);
    }
    
    // Compute E_INV^i mod MOD (E_INV is inverse of E)
    static constexpr int compute_e_inv_power(int i) {
        return pow_mod_constexpr(E_INV, i, MOD);
    }
    
    template<int... Is>
    static constexpr std::array<int, sizeof...(Is)> generate_w_powers(std::integer_sequence<int, Is...>) {
        return { compute_w_power(Is)... };
    }
    
    template<int... Is>
    static constexpr std::array<int, sizeof...(Is)> generate_w_inv_powers(std::integer_sequence<int, Is...>) {
        return { compute_w_inv_power(Is)... };
    }
    
    template<int... Is>
    static constexpr std::array<int, sizeof...(Is)> generate_e_powers(std::integer_sequence<int, Is...>) {
        return { compute_e_power(Is)... };
    }
    
    template<int... Is>
    static constexpr std::array<int, sizeof...(Is)> generate_e_inv_powers(std::integer_sequence<int, Is...>) {
        return { compute_e_inv_power(Is)... };
    }
    
    // Generate tables for NTT/INTT butterfly operations
    static constexpr auto w_powers_half = generate_w_powers(std::make_integer_sequence<int, N/2>{});
    static constexpr auto w_inv_powers_half = generate_w_inv_powers(std::make_integer_sequence<int, N/2>{});
    
    // Generate full-length tables for polynomial multiplication
    static constexpr auto w_powers = generate_w_powers(std::make_integer_sequence<int, N>{});
    static constexpr auto w_inv_powers = generate_w_inv_powers(std::make_integer_sequence<int, N>{});
    static constexpr auto e_powers = generate_e_powers(std::make_integer_sequence<int, N>{});
    static constexpr auto e_inv_powers = generate_e_inv_powers(std::make_integer_sequence<int, N>{});
};

// Define all twiddle factor lookup tables
using TwiddleFactors = TwiddleFactorTable<
    POLYNOMIAL_DEGREE, 
    PRIMITIVE_N_TH_ROOT_OF_UNITY, 
    INVERSE_PRIMITIVE_N_TH_ROOT_OF_UNITY,
    SQUARE_ROOT_OF_W,
    INVERSE_SQUARE_ROOT_OF_W,
    CIPHERTEXT_MODULUS
>;

// Lookup tables for NTT/INTT butterfly operations (smaller size)
constexpr auto W_POWERS_HALF_LUT = TwiddleFactors::w_powers_half;
constexpr auto W_INV_POWERS_HALF_LUT = TwiddleFactors::w_inv_powers_half;

// Full-size lookup tables for polynomial multiplication
constexpr auto W_POWERS_LUT = TwiddleFactors::w_powers;
constexpr auto W_INV_POWERS_LUT = TwiddleFactors::w_inv_powers;
constexpr auto E_POWERS_LUT = TwiddleFactors::e_powers;
constexpr auto E_INV_POWERS_LUT = TwiddleFactors::e_inv_powers;

#endif // CONSTANT_HPP