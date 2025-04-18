#ifndef CONSTANT_HPP
#define CONSTANT_HPP

#include <array>
#include <cstdint>

/** 
 * Rules: 
 * 1. q is a big prime number that satisfies q mod 2n = 1 
 * 3. p << q
 * 
 */
// Integer constants
const int64_t POLYNOMIAL_DEGREE = 128; // N
const int64_t INVERSE_POLYNOMIAL_DEGREE = 16841979; // N^-1 (4*13 mod 17 = 52 mod 17 = 1)
const int64_t PLAINTEXT_MODULUS = 256; // P
const int64_t CIPHERTEXT_MODULUS = 16974593; // Q
const int64_t PRIMITIVE_N_TH_ROOT_OF_UNITY = 908870; // W
const int64_t INVERSE_PRIMITIVE_N_TH_ROOT_OF_UNITY = 12269082; // W^-1 (4*13 mod 17 = 52 mod 17 = 1)
const int64_t SQUARE_ROOT_OF_W = 3259673; // E
const int64_t INVERSE_SQUARE_ROOT_OF_W = 1797420; // E^-1

// const int64_t COEFFICIENT_WIDTH = 8; // E^-1

// const int64_t POLYNOMIAL_DEGREE = 4;         // N
// const int64_t INVERSE_POLYNOMIAL_DEGREE = 13; // N^-1 (4*13 mod 17 = 52 mod 17 = 1)
// const int64_t CIPHERTEXT_MODULUS = 17;       // Q
// const int64_t PRIMITIVE_N_TH_ROOT_OF_UNITY = 4; // W (4^4 mod 17 = 256 mod 17 = 1)
// const int64_t INVERSE_PRIMITIVE_N_TH_ROOT_OF_UNITY = 13; // W^-1 (4*13 mod 17 = 52 mod 17 = 1)
// const int64_t SQUARE_ROOT_OF_W = 2;          // E (2^2 mod 17 = 4 mod 17 = W)
// const int64_t INVERSE_SQUARE_ROOT_OF_W = 9;  // E^-1 (2*9 mod 17 = 18 mod 17 = 1)

// const int64_t POLYNOMIAL_DEGREE = 8; // N
// const int64_t INVERSE_POLYNOMIAL_DEGREE = 211; // N^-1 where N * N^-1 mod Q = 1
// const int64_t PLAINTEXT_MODULUS = 5; // P
// const int64_t CIPHERTEXT_MODULUS = 241; // Q
// const int64_t PRIMITIVE_N_TH_ROOT_OF_UNITY = 30; // W
// const int64_t INVERSE_PRIMITIVE_N_TH_ROOT_OF_UNITY = 233; // W^-1 where w * w^-1 mod Q = 1
// const int64_t SQUARE_ROOT_OF_W = 111; // E
// const int64_t INVERSE_SQUARE_ROOT_OF_W = 76; // E^-1 where E * E^-1 mod Q = 1

// const int64_t COEFFICIENT_WIDTH = 5; // ap_int<COEFFICIENT_WIDTH>

// Compile-time LOG2 calculation
constexpr int64_t log2_constexpr(int64_t n) {
    return (n <= 1) ? 0 : 1 + log2_constexpr(n/2);
}

/**
 * w^4 % q = 2143183
 * 8 10353247
 * 16 6407014
 */

constexpr int64_t LOG2_N = log2_constexpr(POLYNOMIAL_DEGREE);

// Bit-reversal function for compile-time computation
constexpr int64_t bit_reverse(int64_t index, int64_t bits) {
    int64_t result = 0;
    for (int64_t i = 0; i < bits; i++) {
        result = (result << 1) | ((index >> i) & 1);
    }
    return result;
}

// Generate the bit-reversal LUT
template<int64_t N, int64_t BITS>
struct BitReversalTable {
    static constexpr int64_t compute(int64_t i) {
        return bit_reverse(i, BITS);
    }
    
    template<int64_t... Is>
    static constexpr std::array<int64_t, sizeof...(Is)> generate(std::integer_sequence<int64_t, Is...>) {
        return { compute(Is)... };
    }
    
    static constexpr auto table = generate(std::make_integer_sequence<int64_t, N>{});
};

// Define the BIT_REVERSE_LUT using the template
using BitRevTable = BitReversalTable<POLYNOMIAL_DEGREE, LOG2_N>;
constexpr auto BIT_REVERSE_LUT = BitRevTable::table;

// Constexpr modulo function
constexpr int64_t mod_constexpr(int64_t a, int64_t m) {
    int64_t result = a % m;
    return result >= 0 ? result : result + m;
}

// Constexpr modular power function
constexpr int64_t pow_mod_constexpr(int64_t base, int64_t exp, int64_t mod) {
    int64_t result = 1;
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
template<int64_t N, int64_t W, int64_t W_INV, int64_t E, int64_t E_INV, int64_t MOD>
struct TwiddleFactorTable {
    // Compute W^i mod MOD
    static constexpr int64_t compute_w_power(int64_t i) {
        return pow_mod_constexpr(W, i, MOD);
    }
    
    // Compute W_INV^i mod MOD
    static constexpr int64_t compute_w_inv_power(int64_t i) {
        return pow_mod_constexpr(W_INV, i, MOD);
    }
    
    // Compute E^i mod MOD (E is square root of W)
    static constexpr int64_t compute_e_power(int64_t i) {
        return pow_mod_constexpr(E, i, MOD);
    }
    
    // Compute E_INV^i mod MOD (E_INV is inverse of E)
    static constexpr int64_t compute_e_inv_power(int64_t i) {
        return pow_mod_constexpr(E_INV, i, MOD);
    }
    
    template<int64_t... Is>
    static constexpr std::array<int64_t, sizeof...(Is)> generate_w_powers(std::integer_sequence<int64_t, Is...>) {
        return { compute_w_power(Is)... };
    }
    
    template<int64_t... Is>
    static constexpr std::array<int64_t, sizeof...(Is)> generate_w_inv_powers(std::integer_sequence<int64_t, Is...>) {
        return { compute_w_inv_power(Is)... };
    }
    
    template<int64_t... Is>
    static constexpr std::array<int64_t, sizeof...(Is)> generate_e_powers(std::integer_sequence<int64_t, Is...>) {
        return { compute_e_power(Is)... };
    }
    
    template<int64_t... Is>
    static constexpr std::array<int64_t, sizeof...(Is)> generate_e_inv_powers(std::integer_sequence<int64_t, Is...>) {
        return { compute_e_inv_power(Is)... };
    }
    
    // Generate tables for NTT/INTT butterfly operations
    static constexpr auto w_powers_half = generate_w_powers(std::make_integer_sequence<int64_t, N/2>{});
    static constexpr auto w_inv_powers_half = generate_w_inv_powers(std::make_integer_sequence<int64_t, N/2>{});
    
    // Generate full-length tables for polynomial multiplication
    static constexpr auto w_powers = generate_w_powers(std::make_integer_sequence<int64_t, N>{});
    static constexpr auto w_inv_powers = generate_w_inv_powers(std::make_integer_sequence<int64_t, N>{});
    static constexpr auto e_powers = generate_e_powers(std::make_integer_sequence<int64_t, N>{});
    static constexpr auto e_inv_powers = generate_e_inv_powers(std::make_integer_sequence<int64_t, N>{});
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