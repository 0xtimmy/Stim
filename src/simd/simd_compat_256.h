#include <immintrin.h>

#include "simd_util.h"
#include <iostream>

struct simd_word {
    union {
        __m256i val;
        __m128i u128[2];
        uint64_t u64[4];
        uint8_t u8[32];
    };

    inline simd_word() : val(__m256i {}) {}
    inline simd_word(__m256i val) : val(val) {}

    inline static simd_word tile8(uint8_t pattern) { return {_mm256_set1_epi8(pattern)}; }

    inline static simd_word tile16(uint16_t pattern) { return {_mm256_set1_epi16(pattern)}; }

    inline static simd_word tile32(uint32_t pattern) { return {_mm256_set1_epi32(pattern)}; }

    inline static simd_word tile64(uint64_t pattern) { return {_mm256_set1_epi64x(pattern)}; }

    inline operator bool() const {  // NOLINT(hicpp-explicit-conversions)
        return u64[0] | u64[1] | u64[2] | u64[3];
    }

    inline simd_word &operator^=(const simd_word &other) {
        val = _mm256_xor_si256(val, other.val);
        return *this;
    }

    inline simd_word &operator&=(const simd_word &other) {
        val = _mm256_and_si256(val, other.val);
        return *this;
    }

    inline simd_word &operator|=(const simd_word &other) {
        val = _mm256_or_si256(val, other.val);
        return *this;
    }

    inline simd_word operator^(const simd_word &other) const { return {_mm256_xor_si256(val, other.val)}; }

    inline simd_word operator&(const simd_word &other) const { return {_mm256_and_si256(val, other.val)}; }

    inline simd_word operator|(const simd_word &other) const { return {_mm256_or_si256(val, other.val)}; }

    inline simd_word andnot(const simd_word &other) const { return {_mm256_andnot_si256(val, other.val)}; }

    inline simd_word leftshift_tile64(uint8_t offset) { return {_mm256_slli_epi64(val, offset)}; }

    inline simd_word rightshift_tile64(uint8_t offset) { return {_mm256_srli_epi64(val, offset)}; }

    inline uint16_t popcount() const {
        return popcnt64(u64[0]) + popcnt64(u64[1]) + popcnt64(u64[2]) + (uint16_t)popcnt64(u64[3]);
    }

    /// For each 128 bit word pair between the two registers, the byte order goes from this:
    /// [a0 a1 a2 a3 ... a14 a15] [b0 b1 b2 b3 ... b14 b15]
    /// to this:
    /// [a0 b0 a1 b1 ...  a7  b7] [a8 b8 a9 b9 ... a15 b15]
    inline void do_interleave8_tile128(simd_word &other) {
        auto t = _mm256_unpackhi_epi8(val, other.val);
        val = _mm256_unpacklo_epi8(val, other.val);
        other.val = t;
    }
};