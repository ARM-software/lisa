#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

// TODO: remove that, comes from the kernel
#define min(x, y) (x < y ? x : y)
typedef unsigned char unchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

struct string {
	char *start;
	char *end;
};

#define WITH_NULL_TERMINATED(str, f)                                           \
	({                                                                     \
		char last = *(str)->end;                                       \
		*(str)->end = '\0';                                            \
		typeof(f(NULL)) res = f((str)->start);                         \
		*(str)->end = last;                                            \
		res;                                                           \
	})

typedef struct string string;

size_t string_length(string *str)
{
	return (str->end - str->start);
}

string charp2string(char *s)
{
	return (string){ .start = s,
			 .end = (char *)((uintptr_t)s + strlen(s)) };
}

size_t string2charp(string *src, char *dst, size_t max)
{
	size_t n = min(max, string_length(src));
	memcpy(dst, src->start, n);
	dst[n] = '\0';
	return n;
}

enum tag {
	SUCCESS,
	FAILURE,
};

#define PARSE_RESULT(type) parse_result_##type

#define MAKE_PARSE_RESULT_TYPE(type)                                           \
	typedef struct PARSE_RESULT(type) {                                    \
		enum tag tag;                                                  \
		string remainder;                                              \
		type value;                                                    \
	} PARSE_RESULT(type)

#define IS_SUCCESS(res) (res.tag == SUCCESS)

MAKE_PARSE_RESULT_TYPE(string);
MAKE_PARSE_RESULT_TYPE(char);

PARSE_RESULT(string) parse_string(string *input, char *match)
{
	size_t len = strlen(match);
	string remainder;

	if (string_length(input) < len) {
		return (PARSE_RESULT(string)){ .tag = FAILURE,
					       .remainder = *input };
	} else {
		if (!memcmp(input->start, match, len)) {
			return (PARSE_RESULT(
				string)){ .tag = SUCCESS,
					  .remainder =
						  (string){
							  .start =
								  input->start +
								  len,
							  .end = input->end,
						  },
					  .value = {
						  .start = input->start,
						  .end = input->start + len,
					  } };
		} else {
			return (PARSE_RESULT(string)){
				.tag = FAILURE,
				.remainder = *input,
			};
		}
	}
}

PARSE_RESULT(char) __parse_char(string *input, char *allowed, bool revert)
{
	if (input->start < input->end) {
		char input_char = *input->start;
		char c;
		for (c = *allowed; c != '\0'; c = *allowed++) {
			if (revert ? input_char != c : input_char == c) {
				return (PARSE_RESULT(char)){
					.tag = SUCCESS,
					.remainder =
						(string){ .start = input->start + 1,
							.end = input->end },
					.value = c,
				};
			}
		}
	}
	return (PARSE_RESULT(char)){ .tag = FAILURE, .remainder = *input };
}

PARSE_RESULT(char) parse_char(string *input, char *allowed)
{
	return __parse_char(input, allowed, false);
}

PARSE_RESULT(char) parse_char_not_in(string *input, char *disallowed)
{
	return __parse_char(input, disallowed, true);
}

#define APPLY(type, name, parser, ...)                                         \
	PARSE_RESULT(type) name(string *input)                                 \
	{                                                                      \
		return parser(input, __VA_ARGS__);                             \
	}

#define APPLY_AND_PARSE(type, name, parser, ...)                               \
	({                                                                     \
		APPLY(type, name, parser, __VA_ARGS__);                        \
		PARSE(name);                                                   \
	})

#define OR(type, name, parser1, parser2)                                       \
	PARSE_RESULT(type) name(string *input)                                 \
	{                                                                      \
		PARSE_RESULT(type) res1 = parser1(input);                      \
		if (IS_SUCCESS(res1)) {                                        \
			return res1;                                           \
		} else {                                                       \
			return parser2(input);                                 \
		}                                                              \
	}

#define PURE(type, name, _value)                                               \
	PARSE_RESULT(type) name(string *input)                                 \
	{                                                                      \
		return (PARSE_RESULT(type)){ .tag = SUCCESS,                   \
					     .remainder = *input,              \
					     .value = (_value) };              \
	}

#define MAP(f_type, parser_type, name, parser, f)                              \
	PARSE_RESULT(f_type) name(string *input)                               \
	{                                                                      \
		PARSE_RESULT(parser_type) res = parser(input);                 \
		if (IS_SUCCESS(res)) {                                         \
			return (PARSE_RESULT(                                  \
				f_type)){ .tag = SUCCESS,                      \
					  .remainder = res.remainder,          \
					  .value = f(res.value) };             \
		} else {                                                       \
			return (PARSE_RESULT(f_type)){                         \
				.tag = FAILURE, .remainder = res.remainder     \
			};                                                     \
		}                                                              \
	}

#define MAP_STRING(f_type, parser_type, name, parser, f)                       \
	typeof(f(NULL)) __map_string_##f(string str)                           \
	{                                                                      \
		return WITH_NULL_TERMINATED(&str, f);                          \
	}                                                                      \
	MAP(f_type, parser_type, name, parser, __map_string_##f)

#define AT_LEAST(type, name, parser, f, init, n)                               \
	PARSE_RESULT(type) name(string *input)                                 \
	{                                                                      \
		typeof(init) acc = init;                                       \
		typeof(parser(NULL)) res =                                     \
			(typeof(parser(NULL))){ .remainder = *input };         \
		for (size_t i = 0;; i++) {                                     \
			res = parser(&res.remainder);                          \
			if (IS_SUCCESS(res)) {                                 \
				acc = f(acc, res.value);                       \
			} else {                                               \
				return (PARSE_RESULT(type)){                   \
					.tag = i >= n ? SUCCESS : FAILURE,     \
					.remainder = res.remainder,            \
					.value = acc                           \
				};                                             \
			}                                                      \
		}                                                              \
	}

#define MANY(type, name, parser, f, init)                                      \
	AT_LEAST(type, name, parser, f, init, 0)

#define TAKEWHILE_AT_LEAST(type, name, parser, n)                                        \
	PARSE_RESULT(string) name(string *input)                                         \
	{                                                                                \
		char *start = input->start;                                              \
		PARSE_RESULT(type)                                                       \
		res = (PARSE_RESULT(type)){ .remainder = *input };                       \
		for (size_t i = 0;; i++) {                                               \
			res = parser(&res.remainder);                                    \
			if (!IS_SUCCESS(res)) {                                          \
				if (i >= n)                                              \
					return (PARSE_RESULT(string)){                   \
						.tag = SUCCESS,                          \
						.value =                                 \
							(string){                        \
								.start =                 \
									start,           \
								.end = res.remainder     \
									       .start }, \
						.remainder = res.remainder,              \
					};                                               \
				else                                                     \
					return (PARSE_RESULT(string)){                   \
						.tag = FAILURE,                          \
						.remainder = *input,                     \
					};                                               \
			}                                                                \
		}                                                                        \
	}
#define TAKEWHILE(type, name, parser) TAKEWHILE_AT_LEAST(type, name, parser, 0)

#define COUNT_MANY(parser_type, name, parser)                                  \
	int __count_fold_##name(int acc, int x)                                \
	{                                                                      \
		return acc + x;                                                \
	}                                                                      \
	int __count_one_f_##name(parser_type _)                                \
	{                                                                      \
		return 1;                                                      \
	}                                                                      \
	MAP(int, parser_type, __count_one_##name, parser,                      \
	    __count_one_f_##name);                                             \
	MANY(int, name, __count_one_##name, __count_fold_##name, 0)

#define THEN(parser1_type, parser2_type, name, parser1, parser2)               \
	PARSE_RESULT(parser2_type) name(string *input)                         \
	{                                                                      \
		PARSE_RESULT(parser1_type) res = parser1(input);               \
		PARSE_RESULT(parser2_type) res2;                               \
		if (IS_SUCCESS(res)) {                                         \
			res2 = parser2(&res.remainder, res.value);             \
			if (IS_SUCCESS(res2))                                  \
				return res2;                                   \
		}                                                              \
		return (PARSE_RESULT(parser2_type)){ .tag = FAILURE,           \
						     .remainder = *input };    \
	}

#define RIGHT(parser1_type, parser2_type, name, parser1, parser2)              \
	PARSE_RESULT(parser2_type)                                             \
	__discard_then_##parser2(string *input, parser1_type _)                \
	{                                                                      \
		return parser2(input);                                         \
	}                                                                      \
	THEN(parser1_type, parser2_type, name, parser1,                        \
	     __discard_then_##parser2)

#define LEFT(parser1_type, parser2_type, name, parser1, parser2)               \
	PARSE_RESULT(parser1_type)                                             \
	__forward_then_discard_##parser2(string *input, parser1_type value)    \
	{                                                                      \
		PARSE_RESULT(parser2_type) res = parser2(input);               \
		return (PARSE_RESULT(parser1_type)){ .tag = SUCCESS,           \
						     .value = value,           \
						     .remainder =              \
							     res.remainder };  \
	}                                                                      \
	THEN(parser1_type, parser1_type, name, parser1,                        \
	     __forward_then_discard_##parser2)

#define PARSE(parser, ...)                                                     \
	({                                                                     \
		typeof(parser(&__seq_remainder, ##__VA_ARGS__)) res =          \
			parser(&__seq_remainder, ##__VA_ARGS__);               \
		if (!IS_SUCCESS(res))                                          \
			goto __seq_failure;                                    \
		__seq_remainder = res.remainder;                               \
		res.value;                                                     \
	})

#define SEQUENCE(type, name, body, params...)                                  \
	PARSE_RESULT(type) name(string *input, ##params)                       \
	{                                                                      \
		string __seq_remainder = *input;                               \
		string __seq_unmodified_input = *input;                        \
		type __seq_value = (body);                                     \
		return (PARSE_RESULT(type)){                                   \
			.tag = SUCCESS,                                        \
			.remainder = __seq_remainder,                          \
			.value = __seq_value,                                  \
		};                                                             \
	__seq_failure:                                                         \
		return (PARSE_RESULT(type)){                                   \
			.tag = FAILURE,                                        \
			.remainder = __seq_unmodified_input,                   \
		};                                                             \
	}

/* TEST */

MAKE_PARSE_RESULT_TYPE(int);
MAKE_PARSE_RESULT_TYPE(long);
MAKE_PARSE_RESULT_TYPE(ulong);
MAKE_PARSE_RESULT_TYPE(uint);

APPLY(char, parse_space, parse_char, " \n");
COUNT_MANY(char, count_spaces, parse_space);

long tolong(char *s)
{
	char *ptr;
	return strtol(s, &ptr, 10);
}
APPLY(char, parse_digit, parse_char, "0123456789");
TAKEWHILE_AT_LEAST(char, parse_number_string, parse_digit, 1);
MAP_STRING(long, string, parse_long, parse_number_string, tolong);

#define PIXEL6_EMETER_CHAN_NAME_MAX_SIZE 64
typedef struct sample {
	unsigned long ts;
	unsigned long value;
	unsigned int chan;
	char chan_name[PIXEL6_EMETER_CHAN_NAME_MAX_SIZE];
} sample_t;
MAKE_PARSE_RESULT_TYPE(sample_t);

SEQUENCE(sample_t, parse_sample, ({
		 sample_t value;

		 /* CH42 */
		 APPLY_AND_PARSE(string, parse_ch, parse_string, "CH");
		 value.chan = PARSE(parse_long);

		 /* (T=42) */
		 APPLY_AND_PARSE(string, parse_paren_T_eq, parse_string, "(T=");
		 value.ts = PARSE(parse_long);
		 APPLY_AND_PARSE(string, parse_lparen, parse_string, ")");

		 /* [CHAN_NAME] */
		 APPLY_AND_PARSE(string, parse_lbracket, parse_string, "[");
		 APPLY(char, parse_name_char, parse_char_not_in, "]");
		 TAKEWHILE(char, parse_name, parse_name_char);
		 string _name = PARSE(parse_name);
		 string2charp(&_name, value.chan_name,
			      PIXEL6_EMETER_CHAN_NAME_MAX_SIZE);
		 APPLY_AND_PARSE(string, parse_rbracket, parse_string, "]");

		 /* , */
		 APPLY_AND_PARSE(string, parse_comma, parse_string, ", ");

		 /* 12345 */
		 value.value = PARSE(parse_long);

		 value;
	 }))

LEFT(sample_t, int, parse_sample_line, parse_sample, count_spaces)

int process_sample(int nr, sample_t sample)
{
	printf("hello chan=%u, ts=%li chan_name=%s value=%li\n", sample.chan,
	       sample.ts, sample.chan_name, sample.value);
	return nr + 1;
}

SEQUENCE(int, parse_content, ({
		 /* t=12345 */
		 APPLY(string, parse_teq, parse_string, "t=");
		 PARSE(parse_teq);
		 long x = PARSE(parse_long);
		 PARSE(count_spaces);

		 /* Parse all the following sample lines */
		 MANY(int, parse_all_samples, parse_sample_line, process_sample,
		      0);
		 PARSE(parse_all_samples);
	 }))

#define SAMPLE                                                                 \
	"t=473848\nCH42(T=473848)[S10M_VDD_TPU], 3161249\nCH1(T=473848)[VSYS_PWR_MODEM], 48480309\nCH2(T=473848)[VSYS_PWR_RFFE], 9594393\nCH3(T=473848)[S2M_VDD_CPUCL2], 28071872\nCH4(T=473848)[S3M_VDD_CPUCL1], 17477139\nCH5(T=473848)[S4M_VDD_CPUCL0], 113447446\nCH6(T=473848)[S5M_VDD_INT], 12543588\nCH7(T=473848)[S1M_VDD_MIF], 25901660\n"

int main()
{
	char *content = strdup(SAMPLE);
	string input = charp2string(content);
	PARSE_RESULT(int) res = parse_content(&input);
	if (IS_SUCCESS(res)) {
		printf("parsed %i samples\n", res.value);
	} else {
		printf("Failed to parse content\n");
	}
	return 0;
}
