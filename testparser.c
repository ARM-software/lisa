#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

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
MAKE_PARSE_RESULT_TYPE(int);
MAKE_PARSE_RESULT_TYPE(long);
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

static inline PARSE_RESULT(char) parse_char(string *input, char *allowed)
{
	char input_char = *input->start;
	char c;
	for (c = *allowed; c != '\0'; c = *allowed++) {
		if (input_char == c) {
			return (PARSE_RESULT(char)){
				.tag = SUCCESS,
				.remainder =
					(string){ .start = input->start + 1,
						  .end = input->end },
				.value = c,
			};
		}
	}

	return (PARSE_RESULT(char)){ .tag = FAILURE, .remainder = *input };
}

#define APPLY(type, name, parser, ...)                                         \
	inline PARSE_RESULT(type) name(string *input)                          \
	{                                                                      \
		return parser(input, __VA_ARGS__);                             \
	}

#define STATIC_APPLY(type, name, parser, args...)                              \
	static APPLY(type, name, parser, args)

#define OR(type, name, p1, p2)                                                 \
	static inline PARSE_RESULT(type) name(string *input)                   \
	{                                                                      \
		PARSE_RESULT(type) res1 = p1(input);                           \
		if (IS_SUCCESS(res1)) {                                        \
			return res1;                                           \
		} else {                                                       \
			return p2(input);                                      \
		}                                                              \
	}

#define PURE(type, name, _value)                                               \
	static inline PARSE_RESULT(type) name(string *input)                   \
	{                                                                      \
		return (PARSE_RESULT(type)){ .tag = SUCCESS,                   \
					     .remainder = *input,              \
					     .value = (_value) };              \
	}

#define MAP(f_type, parser_type, name, parser, f)                              \
	static PARSE_RESULT(f_type) name(string *input)                        \
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
	static inline typeof(f(NULL)) __map_string_##f(string str)             \
	{                                                                      \
		return WITH_NULL_TERMINATED(&str, f);                          \
	}                                                                      \
	MAP(f_type, parser_type, name, parser, __map_string_##f)

#define AT_LEAST(type, name, parser, f, init, n)                               \
	static PARSE_RESULT(type) name(string *input)                          \
	{                                                                      \
		typeof(init) acc = init;                                       \
		PARSE_RESULT(type)                                             \
		res = (PARSE_RESULT(type)){ .remainder = *input };             \
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
	static PARSE_RESULT(string) name(string *input)                                  \
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
	static int __count_fold_##name(int acc, int x)                         \
	{                                                                      \
		return acc + x;                                                \
	}                                                                      \
	static int __count_one_f_##name(parser_type _)                         \
	{                                                                      \
		return 1;                                                      \
	}                                                                      \
	MAP(int, parser_type, __count_one_##name, parser,                      \
	    __count_one_f_##name);                                             \
	MANY(int, name, __count_one_##name, __count_fold_##name, 0)

#define THEN(parser1_type, parser2_type, name, parser1, parser2)               \
	static PARSE_RESULT(parser2_type) name(string *input)                  \
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

#define DISCARD_THEN(parser1_type, parser2_type, name, parser1, parser2)       \
	static PARSE_RESULT(parser2_type)                                      \
		__discard_then_##parser2(string *input, parser1_type _)        \
	{                                                                      \
		return parser2(input);                                         \
	}                                                                      \
	THEN(parser1_type, parser2_type, name, parser1,                        \
	     __discard_then_##parser2)

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
	static PARSE_RESULT(type) name(string *input, ##params)                \
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

/* char curr; */

/* while (1) { */
/* 	curr = str->start; */
/* 	str->start++; */
/* 	if (str->start > str->end) */
/* 		break; */
/* } */

int myf(char *s)
{
	printf("got %s\n", s);
	return 44;
}

int make_int(string str)
{
	return WITH_NULL_TERMINATED(&str, myf);
}

static long tolong(char *s)
{
	char *ptr;
	return strtol(s, &ptr, 10);
}

STATIC_APPLY(char, parse_space, parse_char, " \n");
COUNT_MANY(char, count_spaces, parse_space);

STATIC_APPLY(string, parse_hello, parse_string, "hello");
STATIC_APPLY(string, parse_bar, parse_string, "bar");
OR(string, parse_hello_or_bar, parse_hello, parse_bar);

MAP(int, string, parse_hello_or_bar_int, parse_hello_or_bar, make_int);

PURE(int, make_45, 45);
OR(int, parse_hello_or_bar_or_else_45, parse_hello_or_bar_int, make_45);

int mymany(int acc, int parsed)
{
	printf("parsed %i, adding\n", parsed);
	return acc + parsed;
}
MANY(int, parse_hello_or_bar_int_many, parse_hello_or_bar_int, mymany, 41);
DISCARD_THEN(int, int, spaces_then_parse_hello_or_bar_int_many, count_spaces,
	     parse_hello_or_bar_int_many)

/* TEST */

STATIC_APPLY(char, parse_digit, parse_char, "0123456789");
TAKEWHILE_AT_LEAST(char, parse_number_string, parse_digit, 1);
MAP_STRING(long, string, parse_long, parse_number_string, tolong);

typedef struct data {
	long x;
	long y;
} data;
MAKE_PARSE_RESULT_TYPE(data);

/* data make_x(long x) */
/* { */
/* 	return (data){ .x = x }; */
/* } */
/* MAP(data, long, parse_x, parse_long, make_x); */

/* // TODO: abstract over that builder pattern */
/* PARSE_RESULT(data) parse_y(string *input, data partial) */
/* { */
/* 	PARSE_RESULT(long) res = parse_long(input); */
/* 	if (IS_SUCCESS(res)) { */
/* 		partial.y = res.value; */
/* 		return (PARSE_RESULT(data)){ .tag = SUCCESS, */
/* 					     .value = partial, */
/* 					     .remainder = res.remainder }; */
/* 	} else { */
/* 		return (PARSE_RESULT(data)){ .tag = FAILURE, */
/* 					     .remainder = res.remainder }; */
/* 	} */
/* } */

/* THEN(data, data, parse_data, parse_x, parse_y); */

SEQUENCE(data, parse_data, ({
		 long x = PARSE(parse_long);
		 PARSE(count_spaces);
		 long y = PARSE(parse_long);
		 (data){ .x = x, .y = y };
	 }))

/* #define SAMPLE "t=123456789 CH11(T=123456789)[CPU_HELLO], 42\nt=123456789 CH11(T=123456789)[RAM], 43" */
#define SAMPLE "1 2 66   \n    hellobar"

void foo(int hell)
{
	inline void bar(void)
	{
		printf("xxx %i\n", hell);
	};
	bar();
};

/* string -> (val, string) */
int main()
{
	char *content = strdup(SAMPLE);
	string input = (string){ .start = content,
				 .end = (char *)((uintptr_t)content +
						 strlen(content)) };

	/* PARSE_RESULT(int) res = spaces_then_parse_hello_or_bar_int_many(&input); */
	/* PARSE_RESULT(int) res = count_spaces(&input); */
	/* PARSE_RESULT(long) res = parse_long(&input); */
	/* printf("%s: %li\n", res.tag == SUCCESS ? "SUCCESS" : "FAILURE", */
	/*        res.value); */

	PARSE_RESULT(data) res = parse_data(&input);
	printf("%s: x=%li y=%li\n", res.tag == SUCCESS ? "SUCCESS" : "FAILURE",
	       res.value.x, res.value.y);
	return 0;
}

int main2()
{
	const char sep[] = "\n";
	char *token;

	unsigned long long ts;
	unsigned long chan;
	unsigned long cpu;
	char chan_name[40] = { '\0' };
	unsigned long long value;

	char *content = strdup(SAMPLE);

	while (token) {
		token = strsep(&content, sep);
		if (!token)
			break;
		sscanf(token, "t=%llu CH%lu(T=%llu)[%s%llu", &ts, &chan, &ts,
		       (char *)&chan_name, &value);
		printf("result: t=%llu CH%lu(T=%llu)[%s %llu\n", ts, chan, ts,
		       chan_name, value);
	}

	return 0;
}
