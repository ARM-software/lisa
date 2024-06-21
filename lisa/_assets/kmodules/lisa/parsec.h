/* SPDX-License-Identifier: GPL-2.0 */

#ifndef _PARSEC_H
#define _PARSEC_H

#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/string.h>

/**
 * struct parse_buffer - Input buffer to parsers defined in that header.
 * @data: Pointer to the data.
 * @size: Size of the data buffer.
 * @capacity: Maximum size of the data pointed. This can be larger than @size
 *            when describing a window into the buffer.
 * @private: Pointer to additional data to be passed down the parser chain.
 */
typedef struct parse_buffer {
	u8 *data;
	size_t size;
	size_t capacity;
	void *private;
} parse_buffer;

/**
 * WITH_NULL_TERMINATED() - Call a function taking a const char * as first
 * parameter.
 * @buff: Buffer to be turned into a null-terminated string.
 * @f: Function to call.
 * @__VA_ARGS__: Extra parameters passed to the function.
 */
#define WITH_NULL_TERMINATED(buff, f, ...)				\
	({								\
		u8 *data;						\
		u8 *end;						\
		u8 last;						\
		bool allocate = (buff)->size + 1 > (buff)->capacity;	\
		if (unlikely(allocate)) {				\
			data = kmalloc((buff)->size + 1, GFP_KERNEL);	\
			BUG_ON(!data);					\
			end = data + (buff)->size;			\
			memcpy(data, (buff)->data, (buff)->size);	\
		} else {						\
			data = (buff)->data;				\
			end = data + (buff)->size;			\
			last = *end;					\
		}							\
		*end = '\0';						\
		typeof(f(NULL)) res = f((const char *)data, ##__VA_ARGS__); \
		if (unlikely(allocate))					\
			kfree(data);					\
		else							\
			*end = last;					\
		res;							\
	})

/**
 * charp2parse_buffer() - Convert a null-terminated string to struct parse_buffer.
 * @s: Null terminated string.
 * @private: Pointer to additional data to be passed down the parser chain.
 */
static inline parse_buffer charp2parse_buffer(char *s, void *private)
{
	size_t size = strlen(s) + 1;
	return (parse_buffer){ .data = (u8 *)s,
			       .private = private,
			       .size = size,
			       .capacity = size };
}

/**
 * parse_buffer2charp() - Copy a struct parse_buffer into a null-terminated string.
 * @src: Buffer to copy from.
 * @dst: Null terminated string to copy into.
 * @max: Maximum number of bytes to copy.
 *
 * The @dst string is guaranteed to be null-terminated after calling this function.
 */
static inline size_t parse_buffer2charp(parse_buffer *src, char *dst,
					size_t max)
{
	// Ensure we always provide a null-terminated string, even if the
	// buffer is empty.
	if (max)
		dst[0] = '\0';

	size_t to_copy = min(max, src->size);
	size_t to_zero = min(to_copy, max - 1);
	if (to_copy) {
		memcpy(dst, src->data, to_copy);
		dst[to_zero] = '\0';
	}
	return to_copy;
}

/**
 * parse_buffer_strdup() - Copy the parse_buffer into a newly allocated string.
 * @src: Buffer to copy from.
 *
 * The string string is guaranteed to be null-terminated.
 */
static inline char *parse_buffer_strdup(parse_buffer *src) {
	size_t size = src->size + 1;
	char *s = NULL;
	if (size) {
		s = kmalloc(size, GFP_KERNEL);
		if (s) {
			parse_buffer2charp(src, s, size);
		}
	}
	return s;
}

/**
 * enum parse_result_tag - Tag indicating the result of a parser.
 * @PARSE_SUCCESS: The parse was successful and the value parsed is therefore
 * meaningful.
 * @PARSE_FAILURE: The parse failed and the value should not be inspected.
 */
enum parse_result_tag {
	PARSE_SUCCESS,
	PARSE_FAILURE,
};

/**
 * PARSE_RESULT() - Wrap a type name to turn it into the result type of a parser.
 * @type: Type to wrap. This must be a single identifier, e.g. unsigned char
 * needs to be referred to as unchar instead.
 *
 * Note: Before being used with PARSE_RESULT(type), a wrapper type needs to be
 * defined using DEFINE_PARSE_RESULT_TYPE(type).
 */
#define PARSE_RESULT(type) parse_result_##type

/**
 * DEFINE_PARSE_RESULT_TYPE() - Define a wrapper so that @type can be used in PARSE_RESULT(@type).
 * @type: Type to wrap. This must be a single identifier, e.g. unsigned char
 * needs to be referred to as unchar instead.
 */
#define DEFINE_PARSE_RESULT_TYPE(type)                                         \
	typedef struct PARSE_RESULT(type) {                                    \
		parse_buffer remainder;                                        \
		enum parse_result_tag tag;                                     \
		type value;                                                    \
	} PARSE_RESULT(type)

/* Define only the types used in this header. Users are responsible for any
 * other type
 */
DEFINE_PARSE_RESULT_TYPE(parse_buffer);
DEFINE_PARSE_RESULT_TYPE(u8);
DEFINE_PARSE_RESULT_TYPE(u64);
DEFINE_PARSE_RESULT_TYPE(int);
DEFINE_PARSE_RESULT_TYPE(ulong);
typedef char *str;
DEFINE_PARSE_RESULT_TYPE(str);

/**
 * IS_SUCCESS() - Evaluates to true if the parse result is successful.
 * @res: Value of type PARSE_RESULT(type)
 */
#define IS_SUCCESS(res) (res.tag == PARSE_SUCCESS)

/**
 * parse_string() - Recognize the given string.
 * @input: parse_buffer input to the parser.
 * @match: String to match @input against.
 */
static inline PARSE_RESULT(parse_buffer)
	parse_string(parse_buffer *input, const char *match)
{
	size_t len = strlen(match);
	if (input->size < len) {
		return (PARSE_RESULT(parse_buffer)){ .tag = PARSE_FAILURE,
						     .remainder = *input };
	} else {
		if (!memcmp(input->data, match, len)) {
			return (PARSE_RESULT(
				parse_buffer)){ .tag = PARSE_SUCCESS,
						.remainder =
							(parse_buffer){
								.private = input->private,
								.data = input->data +
									len,
								.size = input->size -
									len,
								.capacity =
									input->capacity -
									len,
							},
						.value = {
							.private = input->private,
							.data = input->data,
							.size = len,
							.capacity =
								input->capacity,
						} };
		} else {
			return (PARSE_RESULT(parse_buffer)){
				.tag = PARSE_FAILURE,
				.remainder = *input,
			};
		}
	}
}

static inline PARSE_RESULT(u8)
	__parse_u8_in(parse_buffer *input, const bool lookup[256], bool revert)
{

	if (input->size) {
		u8 input_char = *input->data;
		bool success = lookup[(size_t)input_char];
		success = revert ? !success : success;
		success = input_char == '\0' ? false : success;

		if (success) {
			return (PARSE_RESULT(u8)){
				.tag = PARSE_SUCCESS,
				.remainder =
					(parse_buffer){
						.private = input->private,
						.data = input->data + 1,
						.size = input->size - 1,
						.capacity =
							input->capacity -
							1 },
				.value = input_char,
			};
		}
	}
	return (PARSE_RESULT(u8)){ .tag = PARSE_FAILURE, .remainder = *input };
}


// Build the lookup table in a way that allows the compiler to optimize-out the
// code so the table is initialized only once, given that the lookup table is
// actually stored in a static variable. Clang is very good at optimizing this
// out from experiments.
static inline __attribute__((always_inline)) void __init_char_lookup(const char *allowed, bool lookup[256]) {
	// Using strlen makes the pattern obvious to the compiler that
	// is then able to optimize-out the loop.
	size_t allowed_len = strlen(allowed);
	for (size_t i=0; i < allowed_len; i++) {
		lookup[(size_t)allowed[i]] = true;
	}
}


#define __CHAR_IN(name, allowed, revert) \
	static inline PARSE_RESULT(u8) name(parse_buffer *input) \
	{ \
		static bool lookup[256] = {0}; \
		__init_char_lookup((allowed), lookup); \
		return __parse_u8_in(input, lookup, (revert)); \
	}


/**
 * CHAR_IN() - Recognize one character in the set passed in @allowed.
 * @allowed: Null-terminated string containing the characters to recognize.
 */
#define CHAR_IN(name, allowed) __CHAR_IN(name, allowed, false)

/**
 * CHAR_NOT_IN() - Recognize one character not in the set passed in @allowed.
 * @allowed: Null-terminated string containing the characters to not recognize.
 */
#define CHAR_NOT_IN(name, allowed) __CHAR_IN(name, allowed, true)


static inline PARSE_RESULT(u8)
	__parse_u8(parse_buffer *input, u8 c, bool revert)
{
	if (input->size && (revert ? *input->data != c : *input->data == c)) {
		return (PARSE_RESULT(u8)){
			.tag = PARSE_SUCCESS,
			.remainder =
				(parse_buffer){ .private = input->private,
						.data = input->data + 1,
						.size = input->size - 1,
						.capacity =
							input->capacity - 1 },
			.value = c,
		};
	} else {
		return (PARSE_RESULT(u8)){ .tag = PARSE_FAILURE,
					   .remainder = *input };
	}
}

/**
 * parse_char() - Recognize one character equal to the one given.
 * @input: parse_buffer input to the parser.
 * @c: Character to recognize.
 */
static inline PARSE_RESULT(u8) parse_char(parse_buffer *input, u8 c)
{
	return __parse_u8(input, c, false);
}

/**
 * parse_not_char() - Recognize one character not equal to the one given.
 * @input: parse_buffer input to the parser.
 * @c: Character to not recognize.
 */
static inline PARSE_RESULT(u8) parse_not_char(parse_buffer *input, u8 c)
{
	return __parse_u8(input, c, true);
}

/**
 * APPLY() - Combinator to apply a function to some arguments.
 * @type: Return type of the parsers.
 * @name: Name of the new parser to create.
 * @parser: Function of type: (parse_buffer *input, ...) -> PARSE_RESULT(type)
 * @__VA_ARGS__: Parameters to pass to the @parser after the parse_buffer input.
 */
#define APPLY(type, name, parser, ...)                                         \
	static inline PARSE_RESULT(type) name(parse_buffer *input)             \
	{                                                                      \
		return parser(input, __VA_ARGS__);                             \
	}

/**
 * OR() - Combinator that tries @parser1 then @parser2 if @parser1 failed.
 * @type: Return type of the parsers.
 * @name: Name of the new parser to create.
 * @parser1: First parser to try.
 * @parser2: Second parser to try.
 */
#define OR(type, name, parser1, parser2)                                       \
	static inline PARSE_RESULT(type) name(parse_buffer *input)             \
	{                                                                      \
		PARSE_RESULT(type) res1 = parser1(input);                      \
		if (IS_SUCCESS(res1)) {                                        \
			return res1;                                           \
		} else {                                                       \
			return parser2(input);                                 \
		}                                                              \
	}

/**
 * PURE() - Create a parser that does not consume any input and returns @value.
 * @type: Return type of the parsers.
 * @name: Name of the new parser to create.
 * @value: Value to return.
 *
 * Note: This parser can be used e.g. to terminate a chain of OR() with a parser
 * that cannot fail and just provide a default value.
 */
#define PURE(type, name, _value)                                               \
	static inline PARSE_RESULT(type) name(parse_buffer *input)             \
	{                                                                      \
		return (PARSE_RESULT(type)){ .tag = PARSE_SUCCESS,             \
					     .remainder = *input,              \
					     .value = (_value) };              \
	}

/**
 * MAP() - Combinator that maps a function over the returned value of @parser
 * if it succeeded.
 * @f_type: Return type of the function.
 * @parser_type: Return type of the parser.
 * @name: Name of the new parser to create.
 * @parser: Parser to wrap.
 * @f: Function converting the parser's output.
 */
#define MAP(f_type, parser_type, name, parser, f)                              \
	static inline PARSE_RESULT(f_type) name(parse_buffer *input)           \
	{                                                                      \
		PARSE_RESULT(parser_type) res = parser(input);                 \
		if (IS_SUCCESS(res)) {                                         \
			return (PARSE_RESULT(                                  \
				f_type)){ .tag = PARSE_SUCCESS,                \
					  .remainder = res.remainder,          \
					  .value = f(res.value) };             \
		} else {                                                       \
			return (PARSE_RESULT(                                  \
				f_type)){ .tag = PARSE_FAILURE,                \
					  .remainder = res.remainder };        \
		}                                                              \
	}

/**
 * MAP_PARSE_BUFFER() - Similar to MAP() but specialized to struct parse_buffer.
 *
 * The function will be called with a null-terminated string provided by
 * WITH_NULL_TERMINATED().
 */
#define MAP_PARSE_BUFFER(f_type, parser_type, name, parser, f)                 \
	static inline typeof(f(NULL))                                          \
		__map_parse_buffer_##f(parse_buffer buff)                      \
	{                                                                      \
		return WITH_NULL_TERMINATED(&buff, f);                         \
	}                                                                      \
	MAP(f_type, parser_type, name, parser, __map_parse_buffer_##f)


static inline char *__strdup(const char *src) {
	if (src) {
		size_t size = strlen(src) + 1;
		char *dst = kmalloc(size, GFP_KERNEL);
		if (dst)
			memcpy(dst, src, size);
		return dst;
	} else {
		return NULL;
	}
}

/**
 * STRDUP() - Transforms a parser returning a parse_buffer to a parser
 * returning an char * allocated with kmalloc.
 */
#define STRDUP(name, parser) MAP_PARSE_BUFFER(str, parse_buffer, name, parser, __strdup)

/**
 * PEEK() - Combinator that applies the parser but does not consume any input.
 * @type: Return type of the parser.
 * @name: Name of the new parser to create.
 * @parser: Parser to wrap.
 * @__VA_ARGS__: Extra arguments to pass to the parser.
 */
#define PEEK(type, name, parser, ...)                                          \
	static inline PARSE_RESULT(type) name(parse_buffer *input)             \
	{                                                                      \
		PARSE_RESULT(type) res = parser(input, ##__VA_ARGS__);         \
		res.remainder = *input;                                        \
		return res;                                                    \
	}

/**
 * AT_LEAST() - Combinator that applies the parser at least @n times. If less
 * than @n times match, rewind the input.
 * @type: Return type of @f.
 * @name: Name of the new parser to create.
 * @parser: Parser to wrap.
 * @f: Function folded over the parser's output. Each time a parse is
 * successful, it will be called with 1. the accumulator's value and 2. the
 * value of the parse. The return value is fed back into the accumulator.
 * @__VA_ARGS__: Extra arguments to pass to the parser.
 *
 * Note: The resulting parser takes a 2nd parameter of type @type, which is the
 * initial value of the accumulator maintained by @f.
 */
#define AT_LEAST(type, name, parser, f, n, ...)                                \
	static inline PARSE_RESULT(type) name(parse_buffer *input, type init)  \
	{                                                                      \
		typeof(init) acc = init;                                       \
		typeof(parser(NULL)) res =                                     \
			(typeof(res)){ .remainder = *input };                  \
		for (size_t i = 0;; i++) {                                     \
			res = parser(&res.remainder, ##__VA_ARGS__);           \
			if (IS_SUCCESS(res)) {                                 \
				acc = f(acc, res.value);                       \
			} else {                                               \
				return (PARSE_RESULT(type)){                   \
					.tag = i >= n ? PARSE_SUCCESS :        \
							PARSE_FAILURE,         \
					.remainder = res.remainder,            \
					.value = acc                           \
				};                                             \
			}                                                      \
		}                                                              \
	}

/**
 * MANY() - Same as AT_LEAST() with n=0
 */
#define MANY(type, name, parser, f, ...) AT_LEAST(type, name, parser, f, 0, ##__VA_ARGS__)

/**
 * TAKEWHILE_AT_LEAST() - Combinator that applies @parser at least @n times and
 * returns a parse_buffer spanning the recognized input.
 * @type: Return type of @parser.
 * @name: Name of the new parser to create.
 * @parser: Parser to wrap.
 * @n: Minimum number of times @parser need to be successful in order to succeed.
 */
#define TAKEWHILE_AT_LEAST(type, name, parser, n)                                          \
	static inline PARSE_RESULT(parse_buffer) name(parse_buffer *input)                 \
	{                                                                                  \
		PARSE_RESULT(type)                                                         \
		res = (PARSE_RESULT(type)){ .remainder = *input };                         \
		for (size_t i = 0;; i++) {                                                 \
			res = parser(&res.remainder);                                      \
			if (!IS_SUCCESS(res)) {                                            \
				if (i >= n)                                                \
					return (PARSE_RESULT(parse_buffer)){               \
						.tag = PARSE_SUCCESS,                      \
						.value =                                   \
							(parse_buffer){                    \
								.private = input->private, \
								.data = input->data,       \
								.size = res.remainder      \
										.data -    \
									input->data,       \
								.capacity =                \
									input->capacity }, \
						.remainder = res.remainder,                \
					};                                                 \
				else                                                       \
					return (PARSE_RESULT(parse_buffer)){               \
						.tag = PARSE_FAILURE,                      \
						.remainder = *input,                       \
					};                                                 \
			}                                                                  \
		}                                                                          \
	}
/**
 * TAKEWHILE() - Same as TAKEWHILE_AT_LEAST() with n=0
 */
#define TAKEWHILE(type, name, parser) TAKEWHILE_AT_LEAST(type, name, parser, 0)

/**
 * COUNT_MANY() - Combinator that applies @parser until it fails and returns how many times it succeeded.
 * @parser_type: Return type of @parser.
 * @name: Name of the new parser to create.
 * @parser: Parser to wrap.
 */
#define COUNT_MANY(parser_type, name, parser)                                  \
	static inline int __count_fold_##name(int acc, int x)                  \
	{                                                                      \
		return acc + x;                                                \
	}                                                                      \
	static inline int __count_one_f_##name(parser_type _)                  \
	{                                                                      \
		return 1;                                                      \
	}                                                                      \
	MAP(int, parser_type, __count_one_##name, parser,                      \
	    __count_one_f_##name);                                             \
	MANY(int, __many_##name, __count_one_##name, __count_fold_##name)      \
	static inline PARSE_RESULT(int) name(parse_buffer *input)              \
	{                                                                      \
		return __many_##name(input, 0);                                \
	}

/**
 * THEN() - Combinator that applies @parser1, applies @parser2 with the return
 * value of @parser1 and returns the result of @parser2.
 * @parser1_type: Return type of @parser1.
 * @parser2_type: Return type of @parser2.
 * @name: Name of the new parser to create.
 * @parser1: First parser to apply.
 * @parser2: Second parser to apply and return the value of. It must take a 2nd
 * parameter that is the return type of @parser1.
 */
#define THEN(parser1_type, parser2_type, name, parser1, parser2)               \
	static inline PARSE_RESULT(parser2_type) name(parse_buffer *input)     \
	{                                                                      \
		PARSE_RESULT(parser1_type) res = parser1(input);               \
		PARSE_RESULT(parser2_type) res2;                               \
		if (IS_SUCCESS(res)) {                                         \
			res2 = parser2(&res.remainder, res.value);             \
			if (IS_SUCCESS(res2))                                  \
				return res2;                                   \
		}                                                              \
		return (PARSE_RESULT(parser2_type)){ .tag = PARSE_FAILURE,     \
						     .remainder = *input };    \
	}

/**
 * RIGHT() - Same as THEN() except @parser1 output is simply discarded, and is not passed to @parser2.
 */
#define RIGHT(parser1_type, parser2_type, name, parser1, parser2)              \
	PARSE_RESULT(parser2_type)                                             \
	static inline __discard_then_##parser2(parse_buffer *input,            \
					       parser1_type _)                 \
	{                                                                      \
		return parser2(input);                                         \
	}                                                                      \
	THEN(parser1_type, parser2_type, name, parser1,                        \
	     __discard_then_##parser2)

/**
 * LEFT() - Same as THEN() except @parser2 output is simply discarded,
 * and @parser1 result is returned instead.
 */
#define LEFT(parser1_type, parser2_type, name, parser1, parser2)               \
	PARSE_RESULT(parser1_type)                                             \
	static inline __forward_then_discard_##parser2(parse_buffer *input,    \
						       parser1_type value)     \
	{                                                                      \
		PARSE_RESULT(parser2_type) res = parser2(input);               \
		return (PARSE_RESULT(parser1_type)){ .tag = PARSE_SUCCESS,     \
						     .value = value,           \
						     .remainder =              \
							     res.remainder };  \
	}                                                                      \
	THEN(parser1_type, parser1_type, name, parser1,                        \
	     __forward_then_discard_##parser2)

/**
 * PARSE() - Apply a parser in the body of SEQUENCE().
 * @parser: Parser to apply
 * @__VA_ARGS__: Arguments to pass to the parser.
 */
#define PARSE(parser, ...)                                                     \
	({                                                                     \
		typeof(parser(&__seq_remainder, ##__VA_ARGS__)) res =          \
			parser(&__seq_remainder, ##__VA_ARGS__);               \
		__seq_remainder = res.remainder;                               \
		if (!IS_SUCCESS(res))                                          \
			goto __seq_failure;                                    \
		res.value;                                                     \
	})

/**
 * SEQUENCE() - Build a custom sequence of parsers in a more friendly way than
 * THEN().
 * @type: Return type of the sequence.
 * @name: Name of the new parser to create.
 * @body: Statement expr containing the custom logic. The last statement will be
 * the value returned by the parser. Private data from parse_buffer can be
 * accessed inside the body through the `private` variable.
 */
#define SEQUENCE(type, name, body, ...)                                        \
	static inline PARSE_RESULT(type)                                       \
		name(parse_buffer *input, ##__VA_ARGS__)                       \
	{                                                                      \
		parse_buffer __seq_remainder = *input;                         \
		parse_buffer __seq_unmodified_input = *input;                  \
		void *private = input->private;                                \
		(void)private;                                                 \
		type __seq_value = (body);                                     \
		return (PARSE_RESULT(type)){                                   \
			.tag = PARSE_SUCCESS,                                  \
			.remainder = __seq_remainder,                          \
			.value = __seq_value,                                  \
		};                                                             \
	__seq_failure:                                                         \
		return (PARSE_RESULT(type)){                                   \
			.tag = PARSE_FAILURE,                                  \
			.remainder = __seq_unmodified_input,                   \
		};                                                             \
	}

/**
 * DISCARD() - Discard the output of the given parser
 * @parser_type: Return type of @parser.
 * @name: Name of the new parser to create.
 * @parser: Parser to wrap.
 */
#define DISCARD(parser_type, name, parser) RIGHT(parser_type, void_t, name, parser, parse_void)

/*
 * Collection of common basic parsers.
 */
typedef struct {
	u8 __internal;
} void_t;
DEFINE_PARSE_RESULT_TYPE(void_t);
PURE(void_t, parse_void, (void_t){0});

CHAR_IN(parse_whitespace, " \n\t");
COUNT_MANY(u8, __count_whitespaces, parse_whitespace);
DISCARD(int, consume_whitespaces, __count_whitespaces);

static inline unsigned long __to_ulong(const char *s)
{
	unsigned long res;
	if (kstrtoul(s, 10, &res))
		return 0;
	else
		return res;
}

static inline unsigned long __to_u64(const char *s)
{
	u64 res;
	if (kstrtou64(s, 10, &res))
		return 0;
	else
		return res;
}

CHAR_IN(parse_digit, "0123456789");
TAKEWHILE_AT_LEAST(u8, parse_number_string, parse_digit, 1);
MAP_PARSE_BUFFER(ulong, parse_buffer, parse_ulong, parse_number_string,
		 __to_ulong);
MAP_PARSE_BUFFER(u64, parse_buffer, parse_u64, parse_number_string,
		 __to_u64);
#endif
