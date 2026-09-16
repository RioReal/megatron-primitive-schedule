#pragma once

#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace testing {

using TestFunc = void (*)();

struct TestCase {
  std::string suite;
  std::string name;
  TestFunc func = nullptr;
};

inline std::vector<TestCase>& Registry() {
  static std::vector<TestCase> tests;
  return tests;
}

inline int& FailureCount() {
  static int failures = 0;
  return failures;
}

inline std::string& TestFilter() {
  static std::string filter;
  return filter;
}

class Registrar {
 public:
  Registrar(const char* suite, const char* name, TestFunc func) {
    Registry().push_back(TestCase{suite, name, func});
  }
};

inline void AddFailure(const char* file, int line, const std::string& message) {
  ++FailureCount();
  std::cerr << file << ":" << line << ": Failure\n" << message << "\n";
}

inline int InitGoogleTest(int* argc, char** argv) {
  if (argc == nullptr || argv == nullptr) return 0;
  const std::string prefix = "--gtest_filter=";
  for (int i = 1; i < *argc; ++i) {
    const std::string arg = argv[i] == nullptr ? "" : argv[i];
    if (arg.rfind(prefix, 0) == 0) {
      TestFilter() = arg.substr(prefix.size());
    }
  }
  return 0;
}

inline bool MatchPattern(const std::string& text, const std::string& pattern) {
  std::size_t text_pos = 0;
  std::size_t pattern_pos = 0;
  std::size_t star_pos = std::string::npos;
  std::size_t match_pos = 0;
  while (text_pos < text.size()) {
    if (pattern_pos < pattern.size() &&
        pattern[pattern_pos] == text[text_pos]) {
      ++text_pos;
      ++pattern_pos;
    } else if (pattern_pos < pattern.size() && pattern[pattern_pos] == '*') {
      star_pos = pattern_pos++;
      match_pos = text_pos;
    } else if (star_pos != std::string::npos) {
      pattern_pos = star_pos + 1;
      text_pos = ++match_pos;
    } else {
      return false;
    }
  }
  while (pattern_pos < pattern.size() && pattern[pattern_pos] == '*') {
    ++pattern_pos;
  }
  return pattern_pos == pattern.size();
}

inline bool MatchesFilter(const TestCase& test) {
  const std::string filter = TestFilter();
  if (filter.empty()) return true;
  const std::string full_name = test.suite + "." + test.name;
  std::size_t start = 0;
  while (start <= filter.size()) {
    const std::size_t colon = filter.find(':', start);
    const std::string pattern = filter.substr(
        start, colon == std::string::npos ? std::string::npos : colon - start);
    if (!pattern.empty() && MatchPattern(full_name, pattern)) return true;
    if (colon == std::string::npos) break;
    start = colon + 1;
  }
  return false;
}

inline int RUN_ALL_TESTS() {
  int selected = 0;
  for (const TestCase& test : Registry()) {
    if (!MatchesFilter(test)) continue;
    ++selected;
    try {
      test.func();
    } catch (const std::exception& error) {
      AddFailure(test.suite.c_str(), 0,
                 test.name + " threw exception: " + error.what());
    }
  }
  if (FailureCount() == 0) {
    std::cout << "[  PASSED  ] " << selected << " tests.\n";
  }
  return FailureCount() == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace testing

#define TEST(SuiteName, TestName)                                             \
  void SuiteName##_##TestName##_Test();                                       \
  static ::testing::Registrar SuiteName##_##TestName##_Registrar(             \
      #SuiteName, #TestName, &SuiteName##_##TestName##_Test);                 \
  void SuiteName##_##TestName##_Test()

#define EXPECT_TRUE(condition)                                                \
  do {                                                                        \
    if (!(condition)) {                                                       \
      ::testing::AddFailure(__FILE__, __LINE__,                               \
                            std::string("Expected true: ") + #condition);    \
    }                                                                         \
  } while (false)

#define EXPECT_FALSE(condition) EXPECT_TRUE(!(condition))

#define EXPECT_EQ(actual, expected)                                           \
  do {                                                                        \
    const auto actual_value = (actual);                                       \
    const auto expected_value = (expected);                                   \
    if (!(actual_value == expected_value)) {                                  \
      std::ostringstream gtest_oss;                                           \
      gtest_oss << "Expected equality of " << #actual << " and "             \
                << #expected << ", actual=" << actual_value                  \
                << ", expected=" << expected_value;                          \
      ::testing::AddFailure(__FILE__, __LINE__, gtest_oss.str());             \
    }                                                                         \
  } while (false)

#define EXPECT_NE(actual, expected)                                           \
  do {                                                                        \
    const auto actual_value = (actual);                                       \
    const auto expected_value = (expected);                                   \
    if (actual_value == expected_value) {                                     \
      std::ostringstream gtest_oss;                                           \
      gtest_oss << "Expected inequality of " << #actual << " and "           \
                << #expected;                                                \
      ::testing::AddFailure(__FILE__, __LINE__, gtest_oss.str());             \
    }                                                                         \
  } while (false)

#define EXPECT_THROW(statement, exception_type)                               \
  do {                                                                        \
    bool gtest_threw = false;                                                 \
    try {                                                                     \
      statement;                                                              \
    } catch (const exception_type&) {                                         \
      gtest_threw = true;                                                     \
    } catch (...) {                                                           \
    }                                                                         \
    if (!gtest_threw) {                                                       \
      ::testing::AddFailure(__FILE__, __LINE__,                               \
                            std::string("Expected throw: ") + #statement);   \
    }                                                                         \
  } while (false)

#define ASSERT_TRUE(condition)                                                \
  do {                                                                        \
    if (!(condition)) {                                                       \
      ::testing::AddFailure(__FILE__, __LINE__,                               \
                            std::string("Expected true: ") + #condition);    \
      return;                                                                 \
    }                                                                         \
  } while (false)

#define ASSERT_EQ(actual, expected)                                           \
  do {                                                                        \
    const auto actual_value = (actual);                                       \
    const auto expected_value = (expected);                                   \
    if (!(actual_value == expected_value)) {                                  \
      std::ostringstream gtest_oss;                                           \
      gtest_oss << "Expected equality of " << #actual << " and "             \
                << #expected << ", actual=" << actual_value                  \
                << ", expected=" << expected_value;                          \
      ::testing::AddFailure(__FILE__, __LINE__, gtest_oss.str());             \
      return;                                                                 \
    }                                                                         \
  } while (false)
