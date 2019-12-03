Current goal:
- write an encoder and decoder given a regex already separated by
    * 1: constant strings
    * 2: regex

- Encoder: works as follows:
    * Given a regex already split into pieces:
    ```
    <# regex pieces (n)> <id_0> <len_0> <regex_0> ... <id_n-1> <len_n-1> <regex_n-1>
    <# disjunctions (m)> <len_0> <disjunction_0> ... <len_m-1> <disjunction_m-1>
    ```
    * all numbers are gamma encoded


- Decoder:
    * Decodes gamma stream, reconstructs regex pieces and disjunctions, reconstructs stream

